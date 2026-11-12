"""事件采集（记忆的原料）。

设计原则（PRD FR-6.3）：
    **只记被采纳的产出**。用户重生成 = 负证据；改了某参数 = 该参数负证据。

事件是原始流水，量大、会过期，**不直接参与生成**。
记忆层（core/memory.py）从事件流蒸馏出结论，那才是生成时用的东西。

事件类型：
    save        保存/使用 → 正证据（最强的正信号）
    regenerate  重生成   → 负证据（说明上一次不合格）
    param_change 改参数  → 该参数值负证据 + 新值正证据候选
    asset_change 换素材  → 该素材负证据
    op_toggle   op 开关  → 该 op 有用/无用
    mem_delete  删记忆   → 明确负证据（用户主动否定）
    mem_edit    改记忆   → 人工修正，hand_edited=True

存储：memories/events.jsonl，一行一条，追加写。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent.parent
MEM_DIR = ROOT / "memories"
EVENTS_PATH = MEM_DIR / "events.jsonl"

KINDS = (
    "save", "regenerate", "param_change", "asset_change",
    "op_toggle", "mem_delete", "mem_edit",
)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _new_id() -> str:
    return "evt_" + uuid.uuid4().hex[:12]


@dataclass
class Event:
    """一条行为事件。"""

    id: str
    kind: str
    ts: str
    recipe_id: str | None = None
    """关联的配方 id（可为空，比如手工删记忆）"""

    base_id: str | None = None
    """基底 id"""

    subtype: str | None = None
    """子类型 precise/distort/stack/rewear"""

    payload: dict = field(default_factory=dict)
    """事件细节。不同 kind 结构不同：
         param_change: {op_index, op, path, old, new}
         asset_change: {op_index, op, old_id, new_id}
         op_toggle:    {op, enabled}
         mem_delete:   {memory_id, memory_type, memory_key}
         mem_edit:     {memory_id, field, old, new}
    """

    def as_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False)


class EventLog:
    """事件日志。追加写，按需读取。"""

    def __init__(self, path: Path = EVENTS_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- 写

    def append(
        self,
        kind: str,
        *,
        recipe_id: str | None = None,
        base_id: str | None = None,
        subtype: str | None = None,
        payload: dict | None = None,
        event_id: str | None = None,
        ts: str | None = None,
    ) -> Event:
        if kind not in KINDS:
            raise ValueError(f"未知事件类型 {kind}，允许: {list(KINDS)}")

        ev = Event(
            id=event_id or _new_id(),
            kind=kind,
            ts=ts or _now(),
            recipe_id=recipe_id,
            base_id=base_id,
            subtype=subtype,
            payload=payload or {},
        )
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(ev.to_json() + "\n")
        return ev

    # ---------------------------------------------------------- 读

    def iter_all(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # 单行损坏不影响整体（手工编辑过的情况）
                    continue

    def all(self) -> list[dict]:
        return list(self.iter_all())

    def count(self, kind: str | None = None) -> int:
        if kind is None:
            return sum(1 for _ in self.iter_all())
        return sum(1 for e in self.iter_all() if e.get("kind") == kind)

    def recent(self, n: int = 20) -> list[dict]:
        buf: list[dict] = []
        for e in self.iter_all():
            buf.append(e)
            if len(buf) > n:
                buf.pop(0)
        return buf

    def for_recipe(self, recipe_id: str) -> list[dict]:
        return [e for e in self.iter_all() if e.get("recipe_id") == recipe_id]

    # ---------------------------------------------------------- 维护

    def rewrite(self, events: list[dict]) -> None:
        """整表重写（用于压缩、清理）。原子替换。"""
        tmp = self.path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        tmp.replace(self.path)

    def compact(self, keep_days: int = 365, keep_max: int = 50000) -> dict:
        """压缩：删掉很久以前的事件，并限制总条数。

        Returns:
            {before, after, dropped_old, dropped_overflow}
        """
        evs = self.all()
        before = len(evs)
        if not evs:
            return {"before": 0, "after": 0, "dropped_old": 0, "dropped_overflow": 0}

        cutoff = datetime.now(timezone.utc).timestamp() - keep_days * 86400

        def _ts(e: dict) -> float:
            try:
                return datetime.fromisoformat(e.get("ts", "")).timestamp()
            except (ValueError, TypeError):
                return 0.0

        kept = [e for e in evs if _ts(e) >= cutoff]
        dropped_old = before - len(kept)

        dropped_overflow = 0
        if len(kept) > keep_max:
            # 保留最近 keep_max 条
            kept.sort(key=_ts)
            dropped_overflow = len(kept) - keep_max
            kept = kept[dropped_overflow:]

        if dropped_old or dropped_overflow:
            self.rewrite(kept)

        return {
            "before": before,
            "after": len(kept),
            "dropped_old": dropped_old,
            "dropped_overflow": dropped_overflow,
        }


# ---------------------------------------------------------------- 便捷函数

_default: EventLog | None = None


def log() -> EventLog:
    global _default
    if _default is None:
        _default = EventLog()
    return _default


def record_save(recipe: dict, *, log_: "EventLog | None" = None, **kw) -> Event:
    """记录「保存」——最强的正信号。"""
    lg = log_ or log()
    ops = recipe.get("ops", [])
    assets = [o.get("asset_id") for o in ops if o.get("asset_id")]
    labels = [o.get("text") for o in ops if o.get("op") == "label" and o.get("text")]
    return lg.append(
        "save",
        recipe_id=recipe.get("recipe_id"),
        base_id=recipe.get("base_id"),
        subtype=recipe.get("subtype"),
        payload={
            "recipe": recipe,
            "asset_ids": assets,
            "labels": labels,
            "slot": _infer_slot(ops),
        },
        **kw,
    )


def record_regenerate(recipe: dict, reason: str = "", *,
                      log_: "EventLog | None" = None, **kw) -> Event:
    """记录「重生成」——负信号。"""
    lg = log_ or log()
    ops = recipe.get("ops", [])
    return lg.append(
        "regenerate",
        recipe_id=recipe.get("recipe_id"),
        base_id=recipe.get("base_id"),
        subtype=recipe.get("subtype"),
        payload={
            "reason": reason,
            "asset_ids": [o.get("asset_id") for o in ops if o.get("asset_id")],
            "slot": _infer_slot(ops),
        },
        **kw,
    )


def record_param_change(
    recipe: dict,
    *,
    op_index: int,
    op: str,
    path: str,
    old: Any,
    new: Any,
    log_: "EventLog | None" = None,
    **kw,
) -> Event:
    """记录参数修改——定位到「哪个 op 的哪个参数」是最强的信息。"""
    lg = log_ or log()
    return lg.append(
        "param_change",
        recipe_id=recipe.get("recipe_id"),
        base_id=recipe.get("base_id"),
        subtype=recipe.get("subtype"),
        payload={
            "op_index": op_index, "op": op, "path": path,
            "old": old, "new": new,
        },
        **kw,
    )


def record_asset_change(
    recipe: dict,
    *,
    op_index: int,
    op: str,
    old_id: str | None,
    new_id: str,
    log_: "EventLog | None" = None,
    **kw,
) -> Event:
    lg = log_ or log()
    return lg.append(
        "asset_change",
        recipe_id=recipe.get("recipe_id"),
        base_id=recipe.get("base_id"),
        subtype=recipe.get("subtype"),
        payload={"op_index": op_index, "op": op,
                 "old_id": old_id, "new_id": new_id},
        **kw,
    )


def record_mem_delete(memory_id: str, memory_type: str, memory_key: str,
                      *, log_: "EventLog | None" = None, **kw) -> Event:
    lg = log_ or log()
    return lg.append(
        "mem_delete",
        payload={"memory_id": memory_id, "memory_type": memory_type,
                 "memory_key": memory_key},
        **kw,
    )


def record_mem_edit(memory_id: str, field_name: str, old: Any, new: Any,
                    *, log_: "EventLog | None" = None, **kw) -> Event:
    lg = log_ or log()
    return lg.append(
        "mem_edit",
        payload={"memory_id": memory_id, "field": field_name,
                 "old": old, "new": new},
        **kw,
    )


def _infer_slot(ops: list[dict]) -> str:
    """从 ops 推断使用的槽位。"""
    names = {o.get("op") for o in ops}
    if "face_fit" in names:
        return "face"
    if "feature_warp" in names:
        return "feature"
    if "texture_wrap" in names:
        return "texture"
    if "sticker" in names:
        return "object"
    return "any"
