"""记忆层（定义层）——项目的核心卖点。

从事件流蒸馏出「稳定结论」，存成一个人能读懂、能手改的文件。

三层职责：
    1. 蒸馏   events -> 候选记忆
    2. 写入   双阈值过滤（频次 ≥2 + 置信度 >0.7）
    3. 淘汰   低置信归档 / 时间衰减 / 显式删除

设计要点（PRD FR-6）：
    · 双阈值：频次 + 置信度，缺一不可
    · 只记被采纳的产出：重生成 = 负证据
    · 冲突不覆盖，加条件（conditions）并存
    · 淘汰是功能，不是可选项
    · 全程可追溯到 event id

存储：memories/memory.yaml —— 单文件，可手改，可备份。
"""

from __future__ import annotations

import copy
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parent.parent
MEM_DIR = ROOT / "memories"
MEM_PATH = MEM_DIR / "memory.yaml"

# ---------------------------------------------------------------- 阈值

WRITE_MIN_EVIDENCE = 2      # 频次阈值
WRITE_MIN_CONFIDENCE = 0.7  # 置信度阈值
ARCHIVE_BELOW = 0.3         # 低于此值归档
DECAY_HALFLIFE_DAYS = 120   # 时间衰减半衰期
STALE_DAYS = 180            # 超过这么久未确认，权重压到 0.3

# 事件对置信度的贡献（正/负）
KIND_WEIGHT = {
    "save": +1.0,
    "regenerate": -0.8,
    "param_change": -0.5,   # 参数被改 = 原值错
    "asset_change": -0.6,
    "op_toggle": -0.3,
    "mem_edit": 0.0,
    "mem_delete": 0.0,
}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _parse_ts(s: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _days_since(s: str) -> float:
    dt = _parse_ts(s)
    if dt is None:
        return 9999.0
    return max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 86400)


def _new_mem_id(mtype: str) -> str:
    return f"mem_{mtype}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------- 置信度模型


def confidence_from(positive: int, negative: int) -> float:
    """从正负证据数算置信度。

    用「平滑比率」而不是简单相除，避免 1 次成功就 100%：
        conf = (pos + 1) / (pos + neg + 2)
    0 证据 = 0.5（中立）；1 正 0 负 = 0.67；2 正 0 负 = 0.75；
    3 正 1 负 = 0.67；5 正 0 负 = 0.875。
    """
    return (positive + 1.0) / (positive + negative + 2.0)


def decay_weight(base_weight: float, last_confirmed: str) -> float:
    """时间衰减。半衰期 DECAY_HALFLIFE_DAYS 天。

    超过 STALE_DAYS 天未确认，直接压到 0.3（PRD 要求）。
    """
    days = _days_since(last_confirmed)
    if days >= STALE_DAYS:
        return min(base_weight, 0.3)
    factor = 0.5 ** (days / DECAY_HALFLIFE_DAYS)
    return max(0.0, min(2.0, base_weight * factor))


# ---------------------------------------------------------------- 记忆库


class MemoryStore:
    """记忆库。单文件 YAML，支持手改。"""

    def __init__(self, path: Path = MEM_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = {}
        self.load()

    # ---------------------------------------------------------- IO

    def load(self) -> None:
        if not self.path.exists():
            self._data = {}
            return
        try:
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"记忆文件格式错误（请检查 YAML 缩进）: {e}") from e

        items = raw.get("memories", []) if isinstance(raw, dict) else []
        self._data = {}
        for m in items:
            if isinstance(m, dict) and m.get("id"):
                self._data[m["id"]] = self._normalize(m)

    def save(self) -> None:
        # 按 type 分组，方便人读
        order = ["combo", "subtype_pref", "hand", "caption", "base", "asset"]
        items = sorted(
            self._data.values(),
            key=lambda m: (order.index(m["type"]) if m["type"] in order else 99,
                           -float(m.get("confidence", 0))))

        doc = {
            "_note": [
                "这是牢图棒的记忆文件。可以直接手改。",
                "改动会被视为人工修正（hand_edited=true），自动流程不再覆盖。",
                "confidence < 0.3 的条目会被归档，不参与生成。",
                f"最后更新: {_now()}",
            ],
            "stats": self.stats(),
            "memories": items,
        }
        tmp = self.path.with_suffix(".yaml.tmp")
        tmp.write_text(
            yaml.safe_dump(doc, allow_unicode=True, sort_keys=False,
                           default_flow_style=False, width=100),
            encoding="utf-8")
        tmp.replace(self.path)

    def _normalize(self, m: dict) -> dict:
        out = copy.deepcopy(m)
        out.setdefault("status", "active")
        out.setdefault("conditions", [])
        out.setdefault("source_events", [])
        out.setdefault("positive_count", 0)
        out.setdefault("negative_count", 0)
        out.setdefault("evidence_count", out.get("positive_count", 0) +
                       out.get("negative_count", 0))
        out.setdefault("hand_edited", False)
        out.setdefault("created_at", _now())
        out.setdefault("last_confirmed", out.get("created_at", _now()))
        out.setdefault("weight", 1.0)
        out["confidence"] = float(out.get("confidence", 0.5))
        return out

    # ---------------------------------------------------------- 查询

    def all(self, *, status: str | None = "active") -> list[dict]:
        items = list(self._data.values())
        if status:
            items = [m for m in items if m.get("status") == status]
        return items

    def get(self, mid: str) -> dict | None:
        return self._data.get(mid)

    def find(self, mtype: str, key: str) -> dict | None:
        for m in self._data.values():
            if m.get("type") == mtype and m.get("key") == key:
                return m
        return None

    def by_type(self, mtype: str, *, status: str | None = "active") -> list[dict]:
        return [m for m in self.all(status=status) if m.get("type") == mtype]

    def effective_weight(self, m: dict) -> float:
        """实际权重。

        设计：
          base_weight  存在 m["weight"]，默认 1.0，prune() 会把它更新为衰减值
          recency      由 last_confirmed 单独算

        最终 = base_weight × recency，但只衰减一次：
          若 m["weight"] 已经是衰减结果（< 1.0），直接用它，
          否则用 recency 现算。
        """
        w = float(m.get("weight", 1.0))
        if w < 0.999:
            return w  # prune 已经算过衰减
        return decay_weight(w, m.get("last_confirmed", m.get("created_at", _now())))

    def stats(self) -> dict:
        active = self.all(status="active")
        archived = [m for m in self._data.values() if m.get("status") == "archived"]
        by_type: dict[str, int] = {}
        for m in active:
            by_type[m["type"]] = by_type.get(m["type"], 0) + 1

        avg_conf = (sum(float(m.get("confidence", 0)) for m in active) / len(active)
                    if active else 0.0)
        return {
            "total": len(self._data),
            "active": len(active),
            "archived": len(archived),
            "by_type": by_type,
            "avg_confidence": round(avg_conf, 3),
        }

    # ---------------------------------------------------------- 写入

    def upsert(
        self,
        mtype: str,
        key: str,
        value: Any,
        *,
        event_id: str | None = None,
        positive: int = 0,
        negative: int = 0,
        condition: str | None = None,
        hand_edited: bool = False,
        absolute: bool = True,
    ) -> dict:
        """插入或更新一条记忆。

        Args:
            absolute: True（默认）表示 positive/negative 是**绝对值**，
                      直接覆盖（幂等，重复 learn 不会累加）；
                      False 表示增量叠加。
            hand_edited: 若已有记忆是人工改过的，不覆盖其值。

        Returns:
            记忆条目。
        """
        existing = self.find(mtype, key)

        if existing is None:
            m = {
                "id": _new_mem_id(mtype),
                "type": mtype,
                "key": key,
                "value": value,
                "confidence": confidence_from(positive, negative),
                "evidence_count": positive + negative,
                "positive_count": positive,
                "negative_count": negative,
                "conditions": [condition] if condition else ["default"],
                "status": "active",
                "weight": 1.0,
                "source_events": [event_id] if event_id else [],
                "hand_edited": hand_edited,
                "created_at": _now(),
                "last_confirmed": _now(),
            }
            self._data[m["id"]] = m
            return m

        # ---- 已存在 ----
        if absolute:
            new_pos = int(positive)
            new_neg = int(negative)
        else:
            new_pos = int(existing.get("positive_count", 0)) + positive
            new_neg = int(existing.get("negative_count", 0)) + negative

        if existing.get("hand_edited") and not hand_edited:
            # 人工改过的：只更新证据数，不动 confidence / value
            existing["positive_count"] = new_pos
            existing["negative_count"] = new_neg
            existing["evidence_count"] = new_pos + new_neg
            if event_id and event_id not in existing.get("source_events", []):
                existing.setdefault("source_events", []).append(event_id)
                existing["source_events"] = existing["source_events"][-40:]
            return existing

        existing["positive_count"] = new_pos
        existing["negative_count"] = new_neg
        existing["evidence_count"] = new_pos + new_neg
        existing["confidence"] = confidence_from(new_pos, new_neg)

        if condition and condition not in existing.setdefault("conditions", []):
            existing["conditions"].append(condition)

        if event_id and event_id not in existing.get("source_events", []):
            existing.setdefault("source_events", []).append(event_id)
            existing["source_events"] = existing["source_events"][-40:]

        existing["last_confirmed"] = _now()
        existing["weight"] = 1.0

        # 值不同 → 以最新值为准，旧值存进 variants（可回溯）
        if value != existing.get("value"):
            existing.setdefault("variants", [])
            old_v = {"value": existing["value"],
                     "condition": (existing.get("conditions") or ["default"])[0],
                     "at": existing.get("last_confirmed", _now()),
                     "superseded": True}
            if old_v["value"] not in [x.get("value") for x in existing["variants"]]:
                existing["variants"].append(old_v)
                existing["variants"] = existing["variants"][-10:]
            # 主值更新为最新
            existing["value"] = value

        return existing

    # ---------------------------------------------------------- 淘汰

    def archive(self, mid: str, reason: str = "") -> bool:
        m = self._data.get(mid)
        if not m:
            return False
        m["status"] = "archived"
        m["archived_at"] = _now()
        if reason:
            m["archive_reason"] = reason
        return True

    def restore(self, mid: str) -> bool:
        m = self._data.get(mid)
        if not m:
            return False
        m["status"] = "active"
        m["last_confirmed"] = _now()
        m.pop("archived_at", None)
        m.pop("archive_reason", None)
        return True

    def delete(self, mid: str) -> bool:
        return self._data.pop(mid, None) is not None

    def prune(self) -> dict:
        """执行淘汰规则（PRD FR-6.4）。

        Returns:
            {archived: [...], recovered: [...], decayed: [...]}
        """
        archived, recovered, decayed = [], [], []

        for m in self._data.values():
            if m.get("hand_edited"):
                continue  # 人工改过的不自动淘汰

            conf = float(m.get("confidence", 0))
            w = self.effective_weight(m)
            m["weight"] = round(w, 3)

            if m.get("status") == "active":
                if conf < ARCHIVE_BELOW:
                    m["status"] = "archived"
                    m["archived_at"] = _now()
                    m["archive_reason"] = f"confidence {conf:.2f} < {ARCHIVE_BELOW}"
                    archived.append(m["id"])
                elif w < 0.35:
                    decayed.append(m["id"])

        return {"archived": archived, "recovered": recovered, "decayed": decayed}

    # ---------------------------------------------------------- 人工编辑

    def edit(self, mid: str, field_name: str, new: Any) -> dict | None:
        m = self._data.get(mid)
        if not m:
            return None
        old = m.get(field_name)
        m[field_name] = new
        m["hand_edited"] = True
        m["edited_at"] = _now()
        m["edit_history"] = m.get("edit_history", [])[-9:] + [
            {"field": field_name, "old": old, "new": new, "at": _now()}]
        return m

    def mark_hand_edited(self, mid: str) -> None:
        m = self._data.get(mid)
        if m:
            m["hand_edited"] = True


# ---------------------------------------------------------------- 全局单例

_default: MemoryStore | None = None


def store() -> MemoryStore:
    global _default
    if _default is None:
        _default = MemoryStore()
    return _default


def reset() -> None:
    """测试用：清空单例。"""
    global _default
    _default = None
