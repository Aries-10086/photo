"""记忆蒸馏：事件流 -> 记忆条目。

这是「学习」的核心步骤。规则化实现（不用模型），因为：
    · 规则可解释、可调试
    · 数据量小时，规则比模型更稳
    · 结论要能被用户读懂，规则天然满足

六类记忆的蒸馏规则（对齐 schema 的 value_shapes）：

    combo        成功组合：asset × base × slot，来自 save 事件
    subtype_pref 惯用子类型：save 的 subtype 分布
    hand         手感参数：param_change 的 new 值（用户调整后保存的）
    caption      配文习惯：save 事件里 label 的文本长度/位置/语气
    base         可下手基底池：save 的 base_id 频次
    asset        素材偏好：save 的 asset_id 频次

写入门槛（PRD FR-6.3）：
    频次 ≥ 2  且  置信度 > 0.7
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Iterable

from core import events as EV
from core.memory import (
    WRITE_MIN_CONFIDENCE,
    WRITE_MIN_EVIDENCE,
    MemoryStore,
    confidence_from,
)

# 按类型的最小证据数。
# 手感参数门槛是 1：用户不会无意义地拖滑块，调一次就是明确信号。
# 组合类门槛是 2：怕偶然成功一次就当成规律。
MIN_EVIDENCE_BY_TYPE = {
    "combo": WRITE_MIN_EVIDENCE,
    "subtype_pref": WRITE_MIN_EVIDENCE,
    "hand": 1,
    "caption": WRITE_MIN_EVIDENCE,
    "base": WRITE_MIN_EVIDENCE,
    "asset": WRITE_MIN_EVIDENCE,
}

# 按类型的最小置信度。
# 手感参数证据少时置信度天然低（1 次 = 2/3），所以门槛也放宽。
MIN_CONF_BY_TYPE = {
    "hand": 0.6,
}


def _min_evidence(mtype: str) -> int:
    return MIN_EVIDENCE_BY_TYPE.get(mtype, WRITE_MIN_EVIDENCE)


def _min_conf(mtype: str) -> float:
    return MIN_CONF_BY_TYPE.get(mtype, WRITE_MIN_CONFIDENCE)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------- 单类蒸馏


def distill_combos(evs: list[dict]) -> list[dict]:
    """成功组合：哪些「素材 × 基底 × 槽位」被保存过。

    正证据 = save；负证据 = regenerate（同一组合的重生成次数）。
    """
    pos: Counter = Counter()
    neg: Counter = Counter()
    first_ev: dict[str, str] = {}
    sample: dict[str, dict] = {}

    for e in evs:
        kind = e.get("kind")
        p = e.get("payload") or {}

        if kind == "save":
            assets = p.get("asset_ids") or []
            slot = p.get("slot", "any")
            base = e.get("base_id")
            if not assets or not base:
                continue
            for a in assets:
                key = f"asset:{a} × base:{base} × slot:{slot}"
                pos[key] += 1
                first_ev.setdefault(key, e.get("id", ""))
                sample.setdefault(key, {
                    "base_id": base,
                    "asset_ids": [a],
                    "slot": slot,
                    "subtype": e.get("subtype"),
                    "sample_recipe_id": e.get("recipe_id"),
                })

        elif kind == "regenerate":
            assets = p.get("asset_ids") or []
            base = e.get("base_id")
            if not assets or not base:
                continue
            # 从 save 事件里已经见过的 slot 推断该基底用过哪些槽位，
            # 只对这些真实存在的 slot 记负证据（避免污染无关 key）
            slots_seen = p.get("slot")
            if slots_seen:
                slots_iter = [slots_seen]
            else:
                slots_iter = sorted({k.rsplit("slot:", 1)[-1]
                                     for k in pos if f"base:{base}" in k})
                if not slots_iter:
                    slots_iter = ["any"]
            for a in assets:
                for slot in slots_iter:
                    key = f"asset:{a} × base:{base} × slot:{slot}"
                    if key in pos:
                        neg[key] += 1

    out = []
    for key, n_pos in pos.items():
        n_neg = neg.get(key, 0)
        conf = confidence_from(n_pos, n_neg)
        out.append({
            "key": key,
            "value": sample[key],
            "positive": n_pos,
            "negative": n_neg,
            "confidence": conf,
            "event_id": first_ev.get(key),
        })
    return out


def distill_subtype_pref(evs: list[dict]) -> list[dict]:
    """惯用子类型：save 事件的 subtype 分布。

    只输出占比 > 25% 的（避免把偶发当成偏好）。
    """
    c: Counter = Counter()
    ev_id: dict[str, str] = {}
    for e in evs:
        if e.get("kind") != "save":
            continue
        st = e.get("subtype")
        if not st:
            continue
        c[st] += 1
        ev_id.setdefault(st, e.get("id", ""))

    total = sum(c.values())
    if total == 0:
        return []

    out = []
    for st, n in c.most_common():
        ratio = n / total
        if ratio < 0.25:
            continue
        conf = confidence_from(n, total - n)
        out.append({
            "key": f"subtype_pref:{st}",
            "value": {"subtype": st, "ratio": round(ratio, 3), "count": n},
            "positive": n,
            "negative": total - n,
            "confidence": conf,
            "event_id": ev_id.get(st),
        })
    return out


# 不进入「手感」记忆的参数：这些是配方结构，不是个人手感。
# 学了它们只会让记忆变脏（每个配方都有 label.text，但内容各不相同）。
HAND_SKIP_PARAMS = {
    "label.text",      # 每次内容不同，不是手感
    "label.anchor",    # 位置偏好归 caption 管
    "label.font",
    "label.color",
    "label.stroke_color",
    "degrade.enabled",  # 开关状态，不是手感
    "asset_id",
    "mask",
    "subject_mask",
    "text",
}


def _is_hand_param(op: str, path: str) -> bool:
    """判断某个 op 参数是否值得作为「手感」记忆。

    标准：它是**连续可调**的数值或明确的选择项，
    且不是每次都会变的文本/标识。
    """
    key = f"{op}.{path}"
    if key in HAND_SKIP_PARAMS:
        return False
    if path in ("asset_id", "mask", "subject_mask"):
        return False
    return True


def distill_hands(evs: list[dict]) -> list[dict]:
    """手感参数：用户最后采纳的参数值。

    证据来源有两类（都算）：
      1. param_change 的 new 值 —— 用户主动调过
      2. save 的配方里该参数的当前值 —— 用户直接用了这个值

    为什么阈值与 combo 不同：
      combo 需要重复验证（怕偶然成功），而**参数被调一次就是明确信号**，
      用户不会无意义地拖滑块。所以手感参数的门槛是 1 次。

    过滤：只学「手感类」参数（数值/比例/模式），跳过文本、素材 id、
    开关状态等配方结构字段。见 HAND_SKIP_PARAMS。
    """
    # recipe_id -> base 值快照（来自 save 的配方 + param_change 覆盖）
    adopted: Counter = Counter()      # (op, path, repr(value)) -> 采纳次数
    changed: Counter = Counter()      # (op, path, repr(value)) -> 调整次数
    ev_of: dict[tuple, str] = {}
    last_change: dict[str, dict] = defaultdict(dict)   # rid -> {(op,path): (new, eid)}

    for e in evs:
        rid = e.get("recipe_id")
        if not rid:
            continue
        kind = e.get("kind")
        p = e.get("payload") or {}

        if kind == "param_change":
            op, path = p.get("op", ""), p.get("path", "")
            if not op or not path or not _is_hand_param(op, path):
                continue
            last_change[rid][(op, path)] = (p.get("new"), e.get("id", ""))

        elif kind == "save":
            rec = p.get("recipe") or {}
            base_vals: dict[tuple[str, str], Any] = {}

            # 从配方里取当前值（这是「采纳值」的可靠来源）
            for op_obj in rec.get("ops", []):
                opname = op_obj.get("op", "")
                for k, v in op_obj.items():
                    if k == "op":
                        continue
                    if not _is_hand_param(opname, k):
                        continue
                    base_vals[(opname, k)] = v
                # degrade 的参数在 params 里
                if opname == "degrade" and isinstance(op_obj.get("params"), dict):
                    for k, v in op_obj["params"].items():
                        if _is_hand_param("degrade", k):
                            base_vals[("degrade", k)] = v

            # 被 param_change 改过的以改动值为准
            for (op, path), (new, eid) in last_change.get(rid, {}).items():
                base_vals[(op, path)] = new
                changed[(op, path, repr(new))] += 1
                ev_of.setdefault((op, path, repr(new)), eid)

            for (op, path), v in base_vals.items():
                if v is None or isinstance(v, (dict, list)):
                    continue
                k = (op, path, repr(v))
                adopted[k] += 1
                ev_of.setdefault(k, e.get("id", ""))

    out = []
    for (op, path, val_repr), n_saved in adopted.items():
        n_changed = changed.get((op, path, val_repr), 0)
        # 证据 = 采纳次数 + 主动调整次数。
        # 每次保存都是独立确认（用户看着结果点了保存），都要计入。
        evidence = n_saved + n_changed
        out.append({
            "key": f"hand:{op}.{path}",
            "value": {
                "op": op, "param_path": path, "value": val_repr,
                "count": n_saved, "changes": n_changed,
            },
            "positive": evidence,
            "negative": 0,
            "confidence": confidence_from(evidence, 0),
            "event_id": ev_of.get((op, path, val_repr)),
        })
    return out


def distill_captions(evs: list[dict]) -> list[dict]:
    """配文习惯：字数、位置、语气。"""
    lengths: list[int] = []
    anchors: Counter = Counter()
    samples: list[str] = []
    ev_id = ""

    for e in evs:
        if e.get("kind") != "save":
            continue
        p = e.get("payload") or {}
        labels = p.get("labels") or []
        recipe = p.get("recipe") or {}
        for ch in labels:
            if not isinstance(ch, str) or not ch.strip():
                continue
            lengths.append(len(ch))
            samples.append(ch)
            ev_id = ev_id or e.get("id", "")
        for op in recipe.get("ops", []):
            if op.get("op") == "label" and op.get("anchor"):
                anchors[op["anchor"]] += 1

    if len(lengths) < WRITE_MIN_EVIDENCE:
        return []

    avg_len = sum(lengths) / len(lengths)
    top_anchor = anchors.most_common(1)[0][0] if anchors else "bottom_right"

    return [{
        "key": "caption:habit",
        "value": {
            "avg_len": round(avg_len, 1),
            "max_len": max(lengths),
            "position": top_anchor,
            "samples": samples[-8:],
            "count": len(lengths),
        },
        "positive": len(lengths),
        "negative": 0,
        "confidence": confidence_from(len(lengths), 0),
        "event_id": ev_id,
    }]


def distill_bases(evs: list[dict]) -> list[dict]:
    """可下手基底池：哪些基底被反复使用。"""
    c: Counter = Counter()
    ev_of: dict[str, str] = {}
    for e in evs:
        if e.get("kind") != "save":
            continue
        b = e.get("base_id")
        if not b:
            continue
        c[b] += 1
        ev_of.setdefault(b, e.get("id", ""))

    out = []
    for b, n in c.items():
        out.append({
            "key": f"base:{b}",
            "value": {"base_id": b, "use_count": n},
            "positive": n,
            "negative": 0,
            "confidence": confidence_from(n, 0),
            "event_id": ev_of.get(b),
        })
    return out


def distill_assets(evs: list[dict]) -> list[dict]:
    """素材偏好：哪些素材被反复使用。"""
    c: Counter = Counter()
    ev_of: dict[str, str] = {}
    for e in evs:
        if e.get("kind") != "save":
            continue
        p = e.get("payload") or {}
        for a in (p.get("asset_ids") or []):
            c[a] += 1
            ev_of.setdefault(a, e.get("id", ""))

    out = []
    for a, n in c.items():
        out.append({
            "key": f"asset:{a}",
            "value": {"asset_id": a, "usable_count": n},
            "positive": n,
            "negative": 0,
            "confidence": confidence_from(n, 0),
            "event_id": ev_of.get(a),
        })
    return out


DISTILLERS = {
    "combo": distill_combos,
    "subtype_pref": distill_subtype_pref,
    "hand": distill_hands,
    "caption": distill_captions,
    "base": distill_bases,
    "asset": distill_assets,
}


# ---------------------------------------------------------------- 主流程


def learn(
    evs: list[dict] | None = None,
    *,
    store: MemoryStore | None = None,
    kinds: Iterable[str] | None = None,
    dry_run: bool = False,
) -> dict:
    """从事件流蒸馏并写入记忆库。

    Returns:
        报告：{candidates, written, skipped, by_type}
    """
    if evs is None:
        evs = EV.log().all()
    if store is None:
        from core import memory
        store = memory.store()

    want = list(kinds) if kinds else list(DISTILLERS)

    report = {
        "events_total": len(evs),
        "candidates": 0,
        "written": 0,
        "skipped_low_evidence": 0,
        "skipped_low_confidence": 0,
        "by_type": {},
        "details": [],
    }

    for mtype in want:
        fn = DISTILLERS.get(mtype)
        if fn is None:
            continue
        cands = fn(evs)
        report["candidates"] += len(cands)
        w = 0

        for c in cands:
            n_ev = c["positive"] + c["negative"]
            conf = float(c["confidence"])
            exists = store.find(mtype, c["key"]) is not None

            # 双阈值只约束「新建」。已存在的条目必须继续更新，
            # 否则负证据永远写不进去（置信度一降就没人管了）。
            if not exists:
                min_ev = _min_evidence(mtype)
                min_cf = _min_conf(mtype)
                if n_ev < min_ev:
                    report["skipped_low_evidence"] += 1
                    report["details"].append({
                        "type": mtype, "key": c["key"], "action": "skip",
                        "reason": f"evidence {n_ev} < {min_ev}"})
                    continue
                if conf < min_cf:
                    report["skipped_low_confidence"] += 1
                    report["details"].append({
                        "type": mtype, "key": c["key"], "action": "skip",
                        "reason": f"confidence {conf:.2f} < {min_cf}"})
                    continue

            if not dry_run:
                store.upsert(
                    mtype, c["key"], c["value"],
                    event_id=c.get("event_id"),
                    positive=c["positive"],
                    negative=c["negative"],
                )
            w += 1
            report["details"].append({
                "type": mtype, "key": c["key"],
                "action": "update" if exists else "write",
                "confidence": round(conf, 3), "evidence": n_ev})

        report["written"] += w
        report["by_type"][mtype] = w

    if not dry_run and report["written"]:
        store.save()

    return report


def relearn_full(*, store: MemoryStore | None = None) -> dict:
    """全量重学：保留人工修正，其余从事件流重建。"""
    if store is None:
        from core import memory
        store = memory.store()

    kept = {mid: m for mid, m in store._data.items() if m.get("hand_edited")}
    store._data = dict(kept)
    rep = learn(store=store)
    rep["preserved_hand_edited"] = len(kept)
    return rep
