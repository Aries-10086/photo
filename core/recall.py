"""记忆检索与建议。

生成时用到哪条记忆、按什么顺序用，全在这里。

核心函数：
    suggest(recipe) -> Suggestion
        给定一份正在编辑的配方，返回：
          · 推荐参数（来自手感记忆）
          · 推荐素材/基底（来自组合记忆）
          · 默认子类型（来自偏好记忆）
          · 每条建议的来源（memory id），支持「为什么这么出」

设计原则（PRD FR-7）：
    每次生成必须能回答「为什么这么出」——列出用到的记忆条目 id。
    所以所有建议都带 source，不允许凭空给值。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.memory import MemoryStore


@dataclass
class Item:
    """一条建议。"""

    kind: str
    """param / asset / base / subtype / caption / combo"""

    key: str
    value: Any
    confidence: float
    weight: float
    memory_id: str
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "kind": self.kind, "key": self.key, "value": self.value,
            "confidence": round(self.confidence, 3),
            "weight": round(self.weight, 3),
            "memory_id": self.memory_id, "reason": self.reason,
        }


@dataclass
class Suggestion:
    """一次建议的完整结果。"""

    items: list[Item] = field(default_factory=list)
    default_subtype: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    """可直接套进配方的参数：{ "face_fit.scale": 1.06, ... }"""

    bases: list[dict] = field(default_factory=list)
    assets: list[dict] = field(default_factory=list)
    caption: dict | None = None

    def as_dict(self) -> dict:
        return {
            "items": [i.as_dict() for i in self.items],
            "default_subtype": self.default_subtype,
            "params": self.params,
            "bases": self.bases,
            "assets": self.assets,
            "caption": self.caption,
            "why_count": len(self.items),
        }

    def why(self) -> list[dict]:
        """「为什么这么出」：只列真正影响了结果的条目。"""
        return [i.as_dict() for i in self.items]


def _eff(store: MemoryStore, m: dict) -> float:
    """实际权重。统一走 store.effective_weight，保证与淘汰逻辑一致。"""
    return store.effective_weight(m)


def suggest(
    recipe: dict | None = None,
    *,
    store: MemoryStore | None = None,
    min_weight: float = 0.4,
    limit: int = 12,
) -> Suggestion:
    """根据当前配方给出建议。

    Args:
        recipe: 正在编辑的配方（可为 None，此时只给全局偏好）。
        store: 记忆库。
        min_weight: 低于此权重的记忆不参与。
        limit: 最多返回多少条建议。

    Returns:
        Suggestion
    """
    if store is None:
        from core import memory
        store = memory.store()

    sug = Suggestion()
    recipe = recipe or {}
    cur_subtype = recipe.get("subtype")
    cur_base = recipe.get("base_id")
    cur_assets = {o.get("asset_id") for o in recipe.get("ops", [])
                  if o.get("asset_id")}

    def usable(m: dict) -> bool:
        return (m.get("status") == "active") and _eff(store, m) >= min_weight

    # ---------------------------------------------- 1. 惯用子类型
    prefs = [m for m in store.by_type("subtype_pref") if usable(m)]
    prefs.sort(key=lambda m: (float(m.get("confidence", 0)),
                              _eff(store, m)), reverse=True)
    if prefs:
        top = prefs[0]
        st = (top.get("value") or {}).get("subtype")
        if st:
            sug.items.append(Item(
                kind="subtype", key=top["key"], value=st,
                confidence=float(top.get("confidence", 0)),
                weight=_eff(store, top), memory_id=top["id"],
                reason="你最常保存这类子类型"))
            # 始终给出偏好，由调用方决定是否覆盖当前值
            sug.default_subtype = st

    # ---------------------------------------------- 2. 手感参数
    hands = [m for m in store.by_type("hand") if usable(m)]
    hands.sort(key=lambda m: (_eff(store, m), float(m.get("confidence", 0))),
               reverse=True)
    for m in hands[:limit]:
        v = m.get("value") or {}
        op = v.get("op")
        path = v.get("param_path")
        if not op or not path:
            continue
        try:
            val = eval(v.get("value"), {"__builtins__": {}}, {})  # noqa: S307
        except Exception:  # noqa: BLE001
            val = v.get("value")
        sug.params[f"{op}.{path}"] = val
        sug.items.append(Item(
            kind="param", key=f"{op}.{path}", value=val,
            confidence=float(m.get("confidence", 0)),
            weight=_eff(store, m), memory_id=m["id"],
            reason=f"你调过 {op}.{path} 并保存了"))

    # ---------------------------------------------- 3. 可下手基底
    for m in store.by_type("base"):
        if not usable(m):
            continue
        v = m.get("value") or {}
        sug.bases.append({
            "base_id": v.get("base_id"),
            "use_count": v.get("use_count", 0),
            "confidence": round(float(m.get("confidence", 0)), 3),
            "memory_id": m["id"],
            "is_current": v.get("base_id") == cur_base,
        })
    sug.bases.sort(key=lambda x: (-x["use_count"], -x["confidence"]))
    sug.bases = sug.bases[:8]

    # ---------------------------------------------- 4. 素材偏好
    for m in store.by_type("asset"):
        if not usable(m):
            continue
        v = m.get("value") or {}
        sug.assets.append({
            "asset_id": v.get("asset_id"),
            "use_count": v.get("usable_count", 0),
            "confidence": round(float(m.get("confidence", 0)), 3),
            "memory_id": m["id"],
            "is_current": v.get("asset_id") in cur_assets,
        })
    sug.assets.sort(key=lambda x: (-x["use_count"], -x["confidence"]))
    sug.assets = sug.assets[:10]

    # ---------------------------------------------- 5. 成功组合（最优先）
    combos = [m for m in store.by_type("combo") if usable(m)]
    # 与当前基底/素材相关的排前面
    def _combo_score(m: dict) -> float:
        v = m.get("value") or {}
        s = _eff(store, m) * float(m.get("confidence", 0))
        if v.get("base_id") == cur_base:
            s += 1.0
        if set(v.get("asset_ids") or []) & cur_assets:
            s += 0.5
        return s

    combos.sort(key=_combo_score, reverse=True)
    for m in combos[:limit]:
        v = m.get("value") or {}
        sug.items.append(Item(
            kind="combo",
            key=m["key"],
            value=v,
            confidence=float(m.get("confidence", 0)),
            weight=_eff(store, m),
            memory_id=m["id"],
            reason="这个组合你保存过"))

    # ---------------------------------------------- 6. 配文习惯
    caps = [m for m in store.by_type("caption") if usable(m)]
    if caps:
        caps.sort(key=lambda m: _eff(store, m), reverse=True)
        m = caps[0]
        sug.caption = {**(m.get("value") or {}), "memory_id": m["id"]}
        sug.items.append(Item(
            kind="caption", key=m["key"], value=m.get("value"),
            confidence=float(m.get("confidence", 0)),
            weight=_eff(store, m), memory_id=m["id"],
            reason="你的配文习惯"))

    return sug


def lookup(
    store: MemoryStore,
    *,
    memory_id: str | None = None,
    kind: str | None = None,
    base_id: str | None = None,
    asset_id: str | None = None,
    status: str | None = "active",
) -> list[dict]:
    """按条件查记忆。"""
    if memory_id:
        m = store.get(memory_id)
        return [m] if m else []

    items = store.all(status=status)
    if kind:
        items = [m for m in items if m.get("type") == kind]

    out = []
    for m in items:
        v = m.get("value") or {}
        if base_id and v.get("base_id") != base_id:
            continue
        if asset_id and asset_id not in (v.get("asset_ids") or
                                         ([v["asset_id"]] if v.get("asset_id") else [])):
            continue
        out.append(m)
    return out


def search_text(store: MemoryStore, q: str, *, limit: int = 30) -> list[dict]:
    """最朴素的文本检索：key / value 里包含 q。"""
    q_low = q.lower()
    out = []
    for m in store.all(status=None):
        blob = (str(m.get("key", "")) + " " +
                str(m.get("value", ""))).lower()
        if q_low in blob:
            out.append(m)
            if len(out) >= limit:
                break
    return out


def summary(store: MemoryStore | None = None) -> dict:
    """记忆总览，用于前端展示与口味画像。"""
    if store is None:
        from core import memory
        store = memory.store()

    st = store.stats()
    subs = store.by_type("subtype_pref")
    top_sub = None
    if subs:
        best = max(subs, key=lambda m: float(m.get("confidence", 0)))
        top_sub = (best.get("value") or {}).get("subtype")

    bases = sorted(store.by_type("base"),
                   key=lambda m: -int((m.get("value") or {}).get("use_count", 0)))
    assets = sorted(store.by_type("asset"),
                    key=lambda m: -int((m.get("value") or {}).get("usable_count", 0)))
    combos = store.by_type("combo")
    caps = store.by_type("caption")

    return {
        "stats": st,
        "top_subtype": top_sub,
        "top_bases": [
            {"base_id": (m.get("value") or {}).get("base_id"),
             "count": (m.get("value") or {}).get("use_count", 0)}
            for m in bases[:5]],
        "top_assets": [
            {"asset_id": (m.get("value") or {}).get("asset_id"),
             "count": (m.get("value") or {}).get("usable_count", 0)}
            for m in assets[:8]],
        "combo_count": len(combos),
        "caption": (caps[0].get("value") if caps else None),
    }
