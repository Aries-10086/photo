"""牢图棒 · Schema 校验与参数规范化

职责：
  1. 加载三张表的 JSON Schema
  2. 校验（jsonschema）
  3. 数值 clamp —— 非法值不报错，而是钳到区间内（LLM 生成时必备）
  4. op 顺序校验 —— JSON Schema 表达不了跨元素顺序

设计取向：**宽容**。业务上「用户/模型给了 1.5 的 opacity」不该导致整条配方报废，
应该钳成 1.0 并记录一条警告。只有结构性错误（缺 op、未知 op、顺序错）才拒绝。
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

SCHEMA_DIR = Path(__file__).parent / "schema"

# op 执行顺序约束：数字越小越先执行
OP_ORDER = {
    "backdrop": -5,      # 背景必须最先铺
    "layer_split": 0,
    "face_fit": 10,
    "feature_warp": 10,
    "texture_wrap": 10,
    "sticker": 10,
    "grade": 15,
    "degrade": 20,
    "label": 30,         # 必须在 degrade 之后
}

KNOWN_OPS = set(OP_ORDER)


@dataclass
class ValidationResult:
    """校验结果。ok=False 时 errors 非空；ok=True 时 warnings 可能非空（已自动修正）。"""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    data: dict[str, Any] | None = None

    def __bool__(self) -> bool:
        return self.ok

    def raise_if_bad(self) -> dict[str, Any]:
        if not self.ok:
            raise ValueError("配方校验失败:\n  - " + "\n  - ".join(self.errors))
        return self.data or {}

    def report(self) -> str:
        lines = ["通过" if self.ok else "失败"]
        for e in self.errors:
            lines.append(f"  ✗ {e}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------- 加载


_cache: dict[str, dict] = {}


def load_schema(name: str) -> dict:
    """name 取 'asset' | 'recipe' | 'memory'。结果缓存。"""
    if name not in _cache:
        path = SCHEMA_DIR / f"{name}.schema.json"
        if not path.exists():
            raise FileNotFoundError(f"schema 不存在: {path}")
        _cache[name] = json.loads(path.read_text(encoding="utf-8"))
    return _cache[name]


def _validator(name: str) -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(load_schema(name))


# ---------------------------------------------------------------- clamp


def _walk_and_clamp(
    data: Any,
    schema: Any,
    root: dict,
    path: str,
    warnings: list[str],
) -> Any:
    """递归遍历 data，按 schema 的 minimum/maximum 钳制数值。

    同时递归进 $ref（只支持本文件内 #/definitions/... 或 #/$defs/...）。
    """
    if not isinstance(schema, dict):
        return data

    # 解析 $ref
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/"):
        node = root
        for part in ref[2:].split("/"):
            node = node.get(part, {})
        merged = {**node}
        # 同级其他约束（如 description）不影响 clamp，直接沿用解析结果
        return _walk_and_clamp(data, merged, root, path, warnings)

    # oneOf：op 的判别联合，按 "op" 字段选分支
    if "oneOf" in schema and isinstance(data, dict):
        tag = data.get("op")
        for branch in schema["oneOf"]:
            consts = [
                v.get("const")
                for v in branch.get("properties", {}).values()
                if isinstance(v, dict) and "const" in v
            ]
        # 上面那段只用于说明；实际按 op 名匹配 title 更稳
        for branch in schema["oneOf"]:
            title = branch.get("title", "")
            if tag and title.startswith(tag):
                return _walk_and_clamp(data, branch, root, path, warnings)
        return data

    # 数值 clamp
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        lo, hi = schema.get("minimum"), schema.get("maximum")
        if lo is not None and data < lo:
            warnings.append(f"{path}: {data} < {lo}，已钳为 {lo}")
            data = lo
        if hi is not None and data > hi:
            warnings.append(f"{path}: {data} > {hi}，已钳为 {hi}")
            data = hi
        if schema.get("type") == "integer" and isinstance(data, float):
            data = int(round(data))
        return data

    # 对象递归
    if isinstance(data, dict):
        props = schema.get("properties", {})
        # allOf / if-then 里也可能有 properties，这里只处理顶层 properties
        for key, val in list(data.items()):
            sub = props.get(key)
            if sub is None:
                # 允许 additionalProperties 时也尝试从 definitions 里找不到，跳过
                continue
            data[key] = _walk_and_clamp(val, sub, root, f"{path}.{key}" if path else key, warnings)
        return data

    # 数组递归
    if isinstance(data, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, val in enumerate(data):
                data[i] = _walk_and_clamp(val, item_schema, root, f"{path}[{i}]", warnings)
        return data

    return data


def clamp(schema_name: str, data: dict) -> tuple[dict, list[str]]:
    """按 schema 钳制数值。返回 (新数据, 警告列表)。不修改入参。"""
    root = load_schema(schema_name)
    d = copy.deepcopy(data)
    warnings: list[str] = []
    d = _walk_and_clamp(d, root, root, "", warnings)
    return d, warnings


# ---------------------------------------------------------------- op 顺序


def check_op_order(ops: list[dict]) -> list[str]:
    """检查 op 序列的合法性。返回错误列表，空表示通过。

    规则：
      1. 不能有未知 op
      2. 最多一个 layer_split，且必须在最前
      3. label 必须在 degrade 之后（若有 degrade）
      4. degrade 最多一个
    """
    errors: list[str] = []
    if not isinstance(ops, list) or not ops:
        return ["ops 为空"]

    names: list[str] = []
    for i, o in enumerate(ops):
        if not isinstance(o, dict) or "op" not in o:
            errors.append(f"ops[{i}] 缺少 op 字段")
            continue
        n = o["op"]
        if n not in KNOWN_OPS:
            errors.append(f"ops[{i}] 未知 op: {n}（允许: {sorted(KNOWN_OPS)}）")
        names.append(n)

    if names.count("layer_split") > 1:
        errors.append("layer_split 最多一个")
    if names and "layer_split" in names and names[0] != "layer_split":
        # backdrop 可以排在 layer_split 之前
        if names[0] != "backdrop":
            errors.append("layer_split 必须是第一个内容 op（backdrop 可更靠前）")
    if names.count("degrade") > 1:
        errors.append("degrade 最多一个")
    if names.count("backdrop") > 1:
        errors.append("backdrop 最多一个")

    if "degrade" in names and "label" in names:
        di, li = names.index("degrade"), names.index("label")
        if li < di:
            errors.append(f"label 在第 {li} 位、degrade 在第 {di} 位：label 必须在 degrade 之后"
                          "（先加字再降质会把字压糊）")

    # 实际的执行顺序单调性检查（warning 级：允许同组内自由排序）
    ranks = [OP_ORDER[n] for n in names if n in OP_ORDER]
    if ranks != sorted(ranks):
        errors.append(f"op 顺序不单调：{names}（应为 layer_split < 内容操作 < degrade < label）")

    return errors


# ---------------------------------------------------------------- 主入口


def validate_asset(data: dict, *, do_clamp: bool = True) -> ValidationResult:
    res = ValidationResult(ok=True)
    d = data
    if do_clamp:
        d, warns = clamp("asset", data)
        res.warnings.extend(warns)

    errs = sorted(_validator("asset").iter_errors(d), key=lambda e: e.path)
    if errs:
        res.ok = False
        res.errors.extend(f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errs)
    res.data = d
    return res


def validate_recipe(data: dict, *, do_clamp: bool = True, check_order: bool = True,
                    draft: bool = False) -> ValidationResult:
    """校验配方。

    Args:
        draft: True 时允许空 ops 与不完整的 degrade —— 校准器/编辑中的中间状态。
               成品保存时必须用 draft=False。
    """
    res = ValidationResult(ok=True)
    d = data
    if do_clamp:
        d, warns = clamp("recipe", data)
        res.warnings.extend(warns)

    errs = sorted(_validator("recipe").iter_errors(d), key=lambda e: e.path)

    if draft:
        # 草稿态：放宽两条会阻碍编辑过程的约束
        errs = [e for e in errs if not (
            "should be non-empty" in e.message and list(e.path)[:1] == ["ops"]
            or (list(e.path)[:2] == ["degrade", "enabled"]
                and "was expected" in e.message)
        )]

    if errs:
        res.ok = False
        res.errors.extend(f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errs)

    if check_order and isinstance(d.get("ops"), list) and d["ops"]:
        order_errs = check_op_order(d["ops"])
        if order_errs:
            res.ok = False
            res.errors.extend(order_errs)

    res.data = d
    return res


def validate_memory(data: dict, *, do_clamp: bool = True) -> ValidationResult:
    res = ValidationResult(ok=True)
    d = data
    if do_clamp:
        d, warns = clamp("memory", data)
        res.warnings.extend(warns)

    errs = sorted(_validator("memory").iter_errors(d), key=lambda e: e.path)
    if errs:
        res.ok = False
        res.errors.extend(f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errs)
    res.data = d
    return res


VALIDATORS = {
    "asset": validate_asset,
    "recipe": validate_recipe,
    "memory": validate_memory,
}


def validate(kind: str, data: dict) -> ValidationResult:
    if kind not in VALIDATORS:
        raise ValueError(f"未知 kind: {kind}（允许: {sorted(VALIDATORS)}）")
    return VALIDATORS[kind](data)
