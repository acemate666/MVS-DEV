"""curve_v3 的 YAML 配置加载入口。

动机：
    - `curve_v3.configs` 中的配置采用 dataclass + 默认值，适合在代码中直接构造。
    - 工程集成时，经常希望用一个可编辑的 YAML 文件覆盖部分字段。

约定：
    - YAML 顶层为一个 mapping，对应 `CurveV3Config` 的字段名（physics/prior/...）。
    - 子节点也是 mapping，对应各子配置 dataclass 的字段名。
    - 未提供的字段使用 dataclass 的默认值。
    - 未知字段会报错，避免“拼写错了但静默无效”。

依赖：
    - 本模块依赖 PyYAML（`pyyaml`）。
"""

from __future__ import annotations

from dataclasses import MISSING, Field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml

from curve_v3.configs import CurveV3Config

_T = TypeVar("_T")


def _as_mapping(x: Any) -> Mapping[str, Any]:
    if x is None:
        return {}
    if isinstance(x, Mapping):
        return x
    raise TypeError(f"YAML 根节点必须是 mapping，实际是：{type(x).__name__}")


def _normalize_value_for_field(f: Field[Any], value: Any) -> Any:
    """尽量把 YAML 读出来的类型归一到与默认值一致的形态。

    说明：
        YAML 里常见写法是 list，但配置里很多字段默认是 tuple。
        这里按“默认值的形态”做一个轻量归一化：
        - 默认是 tuple：把 list/tuple 转成 tuple。
        - 其他类型保持原样。

    注意：
        这里只做结构治理层面的类型归一，不做数值校验（避免改变业务语义）。
    """

    if value is None:
        return None

    default = f.default
    if default is not MISSING and isinstance(default, tuple):
        if isinstance(value, (list, tuple)):
            return tuple(value)

    return value


def _nested_dataclass_type_from_field(f: Field[Any]) -> type | None:
    """尝试从 dataclass Field 推断其嵌套 dataclass 类型。

    说明：
        对于 `CurveV3Config` 这类聚合配置，其字段通常是：
            xxx: XxxConfig = field(default_factory=XxxConfig)
        因此可以通过调用 default_factory 得到实例，再判断是否为 dataclass。

    约束：
        - 只用于解析配置；默认工厂应当是纯函数（本仓库内这些默认工厂均为轻量 dataclass）。
    """

    if f.default_factory is MISSING:  # type: ignore[comparison-overlap]
        return None

    try:
        inst = f.default_factory()  # type: ignore[misc]
    except TypeError:
        return None

    if is_dataclass(inst):
        return type(inst)

    return None


def _dataclass_from_mapping(cls: type[_T], data: Mapping[str, Any]) -> _T:
    if not is_dataclass(cls):
        raise TypeError(f"期望 dataclass 类型，实际是：{cls}")

    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(data.keys()) - allowed)
    if unknown:
        raise KeyError(f"{cls.__name__} 出现未知字段：{unknown}")

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue

        raw = data[f.name]
        nested_cls = _nested_dataclass_type_from_field(f)
        if nested_cls is not None:
            kwargs[f.name] = _dataclass_from_mapping(nested_cls, _as_mapping(raw))
        else:
            kwargs[f.name] = _normalize_value_for_field(f, raw)

    return cls(**kwargs)  # type: ignore[call-arg]


def curve_v3_config_from_dict(data: Mapping[str, Any]) -> CurveV3Config:
    """从 dict（通常来自 YAML）构造 `CurveV3Config`。"""

    return _dataclass_from_mapping(CurveV3Config, data)


def load_curve_v3_config_yaml(path: str | Path) -> CurveV3Config:
    """从 YAML 文件加载 `CurveV3Config`。"""

    p = Path(path)
    text = p.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    return curve_v3_config_from_dict(_as_mapping(payload))
