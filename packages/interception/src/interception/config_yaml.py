"""interception 的 YAML 配置加载入口。

动机：
    `interception.config.InterceptionConfig` 是一个单体 dataclass。
    工程接入时通常希望通过 YAML 提供 y_min/y_max 等参数。

约定：
    - YAML 顶层为 mapping，字段名与 `InterceptionConfig` 一致。
    - 未知字段会报错，避免拼写错误静默失效。
    - y_min/y_max 为必填项（因为 dataclass 无默认值）。

依赖：
    - 本模块依赖 PyYAML（`pyyaml`）。
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

import yaml

from interception.config import InterceptionConfig


def _as_mapping(x: Any) -> Mapping[str, Any]:
    if x is None:
        return {}
    if isinstance(x, Mapping):
        return x
    raise TypeError(f"YAML 根节点必须是 mapping，实际是：{type(x).__name__}")


def interception_config_from_dict(data: Mapping[str, Any]) -> InterceptionConfig:
    """从 dict（通常来自 YAML）构造 `InterceptionConfig`。"""

    allowed = {f.name for f in fields(InterceptionConfig)}
    unknown = sorted(set(data.keys()) - allowed)
    if unknown:
        raise KeyError(f"InterceptionConfig 出现未知字段：{unknown}")

    try:
        return InterceptionConfig(**dict(data))
    except TypeError as e:
        # 主要用于把“缺必填字段”的错误变得更直观。
        raise ValueError(f"InterceptionConfig 构造失败：{e}") from e


def load_interception_config_yaml(path: str | Path) -> InterceptionConfig:
    """从 YAML 文件加载 `InterceptionConfig`。"""

    p = Path(path)
    text = p.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    return interception_config_from_dict(_as_mapping(payload))
