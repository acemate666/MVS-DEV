"""适配层（adapters）。

说明：
    本包用于承载与上游/外部系统对接的“薄适配层”，例如：
    - 多目定位（MVS）输出记录 -> curve_v3 的观测输入。
    - 相机投影/标定模型的接口协议（用于像素域闭环）。

依赖约束：
    - adapters 可以依赖 core（types/config 等），但 core 不应反向依赖具体上游实现。
"""

from __future__ import annotations

__all__ = []
