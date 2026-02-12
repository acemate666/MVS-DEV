"""运行中 ROI 的 int 节点对齐工具（无 SDK 依赖）。

动机：
- `tennis3d-core` 目标是“纯算法库”，不应依赖 `mvs`。
- 但 ROI 控制器需要知道 OffsetX/OffsetY 的范围与步进（inc），以便对齐并 clamp。

说明：
- 在线模式会从 `mvs.sdk.runtime_roi.get_int_node_info()` 读取节点信息，然后把这些信息
  作为纯数据（IntNodeInfo）注入到 ROI 控制器。
- 这里仅提供与 SDK 无关的：数据结构 + clamp_and_align 逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class IntNodeInfo:
    """int 节点信息（与常见 GenICam/MVS int 节点语义对齐）。"""

    cur: int
    vmin: int
    vmax: int
    inc: int


def clamp_and_align(
    value: int,
    *,
    info: IntNodeInfo,
    mode: Literal["nearest", "down"] = "nearest",
) -> int:
    """把目标值裁剪到 [min,max] 并按 inc 对齐。

    说明：
    - OffsetX/OffsetY 常见 inc=4；不对齐会导致写节点失败。
    - mode="nearest" 更适合 offset（减少系统性偏差）；mode="down" 更适合 Width/Height（避免变大）。
    """

    inc = int(info.inc) if int(info.inc) > 0 else 1
    vmin = int(info.vmin)
    vmax = int(info.vmax)

    v = int(value)
    if v < vmin:
        v = vmin
    if v > vmax:
        v = vmax

    if inc <= 1:
        return int(v)

    # 对齐到以 vmin 为起点的步进格。
    rel = int(v - vmin)
    if rel < 0:
        rel = 0

    if str(mode) == "down":
        steps = rel // inc
    else:
        # nearest：使用整数四舍五入（0.5 向上），避免 Python round 的 banker's rounding。
        steps = (rel + inc // 2) // inc

    out = int(vmin + steps * inc)
    if out < vmin:
        out = vmin
    if out > vmax:
        out = vmax
    return int(out)
