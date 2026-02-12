"""posterior 阶段的共享小工具。

说明：
    posterior 的拟合与 tb 搜索在多个模块中会复用一些小逻辑；这里集中放置，
    避免复制粘贴导致口径不一致。
"""

from __future__ import annotations

import numpy as np

from curve_v3.types import BounceEvent


def bounce_event_for_tb(*, bounce: BounceEvent, t_b_rel: float) -> BounceEvent:
    """根据 v^- 将 (t_b, x_b, z_b) 在时间轴上做一阶平移。

    说明：
        posterior 的观测方程以“反弹时刻”为时间零点（tau=t-t_b）。当 t_b 有偏时，
        直接用固定的 bounce.x/z 会把所有 post 点的 dx/dz 也引入系统偏差。

        这里使用 v^- 做一阶平移：
            x_b(tb) = x_b(tb0) + v^-_x * (tb - tb0)
            z_b(tb) = z_b(tb0) + v^-_z * (tb - tb0)

        这是工程化折中：
            - 好处：能显著降低 tb 小偏差对 posterior 的放大效应。
            - 局限：不调整 y（触地高度由 cfg.bounce_contact_y() 约束），也不建模
              v^- 的曲率/加速度误差。

    Args:
        bounce: 原始反弹事件（prefit 输出）。
        t_b_rel: 新的反弹相对时间（秒）。

    Returns:
        平移后的 bounce 副本。
    """

    tb = float(t_b_rel)
    dt = float(tb - float(bounce.t_rel))

    v_minus = np.asarray(bounce.v_minus, dtype=float).reshape(3)
    x = float(bounce.x + float(v_minus[0]) * dt)
    z = float(bounce.z + float(v_minus[2]) * dt)

    return BounceEvent(
        t_rel=tb,
        x=x,
        z=z,
        v_minus=np.asarray(bounce.v_minus, dtype=float),
        y=bounce.y,
        sigma_t_rel=bounce.sigma_t_rel,
        sigma_v_minus=bounce.sigma_v_minus,
        prefit_rms_m=bounce.prefit_rms_m,
    )
