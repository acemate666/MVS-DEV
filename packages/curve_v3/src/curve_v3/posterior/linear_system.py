"""posterior 线性系统构造（内部模块）。

说明：
    把 `H*theta=y` 的构造从拟合逻辑中剥离出来，便于单测与复用。
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from curve_v3.configs import CurveV3Config
from curve_v3.low_snr import weights_from_conf
from curve_v3.low_snr.types import WindowDecisions
from curve_v3.posterior.utils import bounce_event_for_tb
from curve_v3.types import BallObservation, BounceEvent


def build_posterior_linear_system(
    *,
    bounce: BounceEvent,
    post_points: Sequence[BallObservation],
    time_base_abs: float | None,
    cfg: CurveV3Config,
    t_b_rel: float | None = None,
    low_snr: WindowDecisions | None = None,
) -> tuple[np.ndarray, np.ndarray, Literal["v_only", "v+axz"], float, float, float, BounceEvent] | None:
    """构造后验拟合的线性系统 H*theta=y。

    Returns:
        (H, y, mode, t_b, x_b, z_b, bounce2)。若有效点不足则返回 None。

    说明：
        - 当启用 tb 搜索（或显式传入 t_b_rel）时，需要将反弹事件在时间轴上平移
          到新的 t_b；这里返回的 bounce2 就是“对齐到 t_b”的事件副本。
        - bounce2 只调整 (t_rel, x, z)，并保留 v^- 不变：这是一阶近似，用于吸收
          prefit 对 t_b 的小偏差，避免 posterior 的 tau 被系统性放大。
    """

    if time_base_abs is None:
        return None

    pts = list(post_points)[-int(cfg.posterior.max_post_points) :]
    if not pts:
        return None

    # 低 SNR：按 conf 构造每点每轴的行权重；若未开启则 w=1。
    use_low_snr = bool(cfg.low_snr.low_snr_enabled) and (low_snr is not None)
    if use_low_snr:
        confs = [getattr(p, "conf", None) for p in pts]
        wx = weights_from_conf(
            confs,
            sigma0=float(cfg.low_snr.low_snr_sigma_x0_m),
            c_min=float(cfg.low_snr.low_snr_conf_cmin),
        )
        wy = weights_from_conf(
            confs,
            sigma0=float(cfg.low_snr.low_snr_sigma_y0_m),
            c_min=float(cfg.low_snr.low_snr_conf_cmin),
        )
        wz = weights_from_conf(
            confs,
            sigma0=float(cfg.low_snr.low_snr_sigma_z0_m),
            c_min=float(cfg.low_snr.low_snr_conf_cmin),
        )
    else:
        wx = np.ones((len(pts),), dtype=float)
        wy = np.ones((len(pts),), dtype=float)
        wz = np.ones((len(pts),), dtype=float)

    mode_x = str(low_snr.x.mode) if low_snr is not None else "FULL"
    mode_y = str(low_snr.y.mode) if low_snr is not None else "FULL"
    mode_z = str(low_snr.z.mode) if low_snr is not None else "FULL"

    min_tau = float(cfg.posterior.posterior_min_tau_s)
    min_tau = float(max(min_tau, 0.0))

    tb = float(t_b_rel) if t_b_rel is not None else float(bounce.t_rel)
    bounce2 = bounce_event_for_tb(bounce=bounce, t_b_rel=tb)

    t_b = float(bounce2.t_rel)
    x_b = float(bounce2.x)
    z_b = float(bounce2.z)

    rows: list[list[float]] = []
    ys: list[float] = []

    g = float(cfg.physics.gravity)
    y0 = float(cfg.bounce_contact_y())
    mode: Literal["v_only", "v+axz"] = cfg.posterior.fit_params

    for i, p in enumerate(pts):
        t_rel = float(p.t - time_base_abs)
        tau = t_rel - t_b
        if tau <= min_tau:
            continue

        dx = float(p.x - x_b)
        dz = float(p.z - z_b)
        y_rhs = float(p.y + 0.5 * g * tau * tau - y0)

        sx = float(np.sqrt(max(float(wx[i]), 0.0)))
        sy = float(np.sqrt(max(float(wy[i]), 0.0)))
        sz = float(np.sqrt(max(float(wz[i]), 0.0)))

        # IGNORE_AXIS：直接不把该轴观测放进系统（等价 w=0）。
        if mode == "v_only":
            if mode_x != "IGNORE_AXIS":
                rows.append([sx * tau, 0.0, 0.0])
                ys.append(sx * dx)
            if mode_y != "IGNORE_AXIS":
                rows.append([0.0, sy * tau, 0.0])
                ys.append(sy * y_rhs)
            if mode_z != "IGNORE_AXIS":
                rows.append([0.0, 0.0, sz * tau])
                ys.append(sz * dz)
        else:
            # FREEZE_A/STRONG_PRIOR_V：冻结加速度（将对应列置 0）。
            ax_col = 0.5 * tau * tau if mode_x == "FULL" else 0.0
            az_col = 0.5 * tau * tau if mode_z == "FULL" else 0.0

            if mode_x != "IGNORE_AXIS":
                rows.append([sx * tau, 0.0, 0.0, sx * ax_col, 0.0])
                ys.append(sx * dx)
            if mode_y != "IGNORE_AXIS":
                rows.append([0.0, sy * tau, 0.0, 0.0, 0.0])
                ys.append(sy * y_rhs)
            if mode_z != "IGNORE_AXIS":
                rows.append([0.0, 0.0, sz * tau, 0.0, sz * az_col])
                ys.append(sz * dz)

    if not rows:
        return None

    H = np.asarray(rows, dtype=float)
    y_vec = np.asarray(ys, dtype=float)
    return H, y_vec, mode, t_b, x_b, z_b, bounce2


__all__ = [
    "build_posterior_linear_system",
]
