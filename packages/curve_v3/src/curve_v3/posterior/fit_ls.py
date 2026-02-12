"""posterior：不使用候选先验时的 LS/RLS 拟合（内部模块）。"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from curve_v3.configs import CurveV3Config
from curve_v3.low_snr.types import WindowDecisions
from curve_v3.posterior.linear_system import build_posterior_linear_system
from curve_v3.posterior.solve import posterior_obs_sigma_m, solve_map_with_prior
from curve_v3.types import BallObservation, BounceEvent, PosteriorState


def fit_posterior_ls(
    *,
    bounce: BounceEvent,
    post_points: Sequence[BallObservation],
    time_base_abs: float | None,
    low_snr: WindowDecisions | None = None,
    cfg: CurveV3Config,
) -> PosteriorState | None:
    """在不使用候选先验时拟合后验参数。

    说明：
        当前实现只保留 RLS（信息形式递推）这一条代码路径；当 λ=1 时等价于批量 LS。
    """

    if time_base_abs is None:
        return None

    fit_mode = str(cfg.posterior.fit_mode)
    lam = float(cfg.posterior.posterior_rls_lambda) if fit_mode == "rls" else 1.0

    sigma = posterior_obs_sigma_m(cfg)
    min_tau = float(cfg.posterior.posterior_min_tau_s)
    min_tau = float(max(min_tau, 0.0))

    def solve_for_tb(tb_rel: float) -> tuple[PosteriorState, float] | None:
        sys = build_posterior_linear_system(
            bounce=bounce,
            post_points=post_points,
            time_base_abs=time_base_abs,
            cfg=cfg,
            t_b_rel=float(tb_rel),
            low_snr=low_snr,
        )
        if sys is None:
            return None

        H, y_vec, mode, t_b, x_b, z_b, _bounce2 = sys

        # 无先验时：信息形式递推累积正规方程（λ=1 等价于批量 LS）。
        d = int(H.shape[1])
        theta0 = np.zeros((d,), dtype=float)
        Q = np.zeros((d, d), dtype=float)
        theta = solve_map_with_prior(
            H=H,
            y_vec=y_vec,
            theta0=theta0,
            Q=Q,
            sigma_m=float(sigma),
            fit_mode="rls",
            rls_lambda=float(lam),
        )

        if mode == "v_only":
            vx, vy, vz = float(theta[0]), float(theta[1]), float(theta[2])
            ax2, az2 = 0.0, 0.0
        else:
            vx, vy, vz, ax2, az2 = (
                float(theta[0]),
                float(theta[1]),
                float(theta[2]),
                float(theta[3]),
                float(theta[4]),
            )

        # 评分仅用于 tb 搜索（无先验时），保持与线性系统口径一致。
        r = H @ theta - y_vec
        data_term = float(np.dot(r, r) / max(float(sigma) * float(sigma), 1e-12))

        st = PosteriorState(t_b_rel=t_b, x_b=x_b, z_b=z_b, vx=vx, vy=vy, vz=vz, ax=ax2, az=az2)
        return st, float(data_term)

    if bool(cfg.posterior.posterior_optimize_tb):
        tb0 = float(bounce.t_rel)
        window = float(cfg.posterior.posterior_tb_search_window_s)
        step = float(cfg.posterior.posterior_tb_search_step_s)
        if step <= 0:
            step = 0.002

        t_rel_min = min(float(p.t - time_base_abs) for p in post_points)
        tb_lo = max(0.0, tb0 - max(window, 0.0))
        tb_hi = min(tb0 + max(window, 0.0), float(t_rel_min - min_tau))
        if tb_hi < tb_lo:
            tb_hi = tb_lo

        sigma_tb = float(cfg.posterior.posterior_tb_prior_sigma_s)
        sigma_tb2 = max(sigma_tb * sigma_tb, 1e-12)

        best_st: PosteriorState | None = None
        best_j = float("inf")
        tb = float(tb_lo)
        while tb <= tb_hi + 0.5 * step:
            out = solve_for_tb(tb)
            if out is not None:
                st, data_term = out
                tb_pen = (float(tb - tb0) ** 2) / sigma_tb2
                j2 = float(data_term + tb_pen)
                if j2 < best_j:
                    best_j = float(j2)
                    best_st = st
            tb += step

        return best_st

    out0 = solve_for_tb(float(bounce.t_rel))
    if out0 is None:
        return None
    st0, _ = out0
    return st0


__all__ = [
    "fit_posterior_ls",
]
