"""posterior：逐候选 MAP 拟合（内部模块）。

说明：
    这里实现“给定候选先验，对 post 点做 MAP 校正并输出 J_post”的核心逻辑。
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from curve_v3.adapters.camera_rig import CameraRig
from curve_v3.configs import CurveV3Config
from curve_v3.low_snr.types import WindowDecisions
from curve_v3.posterior.linear_system import build_posterior_linear_system
from curve_v3.posterior.pixel_refine import refine_theta_in_pixel_domain
from curve_v3.posterior.solve import posterior_obs_sigma_m, solve_map_with_prior
from curve_v3.posterior.utils import bounce_event_for_tb
from curve_v3.types import BallObservation, BounceEvent, Candidate, PosteriorState


def fit_posterior_map_for_candidate(
    *,
    bounce: BounceEvent,
    post_points: Sequence[BallObservation],
    candidate: Candidate,
    time_base_abs: float | None,
    camera_rig: CameraRig | None = None,
    low_snr: WindowDecisions | None = None,
    cfg: CurveV3Config,
) -> tuple[PosteriorState, float] | None:
    """对单条候选执行后验 MAP 拟合，并返回 J_post。"""

    if time_base_abs is None:
        return None

    # Candidate prior (theta0) and its precision (Q).
    strength = float(cfg.posterior.posterior_prior_strength)
    if strength < 0.0:
        strength = 0.0

    mode: Literal["v_only", "v+axz"] = cfg.posterior.fit_params

    # 低 SNR：可对不同轴施加不同强度的速度先验（STRONG_PRIOR_V）。
    strong_scale = float(cfg.low_snr.low_snr_strong_prior_v_scale)
    strong_scale = float(min(max(strong_scale, 1e-3), 1.0))

    if mode == "v_only":
        theta0 = np.array([candidate.v_plus[0], candidate.v_plus[1], candidate.v_plus[2]], dtype=float)
        if strength > 0.0:
            sigma_v = float(cfg.posterior.posterior_prior_sigma_v)
            sigs = np.array([sigma_v, sigma_v, sigma_v], dtype=float)
            if low_snr is not None:
                if str(low_snr.x.mode) == "STRONG_PRIOR_V":
                    sigs[0] = sigs[0] * strong_scale
                if str(low_snr.y.mode) == "STRONG_PRIOR_V":
                    sigs[1] = sigs[1] * strong_scale
                if str(low_snr.z.mode) == "STRONG_PRIOR_V":
                    sigs[2] = sigs[2] * strong_scale

            q = (1.0 / np.maximum(sigs * sigs, 1e-9)) * float(strength)
            Q = np.diag(q.astype(float))
        else:
            Q = np.zeros((3, 3), dtype=float)
    else:
        theta0 = np.array(
            [candidate.v_plus[0], candidate.v_plus[1], candidate.v_plus[2], candidate.ax, candidate.az],
            dtype=float,
        )
        if strength > 0.0:
            sigma_v = float(cfg.posterior.posterior_prior_sigma_v)
            sigma_a = float(cfg.posterior.posterior_prior_sigma_a)
            sigs_v = np.array([sigma_v, sigma_v, sigma_v], dtype=float)
            if low_snr is not None:
                if str(low_snr.x.mode) == "STRONG_PRIOR_V":
                    sigs_v[0] = sigs_v[0] * strong_scale
                if str(low_snr.y.mode) == "STRONG_PRIOR_V":
                    sigs_v[1] = sigs_v[1] * strong_scale
                if str(low_snr.z.mode) == "STRONG_PRIOR_V":
                    sigs_v[2] = sigs_v[2] * strong_scale

            inv_v2 = 1.0 / np.maximum(sigs_v * sigs_v, 1e-9)
            inv_a2 = 1.0 / max(sigma_a * sigma_a, 1e-9)
            q = np.array([inv_v2[0], inv_v2[1], inv_v2[2], inv_a2, inv_a2], dtype=float) * strength
            Q = np.diag(q.astype(float))
        else:
            Q = np.zeros((5, 5), dtype=float)

    # MAP“求解”与 J_post“评分”使用同一个 σ（观测噪声尺度）。
    sigma = posterior_obs_sigma_m(cfg)
    fit_mode: Literal["rls"] = cfg.posterior.fit_mode

    min_tau = float(cfg.posterior.posterior_min_tau_s)
    min_tau = float(max(min_tau, 0.0))

    def solve_for_tb(tb_rel: float) -> tuple[PosteriorState, float] | None:
        tb_rel = float(tb_rel)

        sys = build_posterior_linear_system(
            bounce=bounce,
            post_points=post_points,
            time_base_abs=time_base_abs,
            cfg=cfg,
            t_b_rel=float(tb_rel),
            low_snr=low_snr,
        )

        if sys is None:
            # 线性系统构造失败时：若像素域可用，则仍尝试用先验作为初值做像素闭环。
            bounce2 = bounce_event_for_tb(bounce=bounce, t_b_rel=tb_rel)
            mode2: Literal["v_only", "v+axz"] = cfg.posterior.fit_params
            x_b2, z_b2 = float(bounce2.x), float(bounce2.z)
            theta_init = np.asarray(theta0, dtype=float).copy()
            H = None
            y_vec = None
        else:
            H, y_vec, mode2, t_b2, x_b2, z_b2, bounce2 = sys

            theta_init = solve_map_with_prior(
                H=H,
                y_vec=y_vec,
                theta0=theta0,
                Q=Q,
                sigma_m=float(sigma),
                fit_mode=fit_mode,
                rls_lambda=float(cfg.posterior.posterior_rls_lambda),
            )

        pixel_enabled = bool(cfg.pixel.pixel_enabled)
        has_2d = any(bool(getattr(p, "obs_2d_by_camera", None)) for p in post_points)
        use_pixel = pixel_enabled and (camera_rig is not None) and has_2d

        if use_pixel:
            assert camera_rig is not None
            out_px = refine_theta_in_pixel_domain(
                bounce=bounce2,
                post_points=post_points,
                candidate=candidate,
                time_base_abs=float(time_base_abs),
                t_b_rel=float(tb_rel),
                theta_init=np.asarray(theta_init, dtype=float),
                theta0=np.asarray(theta0, dtype=float),
                Q=np.asarray(Q, dtype=float),
                mode=mode2,
                camera_rig=camera_rig,
                cfg=cfg,
            )

            if out_px is not None:
                theta, pixel_data_term = out_px
                theta = np.asarray(theta, dtype=float).reshape(-1)

                if mode2 == "v_only":
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

                d = theta - theta0
                prior_term = float(d.T @ Q @ d)
                j_post = float(float(pixel_data_term) + prior_term)
                st = PosteriorState(
                    t_b_rel=float(tb_rel),
                    x_b=float(x_b2),
                    z_b=float(z_b2),
                    vx=vx,
                    vy=vy,
                    vz=vz,
                    ax=ax2,
                    az=az2,
                )
                return st, j_post

            # 若像素域失败：回退到 3D 点域（若线性系统存在）。
            if sys is None:
                return None

        # 3D 点域：使用线性系统的 MAP 结果与 J_post。
        if sys is None or H is None or y_vec is None:
            return None

        theta = np.asarray(theta_init, dtype=float).reshape(-1)
        if mode2 == "v_only":
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

        r = H @ theta - y_vec
        data_term = float(np.dot(r, r) / max(float(sigma) * float(sigma), 1e-12))
        d = theta - theta0
        prior_term = float(d.T @ Q @ d)
        j_post = float(data_term + prior_term)
        st = PosteriorState(
            t_b_rel=float(tb_rel),
            x_b=float(x_b2),
            z_b=float(z_b2),
            vx=vx,
            vy=vy,
            vz=vz,
            ax=ax2,
            az=az2,
        )
        return st, j_post

    # 可选联合估计 tb。
    if bool(cfg.posterior.posterior_optimize_tb):
        tb0 = float(bounce.t_rel)
        window = float(cfg.posterior.posterior_tb_search_window_s)
        step = float(cfg.posterior.posterior_tb_search_step_s)
        if step <= 0:
            step = 0.002

        # 上界：必须早于最早的 post 点，否则 tau<=0 会导致系统退化。
        t_rel_min = min(float(p.t - time_base_abs) for p in post_points)
        tb_lo = max(0.0, tb0 - max(window, 0.0))
        tb_hi = min(tb0 + max(window, 0.0), float(t_rel_min - min_tau))
        if tb_hi < tb_lo:
            tb_hi = tb_lo

        sigma_tb = float(cfg.posterior.posterior_tb_prior_sigma_s)
        sigma_tb2 = max(sigma_tb * sigma_tb, 1e-12)

        best_st: PosteriorState | None = None
        best_j = float("inf")

        # 一维网格搜索：对每个 tb 试算一次 MAP，并叠加 tb 先验惩罚项。
        tb = float(tb_lo)
        while tb <= tb_hi + 0.5 * step:
            out = solve_for_tb(tb)
            if out is not None:
                st, j = out
                tb_pen = (float(tb - tb0) ** 2) / sigma_tb2
                j2 = float(j + tb_pen)
                if j2 < best_j:
                    best_j = float(j2)
                    best_st = st
            tb += step

        if best_st is None:
            return None
        return best_st, float(best_j)

    # 默认：不优化 tb，按 bounce.t_rel 拟合。
    return solve_for_tb(float(bounce.t_rel))


__all__ = [
    "fit_posterior_map_for_candidate",
]
