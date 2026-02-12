"""prior/posterior/corridor 更新步骤。

说明：
    该模块从旧的单体流水线实现中拆出，专注于：
    - 基于 bounce_event 构建 prior candidates
    - 使用 post 段点执行 posterior 融合（可选像素域 refine / 低 SNR 退化）
    - 更新 corridor 输出
    - 在线沉淀（online_prior）回灌

注意：
    这是内部实现细节，不承诺稳定 API。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np

from curve_v3.configs import CurveV3Config
from curve_v3.corridor import build_corridor_by_time
from curve_v3.low_snr import analyze_window
from curve_v3.pipeline.prefit import low_snr_params_from_cfg
from curve_v3.pipeline.types import PostUpdateResult
from curve_v3.posterior.fit_fused import fit_posterior_fused_map
from curve_v3.posterior.anchor import inject_posterior_anchor
from curve_v3.posterior.fusion import reweight_candidates_and_select_best
from curve_v3.prior import PriorModel, build_prior_candidates, maybe_update_online_prior
from curve_v3.types import (
    BallObservation,
    BounceEvent,
    Candidate,
    LowSnrAxisModes,
    PosteriorState,
)

if TYPE_CHECKING:  # pragma: no cover
    from curve_v3.adapters.camera_rig import CameraRig


def extract_post_points_after_land_time(
    *,
    observations: Sequence[BallObservation],
    time_base_abs: float | None,
    t_land: float,
) -> list[BallObservation]:
    """按预测触地时刻把观测点划分出 post 段。"""

    if time_base_abs is None:
        return []

    post: list[BallObservation] = []
    for obs in observations:
        t_rel = float(obs.t - float(time_base_abs))
        if t_rel > float(t_land):
            post.append(obs)
    return post


def update_post_models_and_corridor(
    *,
    cfg: CurveV3Config,
    logger: logging.Logger,
    prior_model: PriorModel | None,
    online_prior,
    camera_rig: "CameraRig | None",
    bounce_event: BounceEvent,
    post_points: Sequence[BallObservation],
    time_base_abs: float | None,
) -> PostUpdateResult:
    """更新 prior 候选、posterior 状态与 corridor 输出。"""

    candidates = build_prior_candidates(
        bounce=bounce_event,
        cfg=cfg,
        prior_model=prior_model,
        online_prior=online_prior,
    )

    best_candidate: Candidate | None = None
    nominal_candidate_id: int | None = None
    posterior_state: PosteriorState | None = None
    posterior_anchor_used = False

    # posterior 低 SNR：用反弹后点做一次退化判别，决定是否忽略/冻结/强先验。
    post_low_snr = None
    low_snr_posterior: LowSnrAxisModes | None = None
    if bool(cfg.low_snr.low_snr_enabled) and post_points:
        pts2 = list(post_points)[-int(cfg.posterior.max_post_points) :]
        post_low_snr = analyze_window(
            xs=np.array([float(p.x) for p in pts2], dtype=float),
            ys=np.array([float(p.y) for p in pts2], dtype=float),
            zs=np.array([float(p.z) for p in pts2], dtype=float),
            confs=[getattr(p, "conf", None) for p in pts2],
            sigma_x0=float(cfg.low_snr.low_snr_sigma_x0_m),
            sigma_y0=float(cfg.low_snr.low_snr_sigma_y0_m),
            sigma_z0=float(cfg.low_snr.low_snr_sigma_z0_m),
            c_min=float(cfg.low_snr.low_snr_conf_cmin),
            params=low_snr_params_from_cfg(cfg),
            disallow_ignore_y=bool(cfg.low_snr.low_snr_disallow_ignore_y),
        )
        low_snr_posterior = LowSnrAxisModes(
            mode_x=post_low_snr.x.mode,
            mode_y=post_low_snr.y.mode,
            mode_z=post_low_snr.z.mode,
        )

    if post_points and candidates:
        (
            candidates_1,
            best_1,
            nominal_id_1,
            posterior_1,
        ) = reweight_candidates_and_select_best(
            bounce=bounce_event,
            candidates=candidates,
            post_points=post_points,
            time_base_abs=time_base_abs,
            camera_rig=camera_rig,
            low_snr=post_low_snr,
            cfg=cfg,
        )

        candidates = candidates_1
        best_candidate = best_1
        nominal_candidate_id = nominal_id_1
        posterior_state = posterior_1

        # 在线沉淀：用融合后的候选权重回灌 prior（docs/curve.md §7）。
        maybe_update_online_prior(
            online_prior=online_prior,
            cfg=cfg,
            candidates=candidates,
            logger=logger,
        )

    # 兜底：如果已经有反弹后点，但未能得到“逐候选 posterior”，则回退为
    # 不带候选先验锚定的 LS/MAP 后验（用于输出/走廊更新）。
    if post_points and posterior_state is None:
        posterior_state = fit_posterior_fused_map(
            bounce=bounce_event,
            post_points=post_points,
            best=None,
            time_base_abs=time_base_abs,
            low_snr=post_low_snr,
            cfg=cfg,
        )

    # 走廊更新：可选将 posterior “锚点候选”注入混合，提升短期走廊一致性。
    candidates_for_corridor = list(candidates)
    if posterior_state is not None and cfg.posterior.posterior_anchor_weight > 0:
        posterior_anchor_used = True
        candidates_for_corridor = inject_posterior_anchor(
            candidates=candidates,
            best=best_candidate,
            posterior=posterior_state,
            cfg=cfg,
        )

    corridor_by_time = build_corridor_by_time(
        bounce=bounce_event,
        candidates=candidates_for_corridor,
        cfg=cfg,
    )

    return PostUpdateResult(
        candidates=candidates,
        best_candidate=best_candidate,
        nominal_candidate_id=nominal_candidate_id,
        posterior_state=posterior_state,
        corridor_by_time=corridor_by_time,
        posterior_anchor_used=bool(posterior_anchor_used),
        low_snr_posterior=low_snr_posterior,
    )
