"""posterior：融合拟合入口（内部模块）。"""

from __future__ import annotations

from typing import Sequence

from curve_v3.configs import CurveV3Config
from curve_v3.low_snr.types import WindowDecisions
from curve_v3.posterior.fit_ls import fit_posterior_ls
from curve_v3.posterior.fit_map import fit_posterior_map_for_candidate
from curve_v3.types import BallObservation, BounceEvent, Candidate, PosteriorState


def fit_posterior_fused_map(
    *,
    bounce: BounceEvent,
    post_points: Sequence[BallObservation],
    best: Candidate | None,
    time_base_abs: float | None,
    low_snr: WindowDecisions | None = None,
    cfg: CurveV3Config,
) -> PosteriorState | None:
    """融合拟合后验状态（可选使用 best 候选作为高斯先验）。"""

    if best is None:
        return fit_posterior_ls(
            bounce=bounce,
            post_points=post_points,
            time_base_abs=time_base_abs,
            low_snr=low_snr,
            cfg=cfg,
        )

    out = fit_posterior_map_for_candidate(
        bounce=bounce,
        post_points=post_points,
        candidate=best,
        time_base_abs=time_base_abs,
        low_snr=low_snr,
        cfg=cfg,
    )
    if out is None:
        return None
    st, _ = out
    return st


__all__ = [
    "fit_posterior_fused_map",
]
