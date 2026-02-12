"""curve_v3 稳定公共 API。

说明：
    本模块用于收敛跨包（例如 `interception`）对 `curve_v3` 的依赖边界。

    约定：
        - 下游集成应优先从 `curve_v3` 包顶层或 `curve_v3.api` import。
        - 避免直接 import `curve_v3.posterior.*` / `curve_v3.prior.*` 等内部子包，
          以减少对内部目录结构与实现细节的耦合。

注意：
    这里的函数是“稳定契约层”。内部实现可以迁移/拆分，但应尽量保持这里的签名与语义稳定。
"""

from __future__ import annotations

from typing import Sequence

from curve_v3.adapters.camera_rig import CameraRig
from curve_v3.configs import CurveV3Config
from curve_v3.low_snr.types import WindowDecisions
from curve_v3.posterior.fit_map import fit_posterior_map_for_candidate as _fit_posterior_map_for_candidate
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
    """对单条候选执行后验 MAP 拟合，并返回 (PosteriorState, J_post)。

    说明：
        这是对 `curve_v3.posterior.fit_map.fit_posterior_map_for_candidate` 的稳定转发。
        下游请通过该函数调用，而不是直接依赖 posterior 子包的路径。
    """

    return _fit_posterior_map_for_candidate(
        bounce=bounce,
        post_points=post_points,
        candidate=candidate,
        time_base_abs=time_base_abs,
        camera_rig=camera_rig,
        low_snr=low_snr,
        cfg=cfg,
    )


__all__ = [
    "fit_posterior_map_for_candidate",
]
