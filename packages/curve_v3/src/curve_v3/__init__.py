"""curve_v3 包入口。

导出内容：
    - CurvePredictorV3：两阶段（prior + posterior）新预测器 API。
    - 常用配置 dataclass：作为跨包集成时的稳定构造入口。
    - 本包不再提供旧版（legacy/curve2.py）兼容适配层。
"""

from curve_v3.configs import (
    BounceDetectorConfig,
    CandidateConfig,
    CorridorConfig,
    CurveV3Config,
    LowSnrConfig,
    OnlinePriorConfig,
    PhysicsConfig,
    PipelineConfig,
    PixelConfig,
    PosteriorConfig,
    PrefitConfig,
    PriorConfig,
    SimplePipelineConfig,
)
from curve_v3.core import CurvePredictorV3
from curve_v3.api import fit_posterior_map_for_candidate
from curve_v3.types import (
    BallObservation,
    BounceEvent,
    Candidate,
    CorridorOnPlane,
    CorridorByTime,
    FusionInfo,
    PrefitFreezeInfo,
    PosteriorState,
)

__all__ = [
    "BallObservation",
    "BounceDetectorConfig",
    "BounceEvent",
    "CandidateConfig",
    "Candidate",
    "CorridorConfig",
    "CorridorByTime",
    "CorridorOnPlane",
    "CurvePredictorV3",
    "CurveV3Config",
    "fit_posterior_map_for_candidate",
    "FusionInfo",
    "LowSnrConfig",
    "OnlinePriorConfig",
    "PhysicsConfig",
    "PipelineConfig",
    "PixelConfig",
    "PosteriorConfig",
    "PrefitConfig",
    "PrefitFreezeInfo",
    "PriorConfig",
    "PosteriorState",
    "SimplePipelineConfig",
]
