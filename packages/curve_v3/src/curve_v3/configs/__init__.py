"""curve_v3 配置包。

说明：
    - 这是一个“纯模型”包，只包含 dataclass 配置定义，不做任何 IO。
    - 下游集成推荐通过 `curve_v3.configs` 导入配置，避免依赖旧路径。
"""

from curve_v3.configs.models import (
    BounceDetectorConfig,
    CandidateConfig,
    CorridorConfig,
    CurveV3Config,
    LegacyConfig,
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

__all__ = [
    "BounceDetectorConfig",
    "CandidateConfig",
    "CorridorConfig",
    "CurveV3Config",
    "LegacyConfig",
    "LowSnrConfig",
    "OnlinePriorConfig",
    "PhysicsConfig",
    "PipelineConfig",
    "PixelConfig",
    "PosteriorConfig",
    "PrefitConfig",
    "PriorConfig",
    "SimplePipelineConfig",
]
