"""curve_v3 的在线更新流水线（内部命名空间）。

说明：
    这里承载 `curve_v3.core.CurvePredictorV3` 的“分步骤更新逻辑”，目标是把
    高耦合的大函数拆成职责更清晰的小模块：

    - prefit：反弹前拟合与 bounce_event 推断
    - post：候选生成、posterior 融合、走廊输出

注意：
    这是内部实现细节，不承诺稳定 API。
"""

from curve_v3.pipeline.post import extract_post_points_after_land_time, update_post_models_and_corridor
from curve_v3.pipeline.prefit import low_snr_params_from_cfg, update_prefit_and_bounce_event
from curve_v3.pipeline.types import PostUpdateResult, PrefitUpdateResult

__all__ = [
    "PostUpdateResult",
    "PrefitUpdateResult",
    "extract_post_points_after_land_time",
    "low_snr_params_from_cfg",
    "update_post_models_and_corridor",
    "update_prefit_and_bounce_event",
]
