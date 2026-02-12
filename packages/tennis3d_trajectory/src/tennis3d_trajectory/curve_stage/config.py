from __future__ import annotations

# 说明：CurveStageConfig 是上游（pipeline/apps）与下游（trajectory stage）之间的纯数据契约。
# 该类型放在 tennis3d-core 中，避免 core 反向依赖 curve_v3/interception。
from tennis3d.curve_stage_config import CurveStageConfig

__all__ = ["CurveStageConfig"]
