from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tennis3d_trajectory.curve_stage.config import CurveStageConfig


@dataclass
class _CurveImports:
    """curve_v3 算法实现的延迟 import 结果（按需填充）。"""

    CurvePredictorV3: Any | None = None
    BallObservation: Any | None = None


def _ensure_curve_imports(cfg: CurveStageConfig) -> _CurveImports:
    """延迟导入 curve_v3。

    说明：
        - 该逻辑集中在一个模块里，便于清晰表达依赖边界。
        - 只在 cfg.enabled=True 且真正处理记录时才会触发。
    """

    out = _CurveImports()

    # 说明：curve_stage 只集成新 curve_v3，不再支持 v2/v3_legacy。
    _ = cfg
    # 依赖边界：跨包仅依赖 curve_v3 的稳定 Public API（包顶层导出）。
    from curve_v3 import BallObservation, CurvePredictorV3

    out.CurvePredictorV3 = CurvePredictorV3
    out.BallObservation = BallObservation

    return out
