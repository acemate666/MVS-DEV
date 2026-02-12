"""tennis3d_trajectory：轨迹后处理（curve stage）。

说明：
- 本包把 `tennis3d-core` 的 3D 定位输出记录（JSON 可序列化 dict）进一步增强为轨迹拟合输出（curve），
  并可选输出击球拦截点（interception）。
- 该包刻意不侵入 `tennis3d.pipeline.core.run_localization_pipeline()` 的内部实现，
  以“后处理 stage”的方式组合到在线/离线入口。
"""

from __future__ import annotations

from tennis3d_trajectory.curve_stage import CurveStage, CurveStageConfig, apply_curve_stage

__all__ = [
    "CurveStage",
    "CurveStageConfig",
    "apply_curve_stage",
]
