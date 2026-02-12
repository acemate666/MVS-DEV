"""tennis3d（tennis3d-core）：网球多相机 3D 定位核心库。

说明：
- `tennis3d-core` 只提供纯算法/契约：几何、定位、流水线核心。
- 采集/在线运行时位于：`mvs` 与 `tennis3d_online`
- 离线入口位于：`tennis3d_offline`
- 轨迹/拦截后处理位于：`tennis3d_trajectory`
"""

from tennis3d.api import build_calibration, run_localization_pipeline

__all__ = ["build_calibration", "run_localization_pipeline"]

