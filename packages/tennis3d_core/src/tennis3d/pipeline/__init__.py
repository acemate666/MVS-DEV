"""在线/离线共用的 3D 定位流水线（纯计算）。

说明：
- 本包只包含“对已组包好的图像组做定位”的核心算法逻辑。
- 组包/采集属于适配层：
    - 离线 captures -> groups：`tennis3d_offline.captures.iter_capture_image_groups`
    - 在线 MVS -> groups：`tennis3d_online.sources.iter_mvs_image_groups`

设计目标：让 `tennis3d-core` 不反向依赖 `mvs`、`tennis3d_online`、`tennis3d_offline`。
"""

from .core import run_localization_pipeline

__all__ = ["run_localization_pipeline"]
