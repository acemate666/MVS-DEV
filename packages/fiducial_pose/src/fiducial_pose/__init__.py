"""fiducial_pose：用人工靶标（例如 AprilTag）估计车辆位姿。

说明：
- 对外 API 仅从 `fiducial_pose.api` 暴露，避免下游耦合内部模块结构。
"""

from fiducial_pose.calib_io import load_camera_intr_extr_from_calib_json
from fiducial_pose.api import (
    CameraIntrinsics,
    TagConfig,
    TagDetection,
    VehiclePose,
    VehiclePoseEstimate,
    WorldToCameraExtrinsics,
    estimate_vehicle_pose_from_tag,
)

__all__ = [
    "load_camera_intr_extr_from_calib_json",
    "CameraIntrinsics",
    "WorldToCameraExtrinsics",
    "TagConfig",
    "TagDetection",
    "VehiclePose",
    "VehiclePoseEstimate",
    "estimate_vehicle_pose_from_tag",
]
