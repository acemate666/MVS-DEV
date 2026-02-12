"""fiducial_pose 对外稳定入口（public API）。

本包目标：
- 输入：相机内外参 + Tag 四角点像素坐标 + Tag->车体安装偏置。
- 输出：车体在世界坐标系下的位置 (x,y,z) 与 yaw。

说明：
- 本包不负责图像采集与 AprilTag 检测；你只需要把 detector 的角点结果喂进来。
- 只要你保证角点顺序与本包 object_points 的顺序一致，PnP 与变换链条就是确定的。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fiducial_pose.pnp import PnPResult, solve_tag_pnp
from fiducial_pose.transforms import compose_T, invert_T, make_T, yaw_from_R_wv
from fiducial_pose.types import (
    CameraIntrinsics,
    TagConfig,
    TagDetection,
    VehiclePose,
    WorldToCameraExtrinsics,
)


@dataclass(frozen=True, slots=True)
class VehiclePoseEstimate:
    """一次位姿估计输出（包含诊断字段）。"""

    pose: VehiclePose
    T_world_from_vehicle: np.ndarray  # (4,4)
    pnp: PnPResult
    tag_id: int


def estimate_vehicle_pose_from_tag(
    *,
    det: TagDetection,
    tag: TagConfig,
    intr: CameraIntrinsics,
    extr: WorldToCameraExtrinsics,
    T_vehicle_from_tag: np.ndarray | None = None,
) -> VehiclePoseEstimate:
    """由单个 Tag 的角点估计车体位姿。

    坐标系链条：
    - 外参给出 world->camera（T_cam_from_world）。
    - PnP 解出 tag->camera（T_cam_from_tag）。
    - 安装偏置给出 vehicle<-tag（T_vehicle_from_tag）。

    目标：
    - 计算 world<-vehicle：
      T_world_from_vehicle = inv(T_cam_from_world) @ T_cam_from_tag @ inv(T_vehicle_from_tag)

    Args:
        det: TagDetection（角点像素坐标）。
        tag: TagConfig（family/size）。目前 family 仅做记录，不参与计算。
        intr: 相机内参。
        extr: 相机外参（world->camera）。
        T_vehicle_from_tag: 固定安装偏置（vehicle<-tag）。
            - placeholder 阶段可传 None（默认单位矩阵）。
            - 后续应由实测/标定给出。

    Returns:
        VehiclePoseEstimate。
    """

    if T_vehicle_from_tag is None:
        T_vehicle_from_tag = np.eye(4, dtype=np.float64)

    T_vehicle_from_tag = np.asarray(T_vehicle_from_tag, dtype=np.float64)
    if T_vehicle_from_tag.shape != (4, 4):
        raise ValueError(f"T_vehicle_from_tag must be (4,4), got {T_vehicle_from_tag.shape}")

    pnp = solve_tag_pnp(corners_px=det.corners_px, intr=intr, tag_size_m=float(tag.size_m))

    T_cam_from_world = make_T(R=np.asarray(extr.R_wc, dtype=np.float64), t=np.asarray(extr.t_wc, dtype=np.float64))
    T_world_from_cam = invert_T(T_cam_from_world)

    T_tag_from_vehicle = invert_T(T_vehicle_from_tag)

    T_world_from_vehicle = compose_T(compose_T(T_world_from_cam, pnp.T_cam_from_tag), T_tag_from_vehicle)

    x, y, z = (
        float(T_world_from_vehicle[0, 3]),
        float(T_world_from_vehicle[1, 3]),
        float(T_world_from_vehicle[2, 3]),
    )
    yaw = yaw_from_R_wv(T_world_from_vehicle[:3, :3])

    return VehiclePoseEstimate(
        pose=VehiclePose(x_m=x, y_m=y, z_m=z, yaw_rad=float(yaw)),
        T_world_from_vehicle=T_world_from_vehicle,
        pnp=pnp,
        tag_id=int(det.tag_id),
    )


__all__ = [
    "VehiclePoseEstimate",
    "estimate_vehicle_pose_from_tag",
    "CameraIntrinsics",
    "WorldToCameraExtrinsics",
    "TagConfig",
    "TagDetection",
    "VehiclePose",
]
