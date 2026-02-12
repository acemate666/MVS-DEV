from __future__ import annotations

import math

import cv2
import numpy as np

from fiducial_pose import (
    CameraIntrinsics,
    TagConfig,
    TagDetection,
    WorldToCameraExtrinsics,
    estimate_vehicle_pose_from_tag,
)
from fiducial_pose.pnp import solve_tag_pnp, tag_object_points


def _rotvec_from_R(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64).reshape(3, 3))
    return np.asarray(rvec, dtype=np.float64).reshape(3)


def _Rz(yaw: float) -> np.ndarray:
    c = float(math.cos(float(yaw)))
    s = float(math.sin(float(yaw)))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rotation_angle(R: np.ndarray) -> float:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    v = (tr - 1.0) * 0.5
    v = min(1.0, max(-1.0, v))
    return float(math.acos(v))


def test_solve_tag_pnp_recovers_pose_on_synthetic_projection() -> None:
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    intr = CameraIntrinsics(K=K, dist=np.zeros(5, dtype=np.float64))

    tag_size_m = 0.16
    obj = tag_object_points(tag_size_m=tag_size_m)

    # 构造一个非退化的姿态（避免极端角度导致平面二义性变差）。
    R_gt, _ = cv2.Rodrigues(np.array([0.2, -0.1, 0.15], dtype=np.float64))
    t_gt = np.array([0.05, -0.02, 1.8], dtype=np.float64)
    rvec_gt = _rotvec_from_R(R_gt)

    img, _ = cv2.projectPoints(obj, rvec_gt, t_gt.reshape(3, 1), K, intr.dist)
    corners_px = np.asarray(img, dtype=np.float64).reshape(4, 2)

    out = solve_tag_pnp(corners_px=corners_px, intr=intr, tag_size_m=tag_size_m)

    T = out.T_cam_from_tag
    R_est = T[:3, :3]
    t_est = T[:3, 3]

    R_err = R_est @ np.asarray(R_gt, dtype=np.float64).T
    ang = _rotation_angle(R_err)

    assert float(np.linalg.norm(t_est - t_gt)) < 1e-3
    assert float(ang) < 1e-3
    assert float(out.reproj_rmse_px) < 1e-6


def test_estimate_vehicle_pose_chain_identity_extrinsics_and_mount() -> None:
    K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    intr = CameraIntrinsics(K=K, dist=np.zeros(5, dtype=np.float64))

    tag = TagConfig(family="tag36h11", size_m=0.10)

    yaw_gt = 0.3
    R_gt = _Rz(yaw_gt)
    t_gt = np.array([0.12, 0.03, 1.5], dtype=np.float64)
    rvec_gt = _rotvec_from_R(R_gt)

    obj = tag_object_points(tag_size_m=float(tag.size_m))
    img, _ = cv2.projectPoints(obj, rvec_gt, t_gt.reshape(3, 1), K, intr.dist)
    corners_px = np.asarray(img, dtype=np.float64).reshape(4, 2)

    det = TagDetection(tag_id=7, corners_px=corners_px)

    extr = WorldToCameraExtrinsics(R_wc=np.eye(3, dtype=np.float64), t_wc=np.zeros(3, dtype=np.float64))

    est = estimate_vehicle_pose_from_tag(det=det, tag=tag, intr=intr, extr=extr, T_vehicle_from_tag=None)

    assert abs(float(est.pose.x_m) - float(t_gt[0])) < 1e-3
    assert abs(float(est.pose.y_m) - float(t_gt[1])) < 1e-3
    assert abs(float(est.pose.z_m) - float(t_gt[2])) < 1e-3

    # yaw 允许 2pi 等价；这里比较到 [-pi, pi]。
    dy = (float(est.pose.yaw_rad) - float(yaw_gt) + math.pi) % (2.0 * math.pi) - math.pi
    assert abs(float(dy)) < 1e-3
