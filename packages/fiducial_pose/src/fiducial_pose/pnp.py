"""PnP 位姿解算：由 Tag 四角点得到 tag->camera 变换。

关键点：
- 本模块只做几何求解；Tag 检测（像素角点提取）由上游负责。
- OpenCV 坐标系约定：
  - 相机坐标系：x 向右，y 向下，z 向前（从相机指向场景）。
  - 像素坐标：u 向右，v 向下。

角点顺序约定（默认）：
- corners_px 与 object_points 的顺序必须一致。
- 本实现默认 object_points 为：
  [(-s/2, -s/2, 0), (s/2, -s/2, 0), (s/2, s/2, 0), (-s/2, s/2, 0)]
  也就是“以 tag 中心为原点、先上左再上右再下右再下左”的顺序，
  其中 y 轴朝下（与像素 y 同向），便于上游按图像位置直觉排序。

如果你的 detector 给的角点顺序不同，请在调用前进行重排。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from fiducial_pose.transforms import make_T
from fiducial_pose.types import CameraIntrinsics, as_np_f64


@dataclass(frozen=True, slots=True)
class PnPResult:
    """PnP 求解结果。"""

    T_cam_from_tag: np.ndarray  # (4,4)
    reproj_rmse_px: float


def tag_object_points(*, tag_size_m: float) -> np.ndarray:
    """构造 tag 4 角点在 tag 坐标系下的 3D 坐标（单位：米）。"""

    s = float(tag_size_m)
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"tag_size_m must be positive, got {tag_size_m}")

    h = 0.5 * s
    # 说明：y 轴朝下（与像素 v 同向）。
    # 顺序：TL, TR, BR, BL
    return np.array(
        [
            [-h, -h, 0.0],
            [h, -h, 0.0],
            [h, h, 0.0],
            [-h, h, 0.0],
        ],
        dtype=np.float64,
    )


def solve_tag_pnp(
    *,
    corners_px: np.ndarray,
    intr: CameraIntrinsics,
    tag_size_m: float,
) -> PnPResult:
    """用四角点解算 tag 在相机坐标系下的位姿。

    Args:
        corners_px: (4,2) 像素坐标（u,v）。必须与 object_points 顺序一致。
        intr: 相机内参与畸变。
        tag_size_m: Tag 边长（米）。

    Returns:
        PnPResult，其中 T_cam_from_tag 表示 tag->camera。
    """

    img_pts = as_np_f64(corners_px, (4, 2))
    obj_pts = tag_object_points(tag_size_m=float(tag_size_m))

    K = as_np_f64(intr.K, (3, 3))
    dist = np.asarray(intr.dist, dtype=np.float64).reshape(-1)

    # 说明：这里默认用 ITERATIVE。
    # IPPE_SQUARE 理论上更适合平面方形标记，但不同 OpenCV 版本/构建下对点序的隐含约定
    # 更严格，容易在“角点顺序已匹配但仍得到错误解”的场景里踩坑。
    # 先保证正确性与可预测性；若后续确实需要提速/提稳，再显式引入可配置的 PnP 策略。
    method = cv2.SOLVEPNP_ITERATIVE

    ok, rvec, tvec = cv2.solvePnP(
        objectPoints=obj_pts,
        imagePoints=img_pts,
        cameraMatrix=K,
        distCoeffs=dist,
        flags=int(method),
    )
    if not bool(ok):
        raise RuntimeError("solvePnP failed")

    R, _ = cv2.Rodrigues(rvec)
    T = make_T(R=np.asarray(R, dtype=np.float64), t=np.asarray(tvec, dtype=np.float64).reshape(3))

    # 计算重投影 RMSE（像素）。
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    proj = np.asarray(proj, dtype=np.float64).reshape(4, 2)
    err = proj - img_pts
    rmse = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))

    return PnPResult(T_cam_from_tag=T, reproj_rmse_px=rmse)
