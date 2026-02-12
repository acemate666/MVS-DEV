"""数据结构：相机/Tag/小车位姿。

说明：
- 本包只负责“位姿解算与坐标变换”的核心算法，不依赖采集链路（例如 mvs）。
- AprilTag 的“检测器”实现不放在本包的强依赖里：
  - 你可以用任意库得到 tag 的 4 角点像素坐标，再交给本包做 PnP 与坐标系链式变换。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """相机内参（OpenCV 口径）。

    Attributes:
        K: 相机内参矩阵 (3,3)。
        dist: 畸变参数 (N,)；若未知可传空或全 0。
    """

    K: np.ndarray
    dist: np.ndarray


@dataclass(frozen=True, slots=True)
class WorldToCameraExtrinsics:
    """相机外参：world -> camera（与 OpenCV/本仓库 tennis3d 一致）。

    约定：
        $X_c = R_{wc} X_w + t_{wc}$

    Attributes:
        R_wc: (3,3)
        t_wc: (3,)
    """

    R_wc: np.ndarray
    t_wc: np.ndarray


@dataclass(frozen=True, slots=True)
class TagConfig:
    """单个 Tag 的物理配置。"""

    family: str
    size_m: float


@dataclass(frozen=True, slots=True)
class TagDetection:
    """Tag 检测结果（只保留后续解算所需的最小信息）。

    说明：
    - corners_px 必须是 4 个角点，顺序需要与 object_points 的定义一致。
      若你的 detector 输出顺序不稳定，请在调用 PnP 前先做排序。
    """

    tag_id: int
    corners_px: np.ndarray  # (4,2)
    decision_margin: float | None = None


@dataclass(frozen=True, slots=True)
class VehiclePose:
    """小车位姿输出（世界坐标系）。

    Attributes:
        x_m, y_m, z_m: 世界坐标（米）。
        yaw_rad: 绕世界 Z 轴的偏航角（弧度）。若坐标系不满足“Z 轴竖直”，请不要使用该字段。
    """

    x_m: float
    y_m: float
    z_m: float
    yaw_rad: float


def as_np_f64(x: np.ndarray | Iterable[float], shape: tuple[int, ...]) -> np.ndarray:
    """把输入转为 float64 ndarray 并校验形状。"""

    a = np.asarray(x, dtype=np.float64)
    a = a.reshape(shape)
    return a
