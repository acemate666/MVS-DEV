"""坐标变换工具：4x4 齐次矩阵。

约定：
- 用 4x4 矩阵表示刚体变换，记作 T_dst_from_src。
- 点从 src 坐标系变换到 dst：X_dst = T_dst_from_src @ X_src（X 为齐次坐标 (4,)）。

注意：
- 本仓库的相机外参默认是 world->camera（见 `WorldToCameraExtrinsics`）。
  若你需要 camera->world，请使用 `invert_T()`。
"""

from __future__ import annotations

import math

import numpy as np


def make_T(*, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """由 R,t 构造 4x4 齐次矩阵。"""

    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    """求刚体变换的逆。"""

    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T must be (4,4), got {T.shape}")

    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R_inv
    out[:3, 3] = t_inv
    return out


def compose_T(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """复合变换：先 B 再 A（即 A @ B）。"""

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape != (4, 4) or B.shape != (4, 4):
        raise ValueError(f"A,B must be (4,4), got {A.shape} and {B.shape}")
    return (A @ B).astype(np.float64)


def yaw_from_R_wv(R_wv: np.ndarray) -> float:
    """从 world<-vehicle 的旋转矩阵提取 yaw（绕世界 Z 轴）。

    前提：
    - 世界坐标系为右手系，Z 轴近似“竖直向上”。
    - yaw 定义为 vehicle 的 x 轴在世界 x-y 平面内的朝向：
      $yaw = atan2(R[1,0], R[0,0])$。

    若你的 world 坐标系并非该约定（例如 Z 不是竖直），请不要使用该函数。
    """

    R = np.asarray(R_wv, dtype=np.float64).reshape(3, 3)
    return float(math.atan2(float(R[1, 0]), float(R[0, 0])))
