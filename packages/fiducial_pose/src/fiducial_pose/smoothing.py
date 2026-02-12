"""时序稳定：简单滤波 + 短时丢失预测。

设计目标：
- 让在线输出更平滑（尤其是 yaw），并在短时丢失 tag 时给出“可用但可识别为预测”的输出。
- 避免引入复杂滤波器（例如 EKF），先用可解释、可快速落地的方案。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from fiducial_pose.types import VehiclePose


def _unwrap_angle_rad(*, angle: float, ref: float) -> float:
    """把 angle 拉到 ref 附近（消除 2pi 跳变）。"""

    a = float(angle)
    r = float(ref)
    # 把差值压到 [-pi, pi]
    d = (a - r + math.pi) % (2.0 * math.pi) - math.pi
    return r + d


@dataclass
class PoseSmoother:
    """对 VehiclePose 做指数平滑，并支持短时预测。"""

    alpha_pos: float = 0.35
    alpha_yaw: float = 0.25
    max_predict_s: float = 0.20

    _last_t: float | None = None
    _last_pose: VehiclePose | None = None
    _vx: float = 0.0
    _vy: float = 0.0
    _vz: float = 0.0
    _vyaw: float = 0.0
    _has_vel: bool = False

    def reset(self) -> None:
        self._last_t = None
        self._last_pose = None
        self._vx = self._vy = self._vz = 0.0
        self._vyaw = 0.0
        self._has_vel = False

    def update(self, *, t_s: float, meas: VehiclePose | None) -> VehiclePose | None:
        """更新滤波器。

        Args:
            t_s: 当前时间戳（秒）。建议用上游统一的 wall/capture 时间。
            meas: 当前测量；若为 None 表示本帧没有有效 tag。

        Returns:
            平滑后的 pose；若连续丢失超过 max_predict_s，则返回 None。
        """

        t = float(t_s)
        if self._last_t is None or self._last_pose is None:
            if meas is None:
                return None
            self._last_t = t
            self._last_pose = meas
            self._has_vel = False
            return meas

        dt = t - float(self._last_t)
        if not math.isfinite(dt) or dt < 0:
            dt = 0.0

        # 丢失：短时用速度外推。
        if meas is None:
            if (not self._has_vel) or dt > float(self.max_predict_s):
                return None
            p = self._last_pose
            pred = VehiclePose(
                x_m=float(p.x_m + self._vx * dt),
                y_m=float(p.y_m + self._vy * dt),
                z_m=float(p.z_m + self._vz * dt),
                yaw_rad=float(p.yaw_rad + self._vyaw * dt),
            )
            self._last_t = t
            self._last_pose = pred
            return pred

        # 有测量：先计算速度（基于上一次输出）。
        prev = self._last_pose
        yaw_meas = _unwrap_angle_rad(angle=float(meas.yaw_rad), ref=float(prev.yaw_rad))

        if dt > 1e-6:
            self._vx = (float(meas.x_m) - float(prev.x_m)) / dt
            self._vy = (float(meas.y_m) - float(prev.y_m)) / dt
            self._vz = (float(meas.z_m) - float(prev.z_m)) / dt
            self._vyaw = (float(yaw_meas) - float(prev.yaw_rad)) / dt
            self._has_vel = True

        ap = float(self.alpha_pos)
        ay = float(self.alpha_yaw)
        ap = min(max(ap, 0.0), 1.0)
        ay = min(max(ay, 0.0), 1.0)

        out = VehiclePose(
            x_m=float(ap * float(meas.x_m) + (1.0 - ap) * float(prev.x_m)),
            y_m=float(ap * float(meas.y_m) + (1.0 - ap) * float(prev.y_m)),
            z_m=float(ap * float(meas.z_m) + (1.0 - ap) * float(prev.z_m)),
            yaw_rad=float(ay * float(yaw_meas) + (1.0 - ay) * float(prev.yaw_rad)),
        )

        self._last_t = t
        self._last_pose = out
        return out
