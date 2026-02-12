"""CurvePredictorV3 的观测缓冲与点级权重构造。

设计目标：
    - 把 `curve_v3.core.CurvePredictorV3` 中“观测缓存 + 权重策略”的横切逻辑拆出来，
      降低 core.py 的职责密度。
    - 不改变任何数值行为：该模块只是把原本在 `CurvePredictorV3.add_observation()` 里的
      数据结构与权重计算封装成一个小对象。

说明：
    - 该缓冲区仅负责：
        1) 维护 time_base_abs 与 t_rel
        2) 缓存原始 BallObservation（用于向 posterior 透传 2D 观测等可选字段）
        3) 构造 prefit 用的 (x,y,z,t_rel) 与 (xw,yw,zw)
    - prefit/posterior/corridor 等算法编排仍然属于 `curve_v3.core`。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from curve_v3.adapters.camera_rig import CameraRig

import numpy as np

from curve_v3.configs import CurveV3Config
from curve_v3.low_snr import weights_from_conf
from curve_v3.types import BallObservation
from curve_v3.utils.prefit_pixel_gating import prefit_pixel_weight_multiplier


@dataclass
class ObservationBuffer:
    """在线观测缓冲区（含点级权重）。

    说明：
        - 该对象是 `CurvePredictorV3` 的内部实现细节，不对外暴露稳定 API。
        - 权重构造遵循 core.py 的既有约定：
            * 时间递增权重与 conf 权重必须解耦，避免 conf 级联爆炸。
            * 像素一致性加权仅在显式开启且具备 CameraRig + 2D 观测时生效。
    """

    fit_x_weight: float
    fit_y_weight: float
    fit_z_weight: float

    time_base_abs: float | None = None

    observations: list[BallObservation] = field(default_factory=list)

    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)
    zs: list[float] = field(default_factory=list)
    ts_rel: list[float] = field(default_factory=list)

    confs: list[float | None] = field(default_factory=list)

    x_ws: list[float] = field(default_factory=list)
    y_ws: list[float] = field(default_factory=list)
    z_ws: list[float] = field(default_factory=list)

    # 时间递增权重（仅随点序号递推，不包含 conf/像素权重）
    w_time_x: float = 1.0
    w_time_y: float = 1.0
    w_time_z: float = 1.0

    def reset(self) -> None:
        """清空缓冲区。"""

        self.time_base_abs = None
        self.observations.clear()

        self.xs.clear()
        self.ys.clear()
        self.zs.clear()
        self.ts_rel.clear()
        self.confs.clear()

        self.x_ws.clear()
        self.y_ws.clear()
        self.z_ws.clear()

        self.w_time_x = 1.0
        self.w_time_y = 1.0
        self.w_time_z = 1.0

    @property
    def size(self) -> int:
        return int(len(self.ts_rel))

    def append(
        self,
        obs: BallObservation,
        *,
        cfg: CurveV3Config,
        camera_rig: CameraRig | None,
    ) -> None:
        """追加一帧观测并构造点级权重。"""

        if self.time_base_abs is None:
            self.time_base_abs = float(obs.t)

        # 保留原始观测（含可选的 2D 信息），供 posterior 使用。
        self.observations.append(obs)

        t_rel = float(float(obs.t) - float(self.time_base_abs))

        self.xs.append(float(obs.x))
        self.ys.append(float(obs.y))
        self.zs.append(float(obs.z))
        self.ts_rel.append(float(t_rel))

        conf = getattr(obs, "conf", None)
        self.confs.append(conf)

        # 时间递增权重（仅与点序相关）。
        if self.x_ws:
            self.w_time_x *= float(self.fit_x_weight)
            self.w_time_y *= float(self.fit_y_weight)
            self.w_time_z *= float(self.fit_z_weight)

        w_time_x = float(self.w_time_x)
        w_time_y = float(self.w_time_y)
        w_time_z = float(self.w_time_z)

        # conf -> 每点权重（仅在显式开启且提供 conf 时启用）。
        if bool(cfg.low_snr.low_snr_enabled) and (conf is not None):
            cmin = float(cfg.low_snr.low_snr_conf_cmin)
            sx0 = float(cfg.low_snr.low_snr_sigma_x0_m)
            sy0 = float(cfg.low_snr.low_snr_sigma_y0_m)
            sz0 = float(cfg.low_snr.low_snr_sigma_z0_m)

            conf_val = float(conf)
            wx = float(weights_from_conf([conf_val], sigma0=sx0, c_min=cmin)[0])
            wy = float(weights_from_conf([conf_val], sigma0=sy0, c_min=cmin)[0])
            wz = float(weights_from_conf([conf_val], sigma0=sz0, c_min=cmin)[0])
        else:
            wx, wy, wz = 1.0, 1.0, 1.0

        # prefit 像素一致性加权（可选）。
        if bool(cfg.prefit.prefit_pixel_enabled) and (camera_rig is not None):
            try:
                w2d = prefit_pixel_weight_multiplier(obs=obs, camera_rig=camera_rig, cfg=cfg)
            except Exception:
                # 约定：任意异常必须严格回退为 1.0，保证不影响旧行为。
                w2d = 1.0
        else:
            w2d = 1.0

        wx = float(wx) * float(w2d)
        wy = float(wy) * float(w2d)
        wz = float(wz) * float(w2d)

        self.x_ws.append(w_time_x * wx)
        self.y_ws.append(w_time_y * wy)
        self.z_ws.append(w_time_z * wz)

    def as_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """以 ndarray 形式返回当前缓冲数据。"""

        t = np.asarray(self.ts_rel, dtype=float)
        xs = np.asarray(self.xs, dtype=float)
        ys = np.asarray(self.ys, dtype=float)
        zs = np.asarray(self.zs, dtype=float)

        xw = np.asarray(self.x_ws, dtype=float)
        yw = np.asarray(self.y_ws, dtype=float)
        zw = np.asarray(self.z_ws, dtype=float)

        return t, xs, ys, zs, xw, yw, zw
