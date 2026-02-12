import unittest

import numpy as np

from curve_v3.adapters.camera_rig import CameraRig
from curve_v3.configs import CurveV3Config, LowSnrConfig, PhysicsConfig, PixelConfig, PrefitConfig
from curve_v3.core import BallObservation, CurvePredictorV3
from curve_v3.types import Obs2D


class _PinholeProjector:
    """极简针孔投影器（用于单测）。"""

    def __init__(self, *, fx: float, fy: float, cx: float, cy: float) -> None:
        self._fx = float(fx)
        self._fy = float(fy)
        self._cx = float(cx)
        self._cy = float(cy)

    def project_world_to_pixel(self, p_world: np.ndarray) -> np.ndarray:
        p = np.asarray(p_world, dtype=float).reshape(3)
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        if z <= 1e-6:
            raise ValueError("point behind camera")
        u = self._fx * (x / z) + self._cx
        v = self._fy * (y / z) + self._cy
        return np.array([u, v], dtype=float)


class TestCurveV3PrefitPixelGating(unittest.TestCase):
    def test_prefit_pixel_gating_reduces_land_time_bias(self):
        """验证：prefit 像素一致性加权能抑制 3D 离群点对 t_land 的影响。"""

        g = 9.8

        cfg_no = CurveV3Config(
            physics=PhysicsConfig(gravity=g),
            low_snr=LowSnrConfig(low_snr_enabled=False),
            pixel=PixelConfig(pixel_enabled=False),
            prefit=PrefitConfig(
                prefit_robust_iters=0,
                prefit_pixel_enabled=False,
            ),
        )
        cfg_yes = CurveV3Config(
            physics=PhysicsConfig(gravity=g),
            low_snr=LowSnrConfig(low_snr_enabled=False),
            pixel=PixelConfig(pixel_enabled=False),
            prefit=PrefitConfig(
                prefit_robust_iters=0,
                prefit_pixel_enabled=True,
                prefit_pixel_gate_tau_px=20.0,
                prefit_pixel_huber_delta_px=5.0,
                prefit_pixel_min_cameras=1,
            ),
        )

        rig = CameraRig(cameras={"cam0": _PinholeProjector(fx=800.0, fy=800.0, cx=640.0, cy=360.0)})

        time_base_abs = 1000.0
        t_land_true = 0.5
        y_contact = float(cfg_no.bounce_contact_y())

        # 构造一条无噪声反弹前轨迹，并在一个时刻注入“3D 离群点”（y 偏大）。
        x0, y0, z0 = 0.2, 1.0, 1.0
        vx, vz = 1.0, 8.0
        vy = (0.5 * g * t_land_true * t_land_true + y_contact - y0) / t_land_true

        sigma_px = 0.5
        cov_uv = (sigma_px * sigma_px) * np.eye(2, dtype=float)

        ts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        obs_list: list[BallObservation] = []
        for t_rel in ts:
            x = x0 + vx * t_rel
            y = y0 + vy * t_rel - 0.5 * g * t_rel * t_rel
            z = z0 + vz * t_rel

            p_true = np.array([x, y, z], dtype=float)
            uv_true = rig.project("cam0", p_true)

            # 在 t=0.4s 处制造一个明显的 3D 离群点：y 偏大，但 2D 仍然是“真值”。
            y_obs = float(y)
            if abs(float(t_rel) - 0.4) < 1e-12:
                y_obs = float(y_obs + 0.25)

            obs_list.append(
                BallObservation(
                    x=float(x),
                    y=float(y_obs),
                    z=float(z),
                    t=float(time_base_abs + float(t_rel)),
                    obs_2d_by_camera={
                        "cam0": Obs2D(uv=np.asarray(uv_true, dtype=float), cov_uv=cov_uv, sigma_px=sigma_px),
                    },
                )
            )

        engine_no = CurvePredictorV3(config=cfg_no, camera_rig=None)
        engine_yes = CurvePredictorV3(config=cfg_yes, camera_rig=rig)

        for o in obs_list:
            engine_no.add_observation(o)
            engine_yes.add_observation(o)

        b0 = engine_no.get_bounce_event()
        b1 = engine_yes.get_bounce_event()
        self.assertIsNotNone(b0)
        self.assertIsNotNone(b1)
        assert b0 is not None
        assert b1 is not None

        err0 = abs(float(b0.t_rel) - float(t_land_true))
        err1 = abs(float(b1.t_rel) - float(t_land_true))

        self.assertLess(err1, err0)


if __name__ == "__main__":
    unittest.main()
