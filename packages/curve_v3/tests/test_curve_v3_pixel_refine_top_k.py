import unittest

import numpy as np

from curve_v3.adapters.camera_rig import CameraRig
from curve_v3.configs import CurveV3Config, PhysicsConfig, PixelConfig, PosteriorConfig
from curve_v3.dynamics import propagate_post_bounce_state
from curve_v3.posterior.fusion import reweight_candidates_and_select_best
from curve_v3.types import BallObservation, BounceEvent, Candidate, Obs2D


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


class TestCurveV3PixelRefineTopK(unittest.TestCase):
    def test_pixel_refine_top_k_limits_candidates(self):
        """验证：开启 pixel_refine_top_k 后，像素闭环只会影响 top-K 候选。"""

        rng = np.random.default_rng(0)

        cfg_all = CurveV3Config(
            physics=PhysicsConfig(gravity=9.8),
            posterior=PosteriorConfig(
                fit_params="v+axz",
                max_post_points=5,
                weight_sigma_m=0.15,
                posterior_prior_strength=50.0,
                posterior_prior_sigma_v=0.2,
                posterior_prior_sigma_a=0.8,
            ),
            pixel=PixelConfig(
                pixel_enabled=True,
                pixel_refine_top_k=None,
                pixel_max_iters=1,
                pixel_huber_delta_px=3.0,
                pixel_lm_damping=1e-3,
                pixel_fd_rel_step=1e-3,
            ),
        )
        cfg_top1 = CurveV3Config(
            physics=PhysicsConfig(gravity=9.8),
            posterior=PosteriorConfig(
                fit_params="v+axz",
                max_post_points=5,
                weight_sigma_m=0.15,
                posterior_prior_strength=50.0,
                posterior_prior_sigma_v=0.2,
                posterior_prior_sigma_a=0.8,
            ),
            pixel=PixelConfig(
                pixel_enabled=True,
                pixel_refine_top_k=1,
                pixel_max_iters=1,
                pixel_huber_delta_px=3.0,
                pixel_lm_damping=1e-3,
                pixel_fd_rel_step=1e-3,
            ),
        )

        rig = CameraRig(cameras={"cam0": _PinholeProjector(fx=800.0, fy=800.0, cx=640.0, cy=360.0)})

        time_base_abs = 1000.0
        t_b_rel = 0.2
        bounce = BounceEvent(
            t_rel=float(t_b_rel),
            x=0.0,
            z=0.0,
            v_minus=np.array([0.0, -3.0, 6.0], dtype=float),
        )

        # 候选 A：与 3D 点域观测一致（因此 3D-only MAP 会更偏向 A）。
        cand_a = Candidate(
            e=0.7,
            kt=0.65,
            weight=0.5,
            v_plus=np.array([0.9, 3.0, 6.9], dtype=float),
            ax=0.0,
            az=0.0,
        )

        # 候选 B：与像素观测一致（因此像素闭环会更偏向 B）。
        cand_b = Candidate(
            e=0.7,
            kt=0.65,
            weight=0.5,
            v_plus=np.array([1.2, 3.2, 6.5], dtype=float),
            ax=0.25,
            az=-0.10,
        )

        sigma_px = 0.5
        cov_uv = (sigma_px * sigma_px) * np.eye(2, dtype=float)

        post_points: list[BallObservation] = []
        for t_rel in (0.25, 0.30, 0.35, 0.40):
            tau = float(t_rel - t_b_rel)

            pos_a, _ = propagate_post_bounce_state(bounce=bounce, candidate=cand_a, tau=tau, cfg=cfg_all)
            pos_b, _ = propagate_post_bounce_state(bounce=bounce, candidate=cand_b, tau=tau, cfg=cfg_all)

            # 3D 点域观测：贴近 A。
            pos_obs = np.asarray(pos_a, dtype=float) + rng.normal(0.0, 0.005, size=(3,))

            # 2D 像素观测：贴近 B。
            uv_b = rig.project("cam0", pos_b)
            uv_obs = np.asarray(uv_b, dtype=float) + rng.normal(0.0, sigma_px, size=(2,))

            post_points.append(
                BallObservation(
                    x=float(pos_obs[0]),
                    y=float(pos_obs[1]),
                    z=float(pos_obs[2]),
                    t=float(time_base_abs + float(t_rel)),
                    conf=1.0,
                    obs_2d_by_camera={
                        "cam0": Obs2D(uv=np.asarray(uv_obs, dtype=float), cov_uv=cov_uv, sigma_px=sigma_px),
                    },
                )
            )

        candidates = [cand_a, cand_b]

        # 全量像素闭环：应选择 B（像素一致性更好）。
        _, _, best_idx_all, _ = reweight_candidates_and_select_best(
            bounce=bounce,
            candidates=candidates,
            post_points=post_points,
            time_base_abs=time_base_abs,
            camera_rig=rig,
            cfg=cfg_all,
        )
        self.assertIsNotNone(best_idx_all)
        assert best_idx_all is not None
        self.assertEqual(int(best_idx_all), 1)

        # Top-1 像素闭环：先按 3D-only 选出 A 做像素精化，B 不会因像素闭环“翻盘”。
        _, _, best_idx_top, _ = reweight_candidates_and_select_best(
            bounce=bounce,
            candidates=candidates,
            post_points=post_points,
            time_base_abs=time_base_abs,
            camera_rig=rig,
            cfg=cfg_top1,
        )
        self.assertIsNotNone(best_idx_top)
        assert best_idx_top is not None
        self.assertEqual(int(best_idx_top), 0)


if __name__ == "__main__":
    unittest.main()
