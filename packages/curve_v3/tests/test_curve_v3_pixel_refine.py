import unittest

import numpy as np

from curve_v3 import fit_posterior_map_for_candidate
from curve_v3.adapters.camera_rig import CameraRig
from curve_v3.configs import CurveV3Config, PhysicsConfig, PixelConfig, PosteriorConfig
from curve_v3.dynamics import propagate_post_bounce_state
from curve_v3.posterior.utils import bounce_event_for_tb
from curve_v3.types import BallObservation, BounceEvent, Candidate, Obs2D


class _PinholeProjector:
    """极简针孔投影器（用于单测）。

    说明：
        - 这里不建模相机位姿/畸变，只提供一个可复现的“世界点 -> 像素”的映射。
        - 该实现仅用于测试像素域闭环能否降低重投影误差。
    """

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


class _AlwaysFailProjector:
    """总是投影失败的投影器（用于测试像素域回退逻辑）。"""

    def project_world_to_pixel(self, p_world: np.ndarray) -> np.ndarray:
        raise ValueError("projection failed")


def _pixel_sse(
    *,
    st,  # PosteriorState
    bounce: BounceEvent,
    post_points: list[BallObservation],
    time_base_abs: float,
    rig: CameraRig,
    cfg: CurveV3Config,
) -> float:
    """计算给定状态下的重投影 SSE（px^2）。"""

    bounce2 = bounce_event_for_tb(bounce=bounce, t_b_rel=float(st.t_b_rel))
    cand = Candidate(
        e=0.7,
        kt=0.65,
        weight=1.0,
        v_plus=np.array([float(st.vx), float(st.vy), float(st.vz)], dtype=float),
        ax=float(getattr(st, "ax", 0.0)),
        az=float(getattr(st, "az", 0.0)),
    )

    sse = 0.0
    for p in post_points:
        obs2d = getattr(p, "obs_2d_by_camera", None)
        if not obs2d:
            continue

        t_rel = float(p.t - float(time_base_abs))
        tau = t_rel - float(st.t_b_rel)
        if tau <= 0.0:
            continue

        pos, _ = propagate_post_bounce_state(bounce=bounce2, candidate=cand, tau=float(tau), cfg=cfg)

        for cam, o in obs2d.items():
            uv_pred = rig.project(str(cam), pos)
            duv = np.asarray(uv_pred, dtype=float).reshape(2) - np.asarray(o.uv, dtype=float).reshape(2)
            sse += float(np.dot(duv, duv))

    return float(sse)


class TestCurveV3PixelRefine(unittest.TestCase):
    def test_pixel_refine_reduces_reprojection_error(self):
        rng = np.random.default_rng(0)

        cfg_3d = CurveV3Config(
            physics=PhysicsConfig(gravity=9.8),
            posterior=PosteriorConfig(
                fit_params="v+axz",
                max_post_points=5,
                weight_sigma_m=0.15,
                posterior_prior_strength=1.0,
                posterior_prior_sigma_v=2.0,
                posterior_prior_sigma_a=8.0,
            ),
            pixel=PixelConfig(pixel_enabled=False),
        )
        cfg_px = CurveV3Config(
            physics=PhysicsConfig(gravity=9.8),
            posterior=PosteriorConfig(
                fit_params="v+axz",
                max_post_points=5,
                weight_sigma_m=0.15,
                posterior_prior_strength=1.0,
                posterior_prior_sigma_v=2.0,
                posterior_prior_sigma_a=8.0,
            ),
            pixel=PixelConfig(
                pixel_enabled=True,
                pixel_max_iters=2,
                pixel_huber_delta_px=3.0,
                pixel_lm_damping=1e-3,
                pixel_fd_rel_step=1e-3,
            ),
        )

        rig = CameraRig(
            cameras={
                "cam0": _PinholeProjector(fx=800.0, fy=800.0, cx=640.0, cy=360.0),
            }
        )

        time_base_abs = 1000.0
        t_b_rel = 0.2

        bounce = BounceEvent(
            t_rel=float(t_b_rel),
            x=0.0,
            z=0.0,
            v_minus=np.array([0.0, -3.0, 6.0], dtype=float),
        )

        # 构造一条“真值”轨迹，用它生成像素观测；同时让 3D 观测带系统性偏差。
        true = Candidate(
            e=0.7,
            kt=0.65,
            weight=1.0,
            v_plus=np.array([1.2, 3.2, 6.5], dtype=float),
            ax=0.25,
            az=-0.10,
        )

        sigma_px = 0.5
        cov_uv = (sigma_px * sigma_px) * np.eye(2, dtype=float)

        post_points: list[BallObservation] = []
        for t_rel in (0.25, 0.30, 0.35, 0.40):
            tau = float(t_rel - t_b_rel)
            pos_true, _ = propagate_post_bounce_state(bounce=bounce, candidate=true, tau=tau, cfg=cfg_px)
            uv_true = rig.project("cam0", pos_true)

            uv_obs = np.asarray(uv_true, dtype=float) + rng.normal(0.0, sigma_px, size=(2,))

            # 3D 观测：加入明显偏差，模拟三角化在遮挡/弱视差下的不稳定。
            pos_obs = np.asarray(pos_true, dtype=float) + np.array([0.08, -0.05, 0.12], dtype=float)

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

        # 初始候选：刻意设得不准。
        cand0 = Candidate(
            e=0.7,
            kt=0.65,
            weight=1.0,
            v_plus=np.array([0.5, 2.0, 7.5], dtype=float),
            ax=0.0,
            az=0.0,
        )

        out_3d = fit_posterior_map_for_candidate(
            bounce=bounce,
            post_points=post_points,
            candidate=cand0,
            time_base_abs=time_base_abs,
            camera_rig=None,
            cfg=cfg_3d,
        )
        self.assertIsNotNone(out_3d)
        assert out_3d is not None
        st_3d, _ = out_3d

        out_px = fit_posterior_map_for_candidate(
            bounce=bounce,
            post_points=post_points,
            candidate=cand0,
            time_base_abs=time_base_abs,
            camera_rig=rig,
            cfg=cfg_px,
        )
        self.assertIsNotNone(out_px)
        assert out_px is not None
        st_px, _ = out_px

        sse_3d = _pixel_sse(st=st_3d, bounce=bounce, post_points=post_points, time_base_abs=time_base_abs, rig=rig, cfg=cfg_px)
        sse_px = _pixel_sse(st=st_px, bounce=bounce, post_points=post_points, time_base_abs=time_base_abs, rig=rig, cfg=cfg_px)

        # 像素域闭环应能显著降低重投影误差（至少比 3D-only 更差不会发生）。
        self.assertLess(sse_px, sse_3d)

    def test_pixel_refine_falls_back_when_projection_fails(self):
        cfg_3d = CurveV3Config(
            physics=PhysicsConfig(gravity=9.8),
            posterior=PosteriorConfig(
                fit_params="v+axz",
                max_post_points=5,
                weight_sigma_m=0.15,
                posterior_prior_strength=1.0,
                posterior_prior_sigma_v=2.0,
                posterior_prior_sigma_a=8.0,
            ),
            pixel=PixelConfig(pixel_enabled=False),
        )
        cfg_px = CurveV3Config(
            physics=PhysicsConfig(gravity=9.8),
            posterior=PosteriorConfig(
                fit_params="v+axz",
                max_post_points=5,
                weight_sigma_m=0.15,
                posterior_prior_strength=1.0,
                posterior_prior_sigma_v=2.0,
                posterior_prior_sigma_a=8.0,
            ),
            pixel=PixelConfig(
                pixel_enabled=True,
                pixel_max_iters=2,
                pixel_huber_delta_px=3.0,
                pixel_lm_damping=1e-3,
                pixel_fd_rel_step=1e-3,
            ),
        )

        rig_fail = CameraRig(cameras={"cam0": _AlwaysFailProjector()})

        time_base_abs = 1000.0
        t_b_rel = 0.2
        bounce = BounceEvent(
            t_rel=float(t_b_rel),
            x=0.0,
            z=0.0,
            v_minus=np.array([0.0, -3.0, 6.0], dtype=float),
        )

        # 构造少量 post 点：像素观测存在，但投影器总失败。
        sigma_px = 0.5
        cov_uv = (sigma_px * sigma_px) * np.eye(2, dtype=float)
        post_points = [
            BallObservation(
                x=0.1,
                y=0.3,
                z=1.0,
                t=float(time_base_abs + 0.25),
                conf=1.0,
                obs_2d_by_camera={
                    "cam0": Obs2D(uv=np.array([640.0, 360.0], dtype=float), cov_uv=cov_uv, sigma_px=sigma_px),
                },
            ),
            BallObservation(
                x=0.2,
                y=0.4,
                z=1.4,
                t=float(time_base_abs + 0.30),
                conf=1.0,
                obs_2d_by_camera={
                    "cam0": Obs2D(uv=np.array([641.0, 361.0], dtype=float), cov_uv=cov_uv, sigma_px=sigma_px),
                },
            ),
        ]

        cand0 = Candidate(
            e=0.7,
            kt=0.65,
            weight=1.0,
            v_plus=np.array([0.5, 2.0, 7.5], dtype=float),
            ax=0.0,
            az=0.0,
        )

        out_3d = fit_posterior_map_for_candidate(
            bounce=bounce,
            post_points=post_points,
            candidate=cand0,
            time_base_abs=time_base_abs,
            camera_rig=None,
            cfg=cfg_3d,
        )
        self.assertIsNotNone(out_3d)
        assert out_3d is not None
        st_3d, _ = out_3d

        out_px = fit_posterior_map_for_candidate(
            bounce=bounce,
            post_points=post_points,
            candidate=cand0,
            time_base_abs=time_base_abs,
            camera_rig=rig_fail,
            cfg=cfg_px,
        )
        self.assertIsNotNone(out_px)
        assert out_px is not None
        st_px, _ = out_px

        # 投影失败时，像素域闭环必须回退到 3D 点域输出（数值应一致）。
        self.assertAlmostEqual(float(st_px.vx), float(st_3d.vx), places=10)
        self.assertAlmostEqual(float(st_px.vy), float(st_3d.vy), places=10)
        self.assertAlmostEqual(float(st_px.vz), float(st_3d.vz), places=10)
        self.assertAlmostEqual(float(st_px.ax), float(st_3d.ax), places=10)
        self.assertAlmostEqual(float(st_px.az), float(st_3d.az), places=10)
