import unittest

import numpy as np

from curve_v3 import fit_posterior_map_for_candidate
from curve_v3.configs import CurveV3Config, CorridorConfig, PhysicsConfig, PosteriorConfig, PriorConfig
from curve_v3.core import BallObservation, CurvePredictorV3
from curve_v3.posterior.fit_fused import fit_posterior_fused_map


def _make_prebounce_observations(
    *,
    g: float,
    t_land: float,
    y_contact: float,
) -> tuple[list[BallObservation], dict]:
    """构造一段简单的反弹前抛体运动观测，使其在 y==y_contact 处落地。"""

    # Pre-bounce parameters.
    x0 = 0.2
    z0 = 1.0
    y0 = 1.0
    vx = 1.0
    vz = 8.0
    vy = (0.5 * g * t_land * t_land + y_contact - y0) / t_land

    obs: list[BallObservation] = []
    ts = np.linspace(0.0, t_land, 6)
    for t in ts:
        x = x0 + vx * t
        y = y0 + vy * t - 0.5 * g * t * t
        z = z0 + vz * t
        obs.append(BallObservation(x=float(x), y=float(y), z=float(z), t=float(t)))

    params = {
        "x0": x0,
        "y0": y0,
        "z0": z0,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "t_land": t_land,
        "y_contact": y_contact,
    }
    return obs, params


class TestCurveV3Fusion(unittest.TestCase):
    def test_best_candidate_and_fused_posterior_anchor(self):
        g = 9.8
        t_land = 0.5

        e_true = 0.70
        kt_true = 0.65

        cfg = CurveV3Config(
            physics=PhysicsConfig(gravity=g),
            prior=PriorConfig(e_bins=(e_true, 0.85), kt_bins=(kt_true, 0.85)),
            posterior=PosteriorConfig(
                fit_params="v_only",
                max_post_points=5,
                weight_sigma_m=0.15,
                posterior_prior_strength=50.0,
                posterior_prior_sigma_v=0.2,
                posterior_anchor_weight=0.0,
            ),
        )

        engine = CurvePredictorV3(config=cfg)

        y_contact = float(cfg.bounce_contact_y())

        pre_obs, pre_p = _make_prebounce_observations(g=g, t_land=t_land, y_contact=y_contact)
        for o in pre_obs:
            engine.add_observation(o)

        bounce = engine.get_bounce_event()
        self.assertIsNotNone(bounce)
        assert bounce is not None

        # Create a couple of post-bounce points using the engine's own candidate model.
        # This avoids over-constraining the test on how the pre-bounce fit estimates v_minus.
        # We generate points using the (e_true, kt_true) hypothesis directly from the
        # engine's candidate builder output.
        candidates_initial = engine.get_prior_candidates()
        self.assertTrue(candidates_initial)
        c0 = next((c for c in candidates_initial if abs(c.e - e_true) < 1e-9 and abs(c.kt - kt_true) < 1e-9), None)
        self.assertIsNotNone(c0)
        assert c0 is not None

        post_pts: list[BallObservation] = []
        for tau in (0.10, 0.20):
            x = bounce.x + float(c0.v_plus[0]) * tau
            y = y_contact + float(c0.v_plus[1]) * tau - 0.5 * g * tau * tau
            z = bounce.z + float(c0.v_plus[2]) * tau
            pt = BallObservation(x=float(x), y=float(y), z=float(z), t=float(t_land + tau))
            post_pts.append(pt)
            engine.add_observation(pt)

        best = engine.get_best_candidate()
        self.assertIsNotNone(best)
        assert best is not None

        # Bounce estimate may shift as more points arrive; always use the latest.
        bounce2 = engine.get_bounce_event()
        self.assertIsNotNone(bounce2)
        assert bounce2 is not None

        # Best candidate must equal argmin of the v1.1 per-candidate posterior objective.
        cands = engine.get_prior_candidates()
        j_posts: list[float] = []
        for c in cands:
            out = fit_posterior_map_for_candidate(
                bounce=bounce2,
                post_points=post_pts,
                candidate=c,
                time_base_abs=engine.time_base_abs,
                cfg=cfg,
            )
            if out is None:
                j_posts.append(float("inf"))
            else:
                _, j = out
                j_posts.append(float(j))

        best_idx = int(np.argmin(np.asarray(j_posts, dtype=float)))
        self.assertAlmostEqual(best.e, float(cands[best_idx].e))
        self.assertAlmostEqual(best.kt, float(cands[best_idx].kt))

        # Now run a fused posterior fit with a single "bad" x measurement and check
        # that MAP anchoring pulls vx back toward the best candidate.
        tau = 0.2
        x_bad = bounce2.x + float(best.v_plus[0]) * tau + 0.4  # large positive x error
        y = y_contact + float(best.v_plus[1]) * tau - 0.5 * g * tau * tau
        z = bounce2.z + float(best.v_plus[2]) * tau
        bad_pt = BallObservation(
            x=float(x_bad),
            y=float(y),
            z=float(z),
            t=float(bounce2.t_rel + tau),
        )

        fused = fit_posterior_fused_map(
            bounce=bounce2,
            post_points=[bad_pt],
            best=best,
            time_base_abs=engine.time_base_abs,
            cfg=cfg,
        )
        self.assertIsNotNone(fused)
        assert fused is not None

        # v_only + 单点情况下的闭式 MAP：
        #   min (1/σ^2) * (x_b + vx*tau - x)^2 + q * (vx - vx0)^2
        # 解为：vx = ((tau^2/σ^2)*vx_ls + q*vx0) / ((tau^2/σ^2) + q)
        vx_ls = (x_bad - bounce2.x) / tau
        q = cfg.posterior.posterior_prior_strength * (1.0 / (cfg.posterior.posterior_prior_sigma_v**2))

        sigma = (
            float(cfg.posterior.posterior_obs_sigma_m)
            if cfg.posterior.posterior_obs_sigma_m is not None
            else float(cfg.posterior.weight_sigma_m)
        )
        sigma = max(sigma, 1e-6)
        tau2_over_sigma2 = (tau * tau) / (sigma * sigma)

        vx_expected = (tau2_over_sigma2 * vx_ls + q * float(best.v_plus[0])) / (tau2_over_sigma2 + q)

        self.assertAlmostEqual(fused.vx, float(vx_expected), places=6)

    def test_bounce_event_freezes_after_post_rise(self):
        """加入反弹后点后，bounce_event 不应随着更多 post 点继续向后漂移。

        这是一类真实数据常见问题：反弹附近有遮挡/缺失时，如果 prefit 误用 post 点，
        触地时刻（t_land）会被推迟，进而拉爆 posterior 的 tau 并导致轨迹更差。
        """

        g = 9.8
        t_land = 0.5

        e_true = 0.70
        kt_true = 0.65

        cfg = CurveV3Config(
            physics=PhysicsConfig(gravity=g),
            prior=PriorConfig(e_bins=(e_true, 0.85), kt_bins=(kt_true, 0.85)),
            posterior=PosteriorConfig(
                fit_params="v_only",
                max_post_points=5,
                weight_sigma_m=0.15,
            ),
        )

        engine = CurvePredictorV3(config=cfg)
        y_contact = float(cfg.bounce_contact_y())

        pre_obs, _ = _make_prebounce_observations(g=g, t_land=t_land, y_contact=y_contact)
        for o in pre_obs:
            engine.add_observation(o)

        candidates_initial = engine.get_prior_candidates()
        self.assertTrue(candidates_initial)
        c0 = next((c for c in candidates_initial if abs(c.e - e_true) < 1e-9 and abs(c.kt - kt_true) < 1e-9), None)
        self.assertIsNotNone(c0)
        assert c0 is not None

        bounce0 = engine.get_bounce_event()
        self.assertIsNotNone(bounce0)
        assert bounce0 is not None

        # 先加入两个反弹后点（应能触发“post 段 y 上升”的冻结判定）。
        for tau in (0.10, 0.20):
            x = bounce0.x + float(c0.v_plus[0]) * tau
            y = y_contact + float(c0.v_plus[1]) * tau - 0.5 * g * tau * tau
            z = bounce0.z + float(c0.v_plus[2]) * tau
            engine.add_observation(BallObservation(x=float(x), y=float(y), z=float(z), t=float(t_land + tau)))

        bounce2 = engine.get_bounce_event()
        self.assertIsNotNone(bounce2)
        assert bounce2 is not None

        best = engine.get_best_candidate() or c0

        # 再加入更多 post 点，bounce_event 应保持不变（不再随 N 继续漂移）。
        base = float(engine.time_base_abs or 0.0)
        for tau in (0.30, 0.40, 0.50):
            x = bounce2.x + float(best.v_plus[0]) * tau
            y = y_contact + float(best.v_plus[1]) * tau - 0.5 * g * tau * tau
            z = bounce2.z + float(best.v_plus[2]) * tau
            engine.add_observation(
                BallObservation(
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    t=float(base + float(bounce2.t_rel) + tau),
                )
            )

        bounce5 = engine.get_bounce_event()
        self.assertIsNotNone(bounce5)
        assert bounce5 is not None

        self.assertAlmostEqual(float(bounce5.t_rel), float(bounce2.t_rel), places=12)

    def test_corridor_by_time_includes_posterior_anchor_acceleration(self):
        g = 9.8
        t_land = 0.5

        e_true = 0.70
        kt_true = 0.65

        # Single prior candidate so corridor mean is easy to reason about.
        cfg = CurveV3Config(
            physics=PhysicsConfig(gravity=g),
            prior=PriorConfig(e_bins=(e_true,), kt_bins=(kt_true,)),
            posterior=PosteriorConfig(
                fit_params="v+axz",
                max_post_points=5,
                weight_sigma_m=0.15,
                posterior_prior_strength=0.0,  # pure LS for posterior
                posterior_anchor_weight=0.5,
            ),
            corridor=CorridorConfig(corridor_dt=0.1, corridor_horizon_s=0.6),
        )

        engine = CurvePredictorV3(config=cfg)
        y_contact = float(cfg.bounce_contact_y())
        pre_obs, pre_p = _make_prebounce_observations(g=g, t_land=t_land, y_contact=y_contact)
        for o in pre_obs:
            engine.add_observation(o)

        bounce = engine.get_bounce_event()
        self.assertIsNotNone(bounce)
        assert bounce is not None

        ax_true = 1.0
        az_true = -0.5

        # Two points are enough to solve v+axz (5 params) with 6 equations.
        cands = engine.get_prior_candidates()
        self.assertEqual(len(cands), 1)
        c0 = cands[0]

        for tau in (0.2, 0.4):
            x = bounce.x + float(c0.v_plus[0]) * tau + 0.5 * ax_true * tau * tau
            y = y_contact + float(c0.v_plus[1]) * tau - 0.5 * g * tau * tau
            z = bounce.z + float(c0.v_plus[2]) * tau + 0.5 * az_true * tau * tau
            engine.add_observation(BallObservation(x=float(x), y=float(y), z=float(z), t=float(t_land + tau)))

        bounce2 = engine.get_bounce_event()
        self.assertIsNotNone(bounce2)
        assert bounce2 is not None

        corridor = engine.get_corridor_by_time()
        self.assertIsNotNone(corridor)
        assert corridor is not None

        # Pick the last corridor sample (largest tau) and check that mu_x deviates from
        # the linear prior by about alpha * 0.5 * ax * tau^2.
        t_rel = float(corridor.t_rel[-1])
        tau = t_rel - float(bounce2.t_rel)
        self.assertGreaterEqual(tau, 0.0)

        posterior = getattr(engine, "_posterior_state", None)
        self.assertIsNotNone(posterior)
        assert posterior is not None

        alpha = float(cfg.posterior.posterior_anchor_weight)
        # Use the latest candidate list (weights may have been scaled) and latest bounce.
        cands2 = engine.get_prior_candidates()
        self.assertEqual(len(cands2), 1)
        c1 = cands2[0]

        x_cand = bounce2.x + float(c1.v_plus[0]) * tau + 0.5 * float(c1.ax) * tau * tau
        x_post = bounce2.x + float(posterior.vx) * tau + 0.5 * float(posterior.ax) * tau * tau
        x_expected = (1.0 - alpha) * x_cand + alpha * x_post

        mu_x = float(corridor.mu_xz[-1, 0])
        self.assertAlmostEqual(mu_x, float(x_expected), places=3)

    def test_corridor_on_plane_y_range_fixed_grid_with_invalid_placeholders(self):
        g = 9.8
        t_land = 0.5

        # Use multiple kt values but a single e so all candidates share the same vy.
        cfg = CurveV3Config(
            physics=PhysicsConfig(gravity=g),
            prior=PriorConfig(e_bins=(0.70,), kt_bins=(0.45, 0.65)),
            corridor=CorridorConfig(corridor_horizon_s=1.5),
        )

        engine = CurvePredictorV3(config=cfg)
        y_contact = float(cfg.bounce_contact_y())
        pre_obs, _ = _make_prebounce_observations(g=g, t_land=t_land, y_contact=y_contact)
        for o in pre_obs:
            engine.add_observation(o)

        y_min = 0.2
        y_max = 3.0
        step = 0.2
        ys = list(np.arange(y_min, y_max + 1e-9, step, dtype=float))

        planes = engine.corridor_on_plane_y_range(y_min=y_min, y_max=y_max, step=step)
        self.assertEqual(len(planes), len(ys))

        # Low plane should be crossed by all candidates.
        self.assertTrue(planes[0].is_valid)
        self.assertAlmostEqual(float(planes[0].valid_ratio), 1.0, places=6)
        self.assertAlmostEqual(float(planes[0].crossing_prob), 1.0, places=6)

        # High plane should be unreachable and represented by an invalid placeholder.
        self.assertFalse(planes[-1].is_valid)
        self.assertAlmostEqual(float(planes[-1].valid_ratio), 0.0, places=6)
        self.assertAlmostEqual(float(planes[-1].crossing_prob), 0.0, places=6)

    def test_ballistic_smoke(self):
        g = 9.8
        t_land = 0.5

        cfg = CurveV3Config(
            physics=PhysicsConfig(gravity=g),
            prior=PriorConfig(e_bins=(0.70,), kt_bins=(0.65,)),
            corridor=CorridorConfig(corridor_dt=0.1, corridor_horizon_s=0.6),
        )

        engine = CurvePredictorV3(config=cfg)
        y_contact = float(cfg.bounce_contact_y())
        pre_obs, _ = _make_prebounce_observations(g=g, t_land=t_land, y_contact=y_contact)
        for o in pre_obs:
            engine.add_observation(o)

        corridor = engine.get_corridor_by_time()
        self.assertIsNotNone(corridor)
        assert corridor is not None
        self.assertTrue(np.isfinite(corridor.mu_xz).all())
        self.assertTrue(np.isfinite(corridor.cov_xz).all())

        planes = engine.corridor_on_plane_y_range(y_min=0.1, y_max=0.5, step=0.1)
        self.assertEqual(len(planes), 5)

    def test_corridor_quantiles_smoke(self):
        g = 9.8
        t_land = 0.5

        cfg = CurveV3Config(
            physics=PhysicsConfig(gravity=g),
            prior=PriorConfig(e_bins=(0.70,), kt_bins=(0.45, 0.85)),
            corridor=CorridorConfig(
                corridor_dt=0.1,
                corridor_horizon_s=0.2,
                corridor_quantile_levels=(0.0, 1.0),
            ),
        )

        engine = CurvePredictorV3(config=cfg)
        y_contact = float(cfg.bounce_contact_y())
        pre_obs, _ = _make_prebounce_observations(g=g, t_land=t_land, y_contact=y_contact)
        for o in pre_obs:
            engine.add_observation(o)

        bounce = engine.get_bounce_event()
        self.assertIsNotNone(bounce)
        assert bounce is not None

        cands = engine.get_prior_candidates()
        self.assertEqual(len(cands), 2)

        corridor = engine.get_corridor_by_time()
        self.assertIsNotNone(corridor)
        assert corridor is not None
        self.assertIsNotNone(corridor.quantiles_xz)
        assert corridor.quantiles_xz is not None

        # For q=(0,1), the quantiles should match min/max over candidates.
        # Use the first positive tau sample.
        self.assertGreaterEqual(len(corridor.t_rel), 2)
        tau = float(corridor.t_rel[1] - bounce.t_rel)
        self.assertGreater(tau, 0.0)

        xs = [float(bounce.x + float(c.v_plus[0]) * tau) for c in cands]
        zs = [float(bounce.z + float(c.v_plus[2]) * tau) for c in cands]

        qxz = corridor.quantiles_xz[1]
        x_min, x_max = float(min(xs)), float(max(xs))
        z_min, z_max = float(min(zs)), float(max(zs))

        self.assertAlmostEqual(float(qxz[0, 0]), x_min, places=6)
        self.assertAlmostEqual(float(qxz[1, 0]), x_max, places=6)
        self.assertAlmostEqual(float(qxz[0, 1]), z_min, places=6)
        self.assertAlmostEqual(float(qxz[1, 1]), z_max, places=6)

    def test_corridor_components_smoke(self):
        g = 9.8
        t_land = 0.5

        cfg = CurveV3Config(
            physics=PhysicsConfig(gravity=g),
            prior=PriorConfig(e_bins=(0.70,), kt_bins=(0.45, 0.65, 0.85)),
            corridor=CorridorConfig(corridor_components_k=2),
        )

        engine = CurvePredictorV3(config=cfg)
        y_contact = float(cfg.bounce_contact_y())
        pre_obs, _ = _make_prebounce_observations(g=g, t_land=t_land, y_contact=y_contact)
        for o in pre_obs:
            engine.add_observation(o)

        # Pick a reachable plane slightly below bounce contact.
        plane = engine.corridor_on_plane_y(float(y_contact - 0.2))
        self.assertIsNotNone(plane)
        assert plane is not None
        self.assertTrue(plane.is_valid)

        comps = plane.components
        self.assertIsNotNone(comps)
        assert comps is not None
        self.assertGreaterEqual(len(comps), 1)

        total_w = float(sum(float(c.weight) for c in comps))
        self.assertAlmostEqual(total_w, float(plane.crossing_prob), places=6)

        for c in comps:
            self.assertTrue(np.isfinite(c.mu_xz).all())
            self.assertTrue(np.isfinite(c.cov_xz).all())
            self.assertTrue(np.isfinite(float(c.t_rel_mu)))
            self.assertTrue(np.isfinite(float(c.t_rel_var)))


if __name__ == "__main__":
    unittest.main()
