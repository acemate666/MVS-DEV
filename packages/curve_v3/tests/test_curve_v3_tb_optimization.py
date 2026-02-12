import unittest

import numpy as np

from curve_v3 import fit_posterior_map_for_candidate
from curve_v3.configs import CurveV3Config, PhysicsConfig, PosteriorConfig
from curve_v3.types import BallObservation, BounceEvent, Candidate


class TestCurveV3TbOptimization(unittest.TestCase):
    def test_posterior_optimize_tb_recovers_tb(self):
        """在 tb 有偏的情况下，开启 tb 搜索应能把 t_b 拉回并降低目标函数。"""

        g = 9.8
        time_base_abs = 100.0
        y_contact = 0.033

        tb_true = 0.50
        tb0 = tb_true + 0.03  # 有偏的 bounce.t_rel

        v_plus_true = np.array([1.2, 3.8, 2.1], dtype=float)
        taus = [0.04, 0.08, 0.12, 0.16]

        post_points: list[BallObservation] = []
        for tau in taus:
            x = 0.0 + float(v_plus_true[0]) * float(tau)
            y = y_contact + float(v_plus_true[1]) * float(tau) - 0.5 * g * float(tau) * float(tau)
            z = 0.0 + float(v_plus_true[2]) * float(tau)
            post_points.append(
                BallObservation(
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    t=float(time_base_abs + tb_true + float(tau)),
                )
            )

        # pre-bounce 速度用于把 (x_b,z_b) 随 tb 平移到一致的位置。
        v_minus = np.array([0.7, -4.0, -0.5], dtype=float)
        x0 = float(v_minus[0]) * float(tb0 - tb_true)
        z0 = float(v_minus[2]) * float(tb0 - tb_true)

        bounce = BounceEvent(
            t_rel=float(tb0),
            x=float(x0),
            z=float(z0),
            v_minus=np.asarray(v_minus, dtype=float),
            y=float(y_contact),
        )

        cand = Candidate(
            e=0.70,
            kt=0.65,
            weight=1.0,
            v_plus=np.asarray(v_plus_true, dtype=float),
            kt_angle_rad=0.0,
            ax=0.0,
            az=0.0,
        )

        cfg_off = CurveV3Config(
            physics=PhysicsConfig(gravity=g, bounce_contact_y_m=y_contact),
            posterior=PosteriorConfig(
                fit_params="v_only",
                max_post_points=5,
                posterior_prior_strength=0.0,
                posterior_optimize_tb=False,
            ),
        )
        out_off = fit_posterior_map_for_candidate(
            bounce=bounce,
            post_points=post_points,
            candidate=cand,
            time_base_abs=time_base_abs,
            cfg=cfg_off,
        )
        self.assertIsNotNone(out_off)
        assert out_off is not None
        st_off, j_off = out_off

        cfg_on = CurveV3Config(
            physics=PhysicsConfig(gravity=g, bounce_contact_y_m=y_contact),
            posterior=PosteriorConfig(
                fit_params="v_only",
                max_post_points=5,
                posterior_prior_strength=0.0,
                posterior_optimize_tb=True,
                posterior_tb_search_window_s=0.06,
                posterior_tb_search_step_s=0.001,
                posterior_tb_prior_sigma_s=0.03,
            ),
        )
        out_on = fit_posterior_map_for_candidate(
            bounce=bounce,
            post_points=post_points,
            candidate=cand,
            time_base_abs=time_base_abs,
            cfg=cfg_on,
        )
        self.assertIsNotNone(out_on)
        assert out_on is not None
        st_on, j_on = out_on

        self.assertLess(abs(float(st_on.t_b_rel) - tb_true), abs(float(st_off.t_b_rel) - tb_true))
        self.assertLess(float(j_on), float(j_off))

    def test_posterior_map_cost_is_finite_smoke(self):
        """解析模型下，posterior 的 J_post 应返回有限值（smoke）。"""

        g = 9.8
        time_base_abs = 100.0
        y_contact = 0.033

        tb = 0.50
        v_minus = np.array([0.5, -4.0, -0.2], dtype=float)
        bounce = BounceEvent(
            t_rel=float(tb),
            x=0.0,
            z=0.0,
            v_minus=np.asarray(v_minus, dtype=float),
            y=float(y_contact),
        )

        cand = Candidate(
            e=0.70,
            kt=0.65,
            weight=1.0,
            v_plus=np.array([1.0, 4.0, 2.0], dtype=float),
            kt_angle_rad=0.0,
            ax=0.0,
            az=0.0,
        )

        post_points: list[BallObservation] = []
        for tau in (0.06, 0.12):
            x = float(cand.v_plus[0]) * float(tau)
            y = y_contact + float(cand.v_plus[1]) * float(tau) - 0.5 * g * float(tau) * float(tau)
            z = float(cand.v_plus[2]) * float(tau)
            post_points.append(
                BallObservation(
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    t=float(time_base_abs + tb + float(tau)),
                )
            )

        cfg = CurveV3Config(
            physics=PhysicsConfig(gravity=g, bounce_contact_y_m=y_contact),
            posterior=PosteriorConfig(
                fit_params="v_only",
                max_post_points=5,
                posterior_prior_strength=0.0,
            ),
        )

        out = fit_posterior_map_for_candidate(
            bounce=bounce,
            post_points=post_points,
            candidate=cand,
            time_base_abs=time_base_abs,
            cfg=cfg,
        )
        self.assertIsNotNone(out)
        assert out is not None
        _, j = out
        self.assertTrue(np.isfinite(float(j)))
