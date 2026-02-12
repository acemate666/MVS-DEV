import unittest

import numpy as np

from curve_v3 import (
    BallObservation,
    BounceEvent,
    Candidate,
    CurveV3Config,
    PhysicsConfig,
    PosteriorConfig,
)
from interception import (
    HitTargetDiagnostics,
    HitTargetResult,
    HitTargetStabilizer,
    InterceptionConfig,
    select_hit_target_prefit_only,
    select_hit_target_with_post,
)


class TestInterceptionTarget(unittest.TestCase):
    def _make_bounce(self, *, t_rel: float, x: float, z: float, y_contact: float) -> BounceEvent:
        return BounceEvent(
            t_rel=float(t_rel),
            x=float(x),
            z=float(z),
            v_minus=np.zeros((3,), dtype=float),
            y=float(y_contact),
        )

    def test_prefit_only_multipeak_not_mean(self):
        """双峰分布下，不应输出均值/中位数，而应选择命中概率最大的单点。"""

        curve_cfg = CurveV3Config(
            physics=PhysicsConfig(gravity=10.0),
            posterior=PosteriorConfig(fit_params="v_only"),
        )
        y_contact = float(curve_cfg.bounce_contact_y())

        bounce = self._make_bounce(t_rel=0.0, x=0.0, z=0.0, y_contact=y_contact)

        # 两个峰：对称分布在 x=±vx*tau。若错误用均值，会落在 x≈0。
        candidates = [
            Candidate(
                e=0.7,
                kt=0.65,
                weight=0.5,
                v_plus=np.array([-1.0, 6.0, 0.0], dtype=float),
                ax=0.0,
                az=0.0,
            ),
            Candidate(
                e=0.7,
                kt=0.65,
                weight=0.5,
                v_plus=np.array([1.0, 6.0, 0.0], dtype=float),
                ax=0.0,
                az=0.0,
            ),
        ]

        cfg = InterceptionConfig(
            y_min=0.5,
            y_max=0.5,
            num_heights=1,
            r_hit_m=0.10,
            min_valid_candidates=1,
            min_crossing_prob=0.0,
            score_alpha_time=0.0,
            score_lambda_width=0.0,
            score_mu_crossing=0.0,
        )

        out = select_hit_target_prefit_only(
            bounce=bounce,
            candidates=candidates,
            time_base_abs=0.0,
            t_now_abs=0.0,
            cfg=cfg,
            curve_cfg=curve_cfg,
        )

        self.assertTrue(out.valid)
        self.assertIsNotNone(out.target)
        assert out.target is not None

        # 目标 x 不应接近 0（均值），而应接近其中一个峰。
        self.assertLess(out.target.x, 0.0)
        self.assertGreater(abs(out.target.x), 0.5)

        self.assertIsNotNone(out.diag.target_y)
        self.assertAlmostEqual(float(out.diag.crossing_prob), 1.0, places=6)
        self.assertAlmostEqual(float(out.diag.p_hit), 0.5, places=6)
        self.assertAlmostEqual(float(out.diag.score or 0.0), 0.5, places=6)
        self.assertTrue(bool(out.diag.multi_peak_flag))

    def test_prefit_only_crossing_prob_too_low_invalid(self):
        """当某高度的穿越概率质量过低时，应降级为 invalid。"""

        curve_cfg = CurveV3Config(
            physics=PhysicsConfig(gravity=10.0),
            posterior=PosteriorConfig(fit_params="v_only"),
        )
        y_contact = float(curve_cfg.bounce_contact_y())
        bounce = self._make_bounce(t_rel=0.0, x=0.0, z=0.0, y_contact=y_contact)

        # 只有一个候选能达到 y=0.5，其余候选 vy 太小无实根。
        candidates = [
            Candidate(
                e=0.7,
                kt=0.65,
                weight=0.30,
                v_plus=np.array([1.0, 6.0, 0.0], dtype=float),
            ),
            Candidate(
                e=0.7,
                kt=0.65,
                weight=0.35,
                v_plus=np.array([1.0, 0.10, 0.0], dtype=float),
            ),
            Candidate(
                e=0.7,
                kt=0.65,
                weight=0.35,
                v_plus=np.array([-1.0, 0.10, 0.0], dtype=float),
            ),
        ]

        cfg = InterceptionConfig(
            y_min=0.5,
            y_max=0.5,
            num_heights=1,
            r_hit_m=0.10,
            min_valid_candidates=1,
            min_crossing_prob=0.40,
            score_alpha_time=0.0,
            score_lambda_width=0.0,
            score_mu_crossing=0.0,
        )

        out = select_hit_target_prefit_only(
            bounce=bounce,
            candidates=candidates,
            time_base_abs=0.0,
            t_now_abs=0.0,
            cfg=cfg,
            curve_cfg=curve_cfg,
        )

        self.assertFalse(out.valid)
        self.assertIsNotNone(out.diag.per_height)
        per = out.diag.per_height
        assert per is not None
        self.assertEqual(len(per), 1)
        self.assertLess(float(per[0].crossing_prob), float(cfg.min_crossing_prob))
        self.assertFalse(per[0].is_valid)

    def test_with_post_reweights_to_consistent_candidate(self):
        """加入 post 点后，应通过 MAP+重加权偏向与观测一致的候选分支。"""

        curve_cfg = CurveV3Config(
            physics=PhysicsConfig(gravity=10.0),
            posterior=PosteriorConfig(
                fit_params="v_only",
                weight_sigma_m=0.05,
                posterior_prior_strength=50.0,
                posterior_prior_sigma_v=0.2,
                posterior_anchor_weight=0.0,
                max_post_points=5,
            ),
        )
        y_contact = float(curve_cfg.bounce_contact_y())
        bounce = self._make_bounce(t_rel=0.0, x=0.0, z=0.0, y_contact=y_contact)

        # 真实分支：vx=+1；干扰分支：vx=-4（需要很大校正才能解释 post 点）。
        cand_true = Candidate(
            e=0.7,
            kt=0.65,
            weight=0.5,
            v_plus=np.array([1.0, 6.0, 0.0], dtype=float),
        )
        cand_bad = Candidate(
            e=0.7,
            kt=0.65,
            weight=0.5,
            v_plus=np.array([-4.0, 6.0, 0.0], dtype=float),
        )

        post_points: list[BallObservation] = []
        for tau in (0.10, 0.20, 0.30):
            x = bounce.x + float(cand_true.v_plus[0]) * float(tau)
            y = y_contact + float(cand_true.v_plus[1]) * float(tau) - 0.5 * float(curve_cfg.physics.gravity) * float(tau) * float(tau)
            z = bounce.z + float(cand_true.v_plus[2]) * float(tau)
            post_points.append(BallObservation(x=float(x), y=float(y), z=float(z), t=float(tau)))

        cfg = InterceptionConfig(
            y_min=0.5,
            y_max=0.5,
            num_heights=1,
            r_hit_m=0.10,
            min_valid_candidates=1,
            min_crossing_prob=0.0,
            score_alpha_time=0.0,
            score_lambda_width=0.0,
            score_mu_crossing=0.0,
        )

        out = select_hit_target_with_post(
            bounce=bounce,
            candidates=[cand_true, cand_bad],
            post_points=post_points,
            time_base_abs=0.0,
            t_now_abs=0.0,
            cfg=cfg,
            curve_cfg=curve_cfg,
        )

        self.assertTrue(out.valid)
        self.assertIsNotNone(out.target)
        self.assertIsNotNone(out.diag.w_max)
        assert out.target is not None
        assert out.diag.w_max is not None

        self.assertGreater(float(out.diag.w_max), 0.80)
        self.assertGreater(float(out.target.x), 0.0)
        self.assertEqual(out.diag.target_source, "map")

    def test_stabilizer_keeps_previous_when_not_improved(self):
        cfg = InterceptionConfig(y_min=0.5, y_max=0.5, num_heights=1)
        stabilizer = HitTargetStabilizer()

        prev = HitTargetResult(
            valid=True,
            reason=None,
            target=None,
            diag=HitTargetDiagnostics(
                target_y=0.5,
                crossing_prob=1.0,
                valid_candidates=5,
                width_xz=1.0,
                p_hit=0.5,
                score=1.0,
                multi_peak_flag=False,
                target_source="phit",
                w_max=0.6,
            ),
        )
        cur = HitTargetResult(
            valid=True,
            reason=None,
            target=None,
            diag=HitTargetDiagnostics(
                target_y=0.5,
                crossing_prob=1.0,
                valid_candidates=5,
                width_xz=0.99,
                p_hit=0.5,
                score=1.0 + float(cfg.hysteresis_score_gain) * 0.5,
                multi_peak_flag=False,
                target_source="phit",
                w_max=0.6,
            ),
        )

        out1 = stabilizer.update(prev, cfg)
        self.assertIs(out1, prev)

        out2 = stabilizer.update(cur, cfg)
        # score 提升不足，应保持上一帧
        self.assertIs(out2, prev)
