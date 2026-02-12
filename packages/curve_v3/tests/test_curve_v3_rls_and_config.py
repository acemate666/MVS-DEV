import unittest

import numpy as np

from curve_v3 import fit_posterior_map_for_candidate
from curve_v3.configs import CurveV3Config, PhysicsConfig, PosteriorConfig, PriorConfig
from curve_v3.prior import build_prior_candidates
from curve_v3.types import BallObservation, BounceEvent, Candidate


class TestCurveV3RlsAndConfig(unittest.TestCase):
    def test_rls_lambda_1_matches_batch_normal_equations(self):
        """验证 RLS（信息形式递推）在 λ=1 时与批量正规方程解一致。"""

        g = 9.8
        y_contact = 0.033

        bounce = BounceEvent(
            t_rel=0.0,
            x=0.2,
            z=1.0,
            v_minus=np.array([1.0, -2.0, 8.0], dtype=float),
            y=y_contact,
        )

        # 生成一组“无噪声”的反弹后点，用于构造线性系统。
        vx_true, vy_true, vz_true = 1.2, 2.3, 3.4
        ax_true, az_true = 0.8, -0.6
        taus = [0.08, 0.16, 0.24]
        post_points: list[BallObservation] = []
        for tau in taus:
            x = bounce.x + vx_true * tau + 0.5 * ax_true * tau * tau
            y = y_contact + vy_true * tau - 0.5 * g * tau * tau
            z = bounce.z + vz_true * tau + 0.5 * az_true * tau * tau
            post_points.append(BallObservation(x=float(x), y=float(y), z=float(z), t=float(tau)))

        cand = Candidate(
            e=0.7,
            kt=0.65,
            weight=1.0,
            v_plus=np.array([0.0, 0.0, 0.0], dtype=float),
            ax=0.0,
            az=0.0,
        )

        # 显式设置 posterior_obs_sigma_m，固定 σ 的数值尺度，避免测试受默认回退策略影响。
        cfg_rls = CurveV3Config(
            physics=PhysicsConfig(gravity=g),
            posterior=PosteriorConfig(
                fit_mode="rls",
                fit_params="v+axz",
                posterior_obs_sigma_m=0.2,
                posterior_rls_lambda=1.0,
                posterior_prior_strength=3.0,
                posterior_prior_sigma_v=10.0,
                posterior_prior_sigma_a=10.0,
            ),
        )
        out_rls = fit_posterior_map_for_candidate(
            bounce=bounce,
            post_points=post_points,
            candidate=cand,
            time_base_abs=0.0,
            cfg=cfg_rls,
        )
        self.assertIsNotNone(out_rls)
        assert out_rls is not None

        st_rls, _ = out_rls

        # 构造批量正规方程参考解：
        #   (1/σ^2) H^T H θ + Q θ = (1/σ^2) H^T y + Q θ0
        self.assertIsNotNone(cfg_rls.posterior.posterior_obs_sigma_m)
        assert cfg_rls.posterior.posterior_obs_sigma_m is not None
        sigma = float(cfg_rls.posterior.posterior_obs_sigma_m)
        inv_sigma2 = 1.0 / (sigma * sigma)
        strength = float(cfg_rls.posterior.posterior_prior_strength)

        theta0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        theta0[0:3] = np.asarray(cand.v_plus, dtype=float)

        inv_v2 = 1.0 / max(float(cfg_rls.posterior.posterior_prior_sigma_v) ** 2, 1e-9)
        inv_a2 = 1.0 / max(float(cfg_rls.posterior.posterior_prior_sigma_a) ** 2, 1e-9)
        q = np.array([inv_v2, inv_v2, inv_v2, inv_a2, inv_a2], dtype=float) * strength
        Q = np.diag(q)

        rows: list[list[float]] = []
        ys: list[float] = []
        y0 = float(cfg_rls.bounce_contact_y())
        for p in post_points:
            tau = float(p.t) - float(bounce.t_rel)
            self.assertGreater(tau, 0.0)

            dx = float(p.x - bounce.x)
            dz = float(p.z - bounce.z)
            y_rhs = float(p.y + 0.5 * g * tau * tau - y0)

            rows.append([tau, 0.0, 0.0, 0.5 * tau * tau, 0.0])
            ys.append(dx)
            rows.append([0.0, tau, 0.0, 0.0, 0.0])
            ys.append(y_rhs)
            rows.append([0.0, 0.0, tau, 0.0, 0.5 * tau * tau])
            ys.append(dz)

        H = np.asarray(rows, dtype=float)
        y_vec = np.asarray(ys, dtype=float)

        A = inv_sigma2 * (H.T @ H) + Q
        b = inv_sigma2 * (H.T @ y_vec) + Q @ theta0
        theta_batch = np.linalg.solve(A, b)

        self.assertAlmostEqual(st_rls.vx, float(theta_batch[0]), places=10)
        self.assertAlmostEqual(st_rls.vy, float(theta_batch[1]), places=10)
        self.assertAlmostEqual(st_rls.vz, float(theta_batch[2]), places=10)
        self.assertAlmostEqual(st_rls.ax, float(theta_batch[3]), places=10)
        self.assertAlmostEqual(st_rls.az, float(theta_batch[4]), places=10)

    def test_e_range_clamps_candidates(self):
        """验证 e_range 会对候选的 e 做裁剪，避免异常 bins 导致数值爆炸。"""

        cfg = CurveV3Config(
            prior=PriorConfig(
                e_bins=(-1.0, 10.0),
                kt_bins=(0.5,),
                e_range=(0.1, 1.2),
            )
        )

        bounce = BounceEvent(
            t_rel=0.0,
            x=0.0,
            z=0.0,
            v_minus=np.array([1.0, -2.0, 3.0], dtype=float),
            y=float(cfg.bounce_contact_y()),
        )

        cands = build_prior_candidates(bounce=bounce, cfg=cfg, prior_model=None, online_prior=None)
        self.assertTrue(cands)

        for c in cands:
            self.assertGreaterEqual(float(c.e), 0.1)
            self.assertLessEqual(float(c.e), 1.2)


if __name__ == "__main__":
    unittest.main()
