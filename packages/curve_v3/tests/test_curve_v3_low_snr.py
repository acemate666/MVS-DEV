import math

import numpy as np

from curve_v3 import CurvePredictorV3, fit_posterior_map_for_candidate
from curve_v3.configs import CurveV3Config, LowSnrConfig, PhysicsConfig, PosteriorConfig
from curve_v3.low_snr.types import AxisDecision, WindowDecisions
from curve_v3.types import BallObservation, BounceEvent, Candidate


def test_low_snr_prefit_ignore_x_uses_prior_velocity() -> None:
    """低SNR下 x 轴被 IGNORE 时，prefit 应主要沿用上一窗口的速度先验，避免噪声填充导数。"""

    cfg = CurveV3Config(
        physics=PhysicsConfig(bounce_contact_y_m=0.033),
        low_snr=LowSnrConfig(
            low_snr_enabled=True,
            low_snr_conf_cmin=0.05,
            low_snr_sigma_x0_m=0.2,
            low_snr_sigma_y0_m=0.2,
            low_snr_sigma_z0_m=0.2,
            low_snr_delta_k_ignore=1.0,
            low_snr_delta_k_strong_v=2.0,
            low_snr_delta_k_freeze_a=4.0,
            low_snr_prefit_strong_sigma_v_mps=0.5,
        ),
    )

    engine = CurvePredictorV3(config=cfg)

    dt = 0.02
    g = float(cfg.physics.gravity)

    # y 轴提供稳定时间结构（固定 -g）。
    y0 = 1.0

    # 前 12 点：x 有明显变化，便于形成稳定 v_prior。
    vx_good = 2.0
    z_v = 6.0

    conf = 0.1

    for i in range(12):
        t = float(i) * dt
        x = vx_good * t
        z = z_v * t
        y = y0 - 0.5 * g * t * t
        engine.add_observation(BallObservation(x=float(x), y=float(y), z=float(z), t=float(t), conf=conf))

    b0 = engine.get_bounce_event()
    assert b0 is not None
    v_prior_x = float(np.asarray(b0.v_minus, dtype=float)[0])

    # 后续点：x 几乎不变（低SNR），但 z 仍然正常前进。
    x_hold = vx_good * (11 * dt)
    rng = np.random.default_rng(0)
    for i in range(12, 30):
        t = float(i) * dt
        x = x_hold + float(rng.normal(scale=0.01))
        z = z_v * t
        y = y0 - 0.5 * g * t * t
        engine.add_observation(BallObservation(x=float(x), y=float(y), z=float(z), t=float(t), conf=conf))

    b1 = engine.get_bounce_event()
    assert b1 is not None
    v_final_x = float(np.asarray(b1.v_minus, dtype=float)[0])

    # 核心断言：低SNR下不应把噪声拟成“巨大速度”，应靠近先验。
    assert abs(v_final_x - v_prior_x) < 0.8

    info = engine.get_low_snr_info()
    assert info.prefit is not None
    assert info.prefit.mode_x in ("IGNORE_AXIS", "STRONG_PRIOR_V", "FREEZE_A")


def test_low_snr_posterior_ignore_x_keeps_vx_prior() -> None:
    """posterior 阶段 x 轴被 IGNORE 时，vx 应主要由候选先验决定。"""

    cfg = CurveV3Config(
        physics=PhysicsConfig(bounce_contact_y_m=0.033),
        low_snr=LowSnrConfig(
            low_snr_enabled=True,
            low_snr_conf_cmin=0.05,
            low_snr_sigma_x0_m=0.2,
            low_snr_sigma_y0_m=0.2,
            low_snr_sigma_z0_m=0.2,
        ),
        posterior=PosteriorConfig(
            posterior_prior_strength=1.0,
            posterior_prior_sigma_v=2.0,
            posterior_prior_sigma_a=8.0,
        ),
    )

    bounce = BounceEvent(t_rel=0.0, x=0.0, z=0.0, v_minus=np.array([0.0, -3.0, 6.0], dtype=float))
    cand = Candidate(e=0.7, kt=0.7, weight=1.0, v_plus=np.array([5.0, 2.0, 10.0], dtype=float), ax=0.0, az=0.0)

    # 构造 post 点：y/z 与候选一致，但 x 故意给出极端错误值。
    pts: list[BallObservation] = []
    for tau in (0.10, 0.20, 0.30):
        x = 1000.0
        y = float(cfg.bounce_contact_y()) + 2.0 * tau - 0.5 * float(cfg.physics.gravity) * tau * tau
        z = 10.0 * tau
        pts.append(BallObservation(x=float(x), y=float(y), z=float(z), t=float(tau), conf=0.1))

    low = WindowDecisions(
        x=AxisDecision(mode="IGNORE_AXIS", delta=0.0, sigma_mean=1.0, n=len(pts)),
        y=AxisDecision(mode="FULL", delta=1.0, sigma_mean=1.0, n=len(pts)),
        z=AxisDecision(mode="FULL", delta=1.0, sigma_mean=1.0, n=len(pts)),
    )

    out = fit_posterior_map_for_candidate(
        bounce=bounce,
        post_points=pts,
        candidate=cand,
        time_base_abs=0.0,
        low_snr=low,
        cfg=cfg,
    )
    assert out is not None
    st, _ = out

    assert math.isfinite(float(st.vx))
    assert abs(float(st.vx) - 5.0) < 1.0
