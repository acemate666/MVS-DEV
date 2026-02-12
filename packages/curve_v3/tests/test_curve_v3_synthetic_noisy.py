import numpy as np

from curve_v3 import CurvePredictorV3
from curve_v3.configs import CurveV3Config, CorridorConfig, PhysicsConfig, PosteriorConfig, PriorConfig
from curve_v3.offline.testing.synthetic import (
    SyntheticNoise,
    make_postbounce_observations_from_candidate,
    make_prebounce_observations_ballistic,
)


def test_curve_v3_pipeline_runs_on_noisy_synthetic_data() -> None:
    """用合成带噪数据跑通 curve_v3 主链路（smoke test）。

    关注点：
        - 算法在合理噪声下不崩溃、不产生 NaN/inf。
        - 能产出 bounce_event/candidates/corridor 等关键中间结果。

    说明：
        该测试不追求严格数值精度（噪声会导致 bounce 时刻与候选权重轻微漂移），
        主要用于回归“带噪输入的稳定性”。
    """

    cfg = CurveV3Config(
        physics=PhysicsConfig(gravity=9.8),
        # 减少候选数，让单测更快也更稳定。
        prior=PriorConfig(
            e_bins=(0.70, 0.85),
            kt_bins=(0.65, 0.85),
            kt_angle_bins_rad=(0.0,),
        ),
        posterior=PosteriorConfig(max_post_points=5),
        corridor=CorridorConfig(corridor_dt=0.05, corridor_horizon_s=0.6),
    )

    engine = CurvePredictorV3(config=cfg)

    time_base_abs = 100.0
    t_land_rel = 0.55

    pre_noise = SyntheticNoise(sigma_x_m=0.008, sigma_y_m=0.006, sigma_z_m=0.008, seed=0)
    pre_obs, pre_params = make_prebounce_observations_ballistic(
        cfg=cfg,
        time_base_abs=time_base_abs,
        t_land_rel=t_land_rel,
        num_points=14,
        x0=0.2,
        y0=1.0,
        z0=1.0,
        vx=1.2,
        vz=7.5,
        noise=pre_noise,
        include_conf=True,
    )

    for o in pre_obs:
        engine.add_observation(o)

    bounce = engine.get_bounce_event()
    assert bounce is not None

    # 允许较宽容的误差：在噪声存在时，prefit 与分段检测会让 t_b 有轻微漂移。
    assert abs(float(bounce.t_rel) - float(pre_params["t_land_rel"])) < 0.20

    candidates = engine.get_prior_candidates()
    assert candidates

    # 用其中一个候选生成反弹后真值（模型一致），再叠加观测噪声。
    c0 = candidates[0]

    post_noise = SyntheticNoise(sigma_x_m=0.010, sigma_y_m=0.008, sigma_z_m=0.010, seed=1)
    post_obs = make_postbounce_observations_from_candidate(
        cfg=cfg,
        bounce=bounce,
        candidate=c0,
        time_base_abs=float(engine.time_base_abs or time_base_abs),
        taus=[0.08, 0.16, 0.24],
        noise=post_noise,
        include_conf=True,
    )

    for o in post_obs:
        engine.add_observation(o)

    corridor = engine.get_corridor_by_time()
    assert corridor is not None

    assert np.isfinite(corridor.mu_xz).all()
    assert np.isfinite(corridor.cov_xz).all()

    # point_at_time_rel 作为“随时间查询”的公共接口，必须可用。
    t_query = float(bounce.t_rel) + 0.18
    p = engine.point_at_time_rel(t_query)
    assert p is not None
    assert len(p) == 3
    assert np.isfinite(np.asarray(p, dtype=float)).all()
