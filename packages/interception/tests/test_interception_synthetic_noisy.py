import numpy as np

from curve_v3 import CurvePredictorV3
from curve_v3 import CurveV3Config, PhysicsConfig, PosteriorConfig, PriorConfig
from curve_v3.offline.testing.synthetic import (
    SyntheticNoise,
    make_postbounce_observations_from_candidate,
    make_prebounce_observations_ballistic,
)
from interception import InterceptionConfig, select_hit_target_with_post


def test_interception_select_hit_target_on_noisy_synthetic_data() -> None:
    """用合成带噪数据跑通 interception 目标点选择（smoke test）。

    覆盖点：
        - 使用 curve_v3 的 bounce/candidates 作为上游输入。
        - 加入少量 post 点（N<=5）后，selector 能输出一个合理的击球目标点。

    说明：
        该测试以“稳定运行 + 输出合法性”为主，不对精度做强约束。
    """

    curve_cfg = CurveV3Config(
        physics=PhysicsConfig(gravity=9.8),
        prior=PriorConfig(
            e_bins=(0.70, 0.85),
            kt_bins=(0.65, 0.85),
            kt_angle_bins_rad=(0.0,),
        ),
        posterior=PosteriorConfig(
            max_post_points=5,
            weight_sigma_m=0.15,
            posterior_prior_strength=2.0,
            posterior_prior_sigma_v=2.0,
            posterior_prior_sigma_a=8.0,
            posterior_anchor_weight=0.0,
        ),
    )

    engine = CurvePredictorV3(config=curve_cfg)

    time_base_abs = 200.0
    t_land_rel = 0.55

    pre_obs, _ = make_prebounce_observations_ballistic(
        cfg=curve_cfg,
        time_base_abs=time_base_abs,
        t_land_rel=t_land_rel,
        num_points=14,
        vx=1.2,
        vz=7.5,
        noise=SyntheticNoise(sigma_x_m=0.008, sigma_y_m=0.006, sigma_z_m=0.008, seed=10),
        include_conf=True,
    )

    for o in pre_obs:
        engine.add_observation(o)

    bounce = engine.get_bounce_event()
    assert bounce is not None

    candidates = engine.get_prior_candidates()
    assert candidates

    # 生成少量 post 点（与候选模型一致），并加入噪声。
    c0 = candidates[0]
    post_points = make_postbounce_observations_from_candidate(
        cfg=curve_cfg,
        bounce=bounce,
        candidate=c0,
        time_base_abs=float(engine.time_base_abs or time_base_abs),
        taus=[0.10, 0.20, 0.30],
        noise=SyntheticNoise(sigma_x_m=0.012, sigma_y_m=0.010, sigma_z_m=0.012, seed=11),
        include_conf=True,
    )

    # 这里不要求把 post_points 再喂回 engine（interception 自己会做 posterior+reweight）。
    # 但为了更贴近线上使用习惯，我们仍把它们加入 engine，使 time_base/bounce 更新链路被覆盖。
    for p in post_points:
        engine.add_observation(p)

    bounce2 = engine.get_bounce_event()
    assert bounce2 is not None

    # 使用“当前时刻”为最新观测时间戳（工程上更合理）。
    t_now_abs = float(post_points[-1].t)

    cfg = InterceptionConfig(
        y_min=0.35,
        y_max=0.85,
        num_heights=5,
        # 为了避免少量候选在某高度不可达导致整体验证失败，这里放宽有效性门槛。
        min_crossing_prob=0.0,
        min_valid_candidates=1,
        # 测试里不希望 score 中的 time/width 惩罚导致意外 invalid。
        score_alpha_time=0.0,
        score_lambda_width=0.0,
        score_mu_crossing=0.0,
    )

    out = select_hit_target_with_post(
        bounce=bounce2,
        candidates=engine.get_prior_candidates(),
        post_points=post_points,
        time_base_abs=float(engine.time_base_abs or time_base_abs),
        t_now_abs=float(t_now_abs),
        cfg=cfg,
        curve_cfg=curve_cfg,
    )

    assert out.target is None or (
        np.isfinite(float(out.target.x))
        and np.isfinite(float(out.target.y))
        and np.isfinite(float(out.target.z))
        and np.isfinite(float(out.target.t_abs))
        and np.isfinite(float(out.target.t_rel))
    )

    # 在放宽门槛配置下，应当能产出 valid 目标。
    assert out.valid
    assert out.target is not None

    # 输出的目标必须“在未来”（留给机器人/控制的时间裕度），至少不应早于当前时刻。
    assert float(out.target.t_abs) >= float(t_now_abs)

    # y 必须等于选中的高度平面（不是 bounce_contact_y）。
    assert out.diag.target_y is not None
    assert abs(float(out.target.y) - float(out.diag.target_y)) < 1e-12
