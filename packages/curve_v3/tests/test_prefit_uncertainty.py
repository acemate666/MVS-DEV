import numpy as np

from curve_v3.configs import CurveV3Config, PhysicsConfig, PrefitConfig
from curve_v3.prior.prefit import estimate_bounce_event_from_prefit


def test_prefit_outputs_uncertainty_scales():
    """prefit 应输出轻量不确定度尺度（不要求完整协方差）。"""

    cfg = CurveV3Config(
        physics=PhysicsConfig(gravity=9.8, bounce_contact_y_m=0.033),
        prefit=PrefitConfig(prefit_robust_iters=0),
    )

    tb = 0.50
    y_contact = float(cfg.bounce_contact_y())

    # 构造一个满足 y(tb)=y_contact 的“真实”竖直二次模型（a 固定为 -0.5*g）。
    vy0 = -2.0
    a = -0.5 * float(cfg.physics.gravity)
    b = float(vy0 + float(cfg.physics.gravity) * tb)
    c = float(y_contact - vy0 * tb - 0.5 * float(cfg.physics.gravity) * tb * tb)

    # 仅提供 tb 之前的观测点，通过外推求解触地时刻。
    t_rel = np.linspace(0.10, 0.45, 12, dtype=float)

    # x/z 线性即可。
    xs = 0.20 + 1.50 * t_rel
    zs = -0.10 + 2.00 * t_rel

    ys = a * t_rel * t_rel + b * t_rel + c

    # 加一点小噪声，避免完全零残差导致尺度为 0。
    rng = np.random.default_rng(0)
    xs = xs + rng.normal(0.0, 0.002, size=xs.shape)
    ys = ys + rng.normal(0.0, 0.002, size=ys.shape)
    zs = zs + rng.normal(0.0, 0.002, size=zs.shape)

    w = np.ones_like(t_rel, dtype=float)

    out = estimate_bounce_event_from_prefit(
        t_rel=t_rel,
        xs=xs,
        ys=ys,
        zs=zs,
        xw=w,
        yw=w,
        zw=w,
        cfg=cfg,
    )
    assert out is not None

    _, bounce = out

    assert bounce.y is not None
    assert float(bounce.y) == y_contact

    assert bounce.prefit_rms_m is not None
    assert float(bounce.prefit_rms_m) > 0.0

    assert bounce.sigma_t_rel is not None
    assert float(bounce.sigma_t_rel) > 0.0

    assert bounce.sigma_v_minus is not None
    sv = np.asarray(bounce.sigma_v_minus, dtype=float).reshape(-1)
    assert sv.shape == (3,)
    assert np.all(sv > 0.0)
