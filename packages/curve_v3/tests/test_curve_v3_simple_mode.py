from __future__ import annotations

import numpy as np
import pytest

from curve_v3.configs import CurveV3Config, PipelineConfig, SimplePipelineConfig
from curve_v3.core import CurvePredictorV3
from curve_v3.types import BallObservation


def _obs(*, t_abs: float, x: float, y: float, z: float) -> BallObservation:
    return BallObservation(x=float(x), y=float(y), z=float(z), t=float(t_abs), conf=1.0)


def test_simple_mode_segment_isolation_and_postfit_velocity_fit():
    """验证 simple mode 的关键契约：

    1) prefit/bounce_event 冻结后不再被 post 点污染（严格分段）。
    2) postfit 只使用 post 段点，能拟合出接近真实的 (vx,vy,vz)。

    说明：
        - 这里构造一个“反弹前重力抛体 + 反弹后重力抛体”的合成序列。
        - x/z 均为线性（simple mode 的建模假设）。
    """

    cfg = CurveV3Config(
        pipeline=PipelineConfig(mode="simple"),
        simple=SimplePipelineConfig(
            e=0.70,
            kt=0.65,
            kt_angle_rad=0.0,
            postfit_min_points=2,
            postfit_max_points=8,
        ),
    )
    pred = CurvePredictorV3(config=cfg)

    g = float(cfg.physics.gravity)
    y_contact = float(cfg.bounce_contact_y())

    # 设定一个触地/反弹时刻。
    t_b = 0.10

    # 反弹前：y(t) = y_init + vy0*t - 0.5*g*t^2，且 y(t_b)=y_contact。
    y_init = 0.60
    vy0 = (y_contact - y_init + 0.5 * g * t_b * t_b) / t_b

    # 刻意让 pre/post 的水平速度差异很大，用于验证 postfit 不会混入 pre 点。
    vx_pre, vz_pre = 10.0, -5.0
    vx_post, vz_post = 1.0, 2.0
    vy_post = 3.0

    x_init, z_init = 0.2, -0.1

    # 绝对时间基准（CurvePredictorV3 会把第一帧作为 time_base_abs）。
    t0 = 100.0

    # 反弹前（包含 t_b 点）。
    ts_pre = np.arange(0.0, t_b + 1e-9, 0.01, dtype=float)
    for t in ts_pre:
        x = x_init + vx_pre * t
        z = z_init + vz_pre * t
        y = y_init + vy0 * t - 0.5 * g * t * t
        pred.add_observation(_obs(t_abs=t0 + float(t), x=float(x), y=float(y), z=float(z)))

    # 反弹点（用于构造 post 段连续轨迹）。
    x_b = x_init + vx_pre * t_b
    z_b = z_init + vz_pre * t_b

    # 反弹后：y(tau) = y_contact + vy_post*tau - 0.5*g*tau^2。
    ts_post = np.arange(t_b + 0.01, t_b + 0.08, 0.01, dtype=float)
    for t in ts_post:
        tau = float(t - t_b)
        x = x_b + vx_post * tau
        z = z_b + vz_post * tau
        y = y_contact + vy_post * tau - 0.5 * g * tau * tau
        pred.add_observation(_obs(t_abs=t0 + float(t), x=float(x), y=float(y), z=float(z)))

    # 触发冻结后，prefit/bounce_event 不应再变化。
    freeze = pred.get_prefit_freeze_info()
    assert freeze.is_frozen

    # postfit 速度拟合应接近真实 post 速度（仅用 post 点）。
    st1 = pred.get_posterior_state()
    assert st1 is not None
    assert float(st1.vx) == pytest.approx(vx_post, abs=1e-2)
    assert float(st1.vz) == pytest.approx(vz_post, abs=1e-2)
    assert float(st1.vy) == pytest.approx(vy_post, abs=1e-2)

    pre1 = pred.get_pre_fit_coeffs()
    assert pre1 is not None

    # simple mode 的 x/z 线性拟合以二次系数形式返回（a=0）。
    assert float(pre1["x"][0]) == pytest.approx(0.0, abs=1e-12)
    assert float(pre1["z"][0]) == pytest.approx(0.0, abs=1e-12)
    assert float(pre1["x"][1]) == pytest.approx(vx_pre, abs=1e-6)
    assert float(pre1["z"][1]) == pytest.approx(vz_pre, abs=1e-6)

    # 再追加一些“极端离谱”的 post 点，验证不会污染 prefit。
    for i in range(5):
        t = float(t_b + 0.20 + i * 0.01)
        tau = float(t - t_b)
        x = x_b + 999.0
        z = z_b - 999.0
        y = y_contact + vy_post * tau - 0.5 * g * tau * tau
        pred.add_observation(_obs(t_abs=t0 + t, x=float(x), y=float(y), z=float(z)))

    pre2 = pred.get_pre_fit_coeffs()
    assert pre2 is not None
    for k in ("x", "y", "z", "t_land"):
        np.testing.assert_allclose(pre2[k], pre1[k], rtol=0.0, atol=1e-12)
