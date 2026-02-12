"""合成数据生成工具（带噪）。

设计目标：
    - 用尽量少的依赖（仅 NumPy）生成“可控、可复现”的网球轨迹观测。
    - 产物直接适配 `curve_v3.types.BallObservation`，可喂给 `CurvePredictorV3`。
    - 噪声模型保持简单：各轴独立高斯噪声；可选离群点。

注意：
    - 这里生成的是“球心坐标系”的观测（y 为球心高度），需与 `CurveV3Config.bounce_contact_y()` 口径一致。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from curve_v3.configs import CurveV3Config
from curve_v3.dynamics import propagate_post_bounce_state
from curve_v3.types import BallObservation, BounceEvent, Candidate


@dataclass(frozen=True)
class SyntheticNoise:
    """合成噪声配置。"""

    sigma_x_m: float = 0.01
    sigma_y_m: float = 0.01
    sigma_z_m: float = 0.01

    # 可选离群点（非常少量，避免测试变得不稳定）。
    outlier_prob: float = 0.0
    outlier_scale_mult: float = 6.0

    # 随机种子：保证单测可复现。
    seed: int = 0


def _conf_from_noise_sigma(*, sigma_m: float, sigma0_m: float, c_min: float) -> float:
    """把观测噪声尺度映射为低 SNR 策略需要的 conf。

    说明：
        curve_v3 的 low_snr 权重口径：
            σ = σ0 / sqrt(conf)  =>  conf = (σ0/σ)^2

        因此噪声越大，conf 越小。
    """

    sigma = float(max(float(sigma_m), 1e-9))
    sigma0 = float(max(float(sigma0_m), 1e-9))
    c = (sigma0 / sigma) ** 2
    # conf 是“可信度”，工程上通常希望限制在 (0,1]。
    return float(min(1.0, max(float(c_min), c)))


def _apply_noise_xyz(
    *,
    rng: np.random.Generator,
    x: float,
    y: float,
    z: float,
    noise: SyntheticNoise,
) -> tuple[float, float, float]:
    """对 (x,y,z) 添加高斯噪声，并按概率注入离群点。"""

    sx, sy, sz = float(noise.sigma_x_m), float(noise.sigma_y_m), float(noise.sigma_z_m)
    nx = float(rng.normal(scale=sx)) if sx > 0 else 0.0
    ny = float(rng.normal(scale=sy)) if sy > 0 else 0.0
    nz = float(rng.normal(scale=sz)) if sz > 0 else 0.0

    if float(noise.outlier_prob) > 0.0 and float(rng.random()) < float(noise.outlier_prob):
        mult = float(max(noise.outlier_scale_mult, 1.0))
        nx *= mult
        ny *= mult
        nz *= mult

    return float(x + nx), float(y + ny), float(z + nz)


def make_prebounce_observations_ballistic(
    *,
    cfg: CurveV3Config,
    time_base_abs: float,
    t_land_rel: float,
    num_points: int = 12,
    x0: float = 0.2,
    y0: float = 1.0,
    z0: float = 1.0,
    vx: float = 1.0,
    vz: float = 8.0,
    noise: SyntheticNoise | None = None,
    include_conf: bool = True,
) -> tuple[list[BallObservation], dict[str, float]]:
    """生成一段反弹前观测，使其在 y==y_contact 处落地。"""

    g = float(cfg.physics.gravity)
    y_contact = float(cfg.bounce_contact_y())

    t_land_rel = float(t_land_rel)
    if t_land_rel <= 1e-6:
        raise ValueError("t_land_rel must be > 0")

    n = int(num_points)
    if n < 2:
        raise ValueError("num_points must be >= 2")

    # 令 y(t_land)=y_contact，解出 vy。
    vy = (0.5 * g * t_land_rel * t_land_rel + y_contact - float(y0)) / t_land_rel

    rng = np.random.default_rng(int(noise.seed) if noise is not None else 0)

    # 让最后一点落在 t_land 之前一点点：避免数值上刚好触地导致极端情况。
    ts_rel = np.linspace(0.0, float(t_land_rel), n, dtype=float)

    obs: list[BallObservation] = []

    # 用 x 轴噪声尺度推一个“标量 conf”。如果没有噪声，则用 conf=1。
    conf_val: float | None = None
    if include_conf:
        if noise is None:
            conf_val = 1.0
        else:
            conf_val = _conf_from_noise_sigma(
                sigma_m=float(noise.sigma_x_m),
                sigma0_m=float(cfg.low_snr.low_snr_sigma_x0_m),
                c_min=float(cfg.low_snr.low_snr_conf_cmin),
            )

    for t_rel in ts_rel.tolist():
        t_rel = float(t_rel)
        x = float(x0) + float(vx) * t_rel
        y = float(y0) + float(vy) * t_rel - 0.5 * g * t_rel * t_rel
        z = float(z0) + float(vz) * t_rel

        if noise is not None:
            x, y, z = _apply_noise_xyz(rng=rng, x=x, y=y, z=z, noise=noise)

        obs.append(
            BallObservation(
                x=float(x),
                y=float(y),
                z=float(z),
                t=float(time_base_abs + t_rel),
                conf=conf_val,
            )
        )

    params = {
        "x0": float(x0),
        "y0": float(y0),
        "z0": float(z0),
        "vx": float(vx),
        "vy": float(vy),
        "vz": float(vz),
        "t_land_rel": float(t_land_rel),
        "y_contact": float(y_contact),
        "time_base_abs": float(time_base_abs),
    }
    return obs, params


def make_postbounce_observations_from_candidate(
    *,
    cfg: CurveV3Config,
    bounce: BounceEvent,
    candidate: Candidate,
    time_base_abs: float,
    taus: list[float],
    noise: SyntheticNoise | None = None,
    include_conf: bool = True,
) -> list[BallObservation]:
    """用候选动力学生成反弹后观测（可带噪）。"""

    rng = np.random.default_rng(int(noise.seed) if noise is not None else 0)

    conf_val: float | None = None
    if include_conf:
        if noise is None:
            conf_val = 1.0
        else:
            conf_val = _conf_from_noise_sigma(
                sigma_m=float(noise.sigma_x_m),
                sigma0_m=float(cfg.low_snr.low_snr_sigma_x0_m),
                c_min=float(cfg.low_snr.low_snr_conf_cmin),
            )

    out: list[BallObservation] = []
    for tau in taus:
        tau = float(tau)
        pos, _ = propagate_post_bounce_state(bounce=bounce, candidate=candidate, tau=tau, cfg=cfg)
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

        if noise is not None:
            x, y, z = _apply_noise_xyz(rng=rng, x=x, y=y, z=z, noise=noise)

        out.append(
            BallObservation(
                x=float(x),
                y=float(y),
                z=float(z),
                # BallObservation.t 是绝对时间戳。
                t=float(time_base_abs + float(bounce.t_rel) + tau),
                conf=conf_val,
            )
        )

    return out


__all__ = [
    "SyntheticNoise",
    "make_prebounce_observations_ballistic",
    "make_postbounce_observations_from_candidate",
]
