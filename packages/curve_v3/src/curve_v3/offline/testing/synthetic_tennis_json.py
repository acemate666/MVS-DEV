"""把合成网球观测序列导出为 curve_v3 离线评测所需的 JSON。

定位：
    - 合成观测点生成（带噪）本体在 `curve_v3.offline.testing.synthetic`。
    - 本模块只负责：
        1) 选择一组“合理的默认参数”拼出 pre/post 两段观测；
        2) 以 `curve_v3.offline.real_device_eval.load_observations_from_json()`
           兼容的 schema 写出 JSON。

JSON schema（建议）：
    {
      "meta": { ... },
      "observations": [
        {"t": 100.0, "x": 0.1, "y": 1.1, "z": 0.2, "conf": 1.0},
        ...
      ]
    }

注意：
    - 这里生成的 t 是绝对时间戳（s），符合 `BallObservation.t` 的语义。
    - 该模块属于离线/测试工具，不参与在线主链路。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from curve_v3.configs import CurveV3Config
from curve_v3.offline.testing.synthetic import (
    SyntheticNoise,
    make_postbounce_observations_from_candidate,
    make_prebounce_observations_ballistic,
)
from curve_v3.types import BallObservation, BounceEvent, Candidate


def _rotate_about_axis(v: np.ndarray, axis_hat: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues 公式：绕单位轴 axis_hat 旋转向量 v。"""

    v = np.asarray(v, dtype=float).reshape(3)
    k = np.asarray(axis_hat, dtype=float).reshape(3)

    ang = float(angle_rad)
    c = float(np.cos(ang))
    s = float(np.sin(ang))
    return v * c + np.cross(k, v) * s + k * float(np.dot(k, v)) * (1.0 - c)


def _build_truth_bounce_and_candidate(
    *,
    cfg: CurveV3Config,
    pre_params: dict[str, float],
    e: float,
    kt: float,
    kt_angle_rad: float = 0.0,
) -> tuple[BounceEvent, Candidate]:
    """基于 pre 的解析参数构造一组“生成模型”的 bounce + candidate。

    说明：
        这里的 v^- -> v^+ 映射与 `curve_v3.prior.candidates.build_prior_candidates()`
        内部逻辑保持一致：

            v_plus = -e * v_n + kt * rotate(v_t)

        其中 rotate 表示绕地面法向在切平面内旋转（角度为 kt_angle_rad）。
    """

    g = float(cfg.physics.gravity)
    t_b = float(pre_params["t_land_rel"])

    x_b = float(pre_params["x0"]) + float(pre_params["vx"]) * t_b
    z_b = float(pre_params["z0"]) + float(pre_params["vz"]) * t_b

    y_contact = float(pre_params["y_contact"])

    vx = float(pre_params["vx"])
    vy0 = float(pre_params["vy"])
    vz = float(pre_params["vz"])

    vy_minus = float(vy0 - g * t_b)
    v_minus = np.asarray([vx, vy_minus, vz], dtype=float)

    bounce = BounceEvent(
        t_rel=float(t_b),
        x=float(x_b),
        z=float(z_b),
        v_minus=v_minus,
        y=float(y_contact),
    )

    n_hat = np.asarray(cfg.physics.ground_normal, dtype=float).reshape(3)
    n_norm = float(np.linalg.norm(n_hat))
    if n_norm <= 1e-9:
        n_hat = np.array((0.0, 1.0, 0.0), dtype=float)
    else:
        n_hat = n_hat / n_norm

    v_n = float(np.dot(v_minus, n_hat)) * n_hat
    v_t = v_minus - v_n
    v_t_rot = _rotate_about_axis(v_t, n_hat, float(kt_angle_rad))

    v_plus = -float(e) * v_n + float(kt) * v_t_rot
    cand = Candidate(
        e=float(e),
        kt=float(kt),
        weight=1.0,
        v_plus=np.asarray(v_plus, dtype=float),
        kt_angle_rad=float(kt_angle_rad),
        ax=0.0,
        az=0.0,
    )

    return bounce, cand


def make_synthetic_tennis_observations(
    *,
    cfg: CurveV3Config,
    seed: int = 0,
    sigma_m: float = 0.008,
    t_land_rel: float = 0.23,
    truth_e: float = 0.90,
    truth_kt: float = 0.85,
    num_pre_points: int = 14,
    num_post_points: int = 15,
    post_dt_s: float = 0.05,
    time_base_abs: float | None = None,
    include_conf: bool = True,
) -> tuple[list[BallObservation], dict[str, Any]]:
    """生成一段“pre+post”的合成观测点序列。

    Args:
        cfg: curve_v3 配置。
        seed: 随机种子（用于噪声）。
        sigma_m: 3D 观测噪声标准差（米）。传 0 表示无噪声。
        t_land_rel: 反弹时刻（相对 time_base，秒）。
        truth_e: 生成模型使用的 e。
        truth_kt: 生成模型使用的 kt。
        num_pre_points: pre 段观测点数量。
        num_post_points: post 段观测点数量。
        post_dt_s: post 段采样间隔（秒）。
        time_base_abs: 绝对时间基准；None 时按 seed 生成一个稳定值。
        include_conf: 是否在 BallObservation.conf 中填充置信度。

    Returns:
        (observations, meta)
    """

    seed = int(seed)
    sigma = float(max(float(sigma_m), 0.0))

    if time_base_abs is None:
        time_base_abs = 1000.0 + 10.0 * float(seed)

    noise_pre = None
    noise_post = None
    if sigma > 0.0:
        noise_pre = SyntheticNoise(
            sigma_x_m=sigma,
            sigma_y_m=sigma,
            sigma_z_m=sigma,
            seed=seed * 10 + 1,
        )
        noise_post = SyntheticNoise(
            sigma_x_m=sigma,
            sigma_y_m=sigma,
            sigma_z_m=sigma,
            seed=seed * 10 + 2,
        )

    pre_obs, pre_params = make_prebounce_observations_ballistic(
        cfg=cfg,
        time_base_abs=float(time_base_abs),
        t_land_rel=float(t_land_rel),
        num_points=int(num_pre_points),
        x0=0.2,
        y0=1.0,
        z0=1.0,
        vx=1.2,
        vz=7.5,
        noise=noise_pre,
        include_conf=bool(include_conf),
    )

    bounce_gt, cand_gt = _build_truth_bounce_and_candidate(
        cfg=cfg,
        pre_params=pre_params,
        e=float(truth_e),
        kt=float(truth_kt),
        kt_angle_rad=0.0,
    )

    dt = float(max(float(post_dt_s), 1e-6))
    m = int(max(int(num_post_points), 0))
    taus = [dt * (i + 1) for i in range(m)]

    post_obs = make_postbounce_observations_from_candidate(
        cfg=cfg,
        bounce=bounce_gt,
        candidate=cand_gt,
        time_base_abs=float(time_base_abs),
        taus=taus,
        noise=noise_post,
        include_conf=bool(include_conf),
    )

    obs = list(pre_obs) + list(post_obs)
    obs.sort(key=lambda o: float(o.t))

    meta = {
        "source": "curve_v3.offline.testing.synthetic_tennis_json",
        "seed": int(seed),
        "sigma_m": float(sigma),
        "t_land_rel": float(t_land_rel),
        "truth_e": float(truth_e),
        "truth_kt": float(truth_kt),
        "time_base_abs": float(time_base_abs),
        "num_pre_points": int(num_pre_points),
        "num_post_points": int(num_post_points),
        "post_dt_s": float(dt),
    }

    return obs, meta


def make_synthetic_tennis_trajectory_json(
    *,
    cfg: CurveV3Config,
    seed: int = 0,
    sigma_m: float = 0.008,
    t_land_rel: float = 0.23,
    truth_e: float = 0.90,
    truth_kt: float = 0.85,
    num_pre_points: int = 14,
    num_post_points: int = 15,
    post_dt_s: float = 0.05,
    time_base_abs: float | None = None,
    include_conf: bool = True,
    meta_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成可落盘的轨迹 JSON dict。"""

    obs, meta = make_synthetic_tennis_observations(
        cfg=cfg,
        seed=seed,
        sigma_m=sigma_m,
        t_land_rel=t_land_rel,
        truth_e=truth_e,
        truth_kt=truth_kt,
        num_pre_points=num_pre_points,
        num_post_points=num_post_points,
        post_dt_s=post_dt_s,
        time_base_abs=time_base_abs,
        include_conf=include_conf,
    )

    if meta_overrides:
        meta = {**meta, **dict(meta_overrides)}

    return {
        "meta": meta,
        "observations": [
            {
                "t": float(o.t),
                "x": float(o.x),
                "y": float(o.y),
                "z": float(o.z),
                "conf": None if o.conf is None else float(o.conf),
            }
            for o in obs
        ],
    }


def write_synthetic_tennis_trajectory_json(
    path: str | Path,
    *,
    cfg: CurveV3Config,
    seed: int = 0,
    sigma_m: float = 0.008,
    t_land_rel: float = 0.23,
    truth_e: float = 0.90,
    truth_kt: float = 0.85,
    num_pre_points: int = 14,
    num_post_points: int = 15,
    post_dt_s: float = 0.05,
    time_base_abs: float | None = None,
    include_conf: bool = True,
    meta_overrides: dict[str, Any] | None = None,
) -> Path:
    """生成并写出合成轨迹 JSON。"""

    p = Path(path)
    data = make_synthetic_tennis_trajectory_json(
        cfg=cfg,
        seed=seed,
        sigma_m=sigma_m,
        t_land_rel=t_land_rel,
        truth_e=truth_e,
        truth_kt=truth_kt,
        num_pre_points=num_pre_points,
        num_post_points=num_post_points,
        post_dt_s=post_dt_s,
        time_base_abs=time_base_abs,
        include_conf=include_conf,
        meta_overrides=meta_overrides,
    )

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


__all__ = [
    "make_synthetic_tennis_observations",
    "make_synthetic_tennis_trajectory_json",
    "write_synthetic_tennis_trajectory_json",
]
