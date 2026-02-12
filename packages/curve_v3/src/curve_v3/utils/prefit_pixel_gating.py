"""prefit 阶段的像素一致性点级加权。

目的：
    prefit 阶段的物理模型仍然在 3D 世界坐标里拟合，但在弱视差/遮挡/错关联时，
    三角化出来的 3D 点可能会与多相机 2D 观测明显不一致。

    本模块提供一个“轻量、可回退”的权重乘子：
        w_2d ∈ [0, 1]
    由重投影误差（像素域）决定，用于在 prefit 里降低该点对拟合的影响。

设计约束：
    - 不引入第三方依赖。
    - 默认关闭（由 cfg.prefit.prefit_pixel_enabled 控制），避免在未接入 CameraRig/2D 时改变行为。
    - 数值鲁棒：投影失败/协方差不良/NaN 时直接回退为“不过滤”。
"""

from __future__ import annotations

import numpy as np

from curve_v3.adapters.camera_rig import CameraRig
from curve_v3.configs import CurveV3Config
from curve_v3.types import BallObservation


def _safe_cholesky_invL_2x2(cov: np.ndarray) -> np.ndarray:
    """计算 cov^{-1/2}（2x2），并在数值问题时做轻微正则。

    说明：
        这里复用 posterior 像素闭环同样的“SPD 防御”思路，但实现保持极简，
        仅服务于 prefit 的点级加权。
    """

    c = np.asarray(cov, dtype=float).reshape(2, 2)
    c = 0.5 * (c + c.T)

    tr = float(np.trace(c))
    base = max(tr * 1e-9, 1e-12)

    for k in range(5):
        try:
            L = np.linalg.cholesky(c + (base * (10.0**k)) * np.eye(2, dtype=float))
            invL = np.linalg.inv(L)
            return np.asarray(invL, dtype=float)
        except np.linalg.LinAlgError:
            continue

    w, V = np.linalg.eigh(c + base * np.eye(2, dtype=float))
    w = np.maximum(w, base)
    inv_sqrt = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    return np.asarray(inv_sqrt, dtype=float)


def _huber_weight(norm_px: float, *, delta_px: float) -> float:
    """Huber 的 IRLS 权重（对“像素等效残差范数”）。"""

    d = float(max(delta_px, 1e-6))
    n = float(norm_px)
    if not np.isfinite(n) or n <= 0.0:
        return 1.0
    if n <= d:
        return 1.0
    return float(d / n)


def prefit_pixel_weight_multiplier(
    *,
    obs: BallObservation,
    camera_rig: CameraRig,
    cfg: CurveV3Config,
) -> float:
    """计算单帧观测在 prefit 中的像素一致性权重乘子。

    Args:
        obs: 单帧观测（携带 3D 点以及可选的 per-camera 2D 观测）。
        camera_rig: 上层注入的投影器集合。
        cfg: 配置。

    Returns:
        权重乘子 w_2d ∈ [0, 1]。

    规则：
        - 若没有可用 2D 观测，或全部投影失败：返回 1.0（不改变权重）。
        - 单观测门控：norm_px_equiv > gate_tau_px 时剔除。
        - 帧级门控：通过门控的相机数 < K 时返回 0.0。
        - 其余观测用 Huber 权重做软鲁棒，返回平均权重。

    重要：
        该函数只提供“权重乘子”，不直接修改 3D 点，不改变 prefit 的模型形式。
    """

    if not bool(cfg.prefit.prefit_pixel_enabled):
        return 1.0

    obs2d = getattr(obs, "obs_2d_by_camera", None)
    if not obs2d:
        return 1.0

    gate_tau_px = float(cfg.prefit.prefit_pixel_gate_tau_px)
    huber_delta_px = float(cfg.prefit.prefit_pixel_huber_delta_px)
    min_cams = int(cfg.prefit.prefit_pixel_min_cameras)
    min_cams = int(max(min_cams, 1))

    p_world = np.array([float(obs.x), float(obs.y), float(obs.z)], dtype=float)

    weights: list[float] = []
    num_projected = 0

    for cam, o in obs2d.items():
        if o is None:
            continue

        try:
            uv_obs = np.asarray(o.uv, dtype=float).reshape(2)
            cov_uv = np.asarray(o.cov_uv, dtype=float).reshape(2, 2)
            sigma_px = float(getattr(o, "sigma_px", 1.0))
        except Exception:
            continue

        if not np.all(np.isfinite(uv_obs)):
            continue
        if not np.isfinite(sigma_px):
            continue

        try:
            uv_pred = camera_rig.project(str(cam), p_world)
        except Exception:
            continue

        uv_pred = np.asarray(uv_pred, dtype=float).reshape(2)
        if not np.all(np.isfinite(uv_pred)):
            continue

        num_projected += 1

        duv = uv_pred - uv_obs

        # 默认用协方差白化；若协方差不可用，则退化为欧氏范数。
        norm_px_equiv: float
        if np.all(np.isfinite(cov_uv)):
            invL = _safe_cholesky_invL_2x2(cov_uv)
            e = invL @ duv
            if not np.all(np.isfinite(e)):
                continue
            # 在各向同性 cov=σ^2 I 时，该量等价于 ||duv||（单位 px）。
            norm_px_equiv = float(np.linalg.norm(e) * max(float(sigma_px), 1e-6))
        else:
            norm_px_equiv = float(np.linalg.norm(duv))

        if gate_tau_px > 0.0 and norm_px_equiv > gate_tau_px:
            continue

        wi = _huber_weight(norm_px_equiv, delta_px=huber_delta_px)
        weights.append(float(max(wi, 0.0)))

    # 若全部相机都无法投影，则不进行过滤（避免把“上游未接入投影”误判为离群）。
    if num_projected <= 0:
        return 1.0

    if len(weights) < min_cams:
        return 0.0

    w = float(np.mean(np.asarray(weights, dtype=float)))
    if not np.isfinite(w):
        return 1.0

    # 额外防御：限制在 [0,1]。
    return float(min(max(w, 0.0), 1.0))
