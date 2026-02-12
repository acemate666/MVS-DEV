"""像素域闭环（重投影误差最小化）的轻量实现。

该模块的定位：
    - 给 `curve_v3.posterior.fit_map.fit_posterior_map_for_candidate()` 提供一个可选的“二次校正”。
    - 仅在观测携带 2D（`BallObservation.obs_2d_by_camera`）且上层注入 `CameraRig` 时生效。

工程取舍：
    - 在线预算：默认只做 1~2 次 GN/LM 迭代。
    - 不引入第三方依赖：数值雅可比（有限差分），矩阵规模很小（3~5 维）。
    - 鲁棒：用 Huber 抑制坏相机/错关联带来的离群残差。

注意：
    - 这里的目标函数是“像素域 data_term + 参数先验项”。
    - data_term 使用协方差加权（Mahalanobis），因此是无量纲的平方和，可与 3D 归一化 SSE 同域。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from curve_v3.adapters.camera_rig import CameraRig
from curve_v3.configs import CurveV3Config
from curve_v3.dynamics import propagate_post_bounce_state
from curve_v3.types import BallObservation, BounceEvent, Candidate


@dataclass(frozen=True)
class _PixelMeas:
    """单个相机-时刻的像素观测条目。"""

    point_id: int
    camera: str
    tau: float
    uv_obs: np.ndarray  # shape=(2,)
    invL: np.ndarray  # shape=(2,2), cov^{-1/2}，用于白化
    sigma_px: float  # 用于把白化残差范数映射回“像素等效”尺度（门控/Huber）


def _safe_cholesky_invL(cov: np.ndarray) -> np.ndarray:
    """计算 cov^{-1/2}，并在数值问题时做轻微正则。"""

    c = np.asarray(cov, dtype=float).reshape(2, 2)
    # 保证对称。
    c = 0.5 * (c + c.T)

    # 正则强度：按 trace 设一个相对尺度，避免 cov 很小时 eps 过大。
    tr = float(np.trace(c))
    base = max(tr * 1e-9, 1e-12)

    for k in range(5):
        try:
            L = np.linalg.cholesky(c + (base * (10.0**k)) * np.eye(2, dtype=float))
            invL = np.linalg.inv(L)
            return np.asarray(invL, dtype=float)
        except np.linalg.LinAlgError:
            continue

    # 最后兜底：用伪逆近似。
    w, V = np.linalg.eigh(c + base * np.eye(2, dtype=float))
    w = np.maximum(w, base)
    inv_sqrt = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    return np.asarray(inv_sqrt, dtype=float)


def _huber_weight(norm_px: float, *, delta_px: float) -> float:
    """Huber 的 IRLS 权重（对“像素等效残差范数”）。

    约定（与 docs/curve.md §5.6 的 IRLS 写法对齐）：
        w = 1                     , |e| <= delta
        w = delta / |e|           , |e| >  delta

    说明：
        - 这里的 norm_px 是“像素等效”的标量残差范数（单位 px）。
        - 我们在求解里使用 Σ^{-1/2} 白化残差；为了让门控阈值仍以 px 配置，
          会把白化残差的范数乘以一个标量 sigma_px 映射回像素尺度。
    """

    d = float(max(delta_px, 1e-6))
    n = float(norm_px)
    if not np.isfinite(n) or n <= 0.0:
        return 1.0
    if n <= d:
        return 1.0
    return float(d / n)


def _candidate_from_theta(*, candidate: Candidate, theta: np.ndarray, mode: Literal["v_only", "v+axz"]) -> Candidate:
    """用 theta 替换候选中的 (v_plus, ax, az)。"""

    th = np.asarray(theta, dtype=float).reshape(-1)
    v_plus = np.array([float(th[0]), float(th[1]), float(th[2])], dtype=float)

    if mode == "v_only":
        ax = 0.0
        az = 0.0
    else:
        ax = float(th[3])
        az = float(th[4])

    return Candidate(
        e=float(candidate.e),
        kt=float(candidate.kt),
        weight=float(candidate.weight),
        v_plus=v_plus,
        kt_angle_rad=float(getattr(candidate, "kt_angle_rad", 0.0)),
        ax=ax,
        az=az,
    )


def _build_pixel_measurements(
    *,
    post_points: Sequence[BallObservation],
    time_base_abs: float,
    t_b_rel: float,
    cfg: CurveV3Config,
) -> list[_PixelMeas]:
    """从 post_points 抽取可用于像素域闭环的观测条目列表。"""

    min_tau = float(max(float(cfg.posterior.posterior_min_tau_s), 0.0))

    out: list[_PixelMeas] = []
    for point_id, p in enumerate(post_points):
        obs2d = getattr(p, "obs_2d_by_camera", None)
        if not obs2d:
            continue

        t_rel = float(p.t - float(time_base_abs))
        tau = t_rel - float(t_b_rel)
        if tau <= min_tau:
            continue

        for cam, o in obs2d.items():
            if o is None:
                continue
            try:
                uv_obs = np.asarray(o.uv, dtype=float).reshape(2)
                cov_uv = np.asarray(o.cov_uv, dtype=float).reshape(2, 2)
                sigma_px = float(getattr(o, "sigma_px", 1.0))
                if not np.all(np.isfinite(uv_obs)):
                    continue
                if not np.all(np.isfinite(cov_uv)):
                    continue
                if not np.isfinite(sigma_px):
                    continue
            except Exception:
                continue

            invL = _safe_cholesky_invL(cov_uv)
            out.append(
                _PixelMeas(
                    point_id=int(point_id),
                    camera=str(cam),
                    tau=float(tau),
                    uv_obs=uv_obs,
                    invL=invL,
                    sigma_px=float(max(sigma_px, 1e-6)),
                )
            )

    return out


def refine_theta_in_pixel_domain(
    *,
    bounce: BounceEvent,
    post_points: Sequence[BallObservation],
    candidate: Candidate,
    time_base_abs: float,
    t_b_rel: float,
    theta_init: np.ndarray,
    theta0: np.ndarray,
    Q: np.ndarray,
    mode: Literal["v_only", "v+axz"],
    camera_rig: CameraRig,
    cfg: CurveV3Config,
) -> tuple[np.ndarray, float] | None:
    """在像素域对 theta 做少量 GN/LM 迭代。

    Args:
        bounce: 已对齐到 t_b_rel 的 bounce（其 x/z 应与 t_b_rel 一致）。
        post_points: 反弹后 3D 点（每点可携带多相机 2D 观测）。
        candidate: 候选（用于继承 e/kt/weight/kt_angle 等不在 theta 中的字段）。
        time_base_abs: 时间基准（绝对秒）。
        t_b_rel: 反弹相对时间（秒）。
        theta_init: 初始值。
        theta0: MAP 先验均值。
        Q: MAP 先验精度矩阵（对角阵）。
        mode: 参数化。
        camera_rig: 相机投影器集合。
        cfg: 配置。

    Returns:
        (theta_refined, pixel_data_term)。pixel_data_term 是白化+鲁棒后的平方和。
        若没有足够的 2D 观测则返回 None。
    """

    time_base_abs = float(time_base_abs)
    t_b_rel = float(t_b_rel)

    meas = _build_pixel_measurements(post_points=post_points, time_base_abs=time_base_abs, t_b_rel=t_b_rel, cfg=cfg)
    if not meas:
        return None

    theta = np.asarray(theta_init, dtype=float).reshape(-1).copy()
    theta0 = np.asarray(theta0, dtype=float).reshape(-1)
    Q = np.asarray(Q, dtype=float)

    D = int(theta.size)
    if Q.shape != (D, D):
        raise ValueError("Q shape mismatch")

    max_iters = int(cfg.pixel.pixel_max_iters)
    max_iters = int(max(max_iters, 0))
    huber_delta = float(cfg.pixel.pixel_huber_delta_px)
    damping = float(cfg.pixel.pixel_lm_damping)
    damping = float(max(damping, 0.0))
    rel_step = float(cfg.pixel.pixel_fd_rel_step)
    rel_step = float(max(rel_step, 1e-8))

    gate_tau_px = float(cfg.pixel.pixel_gate_tau_px)
    gate_tau_px = float(gate_tau_px)  # <=0 表示关闭

    min_cams = int(cfg.pixel.pixel_min_cameras)
    min_cams = int(max(min_cams, 1))

    def _compute_sqrt_w(th: np.ndarray) -> np.ndarray | None:
        """按当前 theta 计算每条像素观测的 sqrt(w)。

        说明：
            - 单观测门控：超过 gate_tau_px 的观测直接置 0。
            - 帧级门控：同一帧（同一个 3D 点）通过门控的相机数 < K 时，该帧所有观测置 0。
            - 鲁棒：其余观测使用 Huber IRLS 权重。
        """

        th = np.asarray(th, dtype=float).reshape(-1)
        cand2 = _candidate_from_theta(candidate=candidate, theta=th, mode=mode)

        w = np.zeros((len(meas),), dtype=float)

        # 先逐条计算单观测权重。
        for i, m in enumerate(meas):
            pos, _ = propagate_post_bounce_state(bounce=bounce, candidate=cand2, tau=float(m.tau), cfg=cfg)
            try:
                uv_pred = camera_rig.project(m.camera, pos)
            except Exception:
                continue

            uv_pred = np.asarray(uv_pred, dtype=float).reshape(2)
            if not np.all(np.isfinite(uv_pred)):
                continue

            duv = uv_pred - m.uv_obs

            # 白化残差（无量纲）。
            e = m.invL @ duv
            if not np.all(np.isfinite(e)):
                continue

            # 把白化残差范数映射回“像素等效”尺度，便于用 px 阈值做门控/Huber。
            # 在各向同性 Σ=σ_px^2 I 时，该量等价于 ||duv||。
            e_norm = float(np.linalg.norm(e))
            norm_px_equiv = float(e_norm * float(m.sigma_px))

            if gate_tau_px > 0.0 and norm_px_equiv > gate_tau_px:
                continue

            wi = _huber_weight(norm_px_equiv, delta_px=huber_delta)
            w[i] = float(max(wi, 0.0))

        # 帧级门控：统计每个 point_id 的“有效相机数”。
        if min_cams > 1:
            counts: dict[int, int] = {}
            for wi, m in zip(w, meas):
                if wi > 0.0:
                    counts[m.point_id] = counts.get(m.point_id, 0) + 1
            for i, m in enumerate(meas):
                if counts.get(m.point_id, 0) < min_cams:
                    w[i] = 0.0

        if not np.any(w > 0.0):
            # 没有可用的像素观测（全部被门控/投影失败剔除）。
            return None

        return np.sqrt(w)

    def residual_stack(th: np.ndarray, *, sqrt_w: np.ndarray) -> np.ndarray:
        """返回白化且带鲁棒权重的残差向量。"""

        th = np.asarray(th, dtype=float).reshape(-1)
        cand2 = _candidate_from_theta(candidate=candidate, theta=th, mode=mode)

        r = np.zeros((2 * len(meas),), dtype=float)

        for i, m in enumerate(meas):
            sw = float(sqrt_w[i])
            if sw <= 0.0:
                continue
            pos, _ = propagate_post_bounce_state(bounce=bounce, candidate=cand2, tau=float(m.tau), cfg=cfg)
            try:
                uv_pred = camera_rig.project(m.camera, pos)
            except Exception:
                # 不可投影：等价于忽略该条观测。
                continue

            uv_pred = np.asarray(uv_pred, dtype=float).reshape(2)
            if not np.all(np.isfinite(uv_pred)):
                continue

            duv = uv_pred - m.uv_obs
            e = m.invL @ duv
            r[2 * i : 2 * i + 2] = sw * e

        return r

    # 迭代：IRLS + GN/LM（只对 theta 做增量）。
    for _ in range(max_iters):
        sqrt_w = _compute_sqrt_w(theta)
        if sqrt_w is None or sqrt_w.size != len(meas):
            return None

        # 用“冻结的权重”计算残差（用于雅可比/线性化）。
        r = residual_stack(theta, sqrt_w=sqrt_w)
        if not np.all(np.isfinite(r)):
            return None

        # 数值雅可比：每列 2N x D。
        J = np.zeros((r.size, D), dtype=float)
        for k in range(D):
            eps = rel_step * max(1.0, abs(float(theta[k])))
            th1 = theta.copy()
            th1[k] += float(eps)
            r1 = residual_stack(th1, sqrt_w=sqrt_w)
            if not np.all(np.isfinite(r1)):
                return None
            J[:, k] = (r1 - r) / float(eps)

        if not np.all(np.isfinite(J)):
            return None

        A = (J.T @ J) + Q + (float(damping) * np.eye(D, dtype=float))
        g = (J.T @ r) + (Q @ (theta - theta0))

        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(g)):
            return None

        try:
            delta = -np.linalg.solve(A, g)
        except np.linalg.LinAlgError:
            delta, _, _, _ = np.linalg.lstsq(A, -g, rcond=None)

        delta = np.asarray(delta, dtype=float).reshape(-1)
        if not np.all(np.isfinite(delta)):
            return None

        theta = theta + delta

        # 极小预算：不做复杂的 line-search，达到足够小的步长就停。
        if float(np.linalg.norm(delta)) < 1e-6:
            break

    sqrt_w_final = _compute_sqrt_w(theta)
    if sqrt_w_final is None or sqrt_w_final.size != len(meas):
        return None
    r_final = residual_stack(theta, sqrt_w=sqrt_w_final)
    if not np.all(np.isfinite(r_final)):
        return None
    data_term = float(np.dot(r_final, r_final))
    if not np.isfinite(data_term):
        return None

    return theta, data_term
