"""simple mode 的最小流水线实现（内部模块）。

设计目标：
    - 提供一个“最小基线”分支，便于回归、对照和排障。
    - 不引入 prior candidates 网格、posterior 融合、low_snr、pixel refine、online prior 等复杂逻辑。

契约（与用户需求对齐）：
    - prefit：仅使用 PRE 段点拟合 3 条曲线
        - x/z：线性
        - y：固定重力项的二次（a=-0.5*g），等价于线性最小二乘求 (y0, vy)
    - bounce 检测：复用 `PrefitFreezeController` / `BounceTransitionDetector`
        - 一旦检测到 POST 段，冻结 prefit/bounce_event，避免 post 点污染
    - postfit：仅使用 POST 段点拟合 3 条曲线
        - x/z：线性（ax=az=0）
        - y：重力项固定
    - 当 post 点不足时：用简单反弹系数（e/kt/角度）生成一个兜底候选用于预测/走廊。

注意：
    - 该模块为内部实现（下划线文件名），不承诺稳定 API。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np

from curve_v3.pipeline.types import PrefitUpdateResult, PostUpdateResult
from curve_v3.configs import CurveV3Config
from curve_v3.corridor import build_corridor_by_time
from curve_v3.types import (
    BallObservation,
    BounceEvent,
    Candidate,
    CorridorByTime,
    PosteriorState,
)

if TYPE_CHECKING:  # pragma: no cover
    from curve_v3.prefit_freeze import PrefitFreezeController


def _fit_line_v_u0(t: np.ndarray, u: np.ndarray) -> tuple[float, float] | None:
    """拟合 u(t) = v*t + u0。

    Args:
        t: shape=(N,)
        u: shape=(N,)

    Returns:
        (v, u0) 或 None。
    """

    t = np.asarray(t, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)
    if t.size < 2 or u.size != t.size:
        return None

    a = np.stack([t, np.ones_like(t)], axis=1)
    try:
        sol, *_ = np.linalg.lstsq(a, u, rcond=None)
    except np.linalg.LinAlgError:
        return None

    v = float(sol[0])
    u0 = float(sol[1])
    if not np.isfinite(v) or not np.isfinite(u0):
        return None
    return v, u0


def _solve_latest_positive_root_quadratic(a: float, b: float, c: float) -> float | None:
    """求二次方程 a*t^2 + b*t + c = 0 的“最大正实根”。"""

    a = float(a)
    b = float(b)
    c = float(c)
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)):
        return None

    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return None
        t = -c / b
        if np.isfinite(t) and t > 0.0:
            return float(t)
        return None

    disc = b * b - 4.0 * a * c
    if not np.isfinite(disc) or disc < 0.0:
        return None

    s = float(np.sqrt(disc))
    t1 = (-b - s) / (2.0 * a)
    t2 = (-b + s) / (2.0 * a)
    cand = [float(t) for t in (t1, t2) if np.isfinite(t) and t > 0.0]
    if not cand:
        return None
    return float(max(cand))


def _estimate_bounce_event_from_simple_prefit(
    *,
    cfg: CurveV3Config,
    t_rel: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
) -> tuple[dict[str, np.ndarray], BounceEvent] | None:
    """simple prefit：线性 x/z + 固定重力二次 y，解 y(t)=y_contact 得到触地时刻。"""

    t_rel = np.asarray(t_rel, dtype=float).reshape(-1)
    xs = np.asarray(xs, dtype=float).reshape(-1)
    ys = np.asarray(ys, dtype=float).reshape(-1)
    zs = np.asarray(zs, dtype=float).reshape(-1)

    n = int(t_rel.size)
    if n < 5 or xs.size != n or ys.size != n or zs.size != n:
        return None

    g = float(cfg.physics.gravity)
    if not np.isfinite(g) or g <= 1e-6:
        return None

    # x/z：线性最小二乘
    x_fit = _fit_line_v_u0(t_rel, xs)
    z_fit = _fit_line_v_u0(t_rel, zs)
    if x_fit is None or z_fit is None:
        return None
    vx, x0 = x_fit
    vz, z0 = z_fit

    # y：固定重力项，线性最小二乘求 (vy, y0)
    y_lin = ys + 0.5 * g * t_rel * t_rel
    y_fit = _fit_line_v_u0(t_rel, y_lin)
    if y_fit is None:
        return None
    vy, y0 = y_fit

    y_contact = float(cfg.bounce_contact_y())

    # 解：y0 + vy*t - 0.5*g*t^2 = y_contact
    a = -0.5 * g
    b = vy
    c = y0 - y_contact
    t_land = _solve_latest_positive_root_quadratic(a, b, c)
    if t_land is None:
        return None

    # 触地点与入射速度
    x_land = x0 + vx * float(t_land)
    z_land = z0 + vz * float(t_land)

    # vy(t) = vy - g*t
    v_minus = np.array([vx, vy - g * float(t_land), vz], dtype=float)

    pre_coeffs = {
        # 保持与 full mode 一致的形状：二次多项式系数（但 a=0 表示线性）。
        "x": np.array([0.0, vx, x0], dtype=float),
        "y": np.array([a, vy, y0], dtype=float),
        "z": np.array([0.0, vz, z0], dtype=float),
        "t_land": np.array([float(t_land)], dtype=float),
    }

    bounce_event = BounceEvent(
        t_rel=float(t_land),
        x=float(x_land),
        z=float(z_land),
        v_minus=v_minus,
        y=float(y_contact),
    )

    return pre_coeffs, bounce_event


def update_prefit_and_bounce_event_simple(
    *,
    cfg: CurveV3Config,
    prefit_freezer: "PrefitFreezeController",
    prev_pre_coeffs: dict[str, np.ndarray] | None,
    t: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    confs: Sequence[float | None],
    prev_bounce_event: BounceEvent | None,
) -> PrefitUpdateResult:
    """simple mode 的 prefit + bounce_event 更新。"""

    _ = confs  # simple mode 不使用 conf 权重；保留参数仅为对齐调用接口。

    if prefit_freezer.state.is_frozen and prev_pre_coeffs is not None and prev_bounce_event is not None:
        return PrefitUpdateResult(
            pre_coeffs=prev_pre_coeffs,
            bounce_event=prev_bounce_event,
            low_snr_prefit=None,
        )

    t_fit = np.asarray(t, dtype=float)
    xs_fit = np.asarray(xs, dtype=float)
    ys_fit = np.asarray(ys, dtype=float)
    zs_fit = np.asarray(zs, dtype=float)

    # region 检测 prefit/post 切分点
    prefit_freezer.update_cut_index(
        ts=t_fit,
        ys=ys_fit,
        y_contact=float(cfg.bounce_contact_y()),
    )
    # endregion

    # region 根据 cut_index 划分 prefit 段（仅喂给 prefit）
    k = prefit_freezer.prefit_slice_end(n_points=int(t_fit.size))
    if k is not None:
        t_fit = t_fit[:k]
        xs_fit = xs_fit[:k]
        ys_fit = ys_fit[:k]
        zs_fit = zs_fit[:k]
    # endregion

    est = _estimate_bounce_event_from_simple_prefit(
        cfg=cfg,
        t_rel=t_fit,
        xs=xs_fit,
        ys=ys_fit,
        zs=zs_fit,
    )
    if est is None:
        return PrefitUpdateResult(pre_coeffs=None, bounce_event=None, low_snr_prefit=None)

    pre_coeffs, bounce_event = est

    # 与 full mode 保持一致：bounce_event 无效时清理。
    if float(bounce_event.t_rel) <= 0.0:
        return PrefitUpdateResult(pre_coeffs=None, bounce_event=None, low_snr_prefit=None)

    # 若分段检测器给出了 cut_index，则在 prefit 成功后冻结。
    if prefit_freezer.state.cut_index is not None:
        prefit_freezer.freeze()

    return PrefitUpdateResult(pre_coeffs=pre_coeffs, bounce_event=bounce_event, low_snr_prefit=None)


def _rotate_in_tangent_plane(v: np.ndarray, axis_hat: np.ndarray, angle_rad: float) -> np.ndarray:
    """绕法向轴旋转切向向量（Rodrigues 公式）。"""

    ang = float(angle_rad)
    c = float(np.cos(ang))
    s = float(np.sin(ang))
    v = np.asarray(v, dtype=float).reshape(3)
    k = np.asarray(axis_hat, dtype=float).reshape(3)
    return v * c + np.cross(k, v) * s + k * float(np.dot(k, v)) * (1.0 - c)


def _simple_candidate_from_bounce(*, cfg: CurveV3Config, bounce_event: BounceEvent) -> Candidate:
    """用 simple 配置的 (e, kt, angle) 从 v^- 生成一个兜底候选。"""

    v_minus = np.asarray(bounce_event.v_minus, dtype=float).reshape(3)

    n_hat = np.asarray(cfg.physics.ground_normal, dtype=float).reshape(3)
    n_norm = float(np.linalg.norm(n_hat))
    n_hat = n_hat / n_norm if n_norm > 1e-9 else np.array((0.0, 1.0, 0.0), dtype=float)

    v_n = float(np.dot(v_minus, n_hat)) * n_hat
    v_t = v_minus - v_n

    e = float(cfg.simple.e)
    kt = float(cfg.simple.kt)
    ang = float(cfg.simple.kt_angle_rad)

    v_t_rot = _rotate_in_tangent_plane(v_t, n_hat, ang)
    v_plus = -e * v_n + kt * v_t_rot

    return Candidate(
        e=float(e),
        kt=float(kt),
        weight=1.0,
        v_plus=np.asarray(v_plus, dtype=float),
        kt_angle_rad=float(ang),
        ax=0.0,
        az=0.0,
    )


def _fit_postfit_state(
    *,
    cfg: CurveV3Config,
    bounce_event: BounceEvent,
    post_points: Sequence[BallObservation],
    time_base_abs: float | None,
) -> PosteriorState | None:
    """simple postfit：仅用 post 点拟合 (vx, vy, vz)，ax=az=0。"""

    if time_base_abs is None:
        return None

    pts = list(post_points)
    if not pts:
        return None

    # 截断：只取最后 N 个 post 点，避免远期噪声干扰。
    max_n = int(cfg.simple.postfit_max_points)
    if max_n > 0 and len(pts) > max_n:
        pts = pts[-max_n:]

    if len(pts) < int(cfg.simple.postfit_min_points):
        return None

    t_b = float(bounce_event.t_rel)
    x_b = float(bounce_event.x)
    z_b = float(bounce_event.z)
    y0 = float(cfg.bounce_contact_y())
    g = float(cfg.physics.gravity)

    taus: list[float] = []
    dxs: list[float] = []
    dzs: list[float] = []
    dys: list[float] = []

    for p in pts:
        t_rel = float(p.t - float(time_base_abs))
        tau = float(t_rel - t_b)
        if not np.isfinite(tau) or tau <= 0.0:
            continue

        taus.append(tau)
        dxs.append(float(p.x) - x_b)
        dzs.append(float(p.z) - z_b)
        # y(t) = y0 + vy*tau - 0.5*g*tau^2
        dys.append(float(p.y) - y0 + 0.5 * g * tau * tau)

    if len(taus) < int(cfg.simple.postfit_min_points):
        return None

    tau_arr = np.asarray(taus, dtype=float)
    denom = float(np.sum(tau_arr * tau_arr))
    if not np.isfinite(denom) or denom <= 1e-12:
        return None

    vx = float(np.sum(tau_arr * np.asarray(dxs, dtype=float)) / denom)
    vy = float(np.sum(tau_arr * np.asarray(dys, dtype=float)) / denom)
    vz = float(np.sum(tau_arr * np.asarray(dzs, dtype=float)) / denom)

    if not (np.isfinite(vx) and np.isfinite(vy) and np.isfinite(vz)):
        return None

    return PosteriorState(
        t_b_rel=float(t_b),
        x_b=float(x_b),
        z_b=float(z_b),
        vx=float(vx),
        vy=float(vy),
        vz=float(vz),
        ax=0.0,
        az=0.0,
    )


def update_post_models_and_corridor_simple(
    *,
    cfg: CurveV3Config,
    logger: logging.Logger,
    bounce_event: BounceEvent,
    post_points: Sequence[BallObservation],
    time_base_abs: float | None,
) -> PostUpdateResult:
    """simple mode：更新单一候选 + 可选 postfit + corridor。"""

    _ = logger

    # 先给出一个兜底候选（即使 post 点为空，也能用于预测/走廊输出）。
    cand = _simple_candidate_from_bounce(cfg=cfg, bounce_event=bounce_event)

    posterior_state = _fit_postfit_state(
        cfg=cfg,
        bounce_event=bounce_event,
        post_points=post_points,
        time_base_abs=time_base_abs,
    )

    # 若 postfit 可用，则把候选速度更新为 postfit 的速度，使走廊与 point_at_time_rel 对齐。
    if posterior_state is not None:
        cand = Candidate(
            e=float(cand.e),
            kt=float(cand.kt),
            weight=float(cand.weight),
            v_plus=np.asarray([posterior_state.vx, posterior_state.vy, posterior_state.vz], dtype=float),
            kt_angle_rad=float(cand.kt_angle_rad),
            ax=0.0,
            az=0.0,
        )

    candidates = [cand]
    best_candidate = cand
    nominal_candidate_id = 0

    corridor_by_time: CorridorByTime | None = build_corridor_by_time(
        bounce=bounce_event,
        candidates=candidates,
        cfg=cfg,
    )

    return PostUpdateResult(
        candidates=candidates,
        best_candidate=best_candidate,
        nominal_candidate_id=nominal_candidate_id,
        posterior_state=posterior_state,
        corridor_by_time=corridor_by_time,
        posterior_anchor_used=False,
        low_snr_posterior=None,
    )
