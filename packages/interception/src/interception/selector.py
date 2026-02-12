"""击球目标点（拦截点）选择实现。

核心思想（见 `docs/interception.md`）：
    - N=0（只有 prefit）：只依赖 prior 候选与权重，输出“命中概率最大”的单点目标。
    - 1<=N<=5（prefit+post）：对每个候选做 MAP 校正与后验打分，更新权重后再选单点。

关键工程口径：
    - 目标点定义为：球心在高度平面 y==target_y 的“反弹后下降穿越”点。
    - 多峰时避免用均值/中位数：用 r_hit 做命中概率最大化选点。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

# 依赖边界：跨包仅依赖 curve_v3 的稳定 Public API（包顶层导出）。
from curve_v3 import (
    BallObservation,
    BounceEvent,
    Candidate,
    CurveV3Config,
    fit_posterior_map_for_candidate,
)
from interception.math_utils import real_roots_of_quadratic, weighted_quantile_1d, weighted_quantiles_1d

from interception.config import InterceptionConfig
from interception.types import (
    HeightCandidateDiagnostics,
    HitTarget,
    HitTargetDiagnostics,
    HitTargetResult,
)


def _clip01(x: float) -> float:
    """将数值裁剪到 [0,1]。"""

    return float(min(max(float(x), 0.0), 1.0))


def select_hit_target_prefit_only(
    *,
    bounce: BounceEvent,
    candidates: Sequence[Candidate],
    time_base_abs: float,
    t_now_abs: float,
    cfg: InterceptionConfig,
    curve_cfg: CurveV3Config,
) -> HitTargetResult:
    """仅基于 prefit/prior 候选分配击球目标点（N=0）。"""

    return _select_hit_target(
        bounce=bounce,
        candidates=candidates,
        post_points=(),
        time_base_abs=float(time_base_abs),
        t_now_abs=float(t_now_abs),
        cfg=cfg,
        curve_cfg=curve_cfg,
    )


def select_hit_target_with_post(
    *,
    bounce: BounceEvent,
    candidates: Sequence[Candidate],
    post_points: Sequence[BallObservation],
    time_base_abs: float,
    t_now_abs: float,
    cfg: InterceptionConfig,
    curve_cfg: CurveV3Config,
) -> HitTargetResult:
    """基于 prefit + post 点分配击球目标点（1<=N<=5）。"""

    pts = list(post_points)
    return _select_hit_target(
        bounce=bounce,
        candidates=candidates,
        post_points=pts,
        time_base_abs=float(time_base_abs),
        t_now_abs=float(t_now_abs),
        cfg=cfg,
        curve_cfg=curve_cfg,
    )


def _select_hit_target(
    *,
    bounce: BounceEvent,
    candidates: Sequence[Candidate],
    post_points: Sequence[BallObservation],
    time_base_abs: float,
    t_now_abs: float,
    cfg: InterceptionConfig,
    curve_cfg: CurveV3Config,
) -> HitTargetResult:
    """统一入口：支持 N=0 与 N>0。

    说明：
        - 本函数只负责“在当前帧”选择一个最优目标点。
        - 迟滞/稳定（跨帧保持）属于状态逻辑，放在 `interception.stabilizer` 里实现，
          避免 selector 与上游状态机耦合。
    """

    if not candidates:
        return HitTargetResult(
            valid=False,
            reason="no_candidates",
            target=None,
            diag=HitTargetDiagnostics(
                target_y=None,
                crossing_prob=0.0,
                valid_candidates=0,
                width_xz=float("nan"),
                p_hit=0.0,
                score=None,
                multi_peak_flag=None,
                target_source=None,
            ),
        )

    g = float(curve_cfg.physics.gravity)
    if not (g > 0.0):
        return HitTargetResult(
            valid=False,
            reason="invalid_gravity",
            target=None,
            diag=HitTargetDiagnostics(
                target_y=None,
                crossing_prob=0.0,
                valid_candidates=0,
                width_xz=float("nan"),
                p_hit=0.0,
                score=None,
                multi_peak_flag=None,
                target_source=None,
            ),
        )

    # 触地球心高度口径：优先用 bounce.y（若上游提供），否则使用 cfg.bounce_contact_y()。
    y_contact = float(curve_cfg.bounce_contact_y())
    if bounce.y is not None and np.isfinite(float(bounce.y)):
        y_contact = float(bounce.y)

    # 统一归一化候选权重到 sum(w)==1；保持与 corridor/posterior 的权重域一致。
    w0 = np.asarray([float(c.weight) for c in candidates], dtype=float)
    w0 = np.maximum(w0, 0.0)
    sw0 = float(np.sum(w0))
    if sw0 <= 0.0:
        w0 = np.full((len(candidates),), 1.0 / float(len(candidates)), dtype=float)
    else:
        w0 = w0 / sw0

    # N>0：对每候选做 MAP 校正并更新权重。
    # N==0：直接使用 prior 候选（v_plus/ax/az）与 prior 权重。
    states: list[_CandidateState | None]
    weights: np.ndarray
    w_max: float | None = None

    if post_points:
        states, weights = _posterior_update_per_candidate(
            bounce=bounce,
            candidates=candidates,
            post_points=post_points,
            time_base_abs=float(time_base_abs),
            curve_cfg=curve_cfg,
            prior_weights=w0,
        )
        if weights.size > 0:
            w_max = float(np.max(weights))
    else:
        states = []
        for c in candidates:
            # 技术口径：当 N=0（只有 prefit/prior）时，水平方向使用“仅速度”模型，
            # 避免把先验 a_xz 项带入穿越点，导致在缺少 post 约束时产生不必要的漂移。
            states.append(
                _CandidateState(
                    t_b_rel=float(bounce.t_rel),
                    x_b=float(bounce.x),
                    z_b=float(bounce.z),
                    vx=float(c.v_plus[0]),
                    vy=float(c.v_plus[1]),
                    vz=float(c.v_plus[2]),
                    # 说明：这里不是“缺功能”，而是刻意选择更稳健的退化模型。
                    # 当没有 post 点约束时，a_xz 往往只来自先验/噪声，带入会放大漂移。
                    ax=0.0,
                    az=0.0,
                )
            )
        weights = w0

    ys = _sample_heights(cfg)
    if not ys:
        return HitTargetResult(
            valid=False,
            reason="empty_height_grid",
            target=None,
            diag=HitTargetDiagnostics(
                target_y=None,
                crossing_prob=0.0,
                valid_candidates=0,
                width_xz=float("nan"),
                p_hit=0.0,
                score=None,
                multi_peak_flag=None,
                target_source=None,
                w_max=w_max,
            ),
        )

    best_score = -float("inf")
    best_target: HitTarget | None = None
    best_diag: HitTargetDiagnostics | None = None

    per_height: list[HeightCandidateDiagnostics] = []

    for yk in ys:
        eval_out = _evaluate_height(
            target_y=float(yk),
            y_contact=float(y_contact),
            g=float(g),
            eps_tau_s=float(cfg.eps_tau_s),
            r_hit_m=float(cfg.r_hit_m),
            weights=weights,
            states=states,
            time_base_abs=float(time_base_abs),
            t_now_abs=float(t_now_abs),
            cfg=cfg,
        )
        per_height.append(eval_out.per_height)

        if not eval_out.is_valid:
            continue

        if eval_out.score > best_score:
            best_score = float(eval_out.score)
            best_target = eval_out.target
            best_diag = eval_out.diag

    if best_target is None or best_diag is None:
        # 若所有高度都不可用，直接判 invalid。
        return HitTargetResult(
            valid=False,
            reason="no_valid_height_or_low_confidence",
            target=None,
            diag=replace(
                HitTargetDiagnostics(
                    target_y=None,
                    crossing_prob=0.0,
                    valid_candidates=0,
                    width_xz=float("nan"),
                    p_hit=0.0,
                    score=None,
                    multi_peak_flag=None,
                    target_source=None,
                    w_max=w_max,
                ),
                per_height=tuple(per_height),
            ),
        )

    # 将 per-height 诊断注入最终 diag。
    best_diag2 = replace(best_diag, per_height=tuple(per_height), w_max=w_max)

    # 可选增强：若 N>=3 且候选权重已收敛，则切到 MAP 候选在 y* 的穿越点输出。
    # 说明：
    #   - 未收敛时继续用命中概率最大化（稳健）。
    #   - 收敛后用 MAP 更贴真值（但二者在单峰时通常一致）。
    if (
        post_points
        and bool(cfg.map_switch_enabled)
        and w_max is not None
        and w_max >= float(cfg.map_switch_w_max)
        and len(post_points) >= int(cfg.map_switch_min_points)
        and best_diag2.target_y is not None
    ):
        map_i = int(np.argmax(weights)) if weights.size > 0 else -1
        if 0 <= map_i < len(states):
            st = states[map_i]
            if st is not None:
                map_target = _crossing_target_for_state(
                    st=st,
                    target_y=float(best_diag2.target_y),
                    y_contact=float(y_contact),
                    g=float(g),
                    eps_tau_s=float(cfg.eps_tau_s),
                    time_base_abs=float(time_base_abs),
                )
                if map_target is not None:
                    best_target = map_target
                    best_diag2 = replace(best_diag2, target_source="map")

    if best_diag2.target_source is None:
        best_diag2 = replace(best_diag2, target_source="phit")

    # 安全阀：若选中高度仍然低穿越概率/候选过少，则低置信度。
    if best_diag2.crossing_prob < float(cfg.min_crossing_prob):
        return HitTargetResult(
            valid=False,
            reason="crossing_prob_too_low",
            target=None,
            diag=best_diag2,
        )
    if best_diag2.valid_candidates < int(cfg.min_valid_candidates):
        return HitTargetResult(
            valid=False,
            reason="too_few_valid_candidates",
            target=None,
            diag=best_diag2,
        )

    return HitTargetResult(valid=True, reason=None, target=best_target, diag=best_diag2)


@dataclass(frozen=True)
class _CandidateState:
    t_b_rel: float
    x_b: float
    z_b: float
    vx: float
    vy: float
    vz: float
    ax: float
    az: float


@dataclass(frozen=True)
class _HeightEval:
    is_valid: bool
    score: float
    target: HitTarget | None
    diag: HitTargetDiagnostics
    per_height: HeightCandidateDiagnostics


def _sample_heights(cfg: InterceptionConfig) -> list[float]:
    y_min = float(cfg.y_min)
    y_max = float(cfg.y_max)
    if not (np.isfinite(y_min) and np.isfinite(y_max)):
        return []

    if y_max < y_min:
        y_min, y_max = y_max, y_min

    k = int(cfg.num_heights)
    if k <= 1:
        return [0.5 * (y_min + y_max)]
    if abs(y_max - y_min) < 1e-12:
        return [y_min]

    ys = np.linspace(y_min, y_max, k, dtype=float)
    return [float(y) for y in ys.tolist()]


def _solve_downward_crossing_tau(
    *,
    vy: float,
    y_contact: float,
    target_y: float,
    g: float,
    eps_tau_s: float,
) -> float | None:
    """求解下降穿越根 tau（若不可用返回 None）。"""

    a = -0.5 * float(g)
    b = float(vy)
    c = float(y_contact - float(target_y))

    roots = real_roots_of_quadratic(np.array([a, b, c], dtype=float))
    if not roots:
        return None

    eps = float(max(eps_tau_s, 0.0))
    candidates: list[float] = []
    for tau in roots:
        tau = float(tau)
        if tau <= eps:
            continue
        dy = float(vy - float(g) * tau)
        if dy < 0.0:
            candidates.append(tau)

    if not candidates:
        return None

    # 若存在多个满足“下降”的根，取最大的（通常对应上升后下降的第二次穿越）。
    return float(max(candidates))


def _evaluate_height(
    *,
    target_y: float,
    y_contact: float,
    g: float,
    eps_tau_s: float,
    r_hit_m: float,
    weights: np.ndarray,
    states: Sequence[_CandidateState | None],
    time_base_abs: float,
    t_now_abs: float,
    cfg: InterceptionConfig,
) -> _HeightEval:
    """在固定 target_y 上评估并返回该高度的最优单点与诊断。

    核心步骤（与 docs/interception.md 对齐）：
        1) 计算每个候选在 y==target_y 的下降穿越点 (x,z,t)。
        2) 统计 crossing_prob、加权分位数包络 width。
        3) 用命中半径 r_hit 做离散命中概率最大化，选单一目标点。
        4) 计算 Score(y_k)，供跨高度比较。
    """

    xs: list[float] = []
    zs: list[float] = []
    ts_abs: list[float] = []
    ws: list[float] = []

    for w, st in zip(weights.tolist(), states):
        if st is None:
            continue
        w = float(w)
        if w <= 0.0:
            continue

        tau = _solve_downward_crossing_tau(
            vy=float(st.vy),
            y_contact=float(y_contact),
            target_y=float(target_y),
            g=float(g),
            eps_tau_s=float(eps_tau_s),
        )
        if tau is None:
            continue

        x = float(st.x_b + float(st.vx) * tau + 0.5 * float(st.ax) * tau * tau)
        z = float(st.z_b + float(st.vz) * tau + 0.5 * float(st.az) * tau * tau)
        t_rel = float(st.t_b_rel + tau)
        t_abs = float(time_base_abs + t_rel)

        xs.append(x)
        zs.append(z)
        ts_abs.append(t_abs)
        ws.append(w)

    valid = int(len(ws))
    if valid < int(cfg.min_valid_candidates):
        diag0 = HitTargetDiagnostics(
            target_y=None,
            crossing_prob=float(np.sum(ws)) if ws else 0.0,
            valid_candidates=valid,
            width_xz=float("nan"),
            p_hit=0.0,
            score=None,
            multi_peak_flag=None,
            target_source=None,
        )
        return _HeightEval(
            is_valid=False,
            score=-float("inf"),
            target=None,
            diag=diag0,
            per_height=HeightCandidateDiagnostics(
                target_y=float(target_y),
                is_valid=False,
                valid_candidates=valid,
                crossing_prob=float(np.sum(ws)) if ws else 0.0,
                width_xz=float("nan"),
                p_hit=0.0,
                multi_peak_flag=None,
                score=-float("inf"),
                x_best=None,
                z_best=None,
                t_abs_best=None,
            ),
        )

    w_arr = np.asarray(ws, dtype=float)
    crossing_prob = float(np.sum(w_arr))
    if crossing_prob <= 0.0:
        diag0 = HitTargetDiagnostics(
            target_y=None,
            crossing_prob=0.0,
            valid_candidates=valid,
            width_xz=float("nan"),
            p_hit=0.0,
            score=None,
            multi_peak_flag=None,
            target_source=None,
        )
        return _HeightEval(
            is_valid=False,
            score=-float("inf"),
            target=None,
            diag=diag0,
            per_height=HeightCandidateDiagnostics(
                target_y=float(target_y),
                is_valid=False,
                valid_candidates=valid,
                crossing_prob=0.0,
                width_xz=float("nan"),
                p_hit=0.0,
                multi_peak_flag=None,
                score=-float("inf"),
                x_best=None,
                z_best=None,
                t_abs_best=None,
            ),
        )

    if crossing_prob < float(cfg.min_crossing_prob):
        diag0 = HitTargetDiagnostics(
            target_y=None,
            crossing_prob=crossing_prob,
            valid_candidates=valid,
            width_xz=float("nan"),
            p_hit=0.0,
            score=None,
            multi_peak_flag=None,
            target_source=None,
        )
        return _HeightEval(
            is_valid=False,
            score=-float("inf"),
            target=None,
            diag=diag0,
            per_height=HeightCandidateDiagnostics(
                target_y=float(target_y),
                is_valid=False,
                valid_candidates=valid,
                crossing_prob=crossing_prob,
                width_xz=float("nan"),
                p_hit=0.0,
                multi_peak_flag=None,
                score=-float("inf"),
                x_best=None,
                z_best=None,
                t_abs_best=None,
            ),
        )

    w_cond = w_arr / max(crossing_prob, 1e-12)

    x_arr = np.asarray(xs, dtype=float)
    z_arr = np.asarray(zs, dtype=float)
    t_arr = np.asarray(ts_abs, dtype=float)

    q_levels = np.asarray(cfg.quantile_levels, dtype=float).reshape(-1)
    q_levels = q_levels[np.isfinite(q_levels)]
    q_levels = q_levels[(q_levels >= 0.0) & (q_levels <= 1.0)]

    qx = weighted_quantiles_1d(x_arr, w_cond, q_levels) if q_levels.size > 0 else None
    qz = weighted_quantiles_1d(z_arr, w_cond, q_levels) if q_levels.size > 0 else None
    qt = weighted_quantiles_1d(t_arr, w_cond, q_levels) if q_levels.size > 0 else None

    width_xz = float("nan")
    if q_levels.size >= 2 and qx is not None and qz is not None:
        # 走廊宽度口径：优先使用 5% 与 95% 分位（与 docs/interception_tech_spec.md 对齐）。
        # 若调用方修改了 quantile_levels 且不包含 0.05/0.95，则退化为使用最小/最大分位。
        q_low = 0.05
        q_high = 0.95
        has_q_low = bool(np.any(np.isclose(q_levels, q_low)))
        has_q_high = bool(np.any(np.isclose(q_levels, q_high)))
        if not (has_q_low and has_q_high):
            q_low = float(np.min(q_levels))
            q_high = float(np.max(q_levels))

        # 由于 q_levels 可能不是排序的，这里按 level 查索引。
        i_low = int(np.argmin(np.abs(q_levels - q_low)))
        i_high = int(np.argmin(np.abs(q_levels - q_high)))
        width_xz = float((qx[i_high] - qx[i_low]) + (qz[i_high] - qz[i_low]))

    # 时间裕度用保守分位数（默认 10%）。
    t_margin_q = _clip01(float(cfg.time_margin_quantile))
    t_q = float(weighted_quantile_1d(t_arr, w_cond, t_margin_q))
    dt_margin = float(t_q - float(t_now_abs))

    # 命中概率最大化：在离散样本集内选择一个“中心点”。
    r2 = float(r_hit_m) * float(r_hit_m)
    best_i = 0
    best_phit = -1.0
    best_t = float("inf")
    best_wi = -1.0

    # 记录每个样本作为中心点时的命中概率，用于 multi_peak_flag 判定。
    phits: list[float] = []

    for i in range(int(x_arr.size)):
        dx = x_arr - float(x_arr[i])
        dz = z_arr - float(z_arr[i])
        mask = (dx * dx + dz * dz) <= r2
        phit = float(np.sum(w_cond[mask]))
        phits.append(phit)
        ti = float(t_arr[i])
        wi = float(w_cond[i])

        # tie-break：优先更大 phit，其次更早到达（留更多时间），再其次更大本点权重。
        if phit > best_phit + 1e-12:
            best_phit, best_i, best_t, best_wi = phit, i, ti, wi
        elif abs(phit - best_phit) <= 1e-12:
            if ti < best_t - 1e-12:
                best_phit, best_i, best_t, best_wi = phit, i, ti, wi
            elif abs(ti - best_t) <= 1e-12 and wi > best_wi + 1e-12:
                best_phit, best_i, best_t, best_wi = phit, i, ti, wi

    x_best = float(x_arr[best_i])
    z_best = float(z_arr[best_i])
    t_abs_best = float(t_arr[best_i])
    t_rel_best = float(t_abs_best - float(time_base_abs))

    # 高度评分（见 docs/interception.md）。
    alpha = float(cfg.score_alpha_time)
    lam = float(cfg.score_lambda_width)
    mu = float(cfg.score_mu_crossing)
    dt_clip = float(max(0.0, min(dt_margin, float(cfg.score_dt_max_s))))

    width_term = 0.0
    if np.isfinite(width_xz):
        width_term = float(width_xz)

    score = float(best_phit + alpha * dt_clip - lam * width_term - mu * (1.0 - crossing_prob))

    multi_peak_flag = _estimate_multi_peak_flag(
        x_arr=x_arr,
        z_arr=z_arr,
        phits=np.asarray(phits, dtype=float),
        best_i=int(best_i),
        r_hit_m=float(r_hit_m),
        second_phit_threshold=float(cfg.multi_peak_second_phit_threshold),
        separation_r_mult=float(cfg.multi_peak_separation_r_mult),
    )

    target = HitTarget(
        x=x_best,
        y=float(target_y),
        z=z_best,
        t_abs=t_abs_best,
        t_rel=t_rel_best,
    )

    diag = HitTargetDiagnostics(
        target_y=float(target_y),
        crossing_prob=float(crossing_prob),
        valid_candidates=int(valid),
        width_xz=float(width_xz),
        p_hit=float(best_phit),
        score=float(score),
        multi_peak_flag=bool(multi_peak_flag),
        target_source="phit",
        quantile_levels=q_levels if q_levels.size > 0 else None,
        quantiles_x=np.asarray(qx, dtype=float) if qx is not None else None,
        quantiles_z=np.asarray(qz, dtype=float) if qz is not None else None,
        quantiles_t_abs=np.asarray(qt, dtype=float) if qt is not None else None,
    )

    per = HeightCandidateDiagnostics(
        target_y=float(target_y),
        is_valid=True,
        valid_candidates=int(valid),
        crossing_prob=float(crossing_prob),
        width_xz=float(width_xz),
        p_hit=float(best_phit),
        multi_peak_flag=bool(multi_peak_flag),
        score=float(score),
        x_best=float(x_best),
        z_best=float(z_best),
        t_abs_best=float(t_abs_best),
    )

    return _HeightEval(is_valid=True, score=float(score), target=target, diag=diag, per_height=per)


def _posterior_update_per_candidate(
    *,
    bounce: BounceEvent,
    candidates: Sequence[Candidate],
    post_points: Sequence[BallObservation],
    time_base_abs: float,
    curve_cfg: CurveV3Config,
    prior_weights: np.ndarray,
) -> tuple[list[_CandidateState | None], np.ndarray]:
    """对每条候选执行 posterior MAP，得到校正状态与更新后的离散权重。"""

    n = int(len(candidates))
    if n <= 0:
        return [], np.zeros((0,), dtype=float)

    costs = np.full((n,), float("inf"), dtype=float)
    states: list[_CandidateState | None] = [None for _ in range(n)]

    for i, c in enumerate(candidates):
        out = fit_posterior_map_for_candidate(
            bounce=bounce,
            post_points=post_points,
            candidate=c,
            time_base_abs=float(time_base_abs),
            cfg=curve_cfg,
        )
        if out is None:
            continue

        st, j_post = out
        costs[i] = float(j_post)
        states[i] = _CandidateState(
            t_b_rel=float(st.t_b_rel),
            x_b=float(st.x_b),
            z_b=float(st.z_b),
            vx=float(st.vx),
            vy=float(st.vy),
            vz=float(st.vz),
            ax=float(st.ax),
            az=float(st.az),
        )

    prior_w = np.asarray(prior_weights, dtype=float).reshape(-1)
    if prior_w.size != n:
        prior_w = np.full((n,), 1.0 / float(n), dtype=float)

    prior_w = np.maximum(prior_w, 1e-12)

    beta = float(curve_cfg.candidate_likelihood_beta(len(post_points)))
    logw = np.log(prior_w) + (-0.5 * beta) * costs

    m = float(np.max(logw))
    if not np.isfinite(m):
        w = np.full((n,), 1.0 / float(n), dtype=float)
        return states, w

    expw = np.exp(logw - m)
    s = float(np.sum(expw))
    if s <= 0.0 or not np.isfinite(s):
        w = np.full((n,), 1.0 / float(n), dtype=float)
        return states, w

    w = expw / s
    return states, np.asarray(w, dtype=float)


def _estimate_multi_peak_flag(
    *,
    x_arr: np.ndarray,
    z_arr: np.ndarray,
    phits: np.ndarray,
    best_i: int,
    r_hit_m: float,
    second_phit_threshold: float,
    separation_r_mult: float,
) -> bool:
    """启发式判断是否存在明显多峰。

    判定逻辑（尽量简单、可解释）：
        - 已知 best_i 是命中概率最大化选出的“主峰中心”。
        - 若在距离主峰中心足够远（> separation_r_mult * r_hit）的样本里，仍存在一个
          次峰中心，其 phit >= second_phit_threshold，则认为是多峰。

    说明：
        - 这是诊断用 flag，不参与控制决策。
        - M 很小（典型 9/27），用 O(M) 扫描足够。
    """

    if x_arr.size <= 0 or z_arr.size <= 0 or phits.size != x_arr.size:
        return False

    if not (0 <= int(best_i) < int(x_arr.size)):
        return False

    r = float(max(r_hit_m, 0.0))
    if r <= 0.0:
        return False

    sep = float(max(separation_r_mult, 0.0)) * r
    sep2 = sep * sep

    dx = x_arr - float(x_arr[int(best_i)])
    dz = z_arr - float(z_arr[int(best_i)])
    dist2 = dx * dx + dz * dz

    mask_far = dist2 > sep2
    if not bool(np.any(mask_far)):
        return False

    phit2 = float(np.max(phits[mask_far]))
    return bool(phit2 >= float(second_phit_threshold))


def _crossing_target_for_state(
    *,
    st: _CandidateState,
    target_y: float,
    y_contact: float,
    g: float,
    eps_tau_s: float,
    time_base_abs: float,
) -> HitTarget | None:
    """计算单个候选状态在 y==target_y 的下降穿越点（用于 MAP 切换输出）。"""

    tau = _solve_downward_crossing_tau(
        vy=float(st.vy),
        y_contact=float(y_contact),
        target_y=float(target_y),
        g=float(g),
        eps_tau_s=float(eps_tau_s),
    )
    if tau is None:
        return None

    x = float(st.x_b + float(st.vx) * tau + 0.5 * float(st.ax) * tau * tau)
    z = float(st.z_b + float(st.vz) * tau + 0.5 * float(st.az) * tau * tau)
    t_rel = float(st.t_b_rel + tau)
    t_abs = float(time_base_abs + t_rel)

    return HitTarget(x=float(x), y=float(target_y), z=float(z), t_abs=float(t_abs), t_rel=float(t_rel))


__all__ = [
    "select_hit_target_prefit_only",
    "select_hit_target_with_post",
]
