"""curve_v3 v1.1 融合相关的纯逻辑。

说明：
    `CurvePredictorV3` 内部既要维护观测/状态，又要实现 v1.1 的
    “每候选做 MAP 校正再打分”的融合步骤。

    为降低 `core.py` 体量并提升可测试性，这里把“纯计算”的部分抽出：
    - 计算候选的原始轨迹残差（诊断用途）。
    - 对每个候选做 MAP 后验拟合，得到 J_post，并用 log-sum-exp + beta 退火
      更新权重、选择最佳分支。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from curve_v3.adapters.camera_rig import CameraRig
from curve_v3.configs import CurveV3Config
from curve_v3.dynamics import propagate_post_bounce_state
from curve_v3.posterior.fit_map import fit_posterior_map_for_candidate
from curve_v3.low_snr.types import WindowDecisions
from curve_v3.types import BallObservation, BounceEvent, Candidate, PosteriorState


def candidate_costs(
    *,
    bounce: BounceEvent,
    candidates: Sequence[Candidate],
    post_points: Sequence[BallObservation],
    time_base_abs: float | None,
    low_snr: WindowDecisions | None = None,
    cfg: CurveV3Config,
) -> np.ndarray:
    """计算每个候选的归一化 SSE（诊断用途）。

    说明：
        v1.1 的融合流程使用“每候选 posterior MAP”再评分；这里保留一个
        诊断函数：直接用 prior 轨迹残差评分（不做每候选校正）。

    Returns:
        shape=(M,) 的 costs 数组。
    """

    if time_base_abs is None:
        return np.zeros(len(candidates), dtype=float)

    sigma = float(cfg.posterior.weight_sigma_m)
    sigma2 = max(sigma * sigma, 1e-9)

    costs: list[float] = []
    for c in candidates:
        sse = 0.0
        for p in post_points:
            t_rel = float(p.t - time_base_abs)
            tau = t_rel - float(bounce.t_rel)
            if tau <= 0:
                continue

            pos, _ = propagate_post_bounce_state(
                bounce=bounce,
                candidate=c,
                tau=float(tau),
                cfg=cfg,
            )

            dx = float(p.x - float(pos[0]))
            dy = float(p.y - float(pos[1]))
            dz = float(p.z - float(pos[2]))

            # 低 SNR：诊断评分也尊重 IGNORE_AXIS（其余模式不影响 SSE 口径）。
            if low_snr is not None and str(low_snr.x.mode) == "IGNORE_AXIS":
                dx = 0.0
            if low_snr is not None and str(low_snr.y.mode) == "IGNORE_AXIS":
                dy = 0.0
            if low_snr is not None and str(low_snr.z.mode) == "IGNORE_AXIS":
                dz = 0.0

            sse += dx * dx + dy * dy + dz * dz

        costs.append(float(sse / sigma2))

    return np.asarray(costs, dtype=float)


def reweight_candidates_and_select_best(
    *,
    bounce: BounceEvent,
    candidates: Sequence[Candidate],
    post_points: Sequence[BallObservation],
    time_base_abs: float | None,
    camera_rig: CameraRig | None = None,
    low_snr: WindowDecisions | None = None,
    cfg: CurveV3Config,
) -> tuple[list[Candidate], Candidate | None, int | None, PosteriorState | None]:
    """v1.1：逐候选 MAP 校正 -> 打分 -> 重赋权 -> 选最佳。"""

    if time_base_abs is None:
        return list(candidates), None, None, None

    # 像素域闭环的“预算控制”（Top-K）：
    # - 当 pixel_refine_top_k 未设置（None）或 <=0 时，保持历史行为：对所有候选都尝试像素闭环。
    # - 当 pixel_refine_top_k >0 且像素域条件满足时：
    #     1) 先对所有候选做 3D 点域 MAP（禁用像素闭环）得到粗打分；
    #     2) 仅对 Top-K 候选再做一次启用像素闭环的精打分。
    # 这样可以把像素域迭代预算集中在最有希望的分支上，符合 docs/curve.md 的在线算力建议。
    use_pixel_any = bool(cfg.pixel.pixel_enabled) and (camera_rig is not None) and any(
        bool(getattr(p, "obs_2d_by_camera", None)) for p in post_points
    )
    top_k_raw = cfg.pixel.pixel_refine_top_k
    top_k = int(top_k_raw) if top_k_raw is not None else 0
    use_top_k = use_pixel_any and (top_k > 0)

    post_states: list[PosteriorState | None] = [None for _ in candidates]
    costs: list[float] = [float("inf") for _ in candidates]

    def _fit_one(*, cand: Candidate, rig: CameraRig | None) -> tuple[PosteriorState | None, float]:
        out = fit_posterior_map_for_candidate(
            bounce=bounce,
            post_points=post_points,
            candidate=cand,
            time_base_abs=time_base_abs,
            camera_rig=rig,
            low_snr=low_snr,
            cfg=cfg,
        )
        if out is None:
            return None, float("inf")
        st, j = out
        return st, float(j)

    chosen_topk: list[int] | None = None
    costs_coarse: list[float] | None = None
    post_states_coarse: list[PosteriorState | None] | None = None

    if use_top_k:
        # 第一遍：3D-only 粗打分（用于全量候选的权重更新与 top-K 选择）。
        costs_coarse = [float("inf") for _ in candidates]
        post_states_coarse = [None for _ in candidates]
        for i, c in enumerate(candidates):
            st, j = _fit_one(cand=c, rig=None)
            post_states_coarse[i] = st
            costs_coarse[i] = float(j)

        k = int(min(top_k, len(candidates)))
        chosen: list[int] = []
        if k > 0:
            # 先按粗打分从小到大排序；若粗打分全是 inf，则按 prior 权重从大到小补齐。
            idx_all = list(range(len(candidates)))
            idx_finite = [i for i in idx_all if np.isfinite(float(costs_coarse[i]))]
            idx_finite.sort(key=lambda i: float(costs_coarse[i]))

            chosen = list(idx_finite[:k])
            if len(chosen) < k:
                rest = [i for i in idx_all if i not in chosen]
                rest.sort(key=lambda i: float(getattr(candidates[i], "weight", 0.0)), reverse=True)
                chosen.extend(rest[: (k - len(chosen))])

        chosen_topk = chosen

        # 默认输出使用“粗解”；top-K 内若精拟合成功，则用精解覆盖。
        for i in range(len(candidates)):
            post_states[i] = post_states_coarse[i]
            costs[i] = float(costs_coarse[i])

        for i in chosen:
            st, j = _fit_one(cand=candidates[i], rig=camera_rig)
            if st is not None and np.isfinite(float(j)):
                post_states[i] = st
                # 注意：这里的 costs[i] 用于“best 的精细选择”，
                # 权重更新仍应使用 costs_coarse（同一量纲）。
                costs[i] = float(j)
    else:
        # 历史行为：对所有候选都按当前 cfg/camera_rig 条件执行 MAP（包含可选像素闭环）。
        for i, c in enumerate(candidates):
            st, j = _fit_one(cand=c, rig=camera_rig)
            post_states[i] = st
            costs[i] = float(j)

    # 权重更新：
    # - top-K 模式下：使用 3D-only 的 costs_coarse（同一量纲、稳定）。
    # - 非 top-K 模式：使用当前 costs（可能包含像素域闭环代价），保持历史行为。
    costs_for_weights = costs_coarse if (use_top_k and costs_coarse is not None) else costs

    costs_arr = np.asarray(costs_for_weights, dtype=float)
    if costs_arr.size == 0:
        return list(candidates), None, None, None

    # 结合 prior 权重与 likelihood 权重（log-sum-exp 保持数值稳定）。
    prior_w = np.array([float(c.weight) for c in candidates], dtype=float)
    prior_w = np.maximum(prior_w, 1e-12)

    beta = float(cfg.candidate_likelihood_beta(len(post_points)))
    logw = np.log(prior_w) + (-0.5 * beta) * costs_arr

    m = float(np.max(logw))
    if not np.isfinite(m):
        w = np.full_like(prior_w, 1.0 / float(len(candidates)))
    else:
        expw = np.exp(logw - m)
        s = float(np.sum(expw))
        if s <= 0.0 or not np.isfinite(s):
            w = np.full_like(expw, 1.0 / float(len(candidates)))
        else:
            w = expw / s

    updated: list[Candidate] = []
    for c, wi in zip(candidates, w):
        updated.append(
            Candidate(
                e=c.e,
                kt=c.kt,
                weight=float(wi),
                v_plus=np.asarray(c.v_plus, dtype=float),
                kt_angle_rad=float(getattr(c, "kt_angle_rad", 0.0)),
                ax=float(c.ax),
                az=float(c.az),
            )
        )

    # 选 best：
    # - top-K 模式下：只在 chosen_topk 内做“精细 best 选择”，避免把不同量纲的代价混比。
    # - 非 top-K 模式：沿用历史行为（在全体候选上用 costs_arr 选最小）。
    if use_top_k and chosen_topk is not None and len(chosen_topk) > 0:
        fine_costs = np.asarray([float(costs[i]) for i in chosen_topk], dtype=float)
        if np.any(np.isfinite(fine_costs)):
            local = int(np.nanargmin(fine_costs))
            best_idx = int(chosen_topk[local])
        else:
            best_idx = int(np.argmin(costs_arr))
    else:
        # 先按最小 J_post，平局再按更新后权重最大。
        min_cost = float(np.min(costs_arr))
        tie = np.where(np.isclose(costs_arr, min_cost, atol=1e-9, rtol=0.0))[0]
        if tie.size <= 1:
            best_idx = int(np.argmin(costs_arr))
        else:
            best_idx = int(tie[int(np.argmax(w[tie]))])

    best = updated[best_idx]
    best_post = post_states[best_idx]
    return updated, best, best_idx, best_post


__all__ = [
    "candidate_costs",
    "reweight_candidates_and_select_best",
]
