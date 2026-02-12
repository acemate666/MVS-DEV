"""击球目标点（拦截点）选择的类型定义。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HitTarget:
    """单一击球目标点。

    属性:
        x: 目标 x（米）。
        y: 目标 y（米），等于选中的击球高度平面。
        z: 目标 z（米）。
        t_abs: 目标到达时刻（绝对时间，秒）。
        t_rel: 目标到达时刻相对 time_base_abs 的时间（秒）。
    """

    x: float
    y: float
    z: float
    t_abs: float
    t_rel: float


@dataclass(frozen=True)
class HeightCandidateDiagnostics:
    """某个目标高度 y_k 的评估诊断。"""

    target_y: float
    is_valid: bool
    valid_candidates: int
    crossing_prob: float
    width_xz: float
    p_hit: float
    multi_peak_flag: bool | None
    score: float
    x_best: float | None
    z_best: float | None
    t_abs_best: float | None


@dataclass(frozen=True)
class HitTargetDiagnostics:
    """输出诊断信息（用于调参/排障）。"""

    # 选中高度与统计
    target_y: float | None
    crossing_prob: float
    valid_candidates: int
    width_xz: float
    p_hit: float

    # 该高度的综合评分（与 selector 内 Score(y_k) 一致）。
    score: float | None = None

    # 可选诊断：是否存在明显多峰（启发式判定，用于排障/可视化，不参与控制）。
    multi_peak_flag: bool | None = None

    # 目标点来源：
    # - "phit"：命中概率最大化（稳健，默认）。
    # - "map"：收敛后切到 MAP（可选增强）。
    target_source: str | None = None

    # 分位数输出（x,z,t_abs），与 quantile_levels 对齐。
    quantile_levels: np.ndarray | None = None  # shape (Q,)
    quantiles_x: np.ndarray | None = None  # shape (Q,)
    quantiles_z: np.ndarray | None = None  # shape (Q,)
    quantiles_t_abs: np.ndarray | None = None  # shape (Q,)

    # 高度网格上的全量诊断（便于离线回放分析）。
    per_height: tuple[HeightCandidateDiagnostics, ...] = ()

    # N>0 时可选输出：候选权重收敛程度。
    w_max: float | None = None


@dataclass(frozen=True)
class HitTargetResult:
    """击球目标点选择结果。"""

    valid: bool
    reason: str | None
    target: HitTarget | None
    diag: HitTargetDiagnostics
