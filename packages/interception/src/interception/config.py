"""击球目标点（拦截点）选择的配置。

说明：
    该包用于把“反弹后第二段轨迹分布（多候选/后验）”压缩成单一的击球目标点。
    设计依据见 `docs/interception.md`。

约定：
    - 坐标系与 `curve_v3` / `docs/curve.md` 一致：x 向右为正、z 向前为正、y 向上为正。
    - 目标点定义为：球心在高度平面 y==y* 的“反弹后下降穿越”位置与时刻。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InterceptionConfig:
    """击球目标点选择配置。

    属性说明（与 `docs/interception.md` 对齐）：
        y_min: 可击球高度下界（米）。
        y_max: 可击球高度上界（米）。
        num_heights: 在 [y_min, y_max] 上离散采样的高度数 K（建议默认 5）。

        r_hit_m: 命中半径（米）。用于把多峰分布压成单点时，最大化“命中概率质量”。

        quantile_levels: 输出/诊断用的分位数水平（0..1）。
        time_margin_quantile: 计算时间裕度的保守分位数（例如 0.10）。

        score_alpha_time: 评分中“时间裕度”的权重系数。
        score_dt_max_s: 时间裕度参与评分时的截断上限（秒）。
        score_lambda_width: 评分中“走廊宽度”的惩罚系数。
        score_mu_crossing: 评分中“穿越概率质量不足”的惩罚系数。

        min_crossing_prob: 若某高度的穿越概率质量低于该阈值，则直接判为不可用高度。
        min_valid_candidates: 若某高度有效候选样本数少于该阈值，则判为不可用高度。

        eps_tau_s: 选取下降穿越根时的最小 tau（秒），用于排除 tau≈0 的接触根。

        map_switch_enabled: N>0 时是否启用“收敛后切换 MAP”输出（见 docs/interception.md §5.4）。
        map_switch_min_points: 触发切换的最少 post 点数 N。
        map_switch_w_max: 触发切换的 w_max 阈值。

        hysteresis_score_gain: 迟滞阈值：新目标的 score 至少提升多少才允许更新。
        hysteresis_w_max_gain: 迟滞阈值：w_max 至少提升多少才允许更新。
        hysteresis_width_shrink_ratio: 迟滞阈值：width 至少缩小该比例才允许更新。

        multi_peak_second_phit_threshold: multi_peak 的启发式阈值：第二峰的命中概率下限。
        multi_peak_separation_r_mult: multi_peak 的启发式阈值：两峰中心距离需大于该倍数的 r_hit。
    """

    y_min: float
    y_max: float
    num_heights: int = 5

    r_hit_m: float = 0.15

    quantile_levels: tuple[float, ...] = (0.05, 0.50, 0.95)
    time_margin_quantile: float = 0.10

    score_alpha_time: float = 0.35
    score_dt_max_s: float = 1.0
    score_lambda_width: float = 1.0
    score_mu_crossing: float = 1.0

    min_crossing_prob: float = 0.4
    min_valid_candidates: int = 3

    eps_tau_s: float = 1e-3

    # 收敛后切换 MAP（可选增强）。
    map_switch_enabled: bool = True
    map_switch_min_points: int = 3
    map_switch_w_max: float = 0.7

    # 迟滞/稳定（必须，避免 N=1..2 时目标点抖动）。
    hysteresis_score_gain: float = 0.15
    hysteresis_w_max_gain: float = 0.08
    hysteresis_width_shrink_ratio: float = 0.25

    # multi_peak_flag 的启发式判定参数（可选诊断）。
    multi_peak_second_phit_threshold: float = 0.25
    multi_peak_separation_r_mult: float = 2.0
