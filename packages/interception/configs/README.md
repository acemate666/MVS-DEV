# interception 配置说明（YAML）

本目录用于存放 `interception` 的 YAML 配置文件。

- 示例配置：`configs/interception.yaml`
- 配置 dataclass 定义：`src/interception/config.py`
- YAML 加载入口：`src/interception/config_yaml.py`

## 如何加载

`interception` 的 YAML 顶层键对应 `interception.config.InterceptionConfig` 的字段名。

```python
from interception.config_yaml import load_interception_config_yaml

cfg = load_interception_config_yaml("configs/interception.yaml")
```

## 解析约定（非常重要）

这些行为来自 `interception/config_yaml.py` 的实现：

- 顶层必须是 mapping（YAML 里就是“字典/对象”）。
- **未知字段**：会直接报错（`KeyError`），避免“写了但不生效”。
- **必填字段**：`y_min` / `y_max` 在 dataclass 中无默认值，因此 YAML 必须提供；缺失会报错（`ValueError`，底层来自 dataclass 构造失败）。

## 字段说明（与 `docs/interception.md` 对齐）

单位约定：坐标系与 `packages/curve_v3/docs/02_curve.md` 一致：x 向右为正、z 向前为正、y 向上为正。

目标点定义：球心在高度平面 $y==y^*$ 的“反弹后下降穿越”位置与时刻。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `y_min` | `float` | （必填） | 可击球高度下界（米）。 |
| `y_max` | `float` | （必填） | 可击球高度上界（米）。 |
| `num_heights` | `int` | `5` | 在 `[y_min, y_max]` 上离散采样的高度数 K。 |
| `r_hit_m` | `float` | `0.15` | 命中半径（米）。用于把多峰分布压成单点时，最大化“命中概率质量”。 |
| `quantile_levels` | `tuple[float, ...]` | `(0.05, 0.50, 0.95)` | 输出/诊断用的分位数水平（0..1）。 |
| `time_margin_quantile` | `float` | `0.10` | 计算时间裕度的保守分位数（例如 0.10）。 |
| `score_alpha_time` | `float` | `0.35` | 评分中“时间裕度”的权重系数。 |
| `score_dt_max_s` | `float` | `1.0` | 时间裕度参与评分时的截断上限（秒）。 |
| `score_lambda_width` | `float` | `1.0` | 评分中“走廊宽度”的惩罚系数。 |
| `score_mu_crossing` | `float` | `1.0` | 评分中“穿越概率质量不足”的惩罚系数。 |
| `min_crossing_prob` | `float` | `0.4` | 某高度的穿越概率质量低于阈值则判为不可用高度。 |
| `min_valid_candidates` | `int` | `3` | 某高度有效候选样本数少于阈值则判为不可用高度。 |
| `eps_tau_s` | `float` | `1e-3` | 选取下降穿越根时的最小 tau（秒），用于排除 tau≈0 的接触根。 |
| `map_switch_enabled` | `bool` | `True` | 是否启用“收敛后切换 MAP”输出（见 `docs/interception.md §5.4`）。 |
| `map_switch_min_points` | `int` | `3` | 触发切换的最少 post 点数 N。 |
| `map_switch_w_max` | `float` | `0.7` | 触发切换的 w_max 阈值。 |
| `hysteresis_score_gain` | `float` | `0.15` | 迟滞阈值：新目标的 score 至少提升多少才允许更新。 |
| `hysteresis_w_max_gain` | `float` | `0.08` | 迟滞阈值：w_max 至少提升多少才允许更新。 |
| `hysteresis_width_shrink_ratio` | `float` | `0.25` | 迟滞阈值：width 至少缩小该比例才允许更新。 |
| `multi_peak_second_phit_threshold` | `float` | `0.25` | multi_peak 启发式阈值：第二峰的命中概率下限。 |
| `multi_peak_separation_r_mult` | `float` | `2.0` | multi_peak 启发式阈值：两峰中心距离需大于该倍数的 `r_hit_m`。 |

## 如何验证（建议）

- 运行单测以确认 YAML 能被正确解析：
  - `uv run python -m pytest -k config_yaml`
- 或直接运行全量测试（预期输出包含 `passed` 汇总行）：
  - `uv run python -m pytest`
