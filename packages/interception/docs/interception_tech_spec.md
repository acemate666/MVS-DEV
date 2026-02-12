# Interception：单一击球目标点选择（技术规格）

> 面向开源：本文档以“可复现、可实现”为目标，给出 `interception/` 包中“离散候选分布 → 单一击球目标点”的形式化定义、算法流程、参数契约与边界条件。
>
> 相关文档：`docs/interception.md`（需求/直观解释）。实现位置：`interception/`。

## 1. 范围、输入与输出

### 1.1 适用范围

本文仅讨论以下问题：

- 给定反弹事件（bounce）以及反弹后二段轨迹的离散候选集合（带权重），在可击球高度区间内选择一个高度平面 $y=y^*$，并在该平面上输出一个单一目标点 $(x^*,y^*,z^*,t^*)$。

本文不讨论：候选生成（例如 $(e,k_t,\phi)$ 的采样策略）、上游观测噪声建模、以及控制器如何执行击球等。

### 1.2 输入（抽象定义）

- 反弹状态：包含反弹点 $(x_b,z_b)$、反弹时刻 $t_b$（相对时间基准 `time_base_abs` 的相对时刻），以及反弹处球心高度 $y_b$。
- 候选集合：$\{\theta_m\}_{m=1}^M$，每个候选包含反弹后二段的速度 $v^{+(m)}=(v_{x,m},v_{y,m},v_{z,m})$，可选地包含水平加速度 $a_{xz}^{(m)}=(a_{x,m},a_{z,m})$。
- 权重：$w_m\ge 0,\ \sum_{m=1}^M w_m=1$。
- 可选后验观测：反弹后观测点序列（post points），数量 $N$（工程实现中建议 $N\le 5$）。
- 配置：高度搜索区间与评分权重等超参（见 §10.3）。

### 1.3 输出（抽象定义）

算法返回一个结果对象：

- `valid`：是否找到满足置信度约束的可用目标。
- `target`：若 `valid=True`，包含 $(x^*,y^*,z^*)$ 与目标到达时刻 $t^{abs,*}$（绝对时间）。
- `reason`：若 `valid=False`，给出失败原因（字符串枚举，见 §9）。
- `diag`：诊断量（分位数包络、穿越概率质量、宽度、命中概率等），用于分析与调参。

工程实现中，该输出对应 `interception.types.HitTargetResult`（见 §10）。

## 2. 坐标、符号与物理假设

### 2.1 坐标系

坐标系与 `curve_v3` 一致：

- $x$：右为正（m）
- $z$：前为正（m）
- $y$：上为正（m）

重力加速度沿 $-y$ 方向，大小 $g>0$（m/s$^2$）。

### 2.2 时间基准

- `time_base_abs`：绝对时间基准（s）。
- $t_b$：反弹事件相对 `time_base_abs` 的相对时刻（s）。
- $\tau$：相对反弹时刻的相对时间（s），因此绝对时刻 $t^{abs}=time\_base\_abs+(t_b+\tau)$。

### 2.3 物理模型与假设

对反弹后二段的竖直方向采用常加速度模型：

$$
y(\tau)=y_b+v_{y,m}\tau-\tfrac12 g\tau^2.
$$

水平方向在两种模式下计算：

- 仅速度（N=0）：$x(\tau)=x_b+v_{x,m}\tau,\ z(\tau)=z_b+v_{z,m}\tau$。
- 速度+水平加速度（N>0）：$x(\tau)=x_b+v_{x,m}\tau+\tfrac12 a_{x,m}\tau^2,\ z(\tau)=z_b+v_{z,m}\tau+\tfrac12 a_{z,m}\tau^2$。

注：是否启用 $a_{xz}$ 取决于候选状态中是否包含该项，以及调用路径是否提供 post points（实现细节见 §10）。

## 3. 问题形式化

在可击球高度集合 $\mathcal{Y}$（由离散高度网格近似，见 §4.1）上，对每个高度 $y_k$ 定义其可用穿越样本集，并在该样本集上定义命中概率近似 $P_{hit,k}$ 与综合评分 $Score$。

对固定高度 $y_k$，先求单点压缩的最优中心索引：

$$
i_k^*=\arg\max_{i\in\mathcal{M}_k} P_{hit,k}(i).
$$

再跨高度选择最优高度：

$$
y^*=\arg\max_k Score(y_k).
$$

最终目标点取高度 $y^*$ 处的 $(x_{i_{k^*},k^*},y^*,z_{i_{k^*},k^*},t^{abs}_{i_{k^*},k^*})$（其中 $k^*$ 为 $y^*$ 的索引），从而保证 $(x,z,t)$ 来自同一候选轨迹的同一穿越事件。

## 4. Stage1：固定高度的评估与跨高度选择

### 4.1 高度离散

在 $[y_{min},y_{max}]$ 上均匀采样 $K$ 个高度：

$$
y_k=y_{min}+\frac{k}{K-1}(y_{max}-y_{min}),\quad k=0,1,\dots,K-1.
$$

实现对应：`interception.selector._sample_heights(cfg)`。

### 4.2 下降穿越根（核心契约）

对每个候选 $m$ 与高度 $y_k$，求解 $y(\tau)=y_k$ 的实根。为确保选择的是“反弹后下降穿越”，采用如下契约：

- $\tau>\epsilon_{\tau}$（排除 $\tau\approx 0$ 的接触根）。
- $\dot y(\tau)=v_{y,m}-g\tau<0$。

若存在多个满足条件的根，取 $\tau$ 最大者（对应“上升后再下降的穿越”，若该穿越存在）。

实现对应：`interception.selector._solve_downward_crossing_tau(...)`。

### 4.3 固定高度的穿越样本集与穿越概率质量

将满足 §4.2 契约的候选记为有效集合 $\mathcal{M}_k\subseteq\{1,\dots,M\}$。对每个 $m\in\mathcal{M}_k$，计算穿越样本：

$$
s_{m,k}=(x_{m,k},z_{m,k},t^{abs}_{m,k}),
$$

其中 $t^{abs}_{m,k}=time\_base\_abs+(t_b+\tau_{m,k})$。

定义穿越概率质量：

$$
crossing\_prob_k=\sum_{m\in\mathcal{M}_k} w_m.
$$

若 $crossing\_prob_k>0$，定义条件权重：

$$
\bar w_{m,k}=\frac{w_m}{crossing\_prob_k},\quad m\in\mathcal{M}_k.
$$

实现对应：`interception.selector._evaluate_height(...)`。

### 4.4 分位数包络与走廊宽度

对 $\{x_{m,k}\}$、$\{z_{m,k}\}$、$\{t^{abs}_{m,k}\}$ 计算加权分位数（权重 $\bar w_{m,k}$）。记分位数水平为 $q\in\mathcal{Q}$（默认 $\{0.05,0.50,0.95\}$），得到：

$$
x_{k,q},\ z_{k,q},\ t^{abs}_{k,q}.
$$

定义走廊宽度（用于惩罚横向不确定性）：

$$
width_k=(x_{k,0.95}-x_{k,0.05})+(z_{k,0.95}-z_{k,0.05}).
$$

实现对应：使用 `interception.math_utils.weighted_quantiles_1d`。

### 4.5 单点压缩：离散命中概率最大化

为避免多峰分布下均值/中位数落入低概率区域，采用“在样本集上最大化命中概率质量”的离散近似。

令二维位置向量 $p_{m,k}=(x_{m,k},z_{m,k})$。对任意样本索引 $i\in\mathcal{M}_k$，定义命中概率：

$$
P_{hit,k}(i)=\sum_{j\in\mathcal{M}_k}\bar w_{j,k}\,\mathbf{1}(\|p_{j,k}-p_{i,k}\|_2\le r_{hit}).
$$

选择：

$$
i_k^*=\arg\max_{i\in\mathcal{M}_k} P_{hit,k}(i).
$$

目标点取 $(x_{i_k^*,k},y_k,z_{i_k^*,k},t^{abs}_{i_k^*,k})$，从而保证 $(x,z,t)$ 来自同一候选的同一穿越事件。

实现对应：`interception.selector._evaluate_height(...)` 中的 $O(M^2)$ 扫描。

### 4.6 跨高度选择：综合评分

定义时间裕度（保守分位数，默认 $q_t=0.10$）：

$$
\Delta t_k=t^{abs}_{k,q_t}-t_{now}^{abs}.
$$

综合评分定义为：

$$
Score(y_k)=P_{hit,k}(i_k^*)
+\alpha\,\mathrm{clip}(\Delta t_k,0,\Delta t_{max})
-\lambda\,width_k
-\mu\,(1-crossing\_prob_k).
$$

最终选择：

$$
y^*=\arg\max_k Score(y_k).
$$

实现对应：`interception.selector._evaluate_height(...)` 计算每个 $y_k$ 的评分与诊断，`interception.selector._select_hit_target(...)` 取最大者。

### 4.7 可用性判定与降级

对每个高度 $y_k$，若满足任一条件则判为不可用：

- $crossing\_prob_k < min\_crossing\_prob$；
- $|\mathcal{M}_k| < min\_valid\_candidates$。

若所有高度均不可用，则整体返回 `valid=False`，并设置 `reason="no_valid_height_or_low_confidence"`。

## 5. Stage2（可选，N>0）：few-shot 后验更新与重加权

当提供 post points 时，对每个候选进行独立的后验校正与重加权，得到更新后的候选状态与权重，再执行 §4 的 Stage1。

### 5.1 per-candidate MAP 校正

对每个候选 $m$，在其参数空间内进行 MAP（最大后验）估计，得到校正后的状态 $\hat\theta_m$ 以及与 post points 一致性的代价 $J_m$。

实现对应：复用 `curve_v3.posterior.fit_posterior_map_for_candidate(...)`，输出 `PosteriorState`（含 $\hat v^+$、$\hat a_{xz}$）以及代价 $J_{post}$。

### 5.2 权重更新（退火 + log-sum-exp 归一化）

用代价对先验权重做指数型重加权：

$$
\log w_m \leftarrow \log w_m^{prior}-\tfrac12\,\beta(N)\,J_m,
$$

其中 $\beta(N)\in(0,1]$ 为退火系数（随观测点数量 $N$ 变化）。再用 log-sum-exp 做归一化以避免数值下溢。

实现对应：`interception.selector._posterior_update_per_candidate(...)`，退火系数来自 `curve_cfg.candidate_likelihood_beta(N)`。

## 6. 收敛后切换 MAP 输出（可选增强）

当候选权重分布近似单峰时，可将“目标点来源”从 §4.5 的 `"phit"` 切换为 `"map"`：

- 先用 §4.6 的评分仍然选择高度 $y^*$（高度选择逻辑不变）。
- 在已选高度 $y^*$ 上，用 MAP 候选（例如 $w_{max}$ 对应者）的穿越点输出目标。

该增强触发条件由配置指定（例如 $N\ge N_{min}$ 且 $w_{max}\ge w_{thr}$）。

实现对应：`interception.selector._select_hit_target(...)` 内部判断 `cfg.map_switch_*`，并调用 `_crossing_target_for_state(...)` 生成 MAP 穿越点。

## 7. 跨帧稳定（迟滞）

由于 $N$ 较小时（例如 $N=1,2$）后验更新可能不足以稳定目标点，工程实现提供迟滞机制以减少跨帧抖动。

设上一帧稳定输出为 $T_{prev}$，当前帧原始输出为 $T_{raw}$。仅当满足以下任一条件时，用 $T_{raw}$ 更新稳定输出：

1. `score` 至少提升 `hysteresis_score_gain`；
2. `w_max` 至少提升 `hysteresis_w_max_gain`；
3. `width_xz` 至少按比例缩小 `hysteresis_width_shrink_ratio`。

否则保持 $T_{prev}$。

实现对应：`interception.stabilizer.HitTargetStabilizer`。

## 8. 诊断量：multi_peak_flag（启发式，仅诊断）

`multi_peak_flag` 旨在指示“是否存在明显多峰”的可解释诊断，不参与控制决策。

启发式定义：

- 已知主峰中心 $i_k^*$（由 §4.5 的命中概率最大化得到）。
- 在满足 $\|p_{j,k}-p_{i_k^*,k}\|_2 > k_r\,r_{hit}$ 的样本中，若存在某个候选中心 $i'$，其 $P_{hit,k}(i')\ge \tau_{2nd}$，则判定为多峰。

其中 $k_r$ 与 $\tau_{2nd}$ 为可配置阈值。

实现对应：`interception.selector._estimate_multi_peak_flag(...)`。

## 9. 数值稳定、复杂度与失败模式

### 9.1 数值稳定

- 权重进行非负裁剪并归一化。
- 下降穿越根采用 $\epsilon_{\tau}$（实现中 `eps_tau_s`）排除 $\tau\approx 0$ 的接触根。
- Stage2 权重归一化使用 log-sum-exp。

### 9.2 复杂度

记高度数为 $K$、候选数为 $M$：

- 每个高度的命中概率扫描为 $O(M^2)$。
- 总体为 $O(K\cdot M^2)$。

### 9.3 失败模式（reason 枚举）

- `no_candidates`：候选为空。
- `invalid_gravity`：重力参数非法（例如 $g\le 0$）。
- `empty_height_grid`：高度网格为空或不可生成。
- `no_valid_height_or_low_confidence`：所有高度均不可用（穿越概率质量过低或有效候选过少）。
- `crossing_prob_too_low`：最优高度的 `crossing_prob` 仍低于阈值。
- `too_few_valid_candidates`：最优高度的有效候选数仍低于阈值。

上述枚举与 `interception.selector` 的实现保持一致。

## 10. 工程实现对应（接口、类型与默认参数）

### 10.1 对外接口

- `interception.select_hit_target_prefit_only`：仅基于 prefit/prior（$N=0$）。
- `interception.select_hit_target_with_post`：基于 prefit + post points（$1\le N\le 5$）。

### 10.2 输出类型

输出类型为 `interception.types.HitTargetResult`：

- `target`：`interception.types.HitTarget`，字段为 `x,y,z,t_abs,t_rel`。
- `diag`：`interception.types.HitTargetDiagnostics`，包含 `crossing_prob/valid_candidates/width_xz/p_hit/score`、分位数数组，以及可选的 `multi_peak_flag/target_source/w_max`。

### 10.3 配置与默认值

配置类型为 `interception.config.InterceptionConfig`。下表列出关键参数、符号与默认值（默认值来自当前实现）。

| 参数名 | 符号 | 单位 | 含义 | 默认值 |
|---|---:|---:|---|---:|
| `y_min` | $y_{min}$ | m | 可击球高度下界 | （必填） |
| `y_max` | $y_{max}$ | m | 可击球高度上界 | （必填） |
| `num_heights` | $K$ | - | 高度网格数量 | 5 |
| `r_hit_m` | $r_{hit}$ | m | 命中半径 | 0.15 |
| `quantile_levels` | $\mathcal{Q}$ | - | 分位数水平 | (0.05, 0.50, 0.95) |
| `time_margin_quantile` | $q_t$ | - | 时间裕度分位数 | 0.10 |
| `score_alpha_time` | $\alpha$ | - | 时间裕度权重 | 0.35 |
| `score_dt_max_s` | $\Delta t_{max}$ | s | 时间裕度截断上限 | 1.0 |
| `score_lambda_width` | $\lambda$ | - | 宽度惩罚权重 | 1.0 |
| `score_mu_crossing` | $\mu$ | - | 穿越概率不足惩罚权重 | 1.0 |
| `min_crossing_prob` | - | - | 穿越概率质量阈值 | 0.4 |
| `min_valid_candidates` | - | - | 最少有效候选数 | 3 |
| `eps_tau_s` | $\epsilon_{\tau}$ | s | 最小下降穿越时间 | 1e-3 |
| `map_switch_enabled` | - | - | 是否启用 MAP 切换 | True |
| `map_switch_min_points` | $N_{min}$ | - | MAP 切换最少点数 | 3 |
| `map_switch_w_max` | $w_{thr}$ | - | MAP 切换权重阈值 | 0.7 |
| `hysteresis_score_gain` | - | - | 迟滞：score 提升阈值 | 0.15 |
| `hysteresis_w_max_gain` | - | - | 迟滞：w_max 提升阈值 | 0.08 |
| `hysteresis_width_shrink_ratio` | - | - | 迟滞：width 缩小比例阈值 | 0.25 |
| `multi_peak_second_phit_threshold` | $\tau_{2nd}$ | - | 次峰命中概率阈值 | 0.25 |
| `multi_peak_separation_r_mult` | $k_r$ | - | 两峰中心最小间隔倍数 | 2.0 |

## 11. 测试覆盖（仓库现状）

单元测试文件：`tests/test_interception_target.py`，主要覆盖：

- 双峰情况下目标不落在两峰之间的低概率区域（避免均值陷阱）。
- `crossing_prob` 过低时的降级与 reason。
- $N>0$ 时重加权能够偏向一致候选，并在 $N\ge 3$ 时触发 MAP 切换（若启用）。
- `HitTargetStabilizer` 的迟滞保持行为。

## 12. 依赖边界与复用

当前实现复用 `curve_v3` 的以下能力以保持口径一致：

- `curve_v3.posterior.fit_posterior_map_for_candidate`：per-candidate MAP（few-shot）。
- `curve_cfg.candidate_likelihood_beta(N)`：权重退火系数。
- `interception.math_utils.weighted_quantile_1d/weighted_quantiles_1d`：加权分位数（本包内实现，用于避免跨包依赖内部路径）。
- `interception.math_utils.real_roots_of_quadratic`：二次方程实根（本包内实现）。

依赖边界：interception 不依赖 `curve_v3.core` 的流程编排，仅依赖稳定的数据类型与数学/后验工具。

## 13. 潜在扩展方向

- 将 `multi_peak_flag` 扩展为多分量解释（例如输出两个峰中心与权重）。
- 支持椭圆命中半径（方向相关容差）。
- 若上游提供置信度或不确定性尺度，可将其纳入评分或降级条件。
- 提供更贴近实际 pipeline 的集成示例（从上游预测器直接抽取 post points）。
