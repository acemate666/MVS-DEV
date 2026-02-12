# curve_v3 低 SNR（低信噪比）退化策略规格（prefit / posterior 共用）

本文档定义 `curve_v3` 在低信噪比场景下的**权重口径**与**退化模式（mode）**。目标是避免把观测噪声“拟合成速度/加速度”（噪声填充导数），从而破坏时间基准与短窗稳定性。

实现对齐：

- 配置：`curve_v3.configs.LowSnrConfig`（`CurveV3Config.low_snr`）
- 判别与权重：`curve_v3.low_snr.policy`
- prefit 应用：`curve_v3.prior.prefit`
- posterior 应用：`curve_v3.posterior`

相关专题：

- `06_observation_noise_and_weights.md`：观测噪声口径与时间基准风险。
- `03_prefit_and_segmentation.md`：prefit 冻结契约与 gap-freeze。
- `05_stage2_posterior.md`：后验 MAP/RLS 与权重更新。

---

## 1. 输入与基本假设

对一个短窗口（prefit 或 posterior 使用的滑窗），输入为 $N$ 个观测点：

$$
\{(t_i,\ x_i,\ y_i,\ z_i,\ conf_i)\}_{i=1}^N
$$

其中：

- $x/y/z$ 单位 m，$t$ 单位 s；
- `conf` 为点级置信度（无量纲，越大越可信），对应 `curve_v3.types.BallObservation.conf`。

工程假设（刻意简化）：

1) 轴向噪声独立（对角近似），不建模 $x/y/z$ 的相关性；
2) 在短窗口内，位置观测的主要风险来自“信号跨度不足导致导数不可辨识”。

---

## 2. 权重口径：conf → σ → W

对每个点 $i$、每个轴 $u\in\{x,y,z\}$，定义观测噪声尺度（单位 m）：

$$
\sigma_{u,i} = \frac{\sigma_{u0}}{\sqrt{\max(conf_i,\ c_{\min})}}
$$

并定义 WLS 权重：

$$
w_{u,i} = \frac{1}{\sigma_{u,i}^2}.
$$

约束与回退：

- 当 `conf_i is None` 或 `conf_i` 非有限值时，按 `conf_i = 1` 处理（等价于固定噪声尺度 $\sigma_{u,i}=\sigma_{u0}$）。
- $c_{\min}$ 用于避免 $1/\sqrt{conf}$ 发散，配置字段为 `low_snr_conf_cmin`。

重要说明：

- $\sigma_{u0}$ 是“位置观测标准差”（单位 m），不是过程噪声、不是动力学误差方差。
- 即使没有 conf，只要 `low_snr_enabled=True`，仍会按 `conf=1` 生效（可能改变行为）；若需要完全关闭该策略，必须显式关闭开关（见第 5 节）。

---

## 3. 可辨识性启发式：Δu vs. \bar{σ}

对每个轴 $u\in\{x,y,z\}$，计算窗口内的信号跨度：

$$
\Delta u = \max_i(u_i) - \min_i(u_i)
$$

以及平均噪声尺度：

$$
\bar\sigma_u = \frac{1}{N}\sum_{i=1}^N \sigma_{u,i}.
$$

直觉：当 $\Delta u$ 与 $\bar\sigma_u$ 同量级时，该轴的速度/加速度很容易被噪声驱动，导数估计不可用。

据此给出分级退化规则（实现口径为“先更极端的规则优先匹配”）：

- 若 $\Delta u < k_{ignore}\,\bar\sigma_u$：`IGNORE_AXIS`
- 否则若 $\Delta u < k_{strong}\,\bar\sigma_u$ **或** $N < N_{min}$：`STRONG_PRIOR_V`
- 否则若 $\Delta u < k_{freeze}\,\bar\sigma_u$：`FREEZE_A`
- 否则：`FULL`

其中 $(k_{freeze},k_{strong},k_{ignore},N_{min})$ 均为可配置字段（见第 5 节）。

---

## 4. mode 语义（对齐实现）

四种 mode 的语义如下（同一窗口会分别对 x/y/z 产出标签）：

1) `FULL`

- 正常使用观测；允许估计该轴的速度/加速度（具体参数化由阶段决定）。

2) `FREEZE_A`

- 判定“加速度不可辨识”。
- 该轴退化为线性（等价于把该轴二次项冻结为 0）：
  - prefit：按线性拟合该轴；
  - posterior：将该轴对应的 $0.5\,\tau^2$ 列置零（不从观测学习该轴加速度）。

3) `STRONG_PRIOR_V`

- 判定“速度也不稳定”（或点数过少）。
- 该轴冻结加速度，并对速度加入更强先验：

$$
J_{prior}(v_u)=\frac{(v_u - v_{u,prior})^2}{\sigma_{v_u}^2},\quad \sigma_{v_u}\ \text{更小（更强）}
$$

其中：

- prefit 阶段的强先验尺度为 `low_snr_prefit_strong_sigma_v_mps`（单位 m/s）；
- posterior 阶段通过缩放速度先验 $\sigma_v$ 实现，缩放因子为 `low_snr_strong_prior_v_scale`（<1 更强）。

4) `IGNORE_AXIS`

- 判定“该轴几乎无有效信息”。
- 该轴观测残差不进入拟合：
  - posterior：该轴行不加入线性系统；
  - prefit：该轴主要由先验传播/上一稳定估计承担。

该模式的工程含义是“宁可不更新，也不让噪声把导数带跑”。

---

## 4.1 y 轴的特殊约束（时间基准优先）

由于 y 轴通常承载最稳定的重力结构并直接影响 $t_b$ / $\tau$ 的时间基准，本实现默认：

- 不允许 y 轴进入 `IGNORE_AXIS`（字段：`low_snr_disallow_ignore_y=True`）；
- 更倾向于使用 `STRONG_PRIOR_V` 这类“可解释的保守退化”。

---

## 5. 配置字段（以实现为准）

`CurveV3Config.low_snr`（类型 `LowSnrConfig`）包含：

### 5.1 总开关

- `low_snr_enabled`：是否启用低 SNR 策略。

### 5.2 判别窗口与 conf 下限

- `low_snr_prefit_window_points`：prefit 阶段判别窗口长度（仅取末尾 N 点）。
- `low_snr_conf_cmin`：$c_{\min}$。

### 5.3 三轴基础噪声尺度（单位 m）

- `low_snr_sigma_x0_m`
- `low_snr_sigma_y0_m`
- `low_snr_sigma_z0_m`

### 5.4 退化阈值

- `low_snr_delta_k_freeze_a`：$k_{freeze}$
- `low_snr_delta_k_strong_v`：$k_{strong}$
- `low_snr_delta_k_ignore`：$k_{ignore}$
- `low_snr_min_points_for_v`：$N_{min}$
- `low_snr_disallow_ignore_y`：是否禁止 y 轴 `IGNORE_AXIS`

### 5.5 STRONG_PRIOR_V 强度

- `low_snr_strong_prior_v_scale`：posterior 的速度先验缩放（<1 更强）。
- `low_snr_prefit_strong_sigma_v_mps`：prefit 的速度强先验标准差（m/s）。

---

## 5.1 常见用法（契约化说明）

1) **完全关闭低 SNR（回到纯权重/纯拟合历史行为）**：

- 将 `low_snr_enabled=False`。

2) **只保留 conf 加权，不触发退化 mode（冻结/强先验/忽略）**：

- 设 `low_snr_delta_k_freeze_a=0`、`low_snr_delta_k_strong_v=0`、`low_snr_delta_k_ignore=0`；
- 同时设 `low_snr_min_points_for_v=0`（否则点数不足时仍会触发 `STRONG_PRIOR_V`）。

---

## 6. 验证标准（门禁）

### 6.1 前置条件

- Python 版本满足 `packages/curve_v3/pyproject.toml` 的 `requires-python`（>=3.11）。

### 6.2 命令

在 `packages/curve_v3` 目录运行：

- `python -m pytest -k "low_snr"`

### 6.3 预期输出/验收口径

- 相关单测全部通过（例如低 SNR 判别与 mode 退化行为用例）。
- 打开/关闭 `low_snr_enabled` 会以**可复现**的方式改变（或不改变）输出；当仅调整阈值关闭退化时，mode 应保持为 `FULL`（或 y 轴不允许 ignore 的保守约束仍可被验证）。
