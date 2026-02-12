# 2D 观测与像素域闭环：接口契约与回退规则

本文档定义 `curve_v3` 接入多相机 2D 观测时的**工程接口口径**与**可回退行为**。
目标是：在不破坏“纯 3D 点域流程”的前提下，允许在像素域做轻量闭环（B-lite）以提升精度上限与可诊断性。

相关专题：

- `05_stage2_posterior.md`：像素域闭环作为 posterior 的可选增强项（top-K 精修、权重更新口径）。
- `03_prefit_and_segmentation.md`：prefit 的时间基准与分段冻结契约；2D 只允许作为“质量/门控增强”，必须可回退。
- `06_observation_noise_and_weights.md`：观测噪声与权重口径（与像素域/3D 域门控关系）。
- `11_config_knobs.md`：开关与默认值风险说明。

---

## 1. 范围与非目标

范围：

- 2D 观测的**数据字段**与**单位/口径**；
- 相机投影能力的**注入边界**（`CameraRig`）；
- 像素域闭环（posterior/prefit）的**启用条件、门控、失败回退**；
- 用于评审/单测对齐的**最小数学定义**。

非目标：

- 不规定上游如何检测/跟踪球（只消费其输出）；
- 不规定 2D 协方差 `cov_uv` 的估计方法（只要求口径一致与数值可用）；
- 不承诺“只靠像素域优化即可修复所有时间基准/错关联问题”（这些属于系统级数据质量问题）。

---

## 2. 输入数据契约（必须满足）

`curve_v3` 以 `curve_v3.types.BallObservation` 作为单帧输入，以 `curve_v3.types.Obs2D` 承载单相机 2D 观测。

### 2.1 `BallObservation`

| 字段 | 类型 | 单位 | 口径/约束 |
|---|---|---|---|
| `x,y,z` | `float` | m | 世界坐标系球心位置。坐标约定见 `curve_v3.types` 头注释（x 右、y 上、z 前）。 |
| `t` | `float` | s | **物理采集时间戳（绝对秒）**。像素域闭环默认假设各相机在该帧时间上已对齐；若存在显著异步曝光，需在上游先对齐或改造接口。 |
| `conf` | `float \| None` | 无量纲 | 点级置信度（越大越可信）。用于低 SNR 权重与退化判别（见 `07_low_snr_policy.md`）。为 `None` 时等效按 `conf=1` 处理。 |
| `obs_2d_by_camera` | `dict[str, Obs2D] \| None` | - | 可选的每相机 2D 观测集合。为空时必须**完全回退**为纯 3D 点域流程。 |

### 2.2 `Obs2D`

| 字段 | 类型 | 单位 | 口径/约束 |
|---|---|---|---|
| `uv` | `np.ndarray` shape=(2,) | px | 像素坐标 $[u,v]$。必须为有限数。 |
| `cov_uv` | `np.ndarray` shape=(2,2) | px$^2$ | 像素协方差。工程上要求“近似对称正定”。若上游输出不稳定，可在上游或调用侧做轻量正则（例如对角加 $\epsilon$）。 |
| `sigma_px` | `float` | px | 便于调参/日志的标量噪声尺度（仅用于把白化残差映射回 px 等效量纲）。 |
| `cov_source` | `str` | - | 协方差来源说明（可空）。 |

### 2.3 相机标识一致性（强制）

- `obs_2d_by_camera` 的 key（例如 `"cam0"`）必须与 `CameraRig.cameras` 的 key 完全一致。
- 同一条回放序列中，相机 ID 不得漂移（否则像素残差不可解释、且难以复现）。

---

## 3. 系统注入契约：`CameraRig`（必须由上层提供）

像素域闭环只依赖一个最小投影能力：世界系 3D 点 $p\in\mathbb{R}^3$ 投影为像素 $[u,v]$。
该能力以 `curve_v3.adapters.camera_rig.CameraRig` 注入到 `CurvePredictorV3(camera_rig=...)`。

约束：

1) **投影口径一致**：`Obs2D.uv` 的坐标口径必须与 `CameraRig.project_world_to_pixel()` 输出口径一致。
   - 例如：若 `uv` 是“去畸变像素”，投影也必须输出去畸变像素；若 `uv` 是“含畸变原始像素”，投影也必须包含同样畸变模型。
2) **世界坐标一致**：投影使用的世界系必须与 `BallObservation.x/y/z` 一致。
3) **失败允许抛异常**：当点不可投影（背后、超出视野、内部数值异常）时允许抛异常；闭环必须把该观测视为不可用并回退/忽略。

---

## 4. 像素域代价口径（用于评审/单测对齐）

对一个像素观测（第 $i$ 个 3D 点、相机 $j$），定义：

- 观测：$z_{ij} \in \mathbb{R}^2$（`Obs2D.uv`）
- 协方差：$\Sigma_{ij}\in\mathbb{R}^{2\times2}$（`Obs2D.cov_uv`）
- 投影：$\pi_j(\cdot)$（`CameraRig` 提供）
- 轨迹模型给出的 3D 位置：$p(t_i;\theta)$（$\theta$ 的参数化见 `05_stage2_posterior.md`）

像素残差：

$$
r_{ij}(\theta) = \pi_j(p(t_i;\theta)) - z_{ij}.
$$

白化（Cholesky 口径）：令 $\Sigma_{ij}=LL^T$，则

$$
e_{ij}(\theta) = L^{-1}r_{ij}(\theta),\quad e_{ij}\in\mathbb{R}^2.
$$

工程实现会把白化残差范数映射为“像素等效”尺度，用于门控与 Huber：

$$
\|r\|_{px\_equiv} = \|e_{ij}\|_2\cdot \sigma_{px}
$$

其中 $\sigma_{px}$ 来自 `Obs2D.sigma_px`。
在各向同性 $\Sigma=\sigma_{px}^2 I$ 时，上式退化为 $\|r\|_2$。

鲁棒（Huber，IRLS 等效权重，阈值单位 px）：

$$
w_{ij}=\begin{cases}
1,& \|r\|_{px\_equiv}\le \delta_{px}\\
\delta_{px}/\|r\|_{px\_equiv},& \|r\|_{px\_equiv}>\delta_{px}
\end{cases}
$$

最终使用的加权白化残差向量为 $\sqrt{w_{ij}}\,e_{ij}$，并与 MAP 先验项共同构造小维度 GN/LM 更新。

---

## 5. 门控与回退规则（必须保持历史行为）

像素域相关逻辑属于增强项，任何时候都必须满足：**失败可回退、回退后仍能稳定输出**。

### 5.1 posterior 像素域精修（B-lite）

启用条件（同时满足才允许进入像素域）：

1) `cfg.pixel.pixel_enabled=True`；
2) 构造 `CurvePredictorV3` 时注入 `camera_rig`；
3) 至少存在一个 post 点携带 `obs_2d_by_camera`，且通过门控后仍有有效像素观测。

单观测门控（`cfg.pixel.pixel_gate_tau_px`）：

- 当 `pixel_gate_tau_px<=0`：关闭该门控；
- 当 `pixel_gate_tau_px>0`：若 $\|r\|_{px\_equiv} > pixel\_gate\_tau\_px$，该条 (点,相机) 观测被丢弃。

帧级门控（`cfg.pixel.pixel_min_cameras`）：

- 对同一个 3D 点（同一帧）的所有相机观测，若通过单观测门控的相机数 $<K$（$K=pixel\_min\_cameras$），则该帧所有像素观测均丢弃。

失败回退（任一触发即回退到 3D 点域 posterior）：

- 投影异常、残差出现 NaN/Inf；
- 白化不可用（协方差不可分解等）导致有效像素观测为 0；
- 迭代过程中出现数值异常（不可分解、更新非有限等）。

### 5.2 prefit 的像素一致性加权（可选）

prefit 支持使用像素一致性做门控/降权以提升时间基准稳定性，其开关与阈值位于：

- `cfg.prefit.prefit_pixel_enabled`
- `cfg.prefit.prefit_pixel_gate_tau_px`
- `cfg.prefit.prefit_pixel_huber_delta_px`
- `cfg.prefit.prefit_pixel_min_cameras`

该增强同样必须可回退：缺少 `camera_rig` 或缺少 `obs_2d_by_camera` 时不生效。

---

## 6. 配置旋钮速查（字段名以实现为准）

posterior 像素域精修：`CurveV3Config.pixel`：

- `pixel_enabled`：总开关。
- `pixel_max_iters`：GN/LM 迭代上限（固定小值）。
- `pixel_huber_delta_px`：Huber 阈值（px）。
- `pixel_gate_tau_px`：单观测门控阈值（px；<=0 关闭）。
- `pixel_min_cameras`：帧级门控最小有效相机数。
- `pixel_refine_top_k`：只对 top-K 候选做像素精修（节省算力）。

prefit 像素一致性加权：`CurveV3Config.prefit`：

- `prefit_pixel_enabled`：总开关。
- `prefit_pixel_gate_tau_px`：单观测门控阈值（px；<=0 关闭）。
- `prefit_pixel_huber_delta_px`：Huber 阈值（px）。
- `prefit_pixel_min_cameras`：帧级门控最小有效相机数。

---

## 7. 验证标准（门禁）

### 7.1 前置条件

- Python 版本满足 `packages/curve_v3/pyproject.toml` 的 `requires-python`（>=3.11）。
- 若需要验证像素域路径：必须准备可用的 `CameraRig` 与包含 `obs_2d_by_camera` 的输入数据。

### 7.2 命令

在 `packages/curve_v3` 目录运行：

- `python -m pytest -k "pixel_refine"`

### 7.3 预期输出/验收口径

- 相关单测全部通过（例如像素域精修与 top-K 精修用例）。
- 当未注入 `camera_rig` 或未提供 `obs_2d_by_camera` 时：像素域闭环不应被启用，流程应稳定回退到纯 3D 点域（行为必须可复现）。
