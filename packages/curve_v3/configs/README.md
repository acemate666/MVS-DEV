# curve_v3 配置说明（YAML）

本目录用于存放 `curve_v3` 的 YAML 配置文件。

- 示例配置：`configs/curve_v3.yaml`
- 配置 dataclass 定义：`src/curve_v3/configs/models.py`
- YAML 加载入口：`src/curve_v3/config_yaml.py`

## 如何加载

`curve_v3` 的 YAML 顶层键对应 `curve_v3.configs.CurveV3Config` 的字段名（例如 `physics`、`prior`、`posterior` 等）。

```python
from curve_v3.config_yaml import load_curve_v3_config_yaml

cfg = load_curve_v3_config_yaml("configs/curve_v3.yaml")
```

## 解析约定（非常重要）

这些行为来自 `curve_v3/config_yaml.py` 的实现：

- 顶层必须是 mapping（YAML 里就是“字典/对象”）。
- **未提供的字段**：使用 dataclass 的默认值。
- **未知字段**：会直接报错（`KeyError`），避免“写了但不生效”。
- **tuple 归一化**：若某字段的默认值是 `tuple`，则 YAML 中写 `list`/`tuple` 都会被归一化为 `tuple`。
  - 该归一化只做“结构层面”的形态统一，不做数值语义校验（避免改变业务逻辑）。

## 配置结构概览

`CurveV3Config` 按领域拆分为多个子配置（`physics/prior/posterior/...`）。算法模块应尽量只依赖自己负责的子配置，避免形成隐性全局公共 API。

### physics（`PhysicsConfig`）

物理/几何口径相关配置。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `gravity` | `float` | `9.8` | 重力加速度。 |
| `ground_normal` | `tuple[float, float, float]` | `(0.0, 1.0, 0.0)` | 地面法向（默认水平地面，y-up）。 |
| `ball_radius_m` | `float` | `0.033` | 球半径（用于定义触地时刻球心高度）。 |
| `bounce_contact_y_m` | `float \| None` | `None` | 触地/反弹时刻的球心高度；`None` 表示跟随 `ball_radius_m`。 |

补充：`PhysicsConfig.bounce_contact_y()` 会返回实际使用的触地高度。

### prior（`PriorConfig`）

prior（候选生成/反弹参数离散化）相关配置。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `e_bins` | `Sequence[float]` | `(0.55, 0.70, 0.85)` | e 的离散候选 bins。 |
| `kt_bins` | `Sequence[float]` | `(0.45, 0.65, 0.85)` | kt 的离散候选 bins。 |
| `e_range` | `tuple[float, float]` | `(0.05, 1.25)` | e 的裁剪范围（用于兜底防止数值发散）。 |
| `kt_angle_bins_rad` | `Sequence[float]` | `(0.0,)` | kt 映射的可选偏转角 bins（默认不启用偏转，避免候选膨胀/改变历史行为）。 |
| `kt_range` | `tuple[float, float]` | `(-1.2, 1.2)` | kt 的裁剪范围。 |

### candidate（`CandidateConfig`）

候选权重退火相关配置。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `candidate_beta_warmup_points` | `int` | `2` | 前 N 个点使用较小 beta（降低锁错分支风险）。 |
| `candidate_beta_min` | `float` | `0.3` | 预热期 beta 的最小值。 |

补充：`CandidateConfig.likelihood_beta(num_points)` 用于根据点数计算 beta。

### posterior（`PosteriorConfig`）

posterior（反弹后校正/MAP/RLS）相关配置。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `fit_mode` | `Literal["rls"]` | `"rls"` | 后验拟合模式（仅保留递推 rls）。 |
| `fit_params` | `Literal["v_only", "v+axz"]` | `"v+axz"` | 拟合参数集合。 |
| `max_post_points` | `int` | `999` | 第二阶段使用的 post 点上限（默认很大，避免评估回放被裁）。 |
| `weight_sigma_m` | `float` | `0.15` | 候选打分残差尺度（m），用于似然权重。 |
| `posterior_obs_sigma_m` | `float \| None` | `None` | 后验观测噪声（MAP 与 J_post 评分）；`None` 表示沿用 `weight_sigma_m`。 |
| `posterior_prior_strength` | `float` | `1.0` | 后验融合：弱锚点强度（默认温和）。 |
| `posterior_prior_sigma_v` | `float` | `2.0` | 后验融合：速度先验尺度。 |
| `posterior_prior_sigma_a` | `float` | `8.0` | 后验融合：加速度先验尺度。 |
| `posterior_anchor_weight` | `float` | `0.15` | 若 >0，将“后验锚点”作为合成候选注入走廊混合。 |
| `posterior_rls_lambda` | `float` | `1.0` | RLS 参数。 |
| `posterior_min_tau_s` | `float` | `1e-6` | 避免 tau 过小导致的病态（保持与历史实现一致）。 |
| `posterior_optimize_tb` | `bool` | `False` | 方案3：联合估计 tb（默认关闭，需要时显式开启）。 |
| `posterior_tb_search_window_s` | `float` | `0.05` | tb 搜索窗口（秒）。 |
| `posterior_tb_search_step_s` | `float` | `0.002` | tb 搜索步长（秒）。 |
| `posterior_tb_prior_sigma_s` | `float` | `0.03` | tb 的先验尺度（秒）。 |

### pixel（`PixelConfig`）

像素域闭环（重投影误差最小化）相关配置。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `pixel_enabled` | `bool` | `True` | 默认开启，但需要上层注入 `CameraRig` 且观测携带 2D 才会生效。 |
| `pixel_max_iters` | `int` | `2` | 最大迭代次数。 |
| `pixel_huber_delta_px` | `float` | `3.0` | Huber delta（像素）。 |
| `pixel_gate_tau_px` | `float` | `0.0` | gate 阈值（像素）。 |
| `pixel_min_cameras` | `int` | `1` | 最少相机数。 |
| `pixel_refine_top_k` | `int \| None` | `None` | 仅对 top-k 候选做像素 refine（`None` 表示不限制）。 |
| `pixel_lm_damping` | `float` | `1e-3` | LM damping 系数。 |
| `pixel_fd_rel_step` | `float` | `1e-3` | 有限差分相对步长。 |

### prefit（`PrefitConfig`）

prefit（第一段）相关配置。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `prefit_xz_window_points` | `int` | `12` | prefit 阶段水平面窗口点数。 |
| `prefit_robust_iters` | `int` | `1` | 鲁棒重加权迭代次数。 |
| `prefit_robust_delta_m` | `float` | `0.12` | 鲁棒 delta（米）。 |
| `prefit_min_inlier_points` | `int` | `5` | 最少 inlier 点数。 |
| `prefit_pixel_enabled` | `bool` | `True` | prefit 阶段像素一致性加权（需上层注入 CameraRig/2D 才生效）。 |
| `prefit_pixel_gate_tau_px` | `float` | `0.0` | gate 阈值（像素）。 |
| `prefit_pixel_huber_delta_px` | `float` | `5.0` | Huber delta（像素）。 |
| `prefit_pixel_min_cameras` | `int` | `1` | 最少相机数。 |

### bounce_detector（`BounceDetectorConfig`）

分段检测器（冻结/反弹判别）相关配置。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `bounce_detector_v_down_mps` | `float` | `0.6` | prefit 冻结：下落速度阈值（m/s）。 |
| `bounce_detector_v_up_mps` | `float` | `0.4` | prefit 冻结：上升速度阈值（m/s）。 |
| `bounce_detector_eps_y_m` | `float` | `0.04` | near-ground 的 y 容差（米）。 |
| `bounce_detector_down_debounce_s` | `float` | `0.03` | 下落判别去抖（秒）。 |
| `bounce_detector_up_debounce_s` | `float` | `0.03` | 上升判别去抖（秒）。 |
| `bounce_detector_local_min_window` | `int` | `7` | 局部极小窗口大小。 |
| `bounce_detector_min_points` | `int` | `6` | 最少点数。 |
| `bounce_detector_gap_freeze_enabled` | `bool` | `True` | 安全冻结（处理反弹附近不可见）。 |
| `bounce_detector_gap_mult` | `float` | `3.0` | gap 判别倍数。 |
| `bounce_detector_gap_tb_margin_s` | `float` | `0.033` | tb margin（秒，30fps 下约 1 帧）。 |
| `bounce_detector_gap_fit_points` | `int` | `12` | gap fit 使用点数。 |

### corridor（`CorridorConfig`）

走廊输出相关配置。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `corridor_dt` | `float` | `0.05` | 走廊时间步长。 |
| `corridor_horizon_s` | `float` | `1.2` | 走廊预测时域（秒）。 |
| `corridor_quantile_levels` | `Sequence[float]` | `(0.05, 0.95)` | 分位数包络水平（按 `docs/04_stage1_prior.md §4.1` 推荐）。 |
| `corridor_components_k` | `int` | `2` | 多峰走廊混合分量数（K=1~2）。 |

### online_prior（`OnlinePriorConfig`）

在线参数沉淀相关配置（见 `docs/09_data_driven_prior.md §5`）。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `online_prior_enabled` | `bool` | `False` | 是否启用在线 prior。 |
| `online_prior_path` | `str \| None` | `None` | 保存路径。 |
| `online_prior_ema_alpha` | `float` | `0.05` | EMA 系数。 |
| `online_prior_eps` | `float` | `1e-8` | 数值稳定项。 |
| `online_prior_autosave` | `bool` | `True` | 是否自动保存。 |

### low_snr（`LowSnrConfig`）

低 SNR（低信噪比）策略相关配置。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `low_snr_enabled` | `bool` | `True` | 默认开启，但只有当观测提供 `conf` 时才会产生实质影响。 |
| `low_snr_prefit_window_points` | `int` | `7` | prefit 段末尾用于 analyze 的窗口长度（点数）。 |
| `low_snr_conf_cmin` | `float` | `0.05` | conf 下限（避免 `1/sqrt(conf)` 发散）。 |
| `low_snr_sigma_x0_m` | `float` | `0.10` | conf=1 时 x 方向基础噪声 σ0（米）。 |
| `low_snr_sigma_y0_m` | `float` | `0.05` | conf=1 时 y 方向基础噪声 σ0（米）。 |
| `low_snr_sigma_z0_m` | `float` | `0.10` | conf=1 时 z 方向基础噪声 σ0（米）。 |
| `low_snr_delta_k_freeze_a` | `float` | `4.0` | 退化判别阈值（见 `docs/07_low_snr_policy.md`）。 |
| `low_snr_delta_k_strong_v` | `float` | `2.0` | 退化判别阈值（见 `docs/07_low_snr_policy.md`）。 |
| `low_snr_delta_k_ignore` | `float` | `1.0` | 退化判别阈值（见 `docs/07_low_snr_policy.md`）。 |
| `low_snr_min_points_for_v` | `int` | `3` | v 相关策略所需最少点数。 |
| `low_snr_disallow_ignore_y` | `bool` | `True` | 是否禁止 ignore y。 |
| `low_snr_strong_prior_v_scale` | `float` | `0.1` | STRONG_PRIOR_V：速度先验 σ_v 缩放（<1 更强）。 |
| `low_snr_prefit_strong_sigma_v_mps` | `float` | `0.5` | prefit 阶段速度强先验的绝对尺度（m/s）。 |

### legacy（`LegacyConfig`）

legacy 输出/兼容相关配置。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `net_height_1` | `float` | `0.4` | legacy 常量（已在包内固定默认值）。 |
| `net_height_2` | `float` | `1.1` | legacy 常量（已在包内固定默认值）。 |
| `legacy_receive_dt` | `float` | `0.02` | legacy 输出时间步长。 |
| `legacy_too_close_to_land_s` | `float` | `0.03` | legacy 相关阈值（秒）。 |
| `z_speed_range` | `tuple[float, float]` | `(1.0, 27.0)` | z 速度范围裁剪。 |

## 如何验证（建议）

- 运行单测以确认 YAML 能被正确解析：
  - `uv run python -m pytest -k config_yaml`
- 或直接运行全量测试（预期输出包含 `passed` 汇总行）：
  - `uv run python -m pytest`
