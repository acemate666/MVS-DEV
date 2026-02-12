# curve_v3 对外接口（Public API）

本文面向“把 `curve_v3` 当作库集成”的使用者，约定哪些入口是稳定的、应该怎么调用，以及哪些属于内部实现不建议直接依赖。

> 口径说明：本文以当前仓库代码为准（`packages/curve_v3/src/curve_v3`）。若你在业务侧只需要“能用起来”，优先照着本文的“最小集成”与“稳定导出”来写。

---

## 安装与导入

推荐用法：把本仓库作为 mono-repo 使用时，以 editable 方式安装该包。

```bash
uv add --editable ./packages/curve_v3
```

推荐导入路径（稳定）：

- 主要入口类与常用类型：
  - `from curve_v3 import CurvePredictorV3, CurveV3Config`
  - `from curve_v3 import BallObservation, BounceEvent, Candidate`
- 稳定函数契约层（跨包依赖边界）：
  - `from curve_v3 import fit_posterior_map_for_candidate`
- YAML 配置加载（IO 边界在单独模块）：
  - `from curve_v3.config_yaml import load_curve_v3_config_yaml`

不建议的导入方式（内部实现，未来可能重构）：

- `curve_v3.posterior.*`、`curve_v3.prior.*`、`curve_v3.corridor.*` 等子包路径
  - 需要这些能力时，优先看包顶层导出或 `curve_v3.api` 中是否已有稳定转发。

---

## 坐标系、单位与时间口径

这些约定贯穿整个包，是下游集成最容易踩坑的部分：

- 坐标系：$x$ 向右、$y$ 向上、$z$ 向前，重力沿 $-y$。
- 单位：位置米（m），速度 m/s，加速度 m/s²，时间秒（s）。
- `BallObservation.t`：**绝对时间戳**（abs time）。
- `CurvePredictorV3.time_base_abs`：首帧观测的 abs time（用于把后续一切转到相对时间）。
- `BounceEvent.t_rel`：相对 `time_base_abs` 的时间（`t_abs = time_base_abs + t_rel`）。

---

## 稳定导出一览（curve_v3 包顶层）

`curve_v3/__init__.py` 明确导出了以下符号（稳定使用面）：

- 配置（常用 dataclass）：
  - `CurveV3Config`
  - `PhysicsConfig`、`PriorConfig`、`PosteriorConfig`
  - `PrefitConfig`、`PixelConfig`、`LowSnrConfig`
  - `OnlinePriorConfig`
  - `PipelineConfig`、`SimplePipelineConfig`
  - `CorridorConfig`
  - `BounceDetectorConfig`、`CandidateConfig`
- 核心类：`CurvePredictorV3`
- 稳定函数：`fit_posterior_map_for_candidate`
- 常用类型（定义在 `curve_v3/types.py`）：
  - `BallObservation`
  - `BounceEvent`
  - `Candidate`
  - `PosteriorState`
  - `CorridorOnPlane`、`CorridorByTime`
  - `FusionInfo`
  - `PrefitFreezeInfo`

如果你要对外提供“第三方能直接用的接口”，建议只暴露这些符号，不要暴露内部子包。

---

## 核心入口类：CurvePredictorV3

### 构造

```python
from curve_v3 import CurvePredictorV3, CurveV3Config

cfg = CurveV3Config()
pred = CurvePredictorV3(config=cfg)
```

可选参数：

- `prior_model`：用于注入“数据驱动先验权重”（实现位于 `curve_v3.prior`，属于进阶使用）。
- `camera_rig`：若你在观测中携带像素信息并希望启用像素域 refinement（否则不会生效）。

### 在线喂数据

```python
from curve_v3 import BallObservation

pred.add_observation(BallObservation(x=x, y=y, z=z, t=t_abs, conf=conf))
```

- `conf` 是可选项；如果提供，会参与低 SNR 权重/退化判别。

### 常用 getter / 查询方法（稳定）

以下方法在本仓库的上游集成中**确实被依赖**（例如 `tennis3d_core` 的 curve_stage 输出、以及 `interception` 的逐候选 MAP 校正），因此将其视为稳定契约的一部分：

- 时间基准
  - `pred.time_base_abs -> float | None`

- 反弹前/反弹事件
  - `pred.get_bounce_event() -> BounceEvent | None`
  - `pred.get_pre_fit_coeffs() -> dict[str, np.ndarray] | None`

- prior / posterior 状态
  - `pred.get_prior_candidates() -> list[Candidate]`
  - `pred.get_best_candidate() -> Candidate | None`
  - `pred.get_posterior_state() -> PosteriorState | None`
  - `pred.get_post_points() -> list[BallObservation]`

- 走廊（用于下游规划/决策）
  - `pred.get_corridor_by_time() -> CorridorByTime | None`
  - `pred.corridor_on_plane_y(target_y: float) -> CorridorOnPlane | None`
  - `pred.corridor_on_planes_y(target_ys: Sequence[float]) -> list[CorridorOnPlane]`

- 工程输出常用的派生量（`tennis3d_core` 上游依赖）
  - `pred.predicted_land_time_rel() -> float | None`
  - `pred.predicted_second_land_time_rel() -> float | None`
  - `pred.predicted_land_point() -> list[float] | None`（`[x, y, z, t_rel]`）
  - `pred.predicted_land_speed() -> list[float] | None`（`[vx, vy, vz, speed_xz]`）
  - `pred.point_at_time_rel(t_rel: float) -> list[float] | None`（`[x, y, z]`）

- 诊断信息（稳定字段集合，用于排障/回放复现）
  - `pred.get_prefit_freeze_info() -> PrefitFreezeInfo`
  - `pred.get_low_snr_info() -> LowSnrInfo`
  - `pred.get_fusion_info() -> FusionInfo`

### 最小集成示例

下面示例展示“喂入观测 → 等 bounce 出现 → 读候选/走廊”的最短路径：

```python
from curve_v3 import CurvePredictorV3, CurveV3Config, BallObservation

pred = CurvePredictorV3(config=CurveV3Config())

for t_abs, x, y, z, conf in stream:
    pred.add_observation(BallObservation(x=x, y=y, z=z, t=t_abs, conf=conf))

bounce = pred.get_bounce_event()
if bounce is None:
    # 还没有可靠的反弹事件
    ...

cands = pred.get_prior_candidates()      # 反弹后候选（可能为空）
best = pred.get_best_candidate()         # 融合后 best（可能为 None）

corr = pred.corridor_on_plane_y(0.6)     # y=0.6m 的下降穿越统计
if corr is not None:
    # corr 内包含 crossing_prob / 分位数等信息（用于下游决策）
    ...
```

---

## 稳定函数：fit_posterior_map_for_candidate

该函数定义在 `curve_v3/api.py`，目的是为下游（例如 `interception`）提供稳定的 posterior MAP 拟合边界，避免下游直接依赖 `curve_v3.posterior.*` 的内部路径。

推荐调用方式：

```python
from curve_v3 import fit_posterior_map_for_candidate

out = fit_posterior_map_for_candidate(
    bounce=bounce,
    post_points=post_points,
    candidate=cand,
    time_base_abs=time_base_abs,
    cfg=curve_cfg,
)
# out: (PosteriorState, J_post) | None
```

参数要点：

- `post_points`：反弹后观测点（工程建议典型 $N\le 5$）。
- `time_base_abs`：与 `BounceEvent.t_rel` 的基准一致。
- 返回的 `J_post` 可用于候选重赋权/比较。

---

## 配置：CurveV3Config

### 代码内构造

`CurveV3Config` 是聚合配置（`physics/prior/posterior/...`），建议集成侧先从默认值开始，只改你需要的少量旋钮。

常见旋钮（例）：

- 候选网格：`cfg.prior.e_bins`、`cfg.prior.kt_bins`、`cfg.prior.kt_angle_bins_rad`
- 走廊：`cfg.corridor.corridor_dt`、`cfg.corridor.corridor_horizon_s`
- 后验：`cfg.posterior.fit_params`、`cfg.posterior.max_post_points`
- 低 SNR：`cfg.low_snr.low_snr_enabled` 及各阈值

### 从 YAML 加载（推荐用于工程接入）

该包提供了明确的 IO 边界：`curve_v3/config_yaml.py`。

```python
from curve_v3.config_yaml import load_curve_v3_config_yaml

cfg = load_curve_v3_config_yaml("./packages/curve_v3/configs/curve_v3.yaml")
```

约定：

- YAML 顶层必须是 mapping，对应 `CurveV3Config` 的字段名（`physics/prior/...`）。
- 未知字段会报错（避免拼写错误静默无效）。

---

## 兼容性声明

- 本仓库已明确：`curve_v3` **不再提供**旧版 `curve2.Curve` 的兼容适配层。
- 下游集成请以本文列出的“稳定导出”为准。

---

## 你大概率会需要的下一篇文档

- 算法与数据流背景：`02_curve.md`
- 低 SNR 策略：`07_low_snr_policy.md`
- 观测噪声与权重口径：`06_observation_noise_and_weights.md`
