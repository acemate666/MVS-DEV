# interception 对外接口（Public API）

本文面向“把 `interception` 当作库集成”的使用者，说明该包的稳定入口、参数口径、返回值结构，以及与 `curve_v3` 的推荐集成方式。

> 口径说明：本文以当前仓库代码为准（`packages/interception/src/interception`）。

---

## 安装与依赖

`interception` 依赖 `curve_v3`。

在 mono-repo 下推荐用 editable 安装：

```bash
uv add --editable ./packages/curve_v3
uv add --editable ./packages/interception
```

推荐导入路径（稳定）：

```python
from interception import (
    InterceptionConfig,
    HitTarget,
    HitTargetDiagnostics,
    HitTargetResult,
    HitTargetStabilizer,
    select_hit_target_prefit_only,
    select_hit_target_with_post,
)
```

YAML 配置加载（IO 边界在单独模块）：

```python
from interception.config_yaml import load_interception_config_yaml
```

---

## 输入/输出口径（强约定）

### 坐标系与单位

与 `curve_v3` 保持一致：

- 坐标系：$x$ 向右、$y$ 向上、$z$ 向前
- 单位：位置 m，时间 s

### 目标点定义

`interception` 输出的目标点定义为：

- 球心在高度平面 $y=y^*$ 的**反弹后下降穿越**位置与到达时刻。

### 时间基准

- `time_base_abs`：与 `curve_v3.CurvePredictorV3.time_base_abs` 一致
- `BounceEvent.t_rel`：相对 `time_base_abs` 的时间
- 输出 `HitTarget.t_abs` 与 `HitTarget.t_rel` 同时给出

---

## 稳定导出一览（interception 包顶层）

`interception/__init__.py` 明确导出了以下符号（稳定使用面）：

- 配置：`InterceptionConfig`
- 选择器入口：
  - `select_hit_target_prefit_only`（N=0）
  - `select_hit_target_with_post`（1<=N<=5，工程建议 few-shot）
- 稳定器：`HitTargetStabilizer`
- 类型：`HitTarget`、`HitTargetDiagnostics`、`HitTargetResult`

不建议直接依赖的内部模块（未来可能重构）：

- `interception.math_utils`、`interception.selector` 内部私有函数

---

## 两个选择器入口

两个入口都返回 `HitTargetResult`：

- `valid: bool`：是否输出有效目标
- `reason: str | None`：无效原因（如 `no_candidates` / `crossing_prob_too_low` 等）
- `target: HitTarget | None`：有效时给出目标点
- `diag: HitTargetDiagnostics`：诊断信息（用于回放/调参/可视化）

### 1) 仅基于 prefit/prior（N=0）

当你还没有任何反弹后观测点（或不想使用 post 点），使用：

```python
from interception import select_hit_target_prefit_only

out = select_hit_target_prefit_only(
    bounce=bounce,
    candidates=candidates,
    time_base_abs=time_base_abs,
    t_now_abs=t_now_abs,
    cfg=icfg,
    curve_cfg=curve_cfg,
)
```

要点：

- `candidates` 为 `curve_v3` 的 prior 候选（例如来自 `CurvePredictorV3.get_prior_candidates()`）。
- N=0 时会刻意采用更稳健的退化模型（水平轴只用速度，不带先验加速度项），避免缺少约束时漂移。

### 2) 基于 prefit + 少量 post 点（1<=N<=5）

当你已经积累了少量反弹后观测点，使用：

```python
from interception import select_hit_target_with_post

out = select_hit_target_with_post(
    bounce=bounce,
    candidates=candidates,
    post_points=post_points,
    time_base_abs=time_base_abs,
    t_now_abs=t_now_abs,
    cfg=icfg,
    curve_cfg=curve_cfg,
)
```

要点：

- `post_points` 为反弹后观测点（`curve_v3.types.BallObservation`）。
- 内部会对每个候选调用 `curve_v3.fit_posterior_map_for_candidate(...)` 做 MAP 校正并重赋权，然后再做“命中概率最大化”选点。
- 工程上建议只传 few-shot 的点（典型 $N\le 5$），否则会把远期噪声引入决策。

---

## 与 curve_v3 的推荐集成方式

典型数据流（调用侧维护观测缓存）：

1) 使用 `CurvePredictorV3` 在线喂入观测
2) 一旦 `pred.get_bounce_event()` 可用：
   - 取 `bounce = pred.get_bounce_event()`
   - 取 `candidates = pred.get_prior_candidates()`
   - 取 `time_base_abs = pred.time_base_abs`
3) 构造 `post_points`：由调用侧根据时间将“反弹后点”筛出来，并裁剪到最近 $N\le 5$ 个
4) 调用 `select_hit_target_with_post(...)` 或在无 post 点时调用 `select_hit_target_prefit_only(...)`

示例（只演示关键口径，观测缓存逻辑由你实现）：

```python
from curve_v3 import CurvePredictorV3, CurveV3Config, BallObservation
from interception import InterceptionConfig, select_hit_target_with_post

curve_cfg = CurveV3Config()
pred = CurvePredictorV3(config=curve_cfg)
icfg = InterceptionConfig(y_min=0.6, y_max=0.9, num_heights=5)

observations: list[BallObservation] = []

for obs in stream:
    observations.append(obs)
    pred.add_observation(obs)

    bounce = pred.get_bounce_event()
    if bounce is None or pred.time_base_abs is None:
        continue

    # 由调用侧筛选反弹后点（建议再裁剪为最近 5 个）
    t_land_abs = pred.time_base_abs + bounce.t_rel
    post_points = [p for p in observations if p.t >= t_land_abs]
    post_points = post_points[-5:]

    out = select_hit_target_with_post(
        bounce=bounce,
        candidates=pred.get_prior_candidates(),
        post_points=post_points,
        time_base_abs=pred.time_base_abs,
        t_now_abs=obs.t,
        cfg=icfg,
      curve_cfg=curve_cfg,
    )
```

  注意：真实集成时请在构造 `CurvePredictorV3` 时把 `curve_cfg` 保存到你自己的上下文中，并在调用 `interception` 时显式传入，避免依赖 `CurvePredictorV3` 的私有成员。

---

## 跨帧稳定：HitTargetStabilizer

在反弹后早期（尤其 N=1..2）目标可能跳动。若你希望输出更稳定，使用 `HitTargetStabilizer`：

```python
from interception import HitTargetStabilizer

stab = HitTargetStabilizer()

raw = select_hit_target_with_post(...)
stable = stab.update(raw, cfg=icfg)
```

迟滞规则（任一满足则允许更新，否则保持上一帧）：

1) `score` 明显提升（`cfg.hysteresis_score_gain`）
2) `w_max` 明显上升（分布更收敛，`cfg.hysteresis_w_max_gain`）
3) `width_xz` 明显变窄（不确定性显著下降，`cfg.hysteresis_width_shrink_ratio`）

当 `current.valid=False` 时稳定器会直接返回当前帧（不会输出过期目标）。

---

## 配置：InterceptionConfig

`InterceptionConfig` 是一个单体 dataclass，必填字段为：

- `y_min: float`、`y_max: float`

常用旋钮：

- `num_heights`：高度网格采样数（默认 5）
- `r_hit_m`：命中半径（多峰压单点的关键超参）
- `min_crossing_prob` / `min_valid_candidates`：高度可用性门限
- `map_switch_*`：收敛后切换到 MAP 输出（可选增强）
- `hysteresis_*`：跨帧迟滞阈值（与 `HitTargetStabilizer` 配套）

---

## 从 YAML 加载配置

`interception/config_yaml.py` 提供了明确的加载入口：

```python
from interception.config_yaml import load_interception_config_yaml

icfg = load_interception_config_yaml("./packages/interception/configs/interception.yaml")
```

约定：

- YAML 顶层必须是 mapping，字段名与 `InterceptionConfig` 一致
- 未知字段会报错
- 缺少必填项（如 y_min/y_max）会报错

---

## 验证标准（最小自检）

如果你要快速验证集成是否跑通：

- `out.valid == True` 时：
  - `out.target` 非空，且 `t_abs` 单调随帧推进（通常越来越接近当前时刻的未来）
  - `out.diag.crossing_prob >= cfg.min_crossing_prob`
- `out.valid == False` 时：
  - `out.reason` 能解释为什么不可用（例如没有候选、穿越概率太低等）

更完整的回归契约见：

- `packages/interception/tests/test_interception_target.py`
- `packages/interception/tests/test_interception_synthetic_noisy.py`
