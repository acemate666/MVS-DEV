# curve_v3 总览（prior 走廊 + posterior 少点校正）

本文档是 `curve_v3` 文档体系的“入口总览”：只保留两阶段设计的关键工程取舍、输入/输出契约摘要与阅读路线；所有公式推导与细节口径已经拆分到专题文档。

历史工程技术报告 v1.1 已归档为备份：[`99_curve.bak.md`](99_curve.bak.md)（保留原样，不作为推荐入口）。

## 目标与非目标

目标：在反弹后第二段轨迹预测问题中，满足以下工程约束：

- **第一阶段（prior）**：只用反弹前信息，输出“名义预测 + 不确定性走廊”，避免灾难性偏差。
- **第二阶段（posterior）**：出现少量反弹后点（通常 $N\le 5$）时，快速校正并收敛，同时避免早锁错。
- 算力预算固定可控（候选数 $M$ 与解算维度固定），便于嵌入式/实时部署。

非目标：在缺少自旋/接触状态观测的前提下，本系统不承诺“仅凭反弹前这一段即可唯一确定反弹后真值轨迹”。因此第一阶段必须表达不确定性（候选/走廊），并把快速校正留给第二阶段。

关于 drag/Magnus：工程上自旋在线不可观测/不可稳定估计，drag 也容易与时间基准误差/噪声尺度混淆并引入数值敏感点。因此 `curve_v3` 的默认工程方案不把 drag/Magnus 作为在线动力学主干；空气效应导致的系统性偏差优先由：stage1 的候选/走廊覆盖 + stage2 的等效低维校正项（如 $a_{xz}$）在少点下吸收。

## 输入/输出契约摘要

更完整的接口口径与字段表见：

- [`03_prefit_and_segmentation.md`](03_prefit_and_segmentation.md)
- [`04_stage1_prior.md`](04_stage1_prior.md)
- [`05_stage2_posterior.md`](05_stage2_posterior.md)
- [`08_observation_2d.md`](08_observation_2d.md)

### 第一阶段输入（来自反弹前 prefit/分段）

- 反弹事件锚点：$(t_b, p_b, v^-)$（通常不可直接观测，需由反弹前窗口稳健估计）。
- （建议）噪声尺度/不确定度估计：用于走廊宽度与候选权重的初始刻画。
- （建议）场地/区域/球类型等元信息：用于数据驱动先验分桶（见 `09_data_driven_prior.md`）。

工程契约：prefit 只消费反弹前点；进入 post 段后冻结锚点，避免 post 点污染导致 $t_b$ 漂移。

### 第一阶段输出（prior）

- 候选集合（candidate set）：由反弹参数离散化产生（典型 $M\in\{9,27\}$，可配）。
- 走廊（corridor）：对未来位置/到达时间的不确定性集合表示。
  - 工程默认推荐：输出“按拦截平面”的分位数包络或等价风险边界（多峰时更安全）。
  - 可选增强：输出 top-K 混合分量（weight/mean/cov）。

### 第二阶段额外输入（post 观测点）

- 反弹后少量观测点 $\{(t_i, p_i)\}_{i=1}^N$，通常 $1\le N\le 5$。
- N 肯定是越多越好，但是我们希望本工程能够在N较少时快速收敛。

### 第二阶段输出（posterior）

- per-candidate 的校正状态与后验代价（用于诊断）。
- 候选权重更新后的融合输出（建议优先继续输出“按平面”的走廊）。

### Optional 增强：多相机 2D 观测（像素域闭环）

- 输入：每帧 3D 点可附带 `obs_2d_by_camera`（每相机像素观测与协方差）。
- 系统注入边界：`CameraRig`/投影模型必须由运行时注入。
- 回退规则：缺失 `CameraRig` 或缺失 2D 时，像素域闭环自动禁用，回退到纯 3D 点域 stage2。

详见：[`08_observation_2d.md`](08_observation_2d.md)。

## 两阶段算法概要（只保留工程要点）

### 0) Prefit / 分段（时间基准优先）

- 目标：稳健估计反弹事件锚点 $(t_b,p_b,v^-)$。
- 风险：$t_b$ 漂移会放大为“落点离谱”的灾难性偏差。
- 约束：任何利用 post 点的增强估计（例如小窗联合估计 $\hat t_b$）必须作为独立输出，不得回灌修改 prefit 冻结锚点。

详见：[`03_prefit_and_segmentation.md`](03_prefit_and_segmentation.md)。

### 1) Stage1 prior：候选生成与走廊

基于地面基（法向/切向）把反弹等效为少量参数：

- 法向恢复：$v_n^+=-e\,v_n^-$。
- 切向映射：$u^+=k_t R(\phi)u^-$（可选 $\phi$；默认可退化为不使用旋转）。

对 $(e,k_t,\phi)$ 离散化得到候选集合；第二段飞行主干默认仅重力解析：

$$
p(\tau)=p_b+v^+\tau+\tfrac12 g\tau^2,\qquad \tau=t-t_b.
$$

走廊优先表达多峰：工程默认推荐“按拦截平面”的下降穿越输出，而不是对整条曲线做密采样。

详见：[`04_stage1_prior.md`](04_stage1_prior.md)。

### 2) Stage2 posterior：少点强校正与候选融合

对每个候选，在少点 $N\le 5$ 下做带先验的线性 MAP（信息形式递推 / RLS 口径），得到校正量（例如 $\hat v^+$ 与可选的等效项 $\hat a_{xz}$）与后验代价 $J_m^{post}$。

权重更新采用 log-sum-exp 形式并配合温度 $\beta_N$，以减少早锁错：

$$
w_m \propto \exp\bigl(-\beta_N\,J_m^{post}\bigr),\qquad \sum_m w_m=1.
$$

详见：[`05_stage2_posterior.md`](05_stage2_posterior.md)。

## 风险地图（从“现象”到“该看哪篇”）

- 时间基准/分段污染：看 `03_prefit_and_segmentation.md` + `10_validation_and_troubleshooting.md`。
- 观测噪声口径（conf→$\sigma$→$W$）与慢轴风险：看 `06_observation_noise_and_weights.md`。
- 低 SNR 退化策略（prefit/posterior 共用）：看 `07_low_snr_policy.md`。
- 像素域闭环/2D 接口口径/回退门控：看 `08_observation_2d.md`。
- 配置含义与默认值：看 `11_config_knobs.md`。

## 阅读路线与迁移门禁

- 推荐入口：[`00_index.md`](00_index.md)（阅读顺序 + 全局符号）。
- 旧文覆盖门禁：[`90_migration_map.md`](90_migration_map.md)（逐项 checklist）。
- 旧全文备份：[`99_curve.bak.md`](99_curve.bak.md)。

## 前置条件、命令与验证标准

前置条件：

- 已能在本仓库中安装并运行 `curve_v3`（Python >= 3.11）。
- 已安装并使用 `uv` 管理 Python 环境与依赖（用于运行测试与复现）。
- 若要启用 2D/像素域闭环：需要可用的 `CameraRig` 投影注入与正确的时间戳/坐标口径（见 `08_observation_2d.md`）。

命令（验证闭环建议）：

- 在 `packages/curve_v3/` 下运行测试套件（建议至少跑一次全量）：
  - 依赖安装：`uv sync --extra dev`
  - 运行测试：`uv run python -m pytest`

预期输出 / 验证标准：

- 测试通过（证明文档提到的字段名/默认值/关键行为与实现一致）。
- 对应专题文档末尾的“验收口径”可被落地执行（例如：走廊覆盖率/宽度、posterior 收敛曲线、低 SNR mode 触发与回退）。
