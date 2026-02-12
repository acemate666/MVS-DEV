# curve_v3 文档导航

本目录文档面向开源读者，目标是解释 `curve_v3` 的**技术原理、关键假设、工程约束与可验证口径**。

- 文档刻意避免对话体叙述。
- 文档刻意减少代码细节；后续若需要代码级解析，建议另建专门文档。
- `99_curve.bak.md` 是历史备份（保留原样），不作为推荐入口。

## 推荐阅读顺序

1. [`01_public_api.md`](01_public_api.md)：对外接口（Public API）与集成口径。
2. [`02_curve.md`](02_curve.md)：总览（两阶段结构、关键概念、输出口径）。
3. [`03_prefit_and_segmentation.md`](03_prefit_and_segmentation.md)：反弹事件锚点 $(t_b,p_b,v^-)$ 的估计与分段契约（时间基准优先）。
4. [`04_stage1_prior.md`](04_stage1_prior.md)：第一阶段（prior）：候选生成与走廊（corridor）输出。
5. [`05_stage2_posterior.md`](05_stage2_posterior.md)：第二阶段（posterior）：$N\le 5$ 少点强校正、候选后验打分与融合。
6. [`06_observation_noise_and_weights.md`](06_observation_noise_and_weights.md)：观测噪声与权重口径（conf→$\sigma$→$W$）、慢轴/时间基准风险与对策。
7. [`07_low_snr_policy.md`](07_low_snr_policy.md)：低 SNR（低信噪比）退化策略规格（prefit / posterior 共用）。
8. [`08_observation_2d.md`](08_observation_2d.md)：多相机 2D 观测如何提升 3D 轨迹（以及像素域闭环的接口契约与回退规则）。
9. [`09_data_driven_prior.md`](09_data_driven_prior.md)：数据驱动先验（分桶/冷启动/在线沉淀）的可落地形态。
10. [`10_validation_and_troubleshooting.md`](10_validation_and_troubleshooting.md)：验收指标与线上排障清单。
11. [`11_config_knobs.md`](11_config_knobs.md)：关键旋钮与默认值含义（偏工程调参视角）。

## 符号与约定（全套文档通用）

- 反弹事件：$t_b$（反弹时刻）、$p_b$（反弹位置，球心）、$v^-$（反弹前速度）、$v^+$（反弹后初速度）。
- 相对时间：$\tau = t - t_b$。
- 触地球心高度：$y_c = y_{\text{ground}} + r + b_y$（用于触地求根与落点定义；避免硬编码 $y=0$）。
- 两阶段：
  - prior（第一阶段）：仅用反弹前信息输出**候选集合**与**走廊**。
  - posterior（第二阶段）：利用反弹后少量点（通常 $N\le 5$）做强校正，并更新权重/走廊。

## 备份说明

- [`99_curve.bak.md`](99_curve.bak.md) 保留为历史备份（内容较长、组织较旧）。
- 推荐入口是 [`02_curve.md`](02_curve.md) + 本导航页。
