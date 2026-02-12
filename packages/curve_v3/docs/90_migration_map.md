# 旧文档迁移覆盖对照（门禁用）

本文用于回答一个工程化问题：**旧版巨型文档 `99_curve.bak.md`（v1.1）里的关键公式/口径/契约，是否已经被新专题体系覆盖？**

约定：

- `99_curve.bak.md` 保留旧全文作为备份（不改写、不裁剪）。
- `02_curve.md` 已“瘦身”为总览与索引；细节迁移到专题文档。
- 本表只做“覆盖定位”（旧内容应去哪里找），不重复抄写公式。

## 一句话结论（当前状态）

- **旧信息未丢**：`99_curve.bak.md` 仍完整保留。
- **新体系已建立骨架**：prior/posterior/prefit/数据驱动先验/验收/配置等专题已存在。
- **门禁仍需持续维护**：`02_curve.md` 已瘦身落地为总览索引；其余三篇旧稿（2D/低SNR/噪声）已按开源契约风格重写。后续对专题文档的任何改动，都应回到本 checklist 逐项复核，避免口径回退或遗漏。

## 旧 `99_curve.bak.md` → 新专题文档：章节级对照

| 旧 `99_curve.bak.md` 章节 | 新位置（推荐入口） | 说明 |
|---|---|---|
| 0 文档信息；0.1 术语；0.2 符号与约定 | `00_index.md`（符号与阅读顺序） + `02_curve.md`（瘦身后的总览） | 将把“全局口径”集中到索引与总览，避免分散复写。 |
| 1 背景与核心需求 | `02_curve.md`（瘦身后的总览） | 总览保留“为何两阶段/为何不做 drag/Magnus”的工程取舍。 |
| 2 输入输出与坐标约定 | `02_curve.md`（总览） + `03_prefit_and_segmentation.md` + `04_stage1_prior.md` + `05_stage2_posterior.md` + `08_observation_2d.md` | 输入/输出的细节分散在各专题；总览只保留“接口契约摘要”。 |
| 2.2.1 / 2.2.1.1 2D 观测协议与注入契约 | `08_observation_2d.md` | 将收敛为：字段口径、单位、时间戳、相机 ID、回退规则。 |
| 2.4 第一段状态估计模块；附录B（prefit 参考规格） | `03_prefit_and_segmentation.md` | 含 $(t_b,p_b,v^-)$、$y_c$ 口径、分段冻结与 gap-freeze。 |
| 3 方案总览（闭环） | `00_index.md`（阅读顺序） + `02_curve.md`（总览） | 用“导航 + 总览图/清单”替换旧文的长篇闭环叙述。 |
| 4 第一阶段：名义预测 + 走廊 | `04_stage1_prior.md` | 候选生成、走廊表示、按平面输出、下降穿越规则。 |
| 5 第二阶段：少点强校正；5.2 信息形式递推；5.4 log-sum-exp + 温度 $\beta_N$ | `05_stage2_posterior.md` | 线性 MAP / 信息形式（RLS）/ $J^{post}$ / 权重更新与 $\beta_N$。 |
| 5.5 小窗联合估计 $t_b$ | `05_stage2_posterior.md`（增强项） + `03_prefit_and_segmentation.md`（冻结契约） | 明确：不回灌修改 prefit 锚点，只输出独立 $\hat t_b$。 |
| 5.6 像素域闭环精修（B-lite / Top-K） | `05_stage2_posterior.md`（增强项） + `08_observation_2d.md`（接口与门控/回退） | 数学目标在 2D 文档，流程/工程开关在 stage2。 |
| 6 数据驱动先验（分桶/冷启动/在线沉淀） | `09_data_driven_prior.md` | 以“可回滚的小统计表”为目标形态。 |
| 7 在线参数沉淀 | `09_data_driven_prior.md` | 当前与 6 合并：沉淀形式、写入条件、安全阀。 |
| 8 配置项 | `11_config_knobs.md` | 以“旋钮含义/默认值/风险”视角组织。 |
| 9 测试与验收；排障清单；落点指标 | `10_validation_and_troubleshooting.md` | 以“先查时间基准/分段/噪声口径”为排障顺序。 |
| 10 最小落地清单 | `00_index.md` + `10_validation_and_troubleshooting.md` | 入口与验收指标会集中给出。 |
| 11 风险与应对 | `10_validation_and_troubleshooting.md` + `06_observation_noise_and_weights.md` + `07_low_snr_policy.md` | 风险按“噪声/低SNR/接口口径/异常点”分流。 |

## 关键条款 checklist（逐项门禁）

本节把旧 `99_curve.bak.md`（v1.1）里的“容易丢/容易口径漂移”的内容拆成逐条门禁项。每条只做定位，不重复抄写推导。

使用方式：

- 迁移完成前：逐条核对“新文档是否存在明确口径”。
- 迁移完成后：将本 checklist 作为评审/回归的入口（避免改文档时把口径改丢）。

### 全局符号与口径

- [ ] 反弹事件符号：$t_b,p_b,v^-,v^+$ 与相对时间 $\tau=t-t_b$ 的定义一致（入口：`00_index.md`）。
- [ ] 触地球心高度 $y_c=y_{ground}+r+b_y$ 的一致口径（入口：`00_index.md` + `03_prefit_and_segmentation.md` §2）。

### Prefit 与分段冻结（时间基准门禁）

- [ ] “prefit 只消费反弹前点”与“进入 post 段后冻结锚点”契约（`03_prefit_and_segmentation.md` §1、§6）。
- [ ] visibility-gap freeze 兜底策略与目标（隔离污染，不承诺精确）（`03_prefit_and_segmentation.md` §6.3）。

### Stage1 prior（候选与走廊）

- [ ] 法向恢复：$v_n^+=-e\,v_n^-$（`04_stage1_prior.md` §1.1）。
- [ ] 切向等效映射：$u^+=k_t R(\phi)u^-$（`04_stage1_prior.md` §1.2）。
- [ ] 候选离散化与候选数：$M=|e|\cdot|k_t|\cdot|\phi|$（`04_stage1_prior.md` §2）。
- [ ] 第二段飞行主干仅重力解析：$p(\tau)=p_b+v^+\tau+\tfrac12 g\tau^2$（`04_stage1_prior.md` §3）。
- [ ] 走廊多峰表示优先：分位数/包络 vs 混合分量（`04_stage1_prior.md` §4）。
- [ ] 下降穿越（plane corridor）的选择规则：$\tau>0$ 且 $\dot y(\tau)<0$（`04_stage1_prior.md` §5.1；排障入口：`10_validation_and_troubleshooting.md` §2）。

### Stage2 posterior（few-shot 校正与候选融合）

- [ ] 线性观测模型与 $H_i$ 的 5 维参数化（`05_stage2_posterior.md` §2–§3）。
- [ ] MAP 目标：数据项 + 正则项（`05_stage2_posterior.md` §4）。
- [ ] 信息形式递推（RLS / normal equation）：$A\leftarrow A+H^TWH,\ b\leftarrow b+H^TWy$（`05_stage2_posterior.md` §5）。
- [ ] “先校正后打分”的后验代价 $J_m^{post}$ 与可诊断拆分（`05_stage2_posterior.md` §6.1）。
- [ ] 权重更新：log-sum-exp + 温度 $\beta_N$（`05_stage2_posterior.md` §6.2）。
- [ ] 小窗联合估计 $t_b$ 的“独立输出，不回灌修改 prefit 锚点”契约（`05_stage2_posterior.md` §7；`03_prefit_and_segmentation.md` §1）。

### 2D 观测与像素域闭环（接口门禁）

- [ ] 2D 观测的数据契约：`BallObservation.obs_2d_by_camera` / `Obs2D` 字段、单位、相机 ID 一致性（`08_observation_2d.md` §2–§3）。
- [ ] 像素域残差与白化口径：$e=L^{-1}(\pi(p)-z)$（`08_observation_2d.md` §4）。
- [ ] 门控与回退规则：缺失 `CameraRig`/缺失 2D 时必须回退到纯 3D（`08_observation_2d.md` §5）。

### 观测噪声与权重口径

- [ ] `conf→\sigma→W` 的统一口径与单位约束（`06_observation_noise_and_weights.md`）。

### 低 SNR 退化策略

- [ ] 判别指标：$\Delta u$ vs $\bar\sigma$；四档 mode（`07_low_snr_policy.md` §3–§4）。
- [ ] y 轴禁用 IGNORE 的约束与理由（`07_low_snr_policy.md` §4.1）。

### 数据驱动先验（分桶/可回滚）

- [ ] prior table 的交付物定义与 Dirichlet 计数口径（`09_data_driven_prior.md` §1–§3）。
- [ ] 离线冷启动与在线沉淀的安全阀（`09_data_driven_prior.md` §4–§5）。

### 验收与排障

- [ ] 分阶段验收口径（prior 覆盖率/宽度；posterior 收敛/锁错率）（`10_validation_and_troubleshooting.md` §1）。
- [ ] 落点/触地的统一定义（下降穿越 + $y=y_c$）（`10_validation_and_troubleshooting.md` §2）。
- [ ] 排障优先级：先查 $t_b$ 漂移，再查 prefit 残差，再查权重塌缩/像素域一致性（`10_validation_and_troubleshooting.md` §3）。

## 另外三篇旧稿的替代关系

| 旧文档 | 新位置 | 处理方式 |
|---|---|---|
| `2d observation.md` | `08_observation_2d.md` | 已重写为开源契约风格：字段/单位/时间戳/相机 ID/回退规则明确，减少重复与代码。 |
| `curve_low_snr_quick.md` | `07_low_snr_policy.md` | 已收敛为“策略规格 + 配置字段表 + 验证标准”，并与实现字段名对齐。 |
| `curve_观测噪声问题.md` | `06_observation_noise_and_weights.md` | 已重写为“噪声口径与权重构造”专题，并与 lowSNR/后验互链。 |

## 验收标准（迁移完成的判据）

- `02_curve.md`（瘦身版）长度显著缩短，且只承担：总览、符号、阅读路线、关键契约索引。
- `99_curve.bak.md` 保留旧全文不变。
- 读者不需要打开 `99_curve.bak.md` 也能从专题文档找到：
  - stage1 候选/走廊定义与“下降穿越”口径
  - stage2 线性 MAP / 信息形式递推 / log-sum-exp 权重更新 / $\beta_N$
  - $t_b$ 冻结契约 + 小窗联合估计（不回灌）
  - 2D 像素域闭环的接口口径与回退规则
  - conf→$\sigma$→$W$ 的噪声口径与低 SNR 退化策略
- 文档写清：前置条件、可验证标准（见各文末尾）。

## 前置条件、命令与验证标准

前置条件：

- 已存在专题体系入口：`00_index.md`。
- `99_curve.bak.md` 保持不变（用于追溯旧版 v1.1 全文）。

命令（可验证闭环）：

- 文档一致性（人工门禁）：逐条勾选本文件的 checklist，确保每条都能在对应新文档中找到明确口径。
- 工程可运行性（自动门禁）：在 `packages/curve_v3/` 下运行测试套件（建议至少跑一次全量）。
  - 依赖安装：`uv sync --extra dev`
  - 运行测试：`uv run python -m pytest`

预期输出 / 验证标准：

- checklist 全部勾选，且链接可跳转到对应文档/小节。
- 测试通过（用于证明文档提到的字段/默认值/开关没有“写错名”导致不可用）。
