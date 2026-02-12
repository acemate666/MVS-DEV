# 第二阶段（posterior）：$N\le 5$ 少点强校正与候选融合

第二阶段的目标是：当反弹后仅出现少量观测点（典型 $N\le 5$）时，快速把 prior 的候选轨迹校正到任务可用精度，并避免早期锁错分支。

本阶段的工程策略是：

1. **对每个候选先做带先验的校正（MAP）**，得到该候选下的最优连续参数；
2. 再用后验代价对候选进行打分与权重更新（离散后验）；
3. 用更新后的权重更新走廊并输出名义结果。

---

## 1. 观测与相对时间

输入（除 prefit 锚点外）为反弹后观测点：

$$
\{(t_i,p_i)\}_{i=1}^N,\quad 1\le N\le 5
$$

统一使用相对时间：

$$
\tau_i=t_i-t_b
$$

若 $t_b$ 低置信或反弹附近缺帧明显，可启用 6.1 的“小窗联合估计 $t_b$”。

---

## 2. 校正参数化：默认 5 维（可退化）

推荐默认估计：

- $v^+\in\mathbb R^3$
- 水平等效常加速度 $a_{xz}=[a_x,a_z]^\top\in\mathbb R^2$

合并为：

$$
\theta=[v_x^+,v_y^+,v_z^+,a_x,a_z]^\top
$$

退化模式：当点数极少或噪声很大时，可仅估计 $v^+$（3 维），并冻结 $a_{xz}$，以降低过拟合风险。

---

## 3. 线性观测模型（短窗口）

在固定 $t_b,p_b$ 下，对每个点 $p_i$：

$$
 p_i = p_b + v^+\tau_i + \tfrac12\mathbf g\tau_i^2 + \tfrac12[a_x,0,a_z]^\top\tau_i^2
$$

移项得到线性形式：

$$
 y_i = H_i\theta
$$

其中

$$
 y_i = p_i - p_b - \tfrac12\mathbf g\tau_i^2
$$

而 $H_i\in\mathbb R^{3\times 5}$ 为

$$
H_i =
\begin{bmatrix}
\tau_i & 0 & 0 & \tfrac12\tau_i^2 & 0\\
0 & \tau_i & 0 & 0 & 0\\
0 & 0 & \tau_i & 0 & \tfrac12\tau_i^2
\end{bmatrix}
$$

把 $N$ 个点堆叠即可得到一个小型线性系统。

---

## 4. 带先验的 MAP 校正（防止 $N$ 小时过拟合）

对候选 $m$，prior 给出先验中心 $\theta_m^0$（至少包含 $v^{+(m)}$，通常令 $a_{xz}^0=0$）。

采用带正则的最小二乘（MAP）：

$$
\hat\theta_m = \arg\min_\theta\ \|W^{1/2}(H\theta-y)\|^2 + \|\Lambda^{1/2}(\theta-\theta_m^0)\|^2
$$

工程要点：

- $\Lambda$ 必须让系统稳定（SPD）。
- 对 $(a_x,a_z)$ 应更强正则，避免把噪声解释为巨大横向加速度。
- 若观测噪声按轴不同，可用对角 $W$（或每点 $W_i$）。

---

## 5. 推荐统一实现：信息形式递推（RLS / normal equation）

为避免同时维护“批量版”和“递推版”，建议统一使用信息形式维护法方程。

对候选 $m$ 初始化：

$$
A_m\leftarrow\Lambda,\qquad b_m\leftarrow\Lambda\theta_m^0
$$

每来一个观测点（或批量遍历）更新：

$$
A_m\leftarrow A_m + H_i^\top W_i H_i,
\qquad
b_m\leftarrow b_m + H_i^\top W_i y_i
$$

需要输出时解小型 SPD 线性系统：

$$
A_m\hat\theta_m=b_m
$$

实现建议：用 Cholesky 求解，避免显式求逆。

---

## 6. 候选融合：先校正后打分，降低早期锁错

### 6.1 后验代价（建议分解为可诊断项）

对每候选，校正后计算后验代价：

$$
J_m^{post}=\|W^{1/2}(H\hat\theta_m-y)\|^2 + \|\Lambda^{1/2}(\hat\theta_m-\theta_m^0)\|^2
$$

工程上建议同时记录：

- `data_term`（数据项）
- `prior_term`（先验/正则项）

以便判断“看似稳定但其实被正则钉死”的情况。

### 6.2 权重更新：log-sum-exp + 温度 $\beta_N$

设 stage1 给出的先验权重为 $w_m^0$，则更新：

$$
\log \tilde w_m = \log w_m^0 - \tfrac12\beta_N J_m^{post}
$$

归一化必须使用 log-sum-exp（避免下溢）：

$$
 w_m = \exp\Big(\log\tilde w_m - \mathrm{LSE}(\{\log\tilde w_j\})\Big)
$$

$\beta_N$ 是“温度/置信度”旋钮：在 $N$ 很小时取更保守值，避免权重过早塌缩到单候选。

### 6.3 输出策略

- 名义输出：常用两种策略
  - 取 $J_m^{post}$ 最小的候选（优先解释数据）
  - 取 $w_m$ 最大的候选（严格离散后验）
- 风险输出：建议用 $\{w_m\}$ 更新走廊（多峰更安全）。

---

## 7. 增强项 A：小窗联合估计 $t_b$（缺帧场景兜底）

当反弹点不可见导致 $t_b$ 不稳时，可在 $t_b^{(0)}$ 附近做一维小窗搜索：

$$
 t_b\in[t_b^{(0)}-\Delta,\ t_b^{(0)}+\Delta]
$$

对每个候选 $t_b$ 重算 $\tau_i$ 并跑 3–6 节流程，加入弱先验惩罚：

$$
J(t_b)=J_{post}(t_b)+\frac{(t_b-t_b^{(0)})^2}{\sigma_{t_b}^2}
$$

并输出独立字段 $\hat t_b$（不得改写 prefit 冻结的锚点）。

---

## 8. 增强项 B：像素域闭环精修（B-lite，Top-K）

当存在多相机 2D 观测与相机投影模型时，可在 3D 线性 MAP 输出基础上做 1–2 次像素域 GN/LM 精修，以提升精度上限与可诊断性。

工程约束：

- **只对 Top-K 候选做像素精修**（例如 K=1 或 2），避免算力线性放大。
- 任何数值异常必须立即回退到 3D 点域结果，并记录原因码。

接口契约与回退规则见 [`08_observation_2d.md`](08_observation_2d.md)。

---

## 9. 与其它专题的关系

- 观测噪声与低 SNR 退化策略：见 [`06_observation_noise_and_weights.md`](06_observation_noise_and_weights.md) 与 [`07_low_snr_policy.md`](07_low_snr_policy.md)。
- 数据驱动先验（候选权重 $w_m^0$）：见 [`09_data_driven_prior.md`](09_data_driven_prior.md)。
