"""curve_v3 的核心数据结构定义。

把数据类型从 `curve_v3.core` 中拆出，主要目的：

1) 降低 `core.py` 的文件体量，避免“一个文件塞下所有东西”。
2) 降低模块间耦合：其它模块只依赖稳定的数据结构，不必依赖 core 的实现细节。
3) 便于复用与单测：类型层尽量保持轻量（只包含字段与语义说明）。

坐标约定（与 `legacy/curve2.py`、`docs/curve.md` 一致）：
- x：向右为正
- z：向前为正
- y：向上为正
- 重力沿 -y 方向
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# 说明：
#   - curve_v3 大量字段是 numpy 数组；这里用 NDArray 让类型契约更显式。
#   - 我们不在类型层强制 dtype/shape（运行时不做检查），仅约定“通常为浮点数组”。
#   - shape 约束仍通过字段注释说明（例如 shape=(3,)）。
FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class Obs2D:
    """单相机 2D 观测（像素域），用于像素域闭环拟合。

    属性:
        uv: 像素坐标 [u, v]，shape=(2,)，单位 px。
        cov_uv: 像素协方差，shape=(2,2)，单位 px^2。
            - 工程上要求为对称正定；若上游估计不稳定，可在使用处做轻微正则。
        sigma_px: 便于人读的标量噪声尺度（px）。
        cov_source: 协方差来源说明（例如 detector/heuristic）。
        reproj_error_px: 可选诊断字段：若上游提供重投影误差统计，可透传。
    """

    uv: FloatArray
    cov_uv: FloatArray
    sigma_px: float
    cov_source: str = ""
    reproj_error_px: float | None = None


@dataclass(frozen=True)
class BallObservation:
    """单帧球观测。

    属性:
        x: 横向位置（m，右正）。
        y: 高度（m，上正）。
        z: 纵向位置（m，前正）。
        t: 绝对时间戳（s）。
        conf: 可选的点级置信度（无量纲，越大表示越可信）。
            - 该字段用于低 SNR 场景下构造每点权重（见 `curve_v3.low_snr`）。
            - 为 None 时表示未知；上层会回退为常量权重。
        obs_2d_by_camera: 可选的“每相机 2D 观测集合”。
            - 该字段用于像素域闭环（重投影误差最小化）。
            - key 为相机标识（例如 "cam0"），value 为 Obs2D。
            - 当该字段为空时，curve_v3 会回退为纯 3D 点域拟合（保持历史行为）。
    """

    x: float
    y: float
    z: float
    t: float
    conf: float | None = None
    obs_2d_by_camera: dict[str, Obs2D] | None = None


@dataclass(frozen=True)
class BounceEvent:
    """反弹事件（可为预测或检测得到）。"""

    t_rel: float
    x: float
    z: float
    v_minus: FloatArray  # shape (3,)

    # 可选增强：用于对齐 docs/curve.md §2.4 / 附录B 的输出约定。
    # - y：触地时刻的球心高度（通常等于 cfg.bounce_contact_y()）。
    # - sigma_*：轻量不确定度尺度（不要求完整协方差），便于后续构造权重矩阵 R。
    y: float | None = None
    sigma_t_rel: float | None = None
    sigma_v_minus: FloatArray | None = None  # shape (3,)
    prefit_rms_m: float | None = None


@dataclass(frozen=True)
class Candidate:
    """反弹后初始状态的先验候选（prior candidate）。

    Attributes:
        e: 恢复系数（normal restitution）。用于把 v^- 映射到 v^+ 的法向分量。
        kt: 切向映射系数（tangential mapping）。用于描述切向速度在反弹后的保留/衰减。
        weight: 该候选的离散权重（非负）。通常会在融合后归一化到 sum(w)=1。
        v_plus: 反弹后初速度（m/s），shape=(3,)。
        kt_angle_rad: 可选的切平面内偏转角（弧度）。用于表达切向映射的旋转分量。
        ax: x 方向等效常加速度（m/s^2），用于吸收未建模效应（如噪声/弱气动）。
        az: z 方向等效常加速度（m/s^2）。
    """

    e: float
    kt: float
    weight: float
    v_plus: FloatArray  # shape (3,)
    kt_angle_rad: float = 0.0
    ax: float = 0.0
    az: float = 0.0


@dataclass(frozen=True)
class CorridorByTime:
    """按时间输出的走廊统计（仅对 x/z 投影）。"""

    t_rel: FloatArray  # shape (K,)
    mu_xz: FloatArray  # shape (K,2)
    cov_xz: FloatArray  # shape (K,2,2)

    # 可选增强：分位数包络（多峰更安全）。
    # 约定：quantiles_xz[k, i, :] 对应 t_rel[k] 处 level=quantile_levels[i] 的 (x,z)。
    quantile_levels: FloatArray | None = None  # shape (Q,)
    quantiles_xz: FloatArray | None = None  # shape (K,Q,2)


@dataclass(frozen=True)
class CorridorOnPlane:
    """轨迹穿越平面 y==target_y 时的走廊统计。"""

    target_y: float
    mu_xz: FloatArray  # shape (2,)
    cov_xz: FloatArray  # shape (2,2)
    t_rel_mu: float
    t_rel_var: float
    valid_ratio: float
    crossing_prob: float
    is_valid: bool

    # 可选增强：分位数包络。
    # - quantiles_xz[i, :] 对应 level=quantile_levels[i] 的 (x,z)。
    # - quantiles_t_rel[i] 对应 level=quantile_levels[i] 的穿越时刻（相对 time_base）。
    quantile_levels: FloatArray | None = None  # shape (Q,)
    quantiles_xz: FloatArray | None = None  # shape (Q,2)
    quantiles_t_rel: FloatArray | None = None  # shape (Q,)

    # 可选增强：多分量走廊（多峰更可解释）。
    components: tuple["CorridorComponent", ...] | None = None


@dataclass(frozen=True)
class CorridorComponent:
    """走廊的一个混合分量（针对按平面穿越输出）。

    属性:
        weight: 分量权重（与整体候选权重同域；当候选权重和为 1 时，等价于该分量的 crossing 概率）。
        mu_xz: 分量在 (x,z) 的均值。
        cov_xz: 分量在 (x,z) 的 2x2 协方差。
        t_rel_mu: 分量穿越时刻均值（相对 time_base）。
        t_rel_var: 分量穿越时刻方差（相对 time_base）。
        num_candidates: 参与该分量统计的候选数（穿越该平面的子集内计数）。
    """

    weight: float
    mu_xz: FloatArray
    cov_xz: FloatArray
    t_rel_mu: float
    t_rel_var: float
    num_candidates: int


@dataclass(frozen=True)
class PosteriorState:
    """反弹后阶段的后验（posterior）校正状态。"""

    t_b_rel: float
    x_b: float
    z_b: float
    vx: float
    vy: float
    vz: float
    ax: float = 0.0
    az: float = 0.0


@dataclass(frozen=True)
class FusionInfo:
    """候选 -> 后验融合流程的诊断信息。"""

    nominal_candidate_id: int | None
    posterior_anchor_used: bool


@dataclass(frozen=True)
class PrefitFreezeInfo:
    """第一段 prefit 冻结（PRE->POST 分段）诊断信息。

    说明：
        该结构用于对齐 `docs/curve.md` 的接口契约（§2.3.1 / §2.4.4）：
        - 一旦进入反弹后段，就应冻结 prefit 的事件锚点，避免 post 点污染 t_b。
        - 触发时刻与原因码必须可回放复现（便于线上排障）。

    属性:
        is_frozen: 是否已冻结。
        cut_index: pre 段末端索引（包含该点）。None 表示未触发。
        freeze_t_rel: 触发冻结时刻（相对 time_base 的时间，秒）。None 表示未触发。
        freeze_reason: 触发原因码（例如 vy_flip_and_near_ground）。None 表示未触发。
    """

    is_frozen: bool
    cut_index: int | None
    freeze_t_rel: float | None
    freeze_reason: str | None


@dataclass(frozen=True)
class LowSnrAxisModes:
    """低 SNR 模式标签（用于上游诊断/记录）。

    说明：
        - 该结构只包含标签，不包含阈值/权重等细节，避免对外暴露过多内部实现。
        - 标签集合见 `curve_v3.low_snr`：FULL/FREEZE_A/STRONG_PRIOR_V/IGNORE_AXIS。
    """

    mode_x: str
    mode_y: str
    mode_z: str


@dataclass(frozen=True)
class LowSnrInfo:
    """prefit 与 posterior 两阶段的低 SNR 诊断信息。"""

    prefit: LowSnrAxisModes | None
    posterior: LowSnrAxisModes | None
