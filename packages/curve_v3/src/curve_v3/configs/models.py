"""curve_v3 的配置定义。

该包实现两阶段的反弹后轨迹预测：
    - prior：多候选 + 走廊（corridor）。
    - posterior：反弹后少点（N<=5）快速校正。

说明：
    本包不再提供旧版 legacy/curve2 向下兼容接口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence


def _try_get_bot_motion_constant(name: str, default: float) -> float:
    """返回默认常量。

    说明：
        抽离后的 `curve_v3` 作为独立 Python 包使用时，必须与仓库内其它业务模块
        （例如 `hit`）完全解耦。因此这里不再尝试从外部模块读取常量，统一使用
        本包内的默认值。
    """

    _ = name
    return float(default)


@dataclass(frozen=True)
class PhysicsConfig:
    """物理/几何口径相关配置。"""

    gravity: float = 9.8

    # 地面法向：默认水平地面（y-up）。
    ground_normal: tuple[float, float, float] = (0.0, 1.0, 0.0)

    # 球半径：用于定义触地时刻的球心高度（bounce_contact_y）。
    ball_radius_m: float = 0.033

    # 反弹事件：观测 y 为球心高度时，触地/反弹时刻应满足 y(t)=r（球半径）。
    # 为避免默认值与 ball_radius_m 强耦合，这里用 None 表示“跟随 ball_radius_m”。
    bounce_contact_y_m: float | None = None

    def bounce_contact_y(self) -> float:
        """返回反弹时刻的球心高度（m）。

        Returns:
            若配置 bounce_contact_y_m 为 None，则返回 ball_radius_m；否则返回 bounce_contact_y_m。
        """

        if self.bounce_contact_y_m is None:
            return float(self.ball_radius_m)
        return float(self.bounce_contact_y_m)


@dataclass(frozen=True)
class PipelineConfig:
    """选择 curve_v3 的流水线模式。

    说明：
        - full：默认两阶段（prefit + prior candidates + posterior fusion + corridor）。
        - simple：最小基线（prefit 3 曲线 + bounce 检测/冻结 + postfit 3 曲线）。
            - 该模式的目标是提供一个“容易理解、容易回归”的基线实现。
            - 默认值保持为 full，避免改变历史行为。
    """

    mode: Literal["full", "simple"] = "full"


@dataclass(frozen=True)
class SimplePipelineConfig:
    """simple mode 的参数配置。

    设计约束：
        - prefit：x/z 线性；y 固定重力项的二次（a=-0.5*g）。
        - postfit：x/z 线性（ax=az=0）；y 仍由重力项确定。
        - 分段：prefit 只能使用 PRE 段点；postfit 只能使用 POST 段点。
    """

    # 若 post 点不足以做 postfit，则用反弹系数（e/kt/角度）给出一个简单预测候选。
    e: float = 0.70
    kt: float = 0.65
    kt_angle_rad: float = 0.0

    # postfit 最少点数：2 点即可做线性速度估计，但要求 tau 不能全为 0。
    postfit_min_points: int = 2

    # simple mode 下的 post 点截断（避免离线回放时喂入过长序列导致拟合被远期噪声带跑）。
    # 说明：该值仅影响 simple mode；full mode 仍由 posterior.max_post_points 控制。
    postfit_max_points: int = 8


@dataclass(frozen=True)
class PriorConfig:
    """prior（候选生成/反弹参数离散化）相关配置。"""

    e_bins: Sequence[float] = (0.55, 0.70, 0.85)
    kt_bins: Sequence[float] = (0.45, 0.65, 0.85)

    # 参数裁剪：用于兜底防止配置/数据异常导致的数值发散。
    # 说明：e 允许略大于 1 以容纳等效误差，但不建议过大。
    e_range: tuple[float, float] = (0.05, 1.25)

    # 切向映射增强：标量 kt + 可选偏转角（等价于一个 2x2 线性映射：缩放 + 旋转）。
    # 默认保持“不启用偏转”，以避免在未调参场景下候选数膨胀并改变历史行为。
    kt_angle_bins_rad: Sequence[float] = (0.0,)
    kt_range: tuple[float, float] = (-1.2, 1.2)


@dataclass(frozen=True)
class CandidateConfig:
    """候选权重/退火相关配置。"""

    # 候选权重退火：默认开启，前 2 个点用较小 beta 以降低锁错分支风险。
    candidate_beta_warmup_points: int = 2
    candidate_beta_min: float = 0.3

    def likelihood_beta(self, num_points: int) -> float:
        """计算候选似然权重的退火系数 beta。

        设计目标：在反弹后点数较少时，避免 exp(-0.5*J) 的权重过于尖锐导致
        过早锁定错误分支；随着点数增加，beta 逐渐回到 1.0。

        Args:
            num_points: 用于打分/重加权的反弹后点数。

        Returns:
            beta，范围 (0, 1]。
        """

        warmup = int(self.candidate_beta_warmup_points)
        if warmup <= 0:
            return 1.0

        beta_min = float(self.candidate_beta_min)
        beta_min = min(max(beta_min, 1e-3), 1.0)

        n = int(max(num_points, 0))
        # 线性预热：n=0 -> beta_min, n>=warmup -> 1.0
        alpha = min(float(n) / float(warmup), 1.0)
        return float(beta_min + (1.0 - beta_min) * alpha)


@dataclass(frozen=True)
class PosteriorConfig:
    """posterior（反弹后校正/MAP/RLS）相关配置。"""

    # 后验拟合模式：仅保留递推（rls）。
    fit_mode: Literal["rls"] = "rls"
    fit_params: Literal["v_only", "v+axz"] = "v+axz"

    # 工程建议：第二阶段只使用少量 post 点（典型 N<=5）。
    # 默认不强制截断（设为较大值），避免离线回放/评估时被意外裁掉信息。
    max_post_points: int = 999

    # 候选打分的残差尺度（m），用于似然权重。
    weight_sigma_m: float = 0.15

    # 后验观测噪声（用于 MAP 求解与 J_post 评分）；None 表示沿用 weight_sigma_m。
    posterior_obs_sigma_m: float | None = None

    # 后验融合相关参数：
    # 这些默认值刻意设置得比较“温和”，用于提供一个弱锚点；
    # 当反弹后点足够有信息时，不会阻止模型做出较强校正。
    posterior_prior_strength: float = 1.0
    posterior_prior_sigma_v: float = 2.0
    posterior_prior_sigma_a: float = 8.0

    # 若 >0，则将“后验锚点”作为合成候选注入走廊混合。
    posterior_anchor_weight: float = 0.15

    # RLS（信息形式递推）参数。
    posterior_rls_lambda: float = 1.0

    # 后验细节：用于避免 tau 过小导致的病态（保持默认值与历史实现一致）。
    posterior_min_tau_s: float = 1e-6

    # 方案3：联合估计 tb（默认关闭；需要时由上层/实验显式开启）。
    posterior_optimize_tb: bool = False
    posterior_tb_search_window_s: float = 0.05
    posterior_tb_search_step_s: float = 0.002
    posterior_tb_prior_sigma_s: float = 0.03


@dataclass(frozen=True)
class PixelConfig:
    """像素域闭环（重投影误差最小化）相关配置。"""

    # 像素域闭环：默认开启，但需要上层注入 CameraRig 且观测携带 2D 才会生效。
    pixel_enabled: bool = True
    pixel_max_iters: int = 2
    pixel_huber_delta_px: float = 3.0
    pixel_gate_tau_px: float = 0.0
    pixel_min_cameras: int = 1
    pixel_refine_top_k: int | None = None
    pixel_lm_damping: float = 1e-3
    pixel_fd_rel_step: float = 1e-3


@dataclass(frozen=True)
class PrefitConfig:
    """prefit（第一段）相关配置。"""

    # 第一段（prefit）增强：水平面采用等效常加速度（二次）模型 + 1 次鲁棒重加权。
    prefit_xz_window_points: int = 12
    prefit_robust_iters: int = 1
    prefit_robust_delta_m: float = 0.12
    prefit_min_inlier_points: int = 5

    # prefit 阶段像素一致性加权：默认开启（仅当上层注入 CameraRig/2D 时才实际生效）。
    prefit_pixel_enabled: bool = True
    prefit_pixel_gate_tau_px: float = 0.0
    prefit_pixel_huber_delta_px: float = 5.0
    prefit_pixel_min_cameras: int = 1


@dataclass(frozen=True)
class BounceDetectorConfig:
    """分段检测器（冻结/反弹判别）相关配置。"""

    # 分段检测器（prefit 冻结）：按 docs/curve.md §2.4.4 的默认建议值。
    # 说明：实现中使用“时间去抖”以适配不同 FPS。
    bounce_detector_v_down_mps: float = 0.6
    bounce_detector_v_up_mps: float = 0.4
    bounce_detector_eps_y_m: float = 0.04
    bounce_detector_down_debounce_s: float = 0.03
    bounce_detector_up_debounce_s: float = 0.03
    bounce_detector_local_min_window: int = 7
    bounce_detector_min_points: int = 6

    # 安全冻结：用于处理反弹附近不可见导致 near_ground 永不成立的场景。
    bounce_detector_gap_freeze_enabled: bool = True
    bounce_detector_gap_mult: float = 3.0
    bounce_detector_gap_tb_margin_s: float = 0.033  # 30 fps 下约 1 帧，33ms
    bounce_detector_gap_fit_points: int = 12


@dataclass(frozen=True)
class CorridorConfig:
    """corridor（走廊输出）相关配置。"""

    corridor_dt: float = 0.05
    corridor_horizon_s: float = 1.2

    # 走廊表达增强：按 docs/curve.md §4.3 推荐，默认提供分位数包络。
    corridor_quantile_levels: Sequence[float] = (0.05, 0.95)

    # 可选增强：多峰走廊的混合分量表示（K=1~2）。
    corridor_components_k: int = 2


@dataclass(frozen=True)
class OnlinePriorConfig:
    """在线参数沉淀（docs/curve.md §7）相关配置。"""

    online_prior_enabled: bool = False
    online_prior_path: str | None = None
    online_prior_ema_alpha: float = 0.05
    online_prior_eps: float = 1e-8
    online_prior_autosave: bool = True


@dataclass(frozen=True)
class LowSnrConfig:
    """低 SNR（低信噪比）策略相关配置。"""

    # 默认开启，但只有当上层为 BallObservation 提供 conf（置信度）时才会产生实质影响。
    low_snr_enabled: bool = True

    # prefit 阶段 low SNR 判别窗口长度（点数）：仅取 prefit 段末尾 N 点做 analyze_window。
    low_snr_prefit_window_points: int = 7

    # conf 的下限（避免 1/sqrt(conf) 发散）。
    low_snr_conf_cmin: float = 0.05

    # 三轴基础噪声尺度 σ0（米），相当于 conf=1 时的观测标准差。
    low_snr_sigma_x0_m: float = 0.10
    low_snr_sigma_y0_m: float = 0.05
    low_snr_sigma_z0_m: float = 0.10

    # 退化判别阈值（见 docs/low_snr_policy.md）。
    low_snr_delta_k_freeze_a: float = 4.0
    low_snr_delta_k_strong_v: float = 2.0
    low_snr_delta_k_ignore: float = 1.0
    low_snr_min_points_for_v: int = 3
    low_snr_disallow_ignore_y: bool = True

    # STRONG_PRIOR_V 的强度：对速度先验 σ_v 做缩放（<1 表示更强）。
    low_snr_strong_prior_v_scale: float = 0.1

    # prefit 阶段速度强先验的绝对尺度（m/s）。该值越小，越不允许被噪声带跑。
    low_snr_prefit_strong_sigma_v_mps: float = 0.5


@dataclass(frozen=True)
class LegacyConfig:
    """legacy 输出/兼容相关配置。"""

    net_height_1: float = _try_get_bot_motion_constant("NET_HEIGHT_1", 0.4)
    net_height_2: float = _try_get_bot_motion_constant("NET_HEIGHT_2", 1.1)
    legacy_receive_dt: float = 0.02
    legacy_too_close_to_land_s: float = 0.03
    z_speed_range: tuple[float, float] = (1.0, 27.0)


@dataclass(frozen=True)
class CurveV3Config:
    """CurvePredictorV3 的配置聚合。

    说明：
        该配置按领域拆分为多个子配置（physics/prior/posterior/...），以减少跨模块的
        隐性公共 API 面。该类仅作为聚合入口，算法模块应尽量只使用自己负责的子配置。
    """

    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    simple: SimplePipelineConfig = field(default_factory=SimplePipelineConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    candidate: CandidateConfig = field(default_factory=CandidateConfig)
    posterior: PosteriorConfig = field(default_factory=PosteriorConfig)
    pixel: PixelConfig = field(default_factory=PixelConfig)
    prefit: PrefitConfig = field(default_factory=PrefitConfig)
    bounce_detector: BounceDetectorConfig = field(default_factory=BounceDetectorConfig)
    corridor: CorridorConfig = field(default_factory=CorridorConfig)
    online_prior: OnlinePriorConfig = field(default_factory=OnlinePriorConfig)
    low_snr: LowSnrConfig = field(default_factory=LowSnrConfig)
    legacy: LegacyConfig = field(default_factory=LegacyConfig)

    def bounce_contact_y(self) -> float:
        """返回反弹时刻的球心高度（m）。"""

        return float(self.physics.bounce_contact_y())

    def candidate_likelihood_beta(self, num_points: int) -> float:
        """计算候选似然权重的退火系数 beta。"""

        return float(self.candidate.likelihood_beta(num_points))
