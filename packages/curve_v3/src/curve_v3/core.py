"""curve_v3 的核心编排实现。

设计目标（来自 `curve.md`）：
    - 第一阶段（prior）：基于反弹前信息生成多候选，并输出不确定性走廊（corridor）。
    - 第二阶段（posterior）：当出现反弹后少量观测点（N<=5）时，快速校正并融合输出。

坐标约定（与 `legacy/curve2.py` 保持一致）：
    - x：向右为正。
    - z：向前为正。
    - y：向上为正。
    - 重力方向沿 -y。

实现原则：
    - 依赖尽量少（NumPy + 标准库）。
    - API 尽量稳定；本包不再提供旧版 legacy/curve2 兼容层。
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Sequence

import numpy as np

from curve_v3.pipeline import (
    PrefitUpdateResult,
    PostUpdateResult,
    extract_post_points_after_land_time,
    update_post_models_and_corridor,
    update_prefit_and_bounce_event,
)
from curve_v3._simple_pipeline import (
    update_post_models_and_corridor_simple,
    update_prefit_and_bounce_event_simple,
)
from curve_v3.observation_buffer import ObservationBuffer
from curve_v3.prefit_freeze import PrefitFreezeController
from curve_v3.corridor import corridor_on_planes_y as compute_corridor_on_planes_y
from curve_v3.configs import CurveV3Config
from curve_v3.utils import polyval
from curve_v3.posterior.anchor import inject_posterior_anchor, prior_nominal_state
from curve_v3.posterior.fusion import candidate_costs as compute_candidate_costs
from curve_v3.prior import PriorModel
from curve_v3.prior import maybe_init_online_prior
from curve_v3.utils import default_logger
from curve_v3.types import (
    BallObservation,
    BounceEvent,
    Candidate,
    CorridorByTime,
    CorridorOnPlane,
    FusionInfo,
    LowSnrAxisModes,
    LowSnrInfo,
    PrefitFreezeInfo,
    PosteriorState,
)

if TYPE_CHECKING:  # pragma: no cover
    from curve_v3.adapters.camera_rig import CameraRig


class CurvePredictorV3:
    """两阶段反弹预测器（prior + posterior）。

    说明：
        - 这是 v3 的新 API；不再提供旧版 legacy/curve2 兼容适配器。
        - 内部会维护反弹前拟合（用于推断反弹时刻/位置/入射速度），以及反弹后的候选/后验。
    """

    # 时间递增权重：越新的点权重越大（指数增长）。
    # 说明：这是历史版本的工程启发式，配合低 SNR 的 conf 权重能更稳地抑制早期噪声。
    _FIT_X_WEIGHT = 1.01
    _FIT_Y_WEIGHT = 1.01
    _FIT_Z_WEIGHT = 1.01

    def __init__(
        self,
        config: CurveV3Config | None = None,
        logger: logging.Logger | None = None,
        prior_model: PriorModel | None = None,
        camera_rig: "CameraRig | None" = None,
    ) -> None:
        self._cfg = config or CurveV3Config()
        self._logger = logger or default_logger()
        self._prior_model = prior_model
        self._camera_rig = camera_rig

        # 在线权重池属于“跨回合”的经验状态：不应被 reset() 清空。
        self._online_prior = maybe_init_online_prior(cfg=self._cfg, logger=self._logger)
        self.reset()

    def reset(self) -> None:
        # 观测缓存 + 点级权重构造：从 core.py 拆出，降低编排器的职责密度。
        self._buf = ObservationBuffer(
            fit_x_weight=float(self._FIT_X_WEIGHT),
            fit_y_weight=float(self._FIT_Y_WEIGHT),
            fit_z_weight=float(self._FIT_Z_WEIGHT),
        )
        self._buf.reset()

        self._pre_coeffs: dict[str, np.ndarray] | None = None
        self._bounce_event: BounceEvent | None = None

        self._candidates: list[Candidate] = []
        self._best_candidate: Candidate | None = None
        self._nominal_candidate_id: int | None = None
        self._posterior_anchor_used: bool = False
        self._corridor_by_time: CorridorByTime | None = None

        self._post_points: list[BallObservation] = []
        self._posterior_state: PosteriorState | None = None

        # prefit 分段/冻结状态机：一旦确认进入反弹后段，就冻结 prefit/bounce_event，
        # 避免后续 post 点参与 prefit 造成 t_land 漂移。
        self._prefit_freezer = PrefitFreezeController(cfg=self._cfg, min_points=5)

        # 低 SNR 诊断信息：分别记录 prefit 与 posterior 最近一次的 mode 标签。
        self._low_snr_prefit: LowSnrAxisModes | None = None
        self._low_snr_posterior: LowSnrAxisModes | None = None

    @property
    def time_base_abs(self) -> float | None:
        return self._buf.time_base_abs

    @property
    def config(self) -> CurveV3Config:
        """返回预测器使用的配置（只读）。

        说明：
            - 该属性用于下游（例如 interception）读取物理口径/候选网格等参数。
            - CurveV3Config 为 frozen dataclass，因此直接返回引用是安全的。
        """

        return self._cfg

    def add_observation(self, obs: BallObservation) -> None:
        """添加一帧观测并更新内部模型。

        Args:
            obs: 带绝对时间戳的观测。
        """

        self._buf.append(obs, cfg=self._cfg, camera_rig=self._camera_rig)

        self._update_models()

    def _update_models(self) -> None:
        """用累计观测更新 prefit / prior / posterior / corridor。

        设计说明：
            这是在线入口 add_observation() 的主更新点。为了保持“高内聚、低耦合”，
            这里仅做流程编排，具体子步骤拆分到若干私有方法中。
        """

        if self._buf.size < 5:
            return

        t, xs, ys, zs, xw, yw, zw = self._buf.as_arrays()

        if str(self._cfg.pipeline.mode) == "simple":
            prefit_res = update_prefit_and_bounce_event_simple(
                cfg=self._cfg,
                prefit_freezer=self._prefit_freezer,
                prev_pre_coeffs=self._pre_coeffs,
                t=t,
                xs=xs,
                ys=ys,
                zs=zs,
                confs=self._buf.confs,
                prev_bounce_event=self._bounce_event,
            )
        else:
            prefit_res = update_prefit_and_bounce_event(
                cfg=self._cfg,
                prefit_freezer=self._prefit_freezer,
                prev_pre_coeffs=self._pre_coeffs,
                t=t,
                xs=xs,
                ys=ys,
                zs=zs,
                xw=xw,
                yw=yw,
                zw=zw,
                confs=self._buf.confs,
                prev_bounce_event=self._bounce_event,
                prev_low_snr_prefit=self._low_snr_prefit,
            )
        self._pre_coeffs = prefit_res.pre_coeffs
        self._bounce_event = prefit_res.bounce_event
        self._low_snr_prefit = prefit_res.low_snr_prefit

        if self._pre_coeffs is None or self._bounce_event is None:
            return

        t_land = float(self._bounce_event.t_rel)
        if t_land <= 0.0:
            # 说明：bounce_event 无效时清理 prefit/bounce_event，但保留低 SNR 标签。
            self._pre_coeffs = None
            self._bounce_event = None
            return

        self._post_points = extract_post_points_after_land_time(
            observations=self._buf.observations,
            time_base_abs=self._buf.time_base_abs,
            t_land=t_land,
        )

        if str(self._cfg.pipeline.mode) == "simple":
            post_res = update_post_models_and_corridor_simple(
                cfg=self._cfg,
                logger=self._logger,
                bounce_event=self._bounce_event,
                post_points=self._post_points,
                time_base_abs=self._buf.time_base_abs,
            )
        else:
            post_res = update_post_models_and_corridor(
                cfg=self._cfg,
                logger=self._logger,
                prior_model=self._prior_model,
                online_prior=self._online_prior,
                camera_rig=self._camera_rig,
                bounce_event=self._bounce_event,
                post_points=self._post_points,
                time_base_abs=self._buf.time_base_abs,
            )
        self._candidates = post_res.candidates
        self._best_candidate = post_res.best_candidate
        self._nominal_candidate_id = post_res.nominal_candidate_id
        self._posterior_state = post_res.posterior_state
        self._corridor_by_time = post_res.corridor_by_time
        self._posterior_anchor_used = bool(post_res.posterior_anchor_used)
        self._low_snr_posterior = post_res.low_snr_posterior

    def _candidate_costs(
        self,
        bounce: BounceEvent,
        candidates: Sequence[Candidate],
        post_points: Sequence[BallObservation],
    ) -> np.ndarray:
        """诊断用途：计算每个候选的归一化 SSE（不做每候选后验校正）。"""

        return compute_candidate_costs(
            bounce=bounce,
            candidates=candidates,
            post_points=post_points,
            time_base_abs=self._buf.time_base_abs,
            cfg=self._cfg,
        )

    def get_bounce_event(self) -> BounceEvent | None:
        return self._bounce_event

    def get_pre_fit_coeffs(self) -> dict[str, np.ndarray] | None:
        """获取反弹前拟合的多项式系数（用于解耦 legacy 访问私有字段）。

        该接口用于替代对私有字段 `_pre_coeffs` 的直接访问，以降低模块间耦合。

        Returns:
            若反弹前拟合尚不可用，返回 None；否则返回一个 dict：
            - "x": 反弹前 x(t_rel) 多项式系数（2 次：等效常加速度模型）
            - "y": 2 次多项式系数（a 固定为 -0.5*g）
            - "z": 反弹前 z(t_rel) 多项式系数（2 次：等效常加速度模型）
            - "t_land": shape (1,) 的数组，表示预测落地相对时间
        """

        if self._pre_coeffs is None:
            return None

        # 返回 copy，避免外部误改内部状态。
        return {k: np.asarray(v, dtype=float).copy() for k, v in self._pre_coeffs.items()}

    def get_prior_candidates(self) -> list[Candidate]:
        return list(self._candidates)

    def get_best_candidate(self) -> Candidate | None:
        """获取融合后得分最佳的候选（已使用反弹后点重赋权）。"""

        return self._best_candidate

    def get_fusion_info(self) -> FusionInfo:
        """获取 prior/posterior 融合流程的诊断信息。"""

        return FusionInfo(
            nominal_candidate_id=self._nominal_candidate_id,
            posterior_anchor_used=bool(self._posterior_anchor_used),
        )

    def get_prefit_freeze_info(self) -> PrefitFreezeInfo:
        """获取 prefit 冻结（分段）诊断信息。

        说明：
            - 该接口用于对齐 `docs/curve.md` 的“接口契约”与回放复现需求。
            - 不返回内部 detector 的全部状态，仅返回对下游排障有用且稳定的字段。
        """

        st = self._prefit_freezer.state
        return PrefitFreezeInfo(
            is_frozen=bool(st.is_frozen),
            cut_index=st.cut_index,
            freeze_t_rel=st.freeze_t_rel,
            freeze_reason=st.freeze_reason,
        )

    def get_corridor_by_time(self) -> CorridorByTime | None:
        return self._corridor_by_time

    def corridor_on_plane_y(self, target_y: float) -> CorridorOnPlane | None:
        """计算水平平面 y==target_y 的穿越走廊。

        说明：
            - 本仓库坐标系中 y 为高度（y-up）。
            - 若候选轨迹未穿越该高度，会返回 None。

        Args:
            target_y: 目标平面高度（米）。

        Returns:
            走廊统计；不可用时返回 None。
        """

        r = self.corridor_on_planes_y([float(target_y)])[0]
        if not r.is_valid:
            return None
        return r

    def corridor_on_planes_y(self, target_ys: Sequence[float]) -> list[CorridorOnPlane]:
        """批量计算多个水平平面 y==const 的穿越走廊统计。"""

        ys = [float(y) for y in target_ys]
        if not ys:
            return []

        candidates = list(self._candidates)
        if (
            str(self._cfg.pipeline.mode) != "simple"
            and self._posterior_state is not None
            and self._cfg.posterior.posterior_anchor_weight > 0
        ):
            candidates = inject_posterior_anchor(
                candidates=self._candidates,
                best=self._best_candidate,
                posterior=self._posterior_state,
                cfg=self._cfg,
            )

        return compute_corridor_on_planes_y(
            bounce=self._bounce_event,
            candidates=candidates,
            cfg=self._cfg,
            target_ys=ys,
        )

    def corridor_on_plane_y_range(
        self,
        y_min: float,
        y_max: float,
        step: float,
    ) -> list[CorridorOnPlane]:
        """在 [y_min, y_max] 的固定网格上批量计算穿越走廊。

        说明：
            这是对 :meth:`corridor_on_planes_y` 的便捷封装。

        Args:
            y_min: 最小高度（米）。
            y_max: 最大高度（米）。
            step: 网格步长（米），必须 > 0。

        Returns:
            与 y 网格对齐的一组走廊统计。
        """

        y_min = float(y_min)
        y_max = float(y_max)
        step = float(step)
        if step <= 0:
            return []

        if y_max < y_min:
            y_min, y_max = y_max, y_min

        ys = list(np.arange(y_min, y_max + 1e-9, step, dtype=float))
        return self.corridor_on_planes_y(ys)

    def get_posterior_state(self) -> PosteriorState | None:
        return self._posterior_state

    def get_post_points(self) -> list[BallObservation]:
        """返回反弹后点序列（POST 段）。

        说明：
            - 返回 copy，避免外部误改内部状态。
            - 该接口主要用于与 interception 的逐候选 MAP 校正对接。
        """

        return list(self._post_points)

    def get_low_snr_info(self) -> LowSnrInfo:
        """获取低 SNR 的诊断信息（mode 标签）。"""

        return LowSnrInfo(prefit=self._low_snr_prefit, posterior=self._low_snr_posterior)

    def point_at_time_rel(self, t_rel: float) -> list[float] | None:
        """在给定相对时间 t_rel 处查询轨迹点 [x,y,z]。

        说明：
            - t_rel 的时间基准为 `time_base_abs`（第一次观测的时间戳）。
            - 反弹前：使用 prefit 的多项式。
            - 反弹后：优先使用 posterior；否则使用候选混合的加权均值状态。
        """

        if self._pre_coeffs is None or self._bounce_event is None:
            return None

        t_land = float(self._bounce_event.t_rel)
        if t_rel < t_land:
            return [
                polyval(self._pre_coeffs["x"], t_rel),
                polyval(self._pre_coeffs["y"], t_rel),
                polyval(self._pre_coeffs["z"], t_rel),
            ]

        # 反弹后：优先使用 posterior；否则回退为候选混合的加权均值状态。
        state = self._posterior_state or prior_nominal_state(
            bounce=self._bounce_event,
            candidates=self._candidates,
        )
        if state is None:
            return None

        tau = float(t_rel - state.t_b_rel)
        if tau < 0:
            tau = 0.0

        x = state.x_b + state.vx * tau + 0.5 * state.ax * tau * tau
        y0 = float(self._cfg.bounce_contact_y())
        y = y0 + state.vy * tau - 0.5 * float(self._cfg.physics.gravity) * tau * tau
        z = state.z_b + state.vz * tau + 0.5 * state.az * tau * tau
        return [float(x), float(y), float(z)]

    def predicted_land_time_rel(self) -> float | None:
        if self._bounce_event is None:
            return None
        return float(self._bounce_event.t_rel)

    def predicted_second_land_time_rel(self) -> float | None:
        """预测第二次触地的相对时间（相对 time_base_abs）。

        定义：
            - 第一次触地/反弹时刻为 t_b（即 BounceEvent.t_rel）。
            - 第二次触地为反弹后再次回到 y==bounce_contact_y 的时刻。

        说明：
            - 若 posterior 可用，优先使用 posterior_state（更贴近当前观测）。
            - 否则回退为 prior 候选加权的名义状态。
        """

        if self._bounce_event is None:
            return None

        g = float(self._cfg.physics.gravity)
        if not (g > 0.0):
            return None

        state = self._posterior_state or prior_nominal_state(
            bounce=self._bounce_event,
            candidates=self._candidates,
        )
        if state is None:
            return None

        vy = float(state.vy)
        if not (vy > 0.0):
            return None

        return float(state.t_b_rel + 2.0 * vy / g)

    def predicted_land_speed(self) -> list[float] | None:
        if self._bounce_event is None:
            return None
        v = np.asarray(self._bounce_event.v_minus, dtype=float)
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        return [vx, vy, vz, float(math.sqrt(vx * vx + vz * vz))]

    def predicted_land_point(self) -> list[float] | None:
        if self._bounce_event is None:
            return None
        y0 = float(self._cfg.bounce_contact_y())
        return [float(self._bounce_event.x), y0, float(self._bounce_event.z), float(self._bounce_event.t_rel)]
