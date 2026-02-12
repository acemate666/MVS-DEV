"""prefit/bounce_event 更新步骤。

说明：
    该模块从旧的单体流水线实现中拆出，专注于：
    - 反弹前 prefit 拟合
    - bounce_event（触地时刻/位置/入射速度）推断
    - prefit 阶段低 SNR 诊断（可选）

注意：
    - 这是在线主链路的一部分，但仍属于内部实现细节。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from curve_v3.configs import CurveV3Config
from curve_v3.low_snr import LowSnrPolicyParams, analyze_window
from curve_v3.prior import estimate_bounce_event_from_prefit
from curve_v3.types import BounceEvent, LowSnrAxisModes
from curve_v3.pipeline.types import PrefitUpdateResult

if TYPE_CHECKING:  # pragma: no cover
    from curve_v3.prefit_freeze import PrefitFreezeController


def low_snr_params_from_cfg(cfg: CurveV3Config) -> LowSnrPolicyParams:
    """从配置构造低 SNR 策略参数。"""

    return LowSnrPolicyParams(
        delta_k_freeze_a=float(cfg.low_snr.low_snr_delta_k_freeze_a),
        delta_k_strong_v=float(cfg.low_snr.low_snr_delta_k_strong_v),
        delta_k_ignore=float(cfg.low_snr.low_snr_delta_k_ignore),
        min_points_for_v=int(cfg.low_snr.low_snr_min_points_for_v),
    )


def update_prefit_and_bounce_event(
    *,
    cfg: CurveV3Config,
    prefit_freezer: "PrefitFreezeController",
    prev_pre_coeffs: dict[str, np.ndarray] | None,
    t: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    xw: np.ndarray,
    yw: np.ndarray,
    zw: np.ndarray,
    confs: Sequence[float | None],
    prev_bounce_event: BounceEvent | None,
    prev_low_snr_prefit: LowSnrAxisModes | None,
) -> PrefitUpdateResult:
    """更新反弹前拟合（prefit）与反弹事件（bounce_event）。

    说明：
        - prefit 只由“反弹前点”驱动。
        - 一旦检测到进入反弹后段，则冻结 prefit/bounce_event，避免 post 点污染 t_land。
    """

    if prefit_freezer.state.is_frozen and prev_pre_coeffs is not None and prev_bounce_event is not None:
        return PrefitUpdateResult(
            pre_coeffs=prev_pre_coeffs,
            bounce_event=prev_bounce_event,
            low_snr_prefit=prev_low_snr_prefit,
        )

    t_fit = t
    xs_fit = xs
    ys_fit = ys
    zs_fit = zs
    xw_fit = xw
    yw_fit = yw
    zw_fit = zw

    # region 检测 prefit/post 切分点
    prefit_freezer.update_cut_index(
        ts=t,
        ys=ys,
        y_contact=float(cfg.bounce_contact_y()),
    )
    # endregion

    # region 根据 cut_index 划分 prefit 段
    k = prefit_freezer.prefit_slice_end(n_points=int(t.size))
    if k is not None:
        t_fit = t[:k]
        xs_fit = xs[:k]
        ys_fit = ys[:k]
        zs_fit = zs[:k]
        xw_fit = xw[:k]
        yw_fit = yw[:k]
        zw_fit = zw[:k]
    # endregion

    prefit_low_snr = None
    if bool(cfg.low_snr.low_snr_enabled):
        tail_n = int(cfg.low_snr.low_snr_prefit_window_points)
        tail_n = int(max(tail_n, 3))
        tail_start = max(int(t_fit.size) - tail_n, 0)

        confs_fit = list(confs[: int(t_fit.size)])
        prefit_low_snr = analyze_window(
            xs=xs_fit[tail_start:],
            ys=ys_fit[tail_start:],
            zs=zs_fit[tail_start:],
            confs=confs_fit[tail_start:],
            sigma_x0=float(cfg.low_snr.low_snr_sigma_x0_m),
            sigma_y0=float(cfg.low_snr.low_snr_sigma_y0_m),
            sigma_z0=float(cfg.low_snr.low_snr_sigma_z0_m),
            c_min=float(cfg.low_snr.low_snr_conf_cmin),
            params=low_snr_params_from_cfg(cfg),
            disallow_ignore_y=bool(cfg.low_snr.low_snr_disallow_ignore_y),
        )

    pre = estimate_bounce_event_from_prefit(
        t_rel=t_fit,
        xs=xs_fit,
        ys=ys_fit,
        zs=zs_fit,
        xw=xw_fit,
        yw=yw_fit,
        zw=zw_fit,
        # 低 SNR：在与 x/z 拟合同一窗口上做判别与退化动作。
        low_snr=prefit_low_snr,
        v_prior=(np.asarray(prev_bounce_event.v_minus, dtype=float) if prev_bounce_event is not None else None),
        cfg=cfg,
    )
    if pre is None:
        # 与旧实现一致：prefit 失败时，不主动覆写上一次的 low_snr_prefit 诊断信息。
        return PrefitUpdateResult(pre_coeffs=None, bounce_event=None, low_snr_prefit=prev_low_snr_prefit)

    pre_coeffs, bounce_event = pre

    # 记录 prefit 阶段的低 SNR mode（用于上游诊断）。
    if prefit_low_snr is not None:
        low_snr_prefit = LowSnrAxisModes(
            mode_x=prefit_low_snr.x.mode,
            mode_y=prefit_low_snr.y.mode,
            mode_z=prefit_low_snr.z.mode,
        )
    else:
        low_snr_prefit = None

    t_land = float(bounce_event.t_rel)
    if t_land <= 0.0:
        # 与旧实现一致：bounce_event 无效时清理 prefit/bounce_event，但保留 low_snr 标签。
        return PrefitUpdateResult(pre_coeffs=None, bounce_event=None, low_snr_prefit=low_snr_prefit)

    # 若分段检测器给出了 cut_index，则冻结 prefit/bounce_event。
    if prefit_freezer.state.cut_index is not None:
        prefit_freezer.freeze()

    return PrefitUpdateResult(pre_coeffs=pre_coeffs, bounce_event=bounce_event, low_snr_prefit=low_snr_prefit)
