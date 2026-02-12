"""curve_v3 预测器与在线沉淀（online prior）的衔接逻辑。

说明：
    - online prior 属于“增益项”：用于跨回合沉淀候选权重，提高先验质量。
    - 该模块只负责在预测器里初始化/更新/保存 online prior。
    - 任何初始化或写盘失败都不应影响主链路，因此这里采用“失败降级”。
"""

from __future__ import annotations

import logging
from typing import Sequence

from curve_v3.configs import CurveV3Config
from curve_v3.types import Candidate


def maybe_init_online_prior(*, cfg: CurveV3Config, logger: logging.Logger):
    """按配置加载/创建在线沉淀权重池。

    Args:
        cfg: 配置。
        logger: 用于记录告警。

    Returns:
        online prior 实例或 None（未启用或初始化失败）。
    """

    if not bool(cfg.online_prior.online_prior_enabled):
        return None

    try:
        from curve_v3.prior.online_prior import load_or_create_online_prior

        e_bins = [float(x) for x in cfg.prior.e_bins]
        kt_bins = [float(x) for x in cfg.prior.kt_bins]
        ang_bins = [float(a) for a in cfg.prior.kt_angle_bins_rad]
        if not ang_bins:
            ang_bins = [0.0]

        return load_or_create_online_prior(
            path=cfg.online_prior.online_prior_path,
            e_bins=e_bins,
            kt_bins=kt_bins,
            angle_bins_rad=ang_bins,
            ema_alpha=float(cfg.online_prior.online_prior_ema_alpha),
            eps=float(cfg.online_prior.online_prior_eps),
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("online prior disabled due to init error: %s", exc)
        return None


def maybe_update_online_prior(*, online_prior, cfg: CurveV3Config, candidates: Sequence[Candidate], logger: logging.Logger) -> None:
    """用融合后的候选权重更新 online prior，并可选保存到 JSON。"""

    if online_prior is None:
        return

    try:
        online_prior.update_from_candidates(candidates=candidates)

        if bool(cfg.online_prior.online_prior_autosave):
            path = cfg.online_prior.online_prior_path
            if path is not None:
                online_prior.save_json(path)
    except Exception as exc:  # pragma: no cover
        # 在线沉淀属于“增益项”：不应因为写盘失败影响主功能。
        logger.warning("online prior update/save failed: %s", exc)
