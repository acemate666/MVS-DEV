"""posterior 锚点候选与名义状态工具（内部模块）。"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from curve_v3.configs import CurveV3Config
from curve_v3.types import BounceEvent, Candidate, PosteriorState


def inject_posterior_anchor(
    *,
    candidates: Sequence[Candidate],
    posterior: PosteriorState,
    best: Candidate | None,
    cfg: CurveV3Config,
) -> list[Candidate]:
    """把 posterior 结果作为“锚点候选”注入混合，用于更新走廊。"""

    alpha = float(cfg.posterior.posterior_anchor_weight)
    alpha = min(max(alpha, 0.0), 1.0)
    if alpha <= 0.0:
        return list(candidates)

    scaled: list[Candidate] = []
    for c in candidates:
        scaled.append(
            Candidate(
                e=c.e,
                kt=c.kt,
                weight=float(c.weight) * (1.0 - alpha),
                v_plus=np.asarray(c.v_plus, dtype=float),
                kt_angle_rad=float(getattr(c, "kt_angle_rad", 0.0)),
                ax=float(c.ax),
                az=float(c.az),
            )
        )

    e = float(best.e) if best is not None else 0.0
    kt = float(best.kt) if best is not None else 0.0
    kt_ang = float(getattr(best, "kt_angle_rad", 0.0)) if best is not None else 0.0

    anchor = Candidate(
        e=e,
        kt=kt,
        weight=float(alpha),
        v_plus=np.array([posterior.vx, posterior.vy, posterior.vz], dtype=float),
        kt_angle_rad=float(kt_ang),
        ax=float(posterior.ax),
        az=float(posterior.az),
    )

    merged = scaled + [anchor]
    s = float(np.sum([float(c.weight) for c in merged]))
    if s <= 0.0:
        w = 1.0 / float(len(merged))
        return [
            Candidate(
                e=c.e,
                kt=c.kt,
                weight=float(w),
                v_plus=np.asarray(c.v_plus, dtype=float),
                kt_angle_rad=float(getattr(c, "kt_angle_rad", 0.0)),
                ax=float(c.ax),
                az=float(c.az),
            )
            for c in merged
        ]

    return [
        Candidate(
            e=c.e,
            kt=c.kt,
            weight=float(c.weight) / s,
            v_plus=np.asarray(c.v_plus, dtype=float),
            kt_angle_rad=float(getattr(c, "kt_angle_rad", 0.0)),
            ax=float(c.ax),
            az=float(c.az),
        )
        for c in merged
    ]


def prior_nominal_state(*, bounce: BounceEvent, candidates: Sequence[Candidate]) -> PosteriorState | None:
    """用候选混合的加权均值生成反弹后名义状态。"""

    if not candidates:
        return None

    weights = np.array([float(c.weight) for c in candidates], dtype=float)
    weights = weights / max(float(np.sum(weights)), 1e-12)

    v = np.sum(
        np.stack([np.asarray(c.v_plus, dtype=float) for c in candidates], axis=0) * weights[:, None],
        axis=0,
    )
    ax = float(np.sum(weights * np.asarray([float(c.ax) for c in candidates], dtype=float)))
    az = float(np.sum(weights * np.asarray([float(c.az) for c in candidates], dtype=float)))

    return PosteriorState(
        t_b_rel=float(bounce.t_rel),
        x_b=float(bounce.x),
        z_b=float(bounce.z),
        vx=float(v[0]),
        vy=float(v[1]),
        vz=float(v[2]),
        ax=ax,
        az=az,
    )


__all__ = [
    "inject_posterior_anchor",
    "prior_nominal_state",
]
