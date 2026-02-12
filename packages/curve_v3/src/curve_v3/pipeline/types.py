"""流水线步骤之间的共享返回类型。

说明：
    旧的单体流水线实现时代，这些 dataclass 与实现放在同一文件里，导致导入
    面过宽、职责不清。这里把“数据结构”单独抽出，便于：

    - core/simple 两条流水线共享返回类型
    - 未来按步骤拆文件时，避免互相循环依赖
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from curve_v3.types import (
    BounceEvent,
    Candidate,
    CorridorByTime,
    LowSnrAxisModes,
    PosteriorState,
)


@dataclass(frozen=True)
class PrefitUpdateResult:
    """prefit/bounce_event 更新的返回值。"""

    pre_coeffs: dict[str, np.ndarray] | None
    bounce_event: BounceEvent | None
    low_snr_prefit: LowSnrAxisModes | None


@dataclass(frozen=True)
class PostUpdateResult:
    """prior/posterior/corridor 更新的返回值。"""

    candidates: list[Candidate]
    best_candidate: Candidate | None
    nominal_candidate_id: int | None
    posterior_state: PosteriorState | None
    corridor_by_time: CorridorByTime | None
    posterior_anchor_used: bool
    low_snr_posterior: LowSnrAxisModes | None
