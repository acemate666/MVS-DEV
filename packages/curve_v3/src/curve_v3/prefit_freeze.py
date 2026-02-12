"""prefit 分段/冻结状态机。

动机：
    `CurvePredictorV3` 的职责应尽量收敛到“流程编排”。
    prefit 的分段检测（PRE_BOUNCE -> POST_BOUNCE）与冻结信息（cut_index/reason/t）
    属于可独立理解、可独立单测的状态逻辑，适合从 `core.py` 抽离。

设计原则：
    - 只封装状态与切片决策；不做 prefit 拟合本身。
    - 行为保持与历史实现一致：
        - 当首次检测到 cut_index 且 (cut+1)>=min_points 时，记录 freeze_t_rel=ts[-1]。
        - 由调用方在 prefit 成功后显式触发 freeze（避免在 prefit 失败时“误冻结”）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from curve_v3.configs import CurveV3Config
from curve_v3.utils import BounceTransitionDetector


@dataclass
class PrefitFreezeState:
    """prefit 冻结相关的稳定状态。"""

    is_frozen: bool = False
    cut_index: int | None = None
    freeze_reason: str | None = None
    freeze_t_rel: float | None = None


class PrefitFreezeController:
    """封装 prefit 分段检测与冻结信息。"""

    def __init__(self, *, cfg: CurveV3Config, min_points: int = 5) -> None:
        self._detector = BounceTransitionDetector(cfg=cfg)
        self._min_points = int(min_points)
        if self._min_points < 1:
            self._min_points = 1
        self._state = PrefitFreezeState()

    @property
    def state(self) -> PrefitFreezeState:
        return self._state

    def update_cut_index(self, *, ts: np.ndarray, ys: np.ndarray, y_contact: float) -> None:
        """根据最新序列更新 cut_index（只在首次命中时写入）。

        说明：
            - 若已经有 cut_index，则保持不变。
            - freeze_t_rel 记录为“触发 cut 的那一帧”的 t_rel（而不是 cut 点时刻）。
        """

        if self._state.cut_index is not None:
            return

        cut, reason = self._detector.find_cut_index(
            ts=ts,
            ys=ys,
            y_contact=float(y_contact),
        )
        if cut is None:
            return

        cut_i = int(cut)
        if (cut_i + 1) < self._min_points:
            return

        self._state.cut_index = cut_i
        self._state.freeze_reason = str(reason) if reason is not None else None
        # 与历史实现一致：记录“触发冻结”的时刻（当前帧），而不是 cut 点时刻。
        self._state.freeze_t_rel = float(ts[-1])

    def prefit_slice_end(self, *, n_points: int) -> int | None:
        """返回 prefit 段的切片终点 k（exclusive），不可用则返回 None。"""

        if self._state.cut_index is None:
            return None

        k = int(self._state.cut_index) + 1
        if k < self._min_points:
            return None

        n = int(n_points)
        if n <= 0:
            return None
        return int(min(k, n))

    def freeze(self) -> None:
        """冻结 prefit（调用方确认 prefit 成功后触发）。"""

        if self._state.cut_index is None:
            return
        self._state.is_frozen = True
