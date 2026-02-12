"""拦截目标点的迟滞/稳定（跨帧防抖）。

为什么需要它：
    - 在 N=1..2 的早期阶段，后验分布还未收敛，最优高度/最优单点可能来回跳。
    - 控制器通常更希望“少跳、可预期”，而不是每帧追逐一个噪声驱动的最优点。

设计原则：
    - selector 只负责“当前帧”的最优解；稳定性属于状态逻辑，单独封装在本模块。
    - 不引入复杂状态机：只保留上一帧的结果，并基于少量阈值判断是否切换。
"""

from __future__ import annotations

from dataclasses import dataclass

from interception.config import InterceptionConfig
from interception.types import HitTargetResult


@dataclass
class HitTargetStabilizer:
    """对击球目标点进行跨帧稳定。

    用法：
        stabilizer = HitTargetStabilizer()
        raw = select_hit_target_...( ... )
        stable = stabilizer.update(raw, cfg)

    说明：
        - 只有当当前帧与上一帧均为 valid 时才启用迟滞判断。
        - 若当前帧 invalid，则直接返回当前帧（不输出过期目标）。
    """

    last: HitTargetResult | None = None

    def update(self, current: HitTargetResult, cfg: InterceptionConfig) -> HitTargetResult:
        """根据迟滞规则决定是否更新目标点。

        触发“更新”的条件（任一满足即可）：
            1) score 明显提升
            2) w_max 明显上升（分布更收敛）
            3) width_xz 明显变窄（不确定性显著下降）

        Args:
            current: 当前帧 selector 输出。
            cfg: InterceptionConfig，包含迟滞阈值。

        Returns:
            稳定后的 HitTargetResult。
        """

        if self.last is None:
            if current.valid:
                self.last = current
            return current

        prev = self.last

        # 只在“上一帧 valid 且当前帧 valid”时保持上一帧目标；否则直接放行 current。
        if not (prev.valid and current.valid):
            if current.valid:
                self.last = current
            return current

        prev_score = prev.diag.score if prev.diag.score is not None else -1.0e30
        cur_score = current.diag.score if current.diag.score is not None else -1.0e30

        score_gain = float(cur_score - prev_score)
        score_ok = score_gain >= float(cfg.hysteresis_score_gain)

        w_ok = False
        if prev.diag.w_max is not None and current.diag.w_max is not None:
            w_ok = float(current.diag.w_max - prev.diag.w_max) >= float(cfg.hysteresis_w_max_gain)

        width_ok = False
        pw = float(prev.diag.width_xz)
        cw = float(current.diag.width_xz)
        if pw == pw and cw == cw and pw > 1e-12:  # 非 NaN 且 prev 宽度有效
            width_ok = (pw - cw) / pw >= float(cfg.hysteresis_width_shrink_ratio)

        if score_ok or w_ok or width_ok:
            self.last = current
            return current

        # 否则保持上一帧结果。
        return prev
