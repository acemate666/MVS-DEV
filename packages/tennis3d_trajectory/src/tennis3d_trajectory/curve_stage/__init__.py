"""将 3D 定位输出流转换为轨迹拟合（curve）输出的后处理 stage。

设计目标：
- 组合而非侵入：不修改 `tennis3d.pipeline.core.run_localization_pipeline` 的内部逻辑。
- 支持多球：把每帧 0..N 个 3D 点关联成多个 track，每个 track 维护一套拟合器状态。
- 下游友好：输出“落点 + 落地时间 + 置信走廊”的轻量 JSON 结构（可直接写入 jsonl）。

注意：
- 当前多球关联使用简单的最近邻 + gating（工程够用、易解释）。当两球非常接近/交叉时，
  仍可能发生 track 交换；如果你的场景确实高频出现，需要升级为更强的关联（匈牙利/卡尔曼等）。
"""

from __future__ import annotations

import time
from typing import Any, Iterable

from tennis3d_trajectory.curve_stage.config import CurveStageConfig
from tennis3d_trajectory.curve_stage.stage import CurveStage


def apply_curve_stage(records: Iterable[dict[str, Any]], cfg: CurveStageConfig) -> Iterable[dict[str, Any]]:
    """对记录流应用 curve stage（保持 generator 风格）。"""

    # 说明：这里保持“yield 逐条处理”的行为，避免改变上游的流式内存特性。
    if not bool(cfg.enabled):
        for r in records:
            if not isinstance(r, dict):
                continue
            yield r
        return

    stage = CurveStage(cfg)
    for r in records:
        if not isinstance(r, dict):
            continue
        t0 = time.monotonic()
        out = stage.process_record(r)
        dt_ms = 1000.0 * max(0.0, time.monotonic() - t0)

        # 说明：timing_ms 是“毫秒口径视图”，便于下游稳定解析与回归。
        # curve_stage_ms 仅在 curve_enabled 时出现，避免无意改变默认输出体积。
        if isinstance(out, dict):
            tm = out.get("timing_ms")
            if not isinstance(tm, dict):
                tm = {}
                out["timing_ms"] = tm
            tm["curve_stage_ms"] = float(dt_ms)

        yield out


__all__ = [
    "CurveStage",
    "CurveStageConfig",
    "apply_curve_stage",
]
