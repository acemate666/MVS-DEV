"""interception：击球目标点（拦截点）选择。

对外入口：
    - `select_hit_target_prefit_only`: 仅基于 prefit/prior（N=0）。
    - `select_hit_target_with_post`: 基于 prefit + post 点（1<=N<=5）。
"""

from interception.config import InterceptionConfig
from interception.selector import select_hit_target_prefit_only, select_hit_target_with_post
from interception.stabilizer import HitTargetStabilizer
from interception.types import HitTarget, HitTargetDiagnostics, HitTargetResult

__all__ = [
    "InterceptionConfig",
    "HitTarget",
    "HitTargetDiagnostics",
    "HitTargetResult",
    "HitTargetStabilizer",
    "select_hit_target_prefit_only",
    "select_hit_target_with_post",
]
