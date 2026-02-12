"""curve_v3 的离线/评测工具命名空间。

说明：
    该命名空间下的模块不属于 `docs/curve.md` 描述的核心在线算法路径。
    它们可能包含 DB/文件等 IO 逻辑，或用于离线评测与数据准备。
"""

from curve_v3.offline import testing, vl11

__all__ = [
    "testing",
    "vl11",
]

