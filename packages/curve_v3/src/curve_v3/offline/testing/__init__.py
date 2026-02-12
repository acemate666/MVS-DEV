"""curve_v3 的离线/评测用测试辅助工具。

说明：
    该子包用于单元测试/离线调试生成合成数据，不参与 curve_v3 的核心在线算法实现。
"""

from curve_v3.offline.testing.synthetic import (
    SyntheticNoise,
    make_postbounce_observations_from_candidate,
    make_prebounce_observations_ballistic,
)
from curve_v3.offline.testing.synthetic_tennis_json import (
    make_synthetic_tennis_observations,
    make_synthetic_tennis_trajectory_json,
    write_synthetic_tennis_trajectory_json,
)

__all__ = [
    "SyntheticNoise",
    "make_postbounce_observations_from_candidate",
    "make_prebounce_observations_ballistic",
    "make_synthetic_tennis_observations",
    "make_synthetic_tennis_trajectory_json",
    "write_synthetic_tennis_trajectory_json",
]
