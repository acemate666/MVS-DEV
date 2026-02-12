"""tennis3d core 对外稳定调用入口（public API）。

设计目标：
- 让其他程序以稳定的方式 `import tennis3d` / `from tennis3d.api import ...` 调用核心能力。
- 仅暴露“纯算法/纯数据”能力：不反向依赖 detectors / online / offline 等适配层包。
"""

from __future__ import annotations

from pathlib import Path

from tennis3d.geometry.calibration import CalibrationSet, load_calibration
from tennis3d.pipeline.core import run_localization_pipeline


def build_calibration(calib_path: Path) -> CalibrationSet:
    """加载标定文件。

    Args:
        calib_path: 标定文件路径（.json/.yaml/.yml）。

    Returns:
        CalibrationSet。
    """

    return load_calibration(Path(calib_path).resolve())


__all__ = [
    "CalibrationSet",
    "build_calibration",
    "run_localization_pipeline",
]
