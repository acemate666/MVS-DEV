"""相机投影接口（用于像素域闭环）。

说明：
    `curve_v3` 的核心算法不应该依赖具体相机 SDK/标定实现。
    因此这里用 Protocol 定义最小投影能力：
        世界坐标 3D 点 -> 像素坐标 (u, v)

    上游可以用任意方式实现该协议（OpenCV、CUDA、自研模型等），再通过
    `CameraRig` 注入到 `CurvePredictorV3`。

注意：
    - 这里不引入任何第三方依赖。
    - 投影的畸变/去畸变口径由上游实现保证一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np


class CameraProjector(Protocol):
    """单相机投影器协议。"""

    def project_world_to_pixel(self, p_world: np.ndarray) -> np.ndarray:
        """把世界系 3D 点投影到像素坐标。

        Args:
            p_world: 世界系点，shape=(3,)；单位米。

        Returns:
            像素坐标，shape=(2,)，单位像素 [u, v]。

        Raises:
            实现可以在不可投影时抛异常（例如点在相机背后、超出视野等）。
        """

        ...


@dataclass(frozen=True)
class CameraRig:
    """多相机投影器容器。"""

    cameras: Mapping[str, CameraProjector]

    def project(self, camera: str, p_world: np.ndarray) -> np.ndarray:
        """对指定相机做投影。"""

        cam = self.cameras.get(str(camera))
        if cam is None:
            raise KeyError(f"未知相机: {camera!r}")
        uv = cam.project_world_to_pixel(np.asarray(p_world, dtype=float).reshape(3))
        uv = np.asarray(uv, dtype=float).reshape(2)
        return uv
