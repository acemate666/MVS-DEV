"""读取本仓库标定融合 JSON（camera_extrinsics_C_T_B.json 风格）。

背景：
- 仓库里已有标准标定文件：`data/calibration/camera_extrinsics_C_T_B.json`。
- `fiducial_pose` 的核心 API 只依赖最小输入（K/dist/R_wc/t_wc），因此不强绑定 tennis3d-core。
- 但为了减少“上游手动拆字段”的摩擦，本模块提供一个薄的读取工具。

注意：
- 该 JSON 的外参约定为 world->camera：X_c = R_wc X_w + t_wc。
- 若你要 camera->world，请在上层使用 `fiducial_pose.transforms.invert_T()`。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fiducial_pose.types import CameraIntrinsics, WorldToCameraExtrinsics


def _as_mat(x: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.shape != shape:
        raise RuntimeError(f"{name} 形状应为 {shape}，实际为 {a.shape}")
    return a


def _as_vec(x: Any, n: int, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size != n:
        raise RuntimeError(f"{name} 长度应为 {n}，实际为 {a.size}")
    return a.reshape(n)


def load_camera_intr_extr_from_calib_json(
    *,
    calib_json_path: Path,
    camera: str,
) -> tuple[CameraIntrinsics, WorldToCameraExtrinsics]:
    """从 camera_extrinsics_C_T_B.json 风格文件读取单个相机的内外参。

    Args:
        calib_json_path: 标定融合 JSON 路径（包含 cameras 字段）。
        camera: 相机 key（通常是序列号，例如 DA8199303）。

    Returns:
        (intr, extr)

    Raises:
        RuntimeError: 文件缺失、JSON schema 不符合预期、或指定相机不存在。
    """

    p = Path(calib_json_path)
    if not p.exists():
        raise RuntimeError(f"找不到标定文件: {p}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"无法读取标定 JSON: {p}") from exc

    if not isinstance(data, dict):
        raise RuntimeError("标定 JSON 顶层必须是对象（dict）")

    cams = data.get("cameras")
    if not isinstance(cams, dict) or not cams:
        raise RuntimeError("标定 JSON 缺少 cameras 字段或为空")

    cam = cams.get(str(camera))
    if not isinstance(cam, dict):
        avail = ",".join(sorted(str(k) for k in cams.keys()))
        raise RuntimeError(f"标定文件中找不到相机 {camera}（可用：{avail}）")

    K = _as_mat(cam.get("K"), (3, 3), f"{camera}.K")
    dist_raw = cam.get("dist", [])
    dist = np.asarray(dist_raw, dtype=np.float64).reshape(-1)

    R_wc = _as_mat(cam.get("R_wc"), (3, 3), f"{camera}.R_wc")
    t_wc = _as_vec(cam.get("t_wc"), 3, f"{camera}.t_wc")

    return (
        CameraIntrinsics(K=K, dist=dist),
        WorldToCameraExtrinsics(R_wc=R_wc, t_wc=t_wc),
    )
