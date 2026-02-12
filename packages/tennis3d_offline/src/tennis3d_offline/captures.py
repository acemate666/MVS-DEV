"""离线输入源：从 captures 目录产出每组图像。

说明：
- 该模块属于离线 app/adapter 层，因此放在 `tennis3d_offline` 包内。
- `tennis3d-core` 目标是“纯算法库”，不应再直接依赖 `mvs` 或承担 captures I/O。

职责：
- 读取 `captures_dir/metadata.jsonl`（允许混入非 group 记录）。
- 解析每组 frames 列表，并加载对应图像文件（OpenCV BGR）。
- 计算组级别的时间轴信息（capture_t_abs / capture_host_timestamp）。

时间轴策略：
- frame_host_timestamp：对组内 frames[*].host_timestamp 取中位数并归一化为 epoch 秒。
- dev_timestamp_mapping：读取 time_mapping.json（由 mvs.save_time_mappings_json 生成），
  对每台相机用 host_ms ≈ a*dev_ts + b 映射后取中位数，得到 capture_t_abs。

注意：
- 这里刻意不 import `mvs`，而是仅解析其输出的 time_mapping.json 格式，
  以便 `tennis3d-offline` 在“纯离线”场景下也可使用。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from tennis3d.pipeline.time_utils import host_timestamp_to_seconds, median_float, median_int

__all__ = [
    "LinearTimeMapping",
    "iter_capture_image_groups",
    "load_time_mappings_json",
]


def _find_repo_root(start: Path) -> Path:
    """从给定路径向上查找仓库根目录。

    约定：仓库根目录包含 pyproject.toml。
    """

    start = Path(start).resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start


def _iter_metadata_records(meta_path: Path) -> Iterator[dict[str, Any]]:
    """迭代读取 metadata.jsonl 记录（容错：跳过非 JSON 行/非 dict 行）。"""

    p = Path(meta_path).resolve()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = str(line).strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


@dataclass(frozen=True, slots=True)
class LinearTimeMapping:
    """线性时间映射：host_ms = a * dev_ts + b。"""

    a: float
    b: float

    def map_dev_to_host_ms(self, dev_ts: int) -> float:
        return float(self.a) * float(int(dev_ts)) + float(self.b)


def load_time_mappings_json(path: Path) -> dict[str, LinearTimeMapping]:
    """读取 mvs_time_mapping_v1 格式的 time_mapping.json。

    说明：
    - 该格式由 `mvs.session.time_mapping.save_time_mappings_json()` 生成。
    - 这里只解析 `cameras[*].a/b`，其余字段忽略。
    """

    p = Path(path).resolve()
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("time mapping json must be an object")

    schema = str(data.get("schema", ""))
    if schema != "mvs_time_mapping_v1":
        raise ValueError(f"unsupported schema: {schema}")

    host_unit = str(data.get("host_unit", ""))
    if host_unit != "ms_epoch":
        raise ValueError(f"unsupported host_unit: {host_unit} (expected ms_epoch)")

    cams = data.get("cameras")
    if not isinstance(cams, dict):
        raise ValueError("time mapping json missing 'cameras' dict")

    out: dict[str, LinearTimeMapping] = {}
    for serial, v in cams.items():
        if not isinstance(v, dict):
            continue
        a_raw = v.get("a")
        b_raw = v.get("b")
        if a_raw is None or b_raw is None:
            continue
        try:
            out[str(serial)] = LinearTimeMapping(a=float(a_raw), b=float(b_raw))
        except Exception:
            continue

    return out


def iter_capture_image_groups(
    *,
    captures_dir: Path,
    max_groups: int = 0,
    serials: list[str] | None = None,
    time_sync_mode: str = "frame_host_timestamp",
    time_mapping_path: Path | None = None,
) -> Iterator[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    """从 captures/metadata.jsonl 迭代读取每组图像。

    Notes:
    - metadata.jsonl 中可能混有非 group 记录（例如事件/日志），这里会跳过。
    - frames[*].file 可能是绝对路径，也可能是相对路径：
      1) 相对 captures_dir；
      2) 相对仓库根目录（例如以 data/ 开头）。
      本函数会做一次稳健解析，尽量找到真实存在的文件。

    Args:
        captures_dir: captures 目录（包含 metadata.jsonl 与各帧图像文件）。
        max_groups: 最多处理的组数（0 表示不限）。
        serials: Optional 的相机序列号白名单；为 None 时不过滤。
        time_sync_mode: 时间轴策略（frame_host_timestamp / dev_timestamp_mapping）。
        time_mapping_path: 当 time_sync_mode=dev_timestamp_mapping 时使用的映射文件。

    Yields:
        (meta, images_by_camera_serial)
    """

    captures_dir = Path(captures_dir).resolve()
    meta_path = captures_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise RuntimeError(f"metadata.jsonl not found: {meta_path}")

    repo_root = _find_repo_root(captures_dir)
    groups_done = 0

    mappings: dict[str, LinearTimeMapping] | None = None
    if str(time_sync_mode).strip() == "dev_timestamp_mapping":
        if time_mapping_path is None:
            raise RuntimeError("time_sync_mode=dev_timestamp_mapping 需要提供 time_mapping_path")
        mappings = load_time_mappings_json(Path(time_mapping_path).resolve())

    serials_set: set[str] | None = None
    if serials is not None:
        serials_norm = [str(s).strip() for s in (serials or []) if str(s).strip()]
        serials_set = set(serials_norm) if serials_norm else set()

    for rec in _iter_metadata_records(meta_path):
        if "frames" not in rec:
            continue

        frames = rec.get("frames")
        if not isinstance(frames, list) or not frames:
            continue

        host_ts_list: list[int] = []
        for fr in frames:
            if not isinstance(fr, dict):
                continue
            ts_raw = fr.get("host_timestamp")
            if ts_raw is None:
                continue
            try:
                host_ts_list.append(int(ts_raw))
            except Exception:
                pass
        host_ts_med = median_int(host_ts_list)

        capture_t_abs: float | None = host_timestamp_to_seconds(host_ts_med) if host_ts_med is not None else None
        capture_t_source: str | None = "frame_host_timestamp" if capture_t_abs is not None else None

        # Optional：用映射后的时间替代默认 host_timestamp 聚合。
        if mappings is not None:
            mapped_ms_list: list[float] = []
            for fr in frames:
                if not isinstance(fr, dict):
                    continue
                serial = str(fr.get("serial", "")).strip()
                dev_ts = fr.get("dev_timestamp")
                if not serial or dev_ts is None:
                    continue
                m = mappings.get(serial)
                if m is None:
                    continue
                try:
                    mapped_ms_list.append(float(m.map_dev_to_host_ms(int(dev_ts))))
                except Exception:
                    continue

            mapped_ms_med = median_float(mapped_ms_list)
            if mapped_ms_med is not None:
                capture_t_abs = float(mapped_ms_med) / 1000.0
                capture_t_source = "dev_timestamp_mapping"

        images_by_camera: dict[str, np.ndarray] = {}
        for fr in frames:
            if not isinstance(fr, dict):
                continue

            serial = str(fr.get("serial", "")).strip()
            if serials_set is not None and serial not in serials_set:
                continue
            file = fr.get("file")
            if not serial or not isinstance(file, str) or not file:
                continue

            file_path = Path(file)
            if not file_path.is_absolute():
                candidate = (captures_dir / file_path).resolve()
                if candidate.exists():
                    file_path = candidate
                else:
                    candidate2 = (repo_root / file_path).resolve()
                    file_path = candidate2 if candidate2.exists() else candidate

            if not file_path.exists():
                continue

            img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
            if img is None:
                continue

            images_by_camera[serial] = img

        meta = {
            "group_seq": rec.get("group_seq"),
            "group_by": rec.get("group_by"),
            "capture_t_abs": float(capture_t_abs) if capture_t_abs is not None else None,
            "capture_t_source": str(capture_t_source) if capture_t_source is not None else None,
            "capture_host_timestamp": int(host_ts_med) if host_ts_med is not None else None,
        }

        yield meta, images_by_camera

        groups_done += 1
        if int(max_groups) > 0 and groups_done >= int(max_groups):
            break
