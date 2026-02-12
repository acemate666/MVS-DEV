from __future__ import annotations

import math
import time

import numpy as np

from tennis3d.geometry.calibration import CalibrationSet, CameraCalibration
from tennis3d.models import Detection
from tennis3d.pipeline.core import run_localization_pipeline


class _EmptyDetector:
    """用于 pipeline 可观测性字段回归的最小 detector stub。"""

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        _ = img_bgr
        return []


def _make_calib_two_cams() -> CalibrationSet:
    fx = 1000.0
    fy = 1000.0
    cx = 0.0
    cy = 0.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)

    def _cam(name: str, C_w: np.ndarray) -> CameraCalibration:
        t = -(R @ C_w.reshape(3))
        return CameraCalibration(
            name=name,
            image_size=(1920, 1080),
            K=K,
            dist=np.zeros((0,), dtype=np.float64),
            R_wc=R,
            t_wc=t,
        )

    cams = {
        "cam0": _cam("cam0", np.array([0.0, 0.0, 0.0], dtype=np.float64)),
        "cam1": _cam("cam1", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
    }
    return CalibrationSet(cameras=cams)


def test_pipeline_outputs_pipeline_lag_and_timing_ms() -> None:
    calib = _make_calib_two_cams()

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    images = {"cam0": img, "cam1": img}

    # 说明：capture_t_abs 使用 epoch 秒，应该与 created_at 可相减。
    now = time.time()
    meta = {
        "group_index": 0,
        "capture_t_abs": float(now - 0.05),
        "capture_t_source": "test",
    }

    records = run_localization_pipeline(
        groups=[(meta, images)],
        calib=calib,
        detector=_EmptyDetector(),
        min_score=0.25,
        require_views=2,
        max_detections_per_camera=10,
        max_reproj_error_px=8.0,
        max_uv_match_dist_px=25.0,
        merge_dist_m=0.08,
        include_detection_details=False,
    )

    out = next(iter(records))

    lag = out.get("pipeline_lag_ms")
    assert isinstance(lag, (int, float))
    assert math.isfinite(float(lag))
    assert float(lag) >= 0.0
    assert float(lag) < 5000.0

    tm = out.get("timing_ms")
    assert isinstance(tm, dict)
    assert set(tm.keys()) >= {
        "align_ms",
        "detect_ms",
        "localize_ms",
        "pipeline_total_ms",
        "detect_ms_by_camera",
    }
