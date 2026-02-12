from __future__ import annotations

import numpy as np

from tennis3d.geometry.calibration import CalibrationSet, CameraCalibration
from tennis3d.models import Detection
from tennis3d.pipeline.core import run_localization_pipeline


class _BatchCapableDetector:
    """同时实现 detect 与 detect_batch 的 stub，用于验证 pipeline 的 micro-batch 快路径。"""

    def __init__(self) -> None:
        self.detect_calls = 0
        self.batch_calls = 0

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        _ = img_bgr
        self.detect_calls += 1
        return []

    def detect_batch(self, imgs_bgr: list[np.ndarray]) -> list[list[Detection]]:
        _ = imgs_bgr
        self.batch_calls += 1
        return [[] for _ in imgs_bgr]


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


def test_pipeline_uses_detect_batch_across_cameras() -> None:
    calib = _make_calib_two_cams()

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    images = {"cam0": img, "cam1": img}
    meta = {"group_index": 0}

    det = _BatchCapableDetector()
    records = run_localization_pipeline(
        groups=[(meta, images)],
        calib=calib,
        detector=det,
        min_score=0.25,
        require_views=2,
        max_detections_per_camera=10,
        max_reproj_error_px=8.0,
        max_uv_match_dist_px=25.0,
        merge_dist_m=0.08,
        include_detection_details=False,
    )

    _ = next(iter(records))

    assert det.batch_calls == 1
    assert det.detect_calls == 0


def test_pipeline_falls_back_to_detect_for_single_camera() -> None:
    calib = _make_calib_two_cams()

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    images = {"cam0": img}
    meta = {"group_index": 0}

    det = _BatchCapableDetector()
    records = run_localization_pipeline(
        groups=[(meta, images)],
        calib=calib,
        detector=det,
        min_score=0.25,
        require_views=2,
        max_detections_per_camera=10,
        max_reproj_error_px=8.0,
        max_uv_match_dist_px=25.0,
        merge_dist_m=0.08,
        include_detection_details=False,
    )

    _ = next(iter(records))

    assert det.batch_calls == 0
    assert det.detect_calls == 1
