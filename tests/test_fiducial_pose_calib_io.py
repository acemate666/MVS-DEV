from __future__ import annotations

from pathlib import Path

import numpy as np

from fiducial_pose import load_camera_intr_extr_from_calib_json


def test_load_camera_intr_extr_from_repo_calib_json() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    p = repo_root / "data" / "calibration" / "camera_extrinsics_C_T_B.json"

    intr, extr = load_camera_intr_extr_from_calib_json(calib_json_path=p, camera="DA8199303")

    assert intr.K.shape == (3, 3)
    assert extr.R_wc.shape == (3, 3)
    assert extr.t_wc.shape == (3,)

    assert np.isfinite(intr.K).all()
    assert np.isfinite(extr.R_wc).all()
    assert np.isfinite(extr.t_wc).all()
