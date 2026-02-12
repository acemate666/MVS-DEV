from __future__ import annotations

from pathlib import Path

from curve_v3.configs import CurveV3Config
from curve_v3.offline.real_device_eval import load_observations_from_json
from curve_v3.offline.testing.synthetic_tennis_json import write_synthetic_tennis_trajectory_json


def test_write_and_load_synthetic_tennis_json(tmp_path: Path) -> None:
    cfg = CurveV3Config()
    out = tmp_path / "synthetic_tennis_trajectory.json"

    write_synthetic_tennis_trajectory_json(
        out,
        cfg=cfg,
        seed=7,
        sigma_m=0.01,
        t_land_rel=0.23,
        truth_e=0.9,
        truth_kt=0.85,
        num_pre_points=14,
        num_post_points=15,
        post_dt_s=0.05,
        include_conf=True,
        meta_overrides={"source": "pytest"},
    )

    meta, obs = load_observations_from_json(out)

    assert meta["seed"] == 7
    assert meta["source"] == "pytest"

    assert len(obs) == 14 + 15
    assert all(obs[i].t <= obs[i + 1].t for i in range(len(obs) - 1))

    # conf 字段应可选，但在 include_conf=True 时应填充为数值或 None（这里应为数值）。
    assert all(o.conf is None or isinstance(o.conf, float) for o in obs)
