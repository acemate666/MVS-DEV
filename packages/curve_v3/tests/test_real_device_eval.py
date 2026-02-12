from __future__ import annotations

import json
from pathlib import Path

from curve_v3.configs import CurveV3Config
from curve_v3.offline.real_device_eval import load_observations_from_json, run_post_bounce_subsampling_eval


def test_real_device_eval_post_bounce_subsampling_smoke(tmp_path: Path) -> None:
    """实机/回放评测链路冒烟测试。

    目标：
        - 确保 JSON -> 曲线拟合/评估 -> 落盘 的链路不会随着重构而腐烂。
        - 不要求精度（避免在不同数值实现/配置下变脆）。

    约束：
        - 该测试不依赖 matplotlib（CI/纯算法环境也应能跑）。
    """

    sample = Path(__file__).parent / "data" / "sample_tennis_trajectory.json"
    meta, obs = load_observations_from_json(sample)

    assert isinstance(meta, dict)
    assert obs

    out_dir = tmp_path / "real_device_eval"

    report = run_post_bounce_subsampling_eval(
        observations=obs,
        cfg=CurveV3Config(),
        target_plane_y_m=0.9,
        out_dir=out_dir,
        make_plots=False,
    )

    assert (out_dir / "summary.json").exists()
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "cases").exists()

    # 关键字段存在即可。
    assert "meta" in report
    assert "cases" in report
    assert "ns_post_points" in report

    ns = report["ns_post_points"]
    assert 0 in ns
    assert max(ns) >= 1

    # 每个 case 都应落盘。
    for n in ns:
        p = out_dir / "cases" / f"case_n={int(n):03d}.json"
        assert p.exists()

        d = json.loads(p.read_text(encoding="utf-8"))
        assert int(d["n_post_points"]) == int(n)
        assert "plane_y_m" in d
