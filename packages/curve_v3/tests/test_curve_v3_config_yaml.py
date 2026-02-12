from __future__ import annotations

import textwrap

import pytest

from curve_v3.config_yaml import load_curve_v3_config_yaml


def test_load_curve_v3_config_yaml_partial_override(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        textwrap.dedent(
            """
            physics:
              gravity: 9.81
            prior:
              e_bins: [0.60, 0.80]
            corridor:
              corridor_quantile_levels: [0.05, 0.95]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = load_curve_v3_config_yaml(p)

    assert float(cfg.physics.gravity) == pytest.approx(9.81)
    assert list(cfg.prior.e_bins) == [0.60, 0.80]
    assert list(cfg.corridor.corridor_quantile_levels) == [0.05, 0.95]

    # 未覆盖的字段仍应使用默认值
    assert int(cfg.posterior.max_post_points) == 999


def test_load_curve_v3_config_yaml_unknown_key_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("unknown_top_level: 1\n", encoding="utf-8")

    with pytest.raises(KeyError):
        _ = load_curve_v3_config_yaml(p)
