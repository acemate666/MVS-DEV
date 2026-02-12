from __future__ import annotations

import textwrap

import pytest

from interception.config_yaml import load_interception_config_yaml


def test_load_interception_config_yaml_ok(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        textwrap.dedent(
            """
            y_min: 0.35
            y_max: 1.10
            num_heights: 7
            r_hit_m: 0.20
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = load_interception_config_yaml(p)
    assert float(cfg.y_min) == pytest.approx(0.35)
    assert float(cfg.y_max) == pytest.approx(1.10)
    assert int(cfg.num_heights) == 7
    assert float(cfg.r_hit_m) == pytest.approx(0.20)


def test_load_interception_config_yaml_unknown_key_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("y_min: 0.1\ny_max: 1.0\nnope: 1\n", encoding="utf-8")

    with pytest.raises(KeyError):
        _ = load_interception_config_yaml(p)


def test_load_interception_config_yaml_missing_required_raises(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("num_heights: 5\n", encoding="utf-8")

    with pytest.raises(ValueError):
        _ = load_interception_config_yaml(p)
