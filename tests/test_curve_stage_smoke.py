from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from tennis3d.config import load_offline_app_config
from tennis3d_trajectory.curve_stage import CurveStageConfig, apply_curve_stage
from tennis3d_trajectory.curve_stage.stage import CurveStage


def _make_rec(*, t_abs: float, x: float, y: float, z: float) -> dict:
    # 说明：这里构造的是 run_localization_pipeline 的输出子集，足够覆盖 curve stage。
    return {
        "capture_t_abs": float(t_abs),
        "created_at": 123.0,
        "balls": [
            {
                "ball_id": 0,
                "ball_3d_world": [float(x), float(y), float(z)],
                "quality": 1.0,
            }
        ],
    }


def test_curve_stage_adds_curve_field_and_track_id() -> None:
    cfg = CurveStageConfig(
        enabled=True,
        max_tracks=2,
        association_dist_m=10.0,
        max_missed_s=10.0,
    )

    t0 = 1000.0
    records_in = [_make_rec(t_abs=t0 + i * 0.01, x=0.1 * i, y=1.5 - 0.2 * i, z=3.0) for i in range(6)]

    records_out = list(apply_curve_stage(records_in, cfg))
    assert len(records_out) == len(records_in)

    for r in records_out:
        assert "curve" in r
        assert int(r["curve"].get("schema_version")) == 1
        assert r["curve"]["t_source"] == "capture_t_abs"
        assert int(r["curve"]["num_active_tracks"]) == 1

        balls = r.get("balls")
        assert isinstance(balls, list) and balls
        assert isinstance(balls[0], dict)
        assert balls[0].get("curve_track_id") == 1

    last = records_out[-1]
    tu = last["curve"]["track_updates"]
    assert isinstance(tu, list) and len(tu) == 1

    v3 = tu[0].get("v3")
    assert isinstance(v3, dict)
    assert v3.get("time_base_abs") == t0


def test_curve_stage_applies_y_transform_to_curve_input() -> None:
    cfg = CurveStageConfig(
        enabled=True,
        max_tracks=1,
        association_dist_m=10.0,
        max_missed_s=10.0,
        y_offset_m=0.13,
        y_negate=True,
    )

    rec_in = _make_rec(t_abs=1000.0, x=0.0, y=1.0, z=0.0)
    rec_out = list(apply_curve_stage([rec_in], cfg))[0]

    tu = rec_out["curve"]["track_updates"]
    assert isinstance(tu, list) and len(tu) == 1
    last_pos = tu[0].get("last_pos")
    assert isinstance(last_pos, list) and len(last_pos) == 3

    # 期望：y' = -(y - 0.13) = -0.87
    assert abs(float(last_pos[1]) - (-0.87)) < 1e-9


def test_curve_stage_interception_field_present_when_enabled() -> None:
    cfg = CurveStageConfig(
        enabled=True,
        max_tracks=1,
        association_dist_m=10.0,
        max_missed_s=10.0,
        interception_enabled=True,
        interception_y_min=0.7,
        interception_y_max=1.2,
        interception_num_heights=5,
        interception_r_hit_m=0.15,
    )

    t0 = 1000.0
    records_in = [_make_rec(t_abs=t0 + i * 0.01, x=0.1 * i, y=1.5 - 0.2 * i, z=3.0) for i in range(3)]
    records_out = list(apply_curve_stage(records_in, cfg))

    last = records_out[-1]
    tu = last["curve"]["track_updates"]
    assert isinstance(tu, list) and len(tu) == 1

    inter = tu[0].get("interception")
    assert isinstance(inter, dict)
    assert inter.get("valid") is False
    assert inter.get("reason") in {"no_bounce_event", "interception_import_failed", "selector_failed"}


class _StubV3:
    """用于 CurveStage 集成回归的最小 v3 替身。

    说明：
        - 该测试的目标是锁定“curve_stage -> interception”的输出契约与稳定器行为，
          而不是重复验证 curve_v3 的 bounce/candidate 估计算法。
        - 因此这里用一个可控的 stub，返回确定性的 bounce/candidates/config。
    """

    def __init__(
        self,
        *,
        time_base_abs: float,
        curve_cfg: object,
        bounce: object,
        candidates: Sequence[object],
        post_points: Sequence[object] | None = None,
    ) -> None:
        self.time_base_abs = float(time_base_abs)
        self.config = curve_cfg
        self._bounce = bounce
        self._candidates = list(candidates)
        self._post_points = list(post_points) if post_points is not None else []

    def add_observation(self, _obs: object) -> None:
        # 说明：该 stub 不做任何状态更新；上游只要求该方法存在。
        return

    def get_bounce_event(self):
        return self._bounce

    def get_prior_candidates(self):
        return list(self._candidates)

    def get_post_points(self):
        return list(self._post_points)


def _is_finite_number(x: object) -> bool:
    if not isinstance(x, (int, float)):
        return False
    return bool(math.isfinite(float(x)))


def test_curve_stage_interception_can_be_valid_and_never_outputs_stale_target() -> None:
    """端到端回归：至少一次 valid=true，并确保无“过期目标”泄漏。"""

    cfg = CurveStageConfig(
        enabled=True,
        max_tracks=1,
        association_dist_m=10.0,
        max_missed_s=10.0,
        interception_enabled=True,
        interception_y_min=0.7,
        interception_y_max=1.2,
        interception_num_heights=5,
        interception_r_hit_m=0.15,
    )

    stage = CurveStage(cfg)

    # 第 1 帧：只用于创建 track（随后把 v3 替换为 stub）。
    _ = stage.process_record(_make_rec(t_abs=1000.0, x=0.0, y=1.0, z=0.0))
    assert len(stage._tracks) == 1  # noqa: SLF001

    from curve_v3 import BounceEvent, Candidate, CurveV3Config

    curve_cfg = CurveV3Config()
    bounce = BounceEvent(
        t_rel=0.0,
        x=0.0,
        z=0.0,
        v_minus=np.zeros((3,), dtype=float),
        y=float(curve_cfg.bounce_contact_y()),
    )

    # 说明：interception 默认要求 min_valid_candidates>=3，因此这里给 3 个候选确保可达 valid=true。
    good_candidates = [
        Candidate(e=0.70, kt=0.65, weight=1.0, v_plus=np.array([1.0, 6.0, 0.0], dtype=float)),
        Candidate(e=0.72, kt=0.66, weight=1.0, v_plus=np.array([1.0, 6.2, 0.0], dtype=float)),
        Candidate(e=0.68, kt=0.64, weight=1.0, v_plus=np.array([1.0, 5.8, 0.0], dtype=float)),
    ]

    stub = _StubV3(time_base_abs=1000.0, curve_cfg=curve_cfg, bounce=bounce, candidates=good_candidates)
    stage._tracks[0].v3 = stub  # noqa: SLF001

    # 第 2 帧：应当产出 valid=true，并给出有限的 target 数值。
    out2 = stage.process_record(_make_rec(t_abs=1000.01, x=0.01, y=1.0, z=0.0))
    tu2 = out2["curve"]["track_updates"]
    assert isinstance(tu2, list) and len(tu2) == 1
    inter2 = tu2[0].get("interception")
    assert isinstance(inter2, dict)
    assert inter2.get("valid") is True
    assert set(inter2.keys()) >= {"valid", "reason", "target", "diag"}

    tgt2 = inter2.get("target")
    assert isinstance(tgt2, dict)
    for k in ["x", "y", "z", "t_abs", "t_rel"]:
        assert _is_finite_number(tgt2.get(k)), f"target.{k} must be finite"

    # 第 3 帧：构造一个“仍有 bounce/candidates 但无法穿越高度范围”的情况。
    # 期望：稳定器不会复用上一帧的 target（不输出过期目标）。
    bad_candidates = [
        Candidate(e=0.70, kt=0.65, weight=1.0, v_plus=np.array([0.5, 0.5, 0.0], dtype=float)),
        Candidate(e=0.72, kt=0.66, weight=1.0, v_plus=np.array([0.5, 0.6, 0.0], dtype=float)),
        Candidate(e=0.68, kt=0.64, weight=1.0, v_plus=np.array([0.5, 0.4, 0.0], dtype=float)),
    ]
    stub._candidates = bad_candidates  # type: ignore[attr-defined]

    out3 = stage.process_record(_make_rec(t_abs=1000.02, x=0.02, y=1.0, z=0.0))
    tu3 = out3["curve"]["track_updates"]
    inter3 = tu3[0].get("interception")
    assert isinstance(inter3, dict)
    assert inter3.get("valid") is False
    assert inter3.get("target") is None


def test_offline_config_accepts_curve_section(tmp_path: Path) -> None:
    cfg_path = tmp_path / "offline_cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "input": {
                    "captures_dir": "data/captures_master_slave/tennis_test",
                    "calib": "data/calibration/example_triple_camera_calib.json",
                },
                "detector": {
                    "name": "fake",
                },
                "curve": {
                    "enabled": True,
                    "track": {
                        "max_tracks": 2,
                    },
                    "conf": {
                        "from": "constant",
                        "constant": 0.5,
                    },
                    "transform": {
                        "y_offset_m": 0.13,
                        "y_negate": True,
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cfg = load_offline_app_config(cfg_path)
    assert bool(cfg.curve.enabled) is True
    assert int(cfg.curve.max_tracks) == 2
    assert str(cfg.curve.conf_from) == "constant"
    assert float(cfg.curve.constant_conf) == 0.5
    assert float(cfg.curve.y_offset_m) == 0.13
    assert bool(cfg.curve.y_negate) is True
