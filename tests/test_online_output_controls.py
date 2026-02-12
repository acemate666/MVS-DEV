from __future__ import annotations

import io
import json
import math
from pathlib import Path

from tennis3d_online.cli import build_arg_parser
from tennis3d_online.output_loop import run_output_loop
from tennis3d_online.jsonl_writer import _JsonlBufferedWriter
from tennis3d_online.spec import build_spec_from_config
from tennis3d.config import load_online_app_config


def test_build_arg_parser_accepts_terminal_print_mode_none_and_jsonl_flush_flags() -> None:
    # 说明：该测试不连接相机，仅验证 CLI 参数解析。
    p = build_arg_parser()
    args = p.parse_args(
        [
            "--serial",
            "A",
            "--pt-device",
            "cuda:0",
            "--terminal-print-mode",
            "none",
            "--terminal-print-interval-s",
            "2.0",
            "--terminal-status-interval-s",
            "1.0",
            "--terminal-timing",
            "--latest-only",
            "--out-jsonl",
            "data/tools_output/x.jsonl",
            "--out-jsonl-only-when-balls",
            "--out-jsonl-flush-every-records",
            "10",
            "--out-jsonl-flush-interval-s",
            "0.5",
        ]
    )

    assert str(getattr(args, "pt_device")) == "cuda:0"
    assert str(args.terminal_print_mode) == "none"
    assert float(getattr(args, "terminal_print_interval_s", 0.0)) == 2.0
    assert float(args.terminal_status_interval_s) == 1.0
    assert bool(getattr(args, "terminal_timing", False)) is True
    assert bool(getattr(args, "latest_only", False)) is True
    assert str(args.out_jsonl).endswith("x.jsonl")
    assert bool(args.out_jsonl_only_when_balls) is True
    assert int(args.out_jsonl_flush_every_records) == 10
    assert float(args.out_jsonl_flush_interval_s) == 0.5


def test_load_online_app_config_supports_output_controls(tmp_path: Path) -> None:
    # 说明：使用 JSON 配置来避免测试环境对 PyYAML 的依赖。
    cfg_path = tmp_path / "online.json"
    cfg_path.write_text(
        json.dumps(
            {
                "sdk": {
                    "mvimport_dir": "",
                    "dll_dir": "",
                },
                "camera": {
                    "serials": ["A", "B"],
                    "calib": "data/calibration/example_triple_camera_calib.json",
                    "exposure": {"auto": "Off", "time_us": 8000.0},
                    "gain": {"auto": "Off", "value": 6.0},
                },
                "detector": {
                    "pt_device": "cuda:0",
                },
                "run": {
                    "latest_only": True,
                },
                "output": {
                    "terminal_print_mode": "none",
                    "terminal_print_interval_s": 2.0,
                    "terminal_status_interval_s": 1.0,
                    "terminal_timing": True,
                    "out_jsonl": "data/tools_output/x.jsonl",
                    "out_jsonl_only_when_balls": True,
                    "out_jsonl_flush_every_records": 10,
                    "out_jsonl_flush_interval_s": 0.5,
                },
                "trigger": {
                    "trigger_source": "Software",
                    "master_serial": "",
                    "soft_trigger_fps": 5.0,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cfg = load_online_app_config(cfg_path)
    assert cfg.pt_device == "cuda:0"
    assert bool(getattr(cfg, "latest_only", False)) is True
    assert cfg.terminal_print_mode == "none"
    assert float(getattr(cfg, "terminal_print_interval_s", 0.0)) == 2.0
    assert cfg.terminal_status_interval_s == 1.0
    assert bool(getattr(cfg, "terminal_timing", False)) is True
    assert cfg.out_jsonl is not None
    assert Path(cfg.out_jsonl).as_posix().endswith("data/tools_output/x.jsonl")
    assert cfg.out_jsonl_only_when_balls is True
    assert cfg.out_jsonl_flush_every_records == 10
    assert cfg.out_jsonl_flush_interval_s == 0.5

    assert cfg.exposure_auto == "Off"
    assert float(cfg.exposure_time_us or 0.0) == 8000.0
    assert cfg.gain_auto == "Off"
    assert float(cfg.gain or 0.0) == 6.0

    # 说明：spec 是 entry 与 runtime 的稳定边界；这里验证映射不会丢字段。
    spec = build_spec_from_config(cfg)
    assert bool(getattr(spec, "latest_only", False)) is True
    assert spec.exposure_auto == "Off"
    assert float(spec.exposure_time_us or 0.0) == 8000.0
    assert spec.gain_auto == "Off"
    assert float(spec.gain or 0.0) == 6.0
    assert bool(getattr(spec, "terminal_timing", False)) is True
    assert float(getattr(spec, "terminal_print_interval_s", 0.0)) == 2.0


def test_output_loop_throttles_group_prints_by_terminal_print_interval_s(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    # 说明：该测试不连接相机；仅验证逐组终端输出的节流逻辑。
    cfg_path = tmp_path / "online.json"
    cfg_path.write_text(
        json.dumps(
            {
                "sdk": {"mvimport_dir": "", "dll_dir": ""},
                "camera": {
                    "serials": ["A", "B"],
                    "calib": "data/calibration/example_triple_camera_calib.json",
                    "exposure": {"auto": "Off", "time_us": 8000.0},
                    "gain": {"auto": "Off", "value": 6.0},
                },
                "output": {
                    "terminal_print_mode": "all",
                    "terminal_print_interval_s": 1.5,
                    "terminal_status_interval_s": 0.0,
                    "terminal_timing": False,
                },
                "trigger": {"trigger_source": "Software", "master_serial": "", "soft_trigger_fps": 5.0},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cfg = load_online_app_config(cfg_path)
    spec = build_spec_from_config(cfg)

    # 说明：output_loop 内部会调用 time.monotonic：
    # - 1 次用于 last_status_t
    # - 每条 record 2 次（loop_start / record_ready）
    # - 迭代结束（StopIteration）前还会额外取 1 次 loop_start
    # 这里用可控序列让 record_ready 分别为 0.0 / 1.0 / 2.0。
    ts = iter([0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0])

    def _fake_monotonic() -> float:
        return float(next(ts))

    import tennis3d_online.output_loop as output_loop_mod

    monkeypatch.setattr(output_loop_mod.time, "monotonic", _fake_monotonic)

    records = [
        {
            "group_index": 1,
            "balls": [{"ball_3d_world": [0.0, 0.0, 0.0], "num_views": 2}],
        },
        {
            "group_index": 2,
            "balls": [{"ball_3d_world": [0.0, 0.0, 0.0], "num_views": 2}],
        },
        {
            "group_index": 3,
            "balls": [{"ball_3d_world": [0.0, 0.0, 0.0], "num_views": 2}],
        },
    ]

    run_output_loop(records=records, jsonl_writer=None, spec=spec, get_groups_done=lambda: 0)

    out = capsys.readouterr().out

    # interval=1.5s，record_ready=0.0/1.0/2.0 -> 只应打印 group=1 与 group=3。
    assert "group=1 balls=1" in out
    assert "group=3 balls=1" in out
    assert "group=2 balls=1" not in out


def test_output_loop_writes_timing_ms_write_ms_into_jsonl(tmp_path: Path) -> None:
    # 说明：
    # - write_ms 由 output_loop 统计“序列化+写入+flush”的耗时。
    # - 由于 JSONL 是追加写入，无法写完当前行再回填当前行；因此采用 1 条记录的延迟：
    #   第 N 条记录的 timing_ms.write_ms 代表第 N-1 条写盘耗时。

    cfg_path = tmp_path / "online.json"
    cfg_path.write_text(
        json.dumps(
            {
                "sdk": {"mvimport_dir": "", "dll_dir": ""},
                "camera": {
                    "serials": ["A", "B"],
                    "calib": "data/calibration/example_triple_camera_calib.json",
                },
                "detector": {"pt_device": "cpu"},
                "output": {
                    "terminal_print_mode": "none",
                    "terminal_print_interval_s": 0.0,
                    "terminal_status_interval_s": 0.0,
                    "terminal_timing": False,
                    "out_jsonl_only_when_balls": False,
                    "out_jsonl_flush_every_records": 1,
                    "out_jsonl_flush_interval_s": 0.0,
                },
                "trigger": {
                    "trigger_source": "Software",
                    "master_serial": "",
                    "soft_trigger_fps": 0.0,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cfg = load_online_app_config(cfg_path)
    spec = build_spec_from_config(cfg)

    sink = io.StringIO()
    writer = _JsonlBufferedWriter(f=sink, flush_every_records=1, flush_interval_s=0.0)

    records = [
        {
            "group_index": 1,
            "timing_ms": {"pipeline_total_ms": 1.0},
            "balls": [{"ball_3d_world": [0.0, 0.0, 0.0], "num_views": 2}],
        },
        {
            "group_index": 2,
            "timing_ms": {"pipeline_total_ms": 2.0},
            "balls": [{"ball_3d_world": [0.0, 0.0, 0.0], "num_views": 2}],
        },
        {
            "group_index": 3,
            "timing_ms": {"pipeline_total_ms": 3.0},
            "balls": [{"ball_3d_world": [0.0, 0.0, 0.0], "num_views": 2}],
        },
    ]

    run_output_loop(records=records, jsonl_writer=writer, spec=spec, get_groups_done=lambda: 0)

    lines = [ln for ln in sink.getvalue().splitlines() if ln.strip()]
    assert len(lines) == 3

    decoded = [json.loads(ln) for ln in lines]
    for i, rec in enumerate(decoded):
        tm = rec.get("timing_ms")
        assert isinstance(tm, dict)
        assert "write_ms" in tm
        assert tm.get("pipeline_total_ms") == float(i + 1)

    # 第 0 条：没有“上一条写盘”，因此为 null。
    assert decoded[0]["timing_ms"]["write_ms"] is None

    # 第 1/2 条：应为有限非负数（上一条写盘耗时）。
    for rec in decoded[1:]:
        v = rec["timing_ms"]["write_ms"]
        assert isinstance(v, (int, float))
        assert math.isfinite(float(v))
        assert float(v) >= 0.0

