"""curve_v3 离线“实机/回放”评测脚本。

建议从 `packages/curve_v3` 目录运行：
    - 该脚本依赖 `curve_v3` 包可被 import（通常通过 uv/可编辑安装实现）。

功能：
    - 读取球轨迹 JSON
    - 用 curve_v3 拟合并对比不同 post-bounce 点数 N（0,3,5,7,...,all）
    - 评估水平拦截平面 y==0.9m（可配置）
    - 保存 summary.json / summary.csv
    - 可选生成分图（若本机安装 matplotlib）
"""

from __future__ import annotations

import argparse
from pathlib import Path

from curve_v3.config_yaml import load_curve_v3_config_yaml
from curve_v3.configs import CurveV3Config
from curve_v3.offline.real_device_eval import load_observations_from_json, run_post_bounce_subsampling_eval


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="curve-v3-real-device-eval")
    p.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="输入球轨迹 JSON（含 observations: [{t,x,y,z,conf?}, ...]）。",
    )
    p.add_argument(
        "--curve-config",
        type=str,
        default=None,
        help="curve_v3 的 YAML 配置路径（可选）。不提供则使用 CurveV3Config() 默认值。",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="./artifacts/real_device_eval",
        help="输出目录（默认：./artifacts/real_device_eval）。",
    )
    p.add_argument(
        "--plane-y",
        type=float,
        default=0.9,
        help="拦截平面高度 y==const（米），默认 0.9。",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="不生成图像（即使 matplotlib 已安装）。",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg: CurveV3Config
    if args.curve_config:
        cfg = load_curve_v3_config_yaml(Path(args.curve_config))
    else:
        cfg = CurveV3Config()

    meta, obs = load_observations_from_json(args.input_json)
    _ = meta  # 评测函数会把 time_base_abs 以 observations[0].t 为准，这里 meta 仅用于上层展示。

    run_post_bounce_subsampling_eval(
        observations=obs,
        cfg=cfg,
        target_plane_y_m=float(args.plane_y),
        out_dir=Path(args.out_dir),
        make_plots=not bool(args.no_plots),
    )

    print(f"OK: results saved to: {Path(args.out_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
