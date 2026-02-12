"""把在线输出的 3D 点按时序画成曲线/散点图。

用途：
- 从 online 输出的 JSONL/连续 JSON 流里读取每条 record 的 balls[*].ball_3d_world。
- 按时间轴（默认 capture_t_abs）绘制 X/Y/Z 三条序列，便于观察抖动、趋势与异常点。

说明：
- 该脚本做了一个兼容：支持“标准 JSONL（一行一个 JSON）”以及“pretty print 的多行 JSON（多个对象拼接）”。
- 大文件可能点数极多，默认会做数量上限保护，避免一次性把终端/内存/绘图卡死。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


_TIME_FIELD = Literal["capture_t_abs", "created_at", "group_index"]
_COLOR_BY = Literal["none", "track", "ball_id"]


@dataclass(frozen=True, slots=True)
class _Point3D:
    """单个 3D 点观测（用于绘图）。"""

    t: float
    x: float
    y: float
    z: float
    group_index: int | None
    ball_id: int | None
    track_id: int | None


def _iter_json_objects(path: Path) -> Iterator[dict[str, Any]]:
    """从文件中按顺序解析出一个个 JSON object。

    兼容两种格式：
    1) 标准 JSONL：每行一个 JSON object。
    2) 连续 JSON 流：多个 JSON object 以空白分隔，且每个 object 可能是多行 pretty print。

    Raises:
        RuntimeError: 当文件末尾存在无法解析的残留内容时。
    """

    decoder = json.JSONDecoder()
    buf = ""

    with path.open("r", encoding="utf-8", errors="strict") as f:
        while True:
            chunk = f.read(1 << 16)
            if not chunk:
                break

            buf += chunk
            while True:
                s = buf.lstrip()
                if not s:
                    buf = ""
                    break

                try:
                    obj, idx = decoder.raw_decode(s)
                except json.JSONDecodeError:
                    # 说明：buffer 里可能只有半截 JSON，继续读更多内容。
                    buf = s
                    break

                if not isinstance(obj, dict):
                    # 说明：当前线上输出记录应当是 dict；遇到其它类型说明文件格式不符合预期。
                    raise RuntimeError(f"expected JSON object (dict), got: {type(obj).__name__}")

                yield obj
                buf = s[idx:]

    tail = buf.strip()
    if tail:
        # 说明：EOF 后仍有残留但无法构成完整 JSON。
        raise RuntimeError("trailing non-whitespace content after last JSON object")


def _pick_time(rec: dict[str, Any], *, field: _TIME_FIELD, fallback_index: int) -> float:
    """从 record 中取时间轴值。"""

    if field == "group_index":
        gi = rec.get("group_index")
        try:
            if gi is None:
                return float(fallback_index)
            return float(int(gi))
        except Exception:
            return float(fallback_index)

    v = rec.get(field)
    if v is None:
        # 说明：缺字段时回退到 record 的顺序索引。
        return float(fallback_index)

    try:
        return float(v)
    except Exception:
        return float(fallback_index)


def _extract_points(
    *,
    path: Path,
    time_field: _TIME_FIELD,
    max_points: int,
    stride: int,
    min_num_views: int,
) -> list[_Point3D]:
    """从在线输出文件抽取 3D 点序列。"""

    if stride < 1:
        raise ValueError("stride must be >= 1")

    pts: list[_Point3D] = []

    rec_i = -1
    for rec_i, rec in enumerate(_iter_json_objects(path)):
        if rec_i % stride != 0:
            continue

        t = _pick_time(rec, field=time_field, fallback_index=rec_i)

        balls = rec.get("balls") or []
        if not isinstance(balls, list) or not balls:
            continue

        gi = None
        try:
            gi_raw = rec.get("group_index")
            if gi_raw is not None:
                gi = int(gi_raw)
        except Exception:
            gi = None

        for b in balls:
            if not isinstance(b, dict):
                continue

            nv = b.get("num_views")
            if nv is not None:
                try:
                    if int(nv) < int(min_num_views):
                        continue
                except Exception:
                    pass

            xw = b.get("ball_3d_world")
            if not isinstance(xw, (list, tuple)) or len(xw) != 3:
                continue

            try:
                x, y, z = float(xw[0]), float(xw[1]), float(xw[2])
            except Exception:
                continue

            ball_id = None
            try:
                bid_raw = b.get("ball_id")
                if bid_raw is not None:
                    ball_id = int(bid_raw)
            except Exception:
                ball_id = None

            track_id = None
            try:
                tid_raw = b.get("curve_track_id")
                if tid_raw is not None:
                    track_id = int(tid_raw)
            except Exception:
                track_id = None

            pts.append(
                _Point3D(
                    t=float(t),
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    group_index=gi,
                    ball_id=ball_id,
                    track_id=track_id,
                )
            )

            if max_points > 0 and len(pts) >= max_points:
                return pts

    return pts


def _plot_xyz_time_series(
    *,
    pts: list[_Point3D],
    out_path: Path | None,
    title: str,
    time_field: _TIME_FIELD,
    time_relative: bool,
    color_by: _COLOR_BY,
    connect: bool,
) -> None:
    """绘制 X/Y/Z vs t。"""

    if not pts:
        raise RuntimeError("no points to plot")

    # 说明：按输入顺序通常已经是时序；这里不强制 sort，避免大文件额外开销。
    t0 = pts[0].t
    ts = [(p.t - t0) if time_relative else p.t for p in pts]

    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    zs = [p.z for p in pts]

    # 说明：仅当需要保存且不显示时，使用 Agg 后端避免无显示环境报错。
    import matplotlib

    if out_path is not None:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 10), sharex=True)

    def _time_axis_label(*, time_field: _TIME_FIELD, time_relative: bool) -> str:
        """生成英文时间轴标签。

        说明：
        - 为避免对单位做错误假设，绝对时间场景保留字段名作为提示。
        - 相对时间默认按秒理解（t - t0）。
        """

        if time_relative:
            return "Time (s from start)"
        if time_field == "group_index":
            return "Group index"
        return f"Time ({time_field})"

    t_label = _time_axis_label(time_field=time_field, time_relative=time_relative)

    def _color_key(p: _Point3D) -> int | None:
        if color_by == "track":
            return p.track_id
        if color_by == "ball_id":
            return p.ball_id
        return None

    keys = [_color_key(p) for p in pts]
    uniq_keys = sorted({k for k in keys if k is not None})

    if color_by == "none" or not uniq_keys:
        # 单色绘制
        for ax, vv, name in zip(axes, (xs, ys, zs), ("X", "Y", "Z"), strict=True):
            if connect:
                ax.plot(ts, vv, linewidth=0.8)
            else:
                ax.scatter(ts, vv, s=4)
            ax.set_ylabel(f"{name} (m)")
            ax.grid(True, linewidth=0.3)
    else:
        # 按 key 分组着色（track_id 或 ball_id）
        cmap = plt.get_cmap("tab20")
        key_to_color = {k: cmap(i % 20) for i, k in enumerate(uniq_keys)}

        # 为了避免每个点一条 scatter（太慢），按 key 先聚合 index。
        key_to_idx: dict[int, list[int]] = {k: [] for k in uniq_keys}
        for i, k in enumerate(keys):
            if k is None:
                continue
            key_to_idx[int(k)].append(i)

        def _legend_key_name(*, color_by: _COLOR_BY) -> str:
            if color_by == "track":
                return "track_id"
            if color_by == "ball_id":
                return "ball_id"
            return "key"

        legend_key_name = _legend_key_name(color_by=color_by)

        for ax, series, name in zip(axes, (xs, ys, zs), ("X", "Y", "Z"), strict=True):
            for k in uniq_keys:
                idx = key_to_idx.get(int(k)) or []
                if not idx:
                    continue
                t_k = [ts[i] for i in idx]
                v_k = [series[i] for i in idx]
                label = f"{legend_key_name}={k}"
                if connect:
                    ax.plot(t_k, v_k, linewidth=0.8, color=key_to_color[int(k)], label=label)
                else:
                    ax.scatter(t_k, v_k, s=6, color=key_to_color[int(k)], label=label)

            ax.set_ylabel(f"{name} (m)")
            ax.grid(True, linewidth=0.3)

        axes[0].legend(loc="best", fontsize=8)

    axes[-1].set_xlabel(t_label)
    fig.suptitle(title)
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"saved plot: {out_path}")
    else:
        plt.show()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot online points (x/y/z) over time")
    p.add_argument(
        "--jsonl",
        default="data/tools_output/online points.jsonl",
        help="input file path (JSONL or concatenated JSON objects)",
    )
    p.add_argument(
        "--time-field",
        choices=["capture_t_abs", "created_at", "group_index"],
        default="capture_t_abs",
        help="time axis field",
    )
    p.add_argument(
        "--abs-time",
        action="store_true",
        help="use absolute time as x-axis (default: relative time t-t0)",
    )
    p.add_argument(
        "--color-by",
        choices=["none", "track", "ball_id"],
        default="track",
        help="color points by curve_track_id or ball_id",
    )
    p.add_argument(
        "--connect",
        action="store_true",
        help="connect points with lines (instead of scatter)",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=200_000,
        help="max points to plot (0 = unlimited)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="sample every N records (>=1)",
    )
    p.add_argument(
        "--min-num-views",
        type=int,
        default=0,
        help="ignore balls with num_views < N (0 = disable)",
    )
    p.add_argument(
        "--out",
        default="temp/online_points_xyz.png",
        help="output PNG path (empty means show interactive window)",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    in_path = Path(str(args.jsonl)).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    time_field = str(args.time_field)
    if time_field not in ("capture_t_abs", "created_at", "group_index"):
        raise ValueError(f"invalid --time-field: {time_field}")

    time_relative = not bool(args.abs_time)

    out_raw = str(args.out or "").strip()
    out_path = Path(out_raw).expanduser().resolve() if out_raw else None

    pts = _extract_points(
        path=in_path,
        time_field=time_field,  # type: ignore[arg-type]
        max_points=int(args.max_points),
        stride=int(args.stride),
        min_num_views=int(args.min_num_views),
    )

    print(f"points: {len(pts)}")

    _plot_xyz_time_series(
        pts=pts,
        out_path=out_path,
        title=f"online points: {in_path.name} (n={len(pts)})",
        time_field=time_field,  # type: ignore[arg-type]
        time_relative=time_relative,
        color_by=str(args.color_by),  # type: ignore[arg-type]
        connect=bool(args.connect),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
