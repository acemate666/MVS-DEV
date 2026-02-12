from __future__ import annotations

"""轨迹可视化：online_positions_3d*.jsonl。

用途：
- 从 `data/tools_output/online_positions_3d.master_slave.jsonl`（或类似 JSONL）提取：
  - curve.track_updates[*].last_pos（按 track_id 聚合的 3D 轨迹）
  - curve.track_updates[*].episode（若启用 episode 输出，则用时间轴背景块标注 episode 区间）
  - curve.track_updates[*].v3.diagnostics.prefit_freeze.cut_index（标注“prefit 使用了哪些点”）

输出：
- 对每个 track_id 输出一张 PNG：
  - y(t) 轨迹（可选：t_abs / t_rel / index）
  - x-z 平面投影
  - 标注 prefit cut_index（prefit 区间：index <= cut_index）
  - （若可用）标注 predicted_land_time_abs / predicted_land_point

说明：
- 该脚本属于 tools/ 下的工具脚本，不是稳定库 API。
- matplotlib 为可选依赖：未安装时会给出提示并退出（不自动安装）。

示例：
- uv run python tools/plot_online_positions_3d_trajectory.py --track-id 1
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, cast


@dataclass
class _Point3:
    t_abs: float | None
    x: float
    y: float
    z: float


@dataclass
class _FreezeEvent:
    t_abs_snapshot: float | None
    cut_index: int | None
    is_frozen: bool | None
    freeze_reason: str | None
    freeze_t_abs: float | None


@dataclass
class _EpisodeInfo:
    episode_id: int | None
    active: bool | None
    start_t_abs: float | None
    end_t_abs: float | None
    end_reason: str | None


@dataclass
class _TrackSeries:
    track_id: int

    # 说明：obs_index = n_obs - 1。这里用字典是为了兼容 n_obs 可能出现跳跃的情况。
    obs_by_index: dict[int, _Point3] = field(default_factory=dict)

    # 原始观测点（若从 assignments + balls[*].ball_3d_world 能拿到）。
    raw_obs: list[_Point3] = field(default_factory=list)

    freeze_events: list[_FreezeEvent] = field(default_factory=list)
    episodes: list[_EpisodeInfo] = field(default_factory=list)

    # 取“最后一次更新”的预测信息用于标注。
    predicted_land_time_abs: float | None = None
    predicted_land_point: tuple[float, float, float] | None = None


def _iter_json_objects(path: Path) -> Iterator[dict[str, Any]]:
    """从文件中迭代解析 JSON 对象，兼容：

    1) 标准 JSONL：每行一个 JSON。
    2) 非标准：每条 JSON 被 pretty-print 成多行，且多条对象顺序拼接。

    实现：增量 raw_decode，遇到 JSONDecodeError 就继续读更多字符。
    """

    decoder = json.JSONDecoder()
    buf = ""

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            buf += line
            while True:
                s = buf.lstrip()
                if not s:
                    buf = ""
                    break
                try:
                    obj, idx = decoder.raw_decode(s)
                except json.JSONDecodeError:
                    break

                consumed = len(buf) - len(s) + idx
                buf = buf[consumed:]

                if isinstance(obj, dict):
                    yield obj

        tail = buf.strip()
        if tail:
            try:
                obj, _ = decoder.raw_decode(tail)
            except json.JSONDecodeError:
                return
            if isinstance(obj, dict):
                yield obj


def _safe_float(x: object) -> float | None:
    try:
        if x is None:
            return None
        v = float(cast(Any, x))
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _safe_int(x: object) -> int | None:
    try:
        if x is None:
            return None
        return int(cast(Any, x))
    except Exception:
        return None


def _choose_time_axis(rec: dict[str, Any]) -> float | None:
    """选择可用的 epoch 秒（优先 curve.t_abs）。"""

    curve = rec.get("curve")
    if isinstance(curve, dict):
        t_abs = _safe_float(curve.get("t_abs"))
        if t_abs is not None:
            return t_abs

    t = _safe_float(rec.get("capture_t_abs"))
    if t is not None:
        return t

    return _safe_float(rec.get("created_at"))


def _parse_point3(t_abs: float | None, xyz: object) -> _Point3 | None:
    if not (isinstance(xyz, list) and len(xyz) == 3):
        return None
    x = _safe_float(xyz[0])
    y = _safe_float(xyz[1])
    z = _safe_float(xyz[2])
    if x is None or y is None or z is None:
        return None
    return _Point3(t_abs=t_abs, x=float(x), y=float(y), z=float(z))


def _pick_final_cut_index(track: _TrackSeries) -> tuple[int | None, str | None]:
    """从冻结事件中挑一个用于标注的 cut_index。

    约定：优先取最后一次出现的（也更接近“最终使用的 prefit 范围”）。
    """

    for ev in reversed(track.freeze_events):
        if ev.cut_index is not None:
            return ev.cut_index, ev.freeze_reason
    return None, None


def _nearest_index_by_time(track: _TrackSeries, t_abs: float) -> int | None:
    best_i: int | None = None
    best_dt: float | None = None
    for i, p in track.obs_by_index.items():
        if p.t_abs is None:
            continue
        dt = abs(float(p.t_abs) - float(t_abs))
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best_i = i
    return best_i


def _apply_y_transform(y: float, y_negate: bool, y_offset: float) -> float:
    if y_negate:
        y = -y
    return y + float(y_offset)


def _plot_track(
    track: _TrackSeries,
    out_png: Path,
    *,
    x_axis: str,
    show_raw_obs: bool,
    y_negate: bool,
    y_offset: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print("缺少可选依赖 matplotlib，无法画图。")
        print(f"导入失败：{exc}")
        print("如需使用本脚本，请安装 matplotlib（建议用 uv add --group dev matplotlib）。")
        raise SystemExit(2)

    indices = sorted(track.obs_by_index.keys())
    if not indices:
        print(f"track={track.track_id}: 未收集到 last_pos 轨迹点，跳过。")
        return

    t0_abs: float | None = None
    for i in indices:
        t0_abs = track.obs_by_index[i].t_abs
        if t0_abs is not None:
            break

    def x_of(i: int) -> float:
        if x_axis == "index":
            return float(i)
        if x_axis in {"t_abs", "t_rel"}:
            t = track.obs_by_index[i].t_abs
            if t is None:
                # 说明：缺失时间时退化为索引轴，避免画出 NaN。
                return float(i)
            if x_axis == "t_abs":
                return float(t)
            if t0_abs is None:
                return float(i)
            return float(t - t0_abs)
        raise ValueError(f"unknown x_axis={x_axis!r}")

    xs = [x_of(i) for i in indices]
    ys = [_apply_y_transform(track.obs_by_index[i].y, y_negate, y_offset) for i in indices]
    xs_x = [track.obs_by_index[i].x for i in indices]
    xs_z = [track.obs_by_index[i].z for i in indices]

    cut_index, cut_reason = _pick_final_cut_index(track)

    fig, (ax_y, ax_xz) = plt.subplots(1, 2, figsize=(13, 5))

    # y(t) / y(index)
    if cut_index is None:
        ax_y.plot(xs, ys, "-o", linewidth=1.5, markersize=3, label="last_pos")
    else:
        pre_x: list[float] = []
        pre_y: list[float] = []
        post_x: list[float] = []
        post_y: list[float] = []
        for i, x, y in zip(indices, xs, ys):
            if i <= cut_index:
                pre_x.append(float(x))
                pre_y.append(float(y))
            else:
                post_x.append(float(x))
                post_y.append(float(y))

        if pre_x:
            ax_y.plot(pre_x, pre_y, "-o", linewidth=1.5, markersize=3, label="prefit 使用点")
        if post_x:
            ax_y.plot(post_x, post_y, "-o", linewidth=1.0, markersize=3, color="0.6", label="prefit 之后点")

        # cut_index 垂线：优先用“cut_index 对应的 x 坐标”。
        x_cut: float | None = None
        if x_axis == "index":
            x_cut = float(cut_index)
        else:
            p = track.obs_by_index.get(int(cut_index))
            if p is not None and p.t_abs is not None:
                x_cut = float(p.t_abs) if x_axis == "t_abs" else float(p.t_abs - cast(float, t0_abs))
        if x_cut is not None:
            label = f"cut_index={cut_index}"
            if cut_reason:
                label += f" ({cut_reason})"
            ax_y.axvline(x_cut, color="C3", linestyle="--", linewidth=1.2, label=label)

    ax_y.set_title(f"track={track.track_id} y vs {x_axis}")
    ax_y.set_xlabel(x_axis)
    ax_y.set_ylabel("y (m)")
    ax_y.grid(True, alpha=0.3)
    ax_y.legend(loc="best")

    # episode 背景块（只对时间轴更自然；index 轴也尽量映射）
    for ep in track.episodes:
        if ep.start_t_abs is None:
            continue
        end = ep.end_t_abs
        if end is None and ep.active is True:
            # 仍在进行中的 episode：用最后一个点的时间。
            last_t = None
            for i in reversed(indices):
                last_t = track.obs_by_index[i].t_abs
                if last_t is not None:
                    break
            end = last_t
        if end is None:
            continue

        if x_axis in {"t_abs", "t_rel"}:
            x0 = float(ep.start_t_abs)
            x1 = float(end)
            if x_axis == "t_rel" and t0_abs is not None:
                x0 -= float(t0_abs)
                x1 -= float(t0_abs)
        else:
            i0 = _nearest_index_by_time(track, float(ep.start_t_abs))
            i1 = _nearest_index_by_time(track, float(end))
            if i0 is None or i1 is None:
                continue
            x0 = float(min(i0, i1))
            x1 = float(max(i0, i1))

        ax_y.axvspan(x0, x1, color="C2", alpha=0.08)

    # predicted land time（仅对时间轴画竖线；否则信息量不大）
    if track.predicted_land_time_abs is not None and x_axis in {"t_abs", "t_rel"} and t0_abs is not None:
        x_land = float(track.predicted_land_time_abs)
        if x_axis == "t_rel":
            x_land -= float(t0_abs)
        ax_y.axvline(x_land, color="C1", linestyle=":", linewidth=1.2, label="predicted_land_time")

    # x-z 平面
    if cut_index is None:
        ax_xz.plot(xs_x, xs_z, "-o", linewidth=1.2, markersize=3, label="last_pos")
    else:
        pre_xz_x: list[float] = []
        pre_xz_z: list[float] = []
        post_xz_x: list[float] = []
        post_xz_z: list[float] = []
        for i, x, z in zip(indices, xs_x, xs_z):
            if i <= cut_index:
                pre_xz_x.append(float(x))
                pre_xz_z.append(float(z))
            else:
                post_xz_x.append(float(x))
                post_xz_z.append(float(z))
        if pre_xz_x:
            ax_xz.plot(pre_xz_x, pre_xz_z, "-o", linewidth=1.2, markersize=3, label="prefit 使用点")
        if post_xz_x:
            ax_xz.plot(post_xz_x, post_xz_z, "-o", linewidth=0.9, markersize=3, color="0.6", label="prefit 之后点")

    if show_raw_obs and track.raw_obs:
        raw_x = [p.x for p in track.raw_obs]
        raw_z = [p.z for p in track.raw_obs]
        ax_xz.scatter(raw_x, raw_z, s=8, alpha=0.6, label="raw ball_3d_world")

    if track.predicted_land_point is not None:
        lx, ly, lz = track.predicted_land_point
        ax_xz.scatter([lx], [lz], marker="*", s=120, color="C1", label="predicted_land_point")

    ax_xz.set_title(f"track={track.track_id} x-z")
    ax_xz.set_xlabel("x (m)")
    ax_xz.set_ylabel("z (m)")
    ax_xz.grid(True, alpha=0.3)
    ax_xz.axis("equal")
    ax_xz.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"已写出：{out_png}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot trajectory + episode + prefit cut_index from online_positions_3d JSONL")
    ap.add_argument(
        "--in",
        dest="in_path",
        default="data/tools_output/online points.jsonl",
        help="input JSONL path",
    )
    ap.add_argument("--out-dir", default="", help="output directory (default: <input_parent>/trajectory_plots)")
    ap.add_argument("--track-id", type=int, default=None, help="optional track_id filter")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="optional max records to scan (0 = no limit; scans from start)",
    )
    ap.add_argument(
        "--x-axis",
        choices=["t_abs", "t_rel", "index"],
        default="t_rel",
        help="x axis for y-plot",
    )
    ap.add_argument(
        "--show-raw-obs",
        action="store_true",
        help="also plot raw balls[*].ball_3d_world (if assignments are present)",
    )
    ap.add_argument(
        "--max-tracks",
        type=int,
        default=50,
        help="safety cap when track-id is not specified",
    )
    ap.add_argument(
        "--y-negate",
        action="store_true",
        help="flip y axis (some captures may have inverted y)",
    )
    ap.add_argument(
        "--y-offset",
        type=float,
        default=0.0,
        help="add constant offset to y (after optional negate)",
    )
    args = ap.parse_args()

    in_path = Path(str(args.in_path)).resolve()
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    out_dir = Path(str(args.out_dir)).resolve() if str(args.out_dir).strip() else (in_path.parent / "trajectory_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks: dict[int, _TrackSeries] = {}

    n = 0
    for rec in _iter_json_objects(in_path):
        n += 1
        if int(args.limit or 0) > 0 and n > int(args.limit):
            break

        curve = rec.get("curve")
        if not isinstance(curve, dict):
            continue
        track_updates = curve.get("track_updates")
        if not isinstance(track_updates, list):
            continue

        # assignments: ball_id -> track_id
        assignments = curve.get("assignments")
        ball_to_track: dict[int, int] = {}
        if isinstance(assignments, list):
            for a in assignments:
                if not isinstance(a, dict):
                    continue
                bid = _safe_int(a.get("ball_id"))
                tid = _safe_int(a.get("track_id"))
                if bid is None or tid is None:
                    continue
                ball_to_track[int(bid)] = int(tid)

        balls = rec.get("balls")
        t_rec_abs = _choose_time_axis(rec)
        if isinstance(balls, list):
            for bid, b in enumerate(balls):
                if not isinstance(b, dict):
                    continue
                tid = ball_to_track.get(int(bid))
                if tid is None:
                    continue
                if args.track_id is not None and tid != int(args.track_id):
                    continue
                p = _parse_point3(t_rec_abs, b.get("ball_3d_world"))
                if p is None:
                    continue
                if tid not in tracks:
                    tracks[tid] = _TrackSeries(track_id=int(tid))
                tracks[tid].raw_obs.append(p)

        for tu in track_updates:
            if not isinstance(tu, dict):
                continue
            tid = _safe_int(tu.get("track_id"))
            if tid is None:
                continue
            if args.track_id is not None and tid != int(args.track_id):
                continue

            if tid not in tracks:
                tracks[tid] = _TrackSeries(track_id=int(tid))
            tr = tracks[tid]

            n_obs = _safe_int(tu.get("n_obs"))
            last_t_abs = _safe_float(tu.get("last_t_abs"))
            if last_t_abs is None:
                last_t_abs = t_rec_abs
            last_pos = _parse_point3(last_t_abs, tu.get("last_pos"))
            if n_obs is not None and n_obs > 0 and last_pos is not None:
                obs_index = int(n_obs - 1)
                tr.obs_by_index[obs_index] = last_pos

            # episode（可能被禁用；禁用时该字段可能缺失）
            ep = tu.get("episode")
            if isinstance(ep, dict):
                epi = _EpisodeInfo(
                    episode_id=_safe_int(ep.get("episode_id")),
                    active=cast(bool | None, ep.get("active")) if isinstance(ep.get("active"), bool) else None,
                    start_t_abs=_safe_float(ep.get("start_t_abs")),
                    end_t_abs=_safe_float(ep.get("end_t_abs")),
                    end_reason=cast(str | None, ep.get("end_reason")) if isinstance(ep.get("end_reason"), str) else None,
                )
                tr.episodes.append(epi)

            # v3 diagnostics（prefit_freeze）
            v3 = tu.get("v3")
            if isinstance(v3, dict):
                tr.predicted_land_time_abs = _safe_float(v3.get("predicted_land_time_abs"))
                land_pt = v3.get("predicted_land_point")
                if isinstance(land_pt, list) and len(land_pt) == 3:
                    lx = _safe_float(land_pt[0])
                    ly = _safe_float(land_pt[1])
                    lz = _safe_float(land_pt[2])
                    if lx is not None and ly is not None and lz is not None:
                        tr.predicted_land_point = (float(lx), float(ly), float(lz))

                time_base_abs = _safe_float(v3.get("time_base_abs"))
                diag = v3.get("diagnostics")
                if isinstance(diag, dict):
                    pf = diag.get("prefit_freeze")
                    if isinstance(pf, dict):
                        freeze_t_rel = _safe_float(pf.get("freeze_t_rel"))
                        freeze_t_abs = None
                        if time_base_abs is not None and freeze_t_rel is not None:
                            freeze_t_abs = float(time_base_abs + freeze_t_rel)
                        tr.freeze_events.append(
                            _FreezeEvent(
                                t_abs_snapshot=last_t_abs,
                                cut_index=_safe_int(pf.get("cut_index")),
                                is_frozen=cast(bool | None, pf.get("is_frozen")) if isinstance(pf.get("is_frozen"), bool) else None,
                                freeze_reason=cast(str | None, pf.get("freeze_reason")) if isinstance(pf.get("freeze_reason"), str) else None,
                                freeze_t_abs=freeze_t_abs,
                            )
                        )

        if args.track_id is None and len(tracks) > int(args.max_tracks):
            print(f"track 数量超过上限 {int(args.max_tracks)}，请用 --track-id 或调大 --max-tracks。")
            break

    if not tracks:
        print("未找到任何 track_updates。")
        return 0

    # 去重 episode（同一条记录里会不断重复输出 episode 字段；这里按 (id,start,end,active) 简单去重）
    for tr in tracks.values():
        uniq: dict[tuple[object, ...], _EpisodeInfo] = {}
        for ep in tr.episodes:
            key = (ep.episode_id, ep.active, ep.start_t_abs, ep.end_t_abs, ep.end_reason)
            uniq[key] = ep
        tr.episodes = list(uniq.values())

    for tid in sorted(tracks.keys()):
        tr = tracks[tid]
        out_png = out_dir / f"trajectory_track_{tid}.png"
        _plot_track(
            tr,
            out_png,
            x_axis=str(args.x_axis),
            show_raw_obs=bool(args.show_raw_obs),
            y_negate=bool(args.y_negate),
            y_offset=float(args.y_offset),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
