from __future__ import annotations

"""最小可视化：curve corridor + interception。

用途：
- 从在线/离线定位输出的 JSONL 中，提取：
  - curve.track_updates[*].v3.corridor_on_planes_y（走廊候选：target_y + crossing_prob 等）
  - curve.track_updates[*].interception（拦截点：target 与 diag）
- 输出两张简单图：
  1) 最近一次走廊的 crossing_prob vs target_y
  2) 随时间的 interception 目标 y（仅 valid=true 且存在 target 时）

说明：
- 该脚本是“工具脚本”，不属于稳定库 API。
- matplotlib 为可选依赖：若未安装，会给出提示并退出（不自动安装）。

示例：
- uv run python tools/plot_curve_corridor_and_interception.py --in data/tools_output/online_positions_3d.jsonl --out-dir data/tools_output/curve_plots
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterator, cast


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
        # 说明：x 来自反序列化 JSON，运行期通常是 str/int/float。
        # 这里显式 cast，避免静态类型检查误报。
        v = float(cast(Any, x))
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def _choose_time_axis(rec: dict[str, Any]) -> float | None:
    """选择可用时间轴（epoch 秒），用于画 time-series。"""

    curve = rec.get("curve")
    if isinstance(curve, dict):
        t_abs = _safe_float(curve.get("t_abs"))
        if t_abs is not None:
            return t_abs

    t = _safe_float(rec.get("capture_t_abs"))
    if t is not None:
        return t

    return _safe_float(rec.get("created_at"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot curve corridor + interception from JSONL")
    ap.add_argument("--in", dest="in_path", default="data/tools_output/online_positions_3d.master_slave.jsonl", help="input JSONL path")
    ap.add_argument("--out-dir", default="", help="output directory (default: <input_parent>/curve_plots)")
    ap.add_argument("--track-id", type=int, default=None, help="optional track_id filter")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="optional max records to scan (0 = no limit; scans from start)",
    )
    args = ap.parse_args()

    in_path = Path(str(args.in_path)).resolve()
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    out_dir = Path(str(args.out_dir)).resolve() if str(args.out_dir).strip() else (in_path.parent / "curve_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print("缺少可选依赖 matplotlib，无法画图。")
        print(f"导入失败：{exc}")
        print("如需使用本脚本，请安装 matplotlib（建议用 uv add --group dev matplotlib）。")
        return 2

    # interception time-series
    ts: list[float] = []
    target_y: list[float] = []

    # last corridor snapshot
    last_corridor: list[dict[str, Any]] | None = None
    last_corridor_track: int | None = None
    last_corridor_t: float | None = None

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

        t_axis = _choose_time_axis(rec)

        for tu in track_updates:
            if not isinstance(tu, dict):
                continue
            tid_raw = tu.get("track_id")
            tid = None
            try:
                tid = int(tid_raw) if tid_raw is not None else None
            except Exception:
                tid = None

            if args.track_id is not None and tid is not None and tid != int(args.track_id):
                continue
            if args.track_id is not None and tid is None:
                continue

            # corridor
            v3 = tu.get("v3")
            if isinstance(v3, dict):
                corridor = v3.get("corridor_on_planes_y")
                if isinstance(corridor, list) and corridor:
                    # 取“最近一次出现 corridor 的快照”
                    last_corridor = [c for c in corridor if isinstance(c, dict)]
                    last_corridor_track = tid
                    last_corridor_t = t_axis

            # interception
            inter = tu.get("interception")
            if isinstance(inter, dict):
                valid = inter.get("valid")
                if valid is True:
                    tgt = inter.get("target")
                    if isinstance(tgt, dict):
                        y = _safe_float(tgt.get("y"))
                        if y is not None and t_axis is not None:
                            ts.append(float(t_axis))
                            target_y.append(float(y))

    if last_corridor is None:
        print("未找到 corridor_on_planes_y（可能尚未形成 bounce_event 或未启用 curve）。")
    else:
        ys: list[float] = []
        ps: list[float] = []
        valids: list[bool] = []

        for c in last_corridor:
            y = _safe_float(c.get("target_y"))
            p = _safe_float(c.get("crossing_prob"))
            if y is None or p is None:
                continue
            ys.append(float(y))
            ps.append(float(p))
            valids.append(bool(c.get("is_valid", False)))

        if ys and ps:
            # 按 y 排序，曲线更直观
            order = sorted(range(len(ys)), key=lambda i: ys[i])
            ys_s = [ys[i] for i in order]
            ps_s = [ps[i] for i in order]
            val_s = [valids[i] for i in order]

            plt.figure(figsize=(10, 4))
            plt.plot(ys_s, ps_s, "-o", linewidth=1.5, markersize=3)
            # 标注 invalid 点
            for y, p, ok in zip(ys_s, ps_s, val_s):
                if not ok:
                    plt.plot([y], [p], "x", color="red", markersize=6)
            title_parts = ["corridor crossing_prob vs target_y"]
            if last_corridor_track is not None:
                title_parts.append(f"track={last_corridor_track}")
            if last_corridor_t is not None:
                title_parts.append(f"t_abs~{last_corridor_t:.3f}")
            plt.title(" ".join(title_parts))
            plt.xlabel("target_y (m)")
            plt.ylabel("crossing_prob")
            plt.grid(True, alpha=0.3)
            out_png = out_dir / "corridor_crossing_prob.png"
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close()
            print(f"已写出：{out_png}")
        else:
            print("corridor_on_planes_y 存在，但缺少可用的 target_y/crossing_prob。")

    if not ts:
        print("未找到 valid interception target（可能未启用 interception 或尚未产生有效目标）。")
    else:
        plt.figure(figsize=(10, 4))
        plt.plot(ts, target_y, "-o", linewidth=1.5, markersize=3)
        plt.title("interception target_y over time (valid=true)")
        plt.xlabel("t_abs (epoch s)")
        plt.ylabel("target_y (m)")
        plt.grid(True, alpha=0.3)
        out_png = out_dir / "interception_target_y.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"已写出：{out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
