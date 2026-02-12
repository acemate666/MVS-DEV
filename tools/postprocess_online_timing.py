"""在线 JSONL 后处理：统计每个 loop 内的耗时分解。

适用输入：tennis3d_online 在线输出的 records（默认写到 JSONL）。

重点字段（每条 record / loop）：
- out_rec['latency_host']：pipeline 内部耗时分解（毫秒）
  - align_ms：对齐/筛选耗时（通常很小）
  - detect_ms：检测耗时（通常最大）
  - localize_ms：多视角定位耗时
  - total_ms：pipeline 从 pipe_start 到 pipe_end 的总耗时
  - detect_ms_by_camera：每相机 detect 耗时（毫秒）
- out_rec['created_at'] 与 out_rec['capture_t_abs']：可估算“端到端滞后” lag_ms

注意：
- 若你开启了 JSONL 输出，output_loop 会把写盘开销落到 `timing_ms.write_ms`（毫秒）。
    该字段采用 1 条记录的延迟：第 N 条记录的 `write_ms` 代表“第 N-1 条写盘耗时”。
- 某些历史文件可能不是严格的一行一个 JSON（例如每条 record 被 pretty-print 成多行）；
  本脚本用增量 JSON 解码兼容这两种格式。

用法示例：
- uv run python tools/postprocess_online_timing.py --in data/tools_output/online_positions_3d.master_slave.jsonl

输出：
- 控制台打印汇总统计（p50/p90/p99 等）
- 可选导出 CSV（--out-csv）

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, cast


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


def _safe_int(x: object) -> int | None:
    try:
        if x is None:
            return None
        return int(cast(Any, x))
    except Exception:
        return None


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    """返回百分位数（近似：线性插值）。

    Args:
        sorted_vals: 已排序的非空列表。
        p: [0,100]
    """

    if not sorted_vals:
        return None
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])

    # 线性插值：位置落在 [0, n-1]
    n = len(sorted_vals)
    pos = (p / 100.0) * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _iter_json_objects(path: Path) -> Iterator[dict[str, Any]]:
    """从文件中迭代解析 JSON 对象，兼容：

    1) 标准 JSONL：每行一个 JSON（json.dumps 默认无缩进）。
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
                    # 缓冲不够（或刚好读到半个对象），继续读下一行
                    break

                # raw_decode 的 idx 基于 s（lstrip 后的字符串）
                consumed = len(buf) - len(s) + idx
                buf = buf[consumed:]

                if isinstance(obj, dict):
                    yield obj
                # 若不是 dict（理论上不会出现），直接忽略

        # 文件结束后：如果 buf 里还有残留非空白，尝试再解一次
        tail = buf.strip()
        if tail:
            try:
                obj, _ = decoder.raw_decode(tail)
            except json.JSONDecodeError:
                return
            if isinstance(obj, dict):
                yield obj


def _summarize(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    vs = [float(v) for v in values if math.isfinite(float(v))]
    if not vs:
        return None
    vs.sort()

    out: dict[str, float] = {
        "n": float(len(vs)),
        "min": float(vs[0]),
        "p50": float(_percentile(vs, 50.0) or vs[0]),
        "p90": float(_percentile(vs, 90.0) or vs[-1]),
        "p95": float(_percentile(vs, 95.0) or vs[-1]),
        "p99": float(_percentile(vs, 99.0) or vs[-1]),
        "max": float(vs[-1]),
        "mean": float(statistics.fmean(vs)),
    }
    return out


def _fmt_ms_stat(name: str, stat: dict[str, float] | None) -> str:
    if not stat:
        return f"{name}: <empty>"
    return (
        f"{name}: n={int(stat['n'])} "
        f"min={stat['min']:.2f} p50={stat['p50']:.2f} p90={stat['p90']:.2f} "
        f"p95={stat['p95']:.2f} p99={stat['p99']:.2f} max={stat['max']:.2f} mean={stat['mean']:.2f}"
    )


def _fmt_count_stat(name: str, stat: dict[str, float] | None) -> str:
    """格式化“计数类”统计（跳组/丢帧/队列等）。"""

    if not stat:
        return f"{name}: <empty>"
    return (
        f"{name}: n={int(stat['n'])} "
        f"min={stat['min']:.0f} p50={stat['p50']:.0f} p90={stat['p90']:.0f} "
        f"p95={stat['p95']:.0f} p99={stat['p99']:.0f} max={stat['max']:.0f} mean={stat['mean']:.2f}"
    )


def _iter_records(
    *,
    objs: Iterable[dict[str, Any]],
    start_group: int | None,
    end_group: int | None,
) -> Iterator[dict[str, Any]]:
    for rec in objs:
        gi_raw = rec.get("group_index")
        try:
            gi = int(gi_raw) if gi_raw is not None else None
        except Exception:
            gi = None

        if gi is not None and start_group is not None and gi < int(start_group):
            continue
        if gi is not None and end_group is not None and gi > int(end_group):
            continue
        yield rec


def main() -> int:
    ap = argparse.ArgumentParser(description="Postprocess tennis3d_online timing JSONL")
    ap.add_argument("--in", dest="in_path", required=True, help="input JSONL path")
    ap.add_argument("--out-csv", default="", help="optional output CSV path")
    ap.add_argument("--start-group", type=int, default=None, help="optional group_index lower bound")
    ap.add_argument("--end-group", type=int, default=None, help="optional group_index upper bound")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="optional max records to process (0 = no limit)",
    )
    args = ap.parse_args()

    in_path = Path(str(args.in_path)).resolve()
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    out_csv = str(args.out_csv).strip()
    out_csv_path = Path(out_csv).resolve() if out_csv else None

    # 汇总
    align_ms: list[float] = []
    detect_ms: list[float] = []
    localize_ms: list[float] = []
    total_ms: list[float] = []
    lag_ms: list[float] = []
    write_ms: list[float] = []

    # latest-only（可选）：跳组/丢帧量化
    latest_only_skipped_groups: list[float] = []
    latest_only_skipped_total_max: int | None = None

    # 软触发（可选）
    # - send->arrival：更接近“触发下发到帧到达应用层”的端到端延迟（含曝光/读出/传输/SDK）
    # - send->group_ready：还会包含主机侧排队/积压（backlog）
    send_to_arrival_median_ms: list[float] = []
    send_to_group_ready_ms: list[float] = []

    # 组包侧 backlog（可选）：group_ready - arrival_median。
    # 该值越大，说明 frame_queue/assembler/pipeline 存在积压，处理已落后于采集。
    group_ready_minus_arrival_median_ms: list[float] = []

    # MVS 组包器诊断（可选）
    assembler_dropped_groups: list[float] = []
    assembler_pending_groups: list[float] = []

    send_to_arrival_by_cam_ms: dict[str, list[float]] = defaultdict(list)

    det_by_cam_ms: dict[str, list[float]] = defaultdict(list)

    # 采集侧“每相机 event”统计（单位：ms）
    # - arrival_delta_ms：相机到达时刻相对该组到达中位数的偏差（越小越同步）
    # - ready_minus_arrival_ms：从该相机到达到组包 ready 的等待时间（越大可能表示排队/积压/迟到）
    # - time_mapping_delta_ms：该相机映射到 host_ms 后相对中位数的偏差
    arrival_delta_by_cam_ms: dict[str, list[float]] = defaultdict(list)
    ready_minus_arrival_by_cam_ms: dict[str, list[float]] = defaultdict(list)
    time_mapping_delta_by_cam_ms: dict[str, list[float]] = defaultdict(list)

    # 导出 CSV 时的列：相机列动态扩展
    rows: list[dict[str, Any]] = []
    cam_keys_seen: set[str] = set()

    objs = _iter_json_objects(in_path)
    records = _iter_records(
        objs=objs,
        start_group=args.start_group,
        end_group=args.end_group,
    )

    processed = 0
    for rec in records:
        lat = rec.get("latency_host")
        if not isinstance(lat, dict):
            continue

        gi = rec.get("group_index")
        balls = rec.get("balls") or []
        balls_n = len(balls) if isinstance(balls, list) else 0

        a = _safe_float(lat.get("align_ms"))
        d = _safe_float(lat.get("detect_ms"))
        l = _safe_float(lat.get("localize_ms"))
        t = _safe_float(lat.get("total_ms"))

        if a is not None:
            align_ms.append(a)
        if d is not None:
            detect_ms.append(d)
        if l is not None:
            localize_ms.append(l)
        if t is not None:
            total_ms.append(t)

        # 端到端滞后：created_at(epoch) - capture_t_abs(epoch)
        ca = _safe_float(rec.get("created_at"))
        ta = _safe_float(rec.get("capture_t_abs"))
        if ca is not None and ta is not None:
            lag_ms.append((ca - ta) * 1000.0)

        write_ms_cur = None
        tm = rec.get("timing_ms")
        if isinstance(tm, dict):
            w = _safe_float(tm.get("write_ms"))
            write_ms_cur = w
            if w is not None:
                write_ms.append(w)

        # 软触发：send -> arrival / ready
        s_arr_med = _safe_float(rec.get("soft_trigger_send_to_arrival_median_ms"))
        if s_arr_med is not None:
            send_to_arrival_median_ms.append(s_arr_med)

        s_ready = _safe_float(rec.get("soft_trigger_send_to_group_ready_ms"))
        if s_ready is not None:
            send_to_group_ready_ms.append(s_ready)

        s_by_cam = rec.get("soft_trigger_send_to_arrival_ms_by_camera")
        if isinstance(s_by_cam, dict):
            for k, v in s_by_cam.items():
                kk = str(k)
                cam_keys_seen.add(kk)
                fv = _safe_float(v)
                if fv is None:
                    continue
                send_to_arrival_by_cam_ms[kk].append(fv)

        dbc = lat.get("detect_ms_by_camera")
        if isinstance(dbc, dict):
            for k, v in dbc.items():
                fv = _safe_float(v)
                if fv is None:
                    continue
                kk = str(k)
                det_by_cam_ms[kk].append(fv)
                cam_keys_seen.add(kk)

        # 采集侧每相机 event
        arrival_by_cam = rec.get("capture_arrival_monotonic_by_camera")
        arrival_median = _safe_float(rec.get("capture_arrival_monotonic_median"))
        group_ready = _safe_float(rec.get("capture_group_ready_monotonic"))

        gr_minus_arr_med = _safe_float(rec.get("capture_group_ready_minus_arrival_median_ms"))
        if gr_minus_arr_med is not None:
            group_ready_minus_arrival_median_ms.append(gr_minus_arr_med)

        # latest-only：每条 record 的跳组数（若启用）
        sk = _safe_float(rec.get("latest_only_skipped_groups"))
        if sk is not None:
            latest_only_skipped_groups.append(sk)
        sk_tot_raw = rec.get("latest_only_skipped_groups_total")
        sk_tot = _safe_int(sk_tot_raw)
        if sk_tot is not None:
            latest_only_skipped_total_max = int(sk_tot)

        # 组包器诊断：dropped/pending
        ad = _safe_float(rec.get("mvs_assembler_dropped_groups"))
        if ad is not None:
            assembler_dropped_groups.append(ad)
        apn = _safe_float(rec.get("mvs_assembler_pending_groups"))
        if apn is not None:
            assembler_pending_groups.append(apn)
        if isinstance(arrival_by_cam, dict):
            for k, v in arrival_by_cam.items():
                kk = str(k)
                cam_keys_seen.add(kk)
                av = _safe_float(v)
                if av is None:
                    continue
                if arrival_median is not None:
                    arrival_delta_by_cam_ms[kk].append((av - arrival_median) * 1000.0)
                if group_ready is not None:
                    ready_minus_arrival_by_cam_ms[kk].append((group_ready - av) * 1000.0)

        # 时间映射：优先使用已经计算好的 delta_to_median（单位 ms）
        map_delta = rec.get("time_mapping_host_ms_delta_to_median_by_camera")
        if isinstance(map_delta, dict):
            for k, v in map_delta.items():
                kk = str(k)
                cam_keys_seen.add(kk)
                fv = _safe_float(v)
                if fv is None:
                    continue
                time_mapping_delta_by_cam_ms[kk].append(fv)

        row: dict[str, Any] = {
            "group_index": gi,
            "balls": balls_n,
            "align_ms": a,
            "detect_ms": d,
            "localize_ms": l,
            "total_ms": t,
            "lag_ms": (lag_ms[-1] if (ca is not None and ta is not None and lag_ms) else None),
            "write_ms": write_ms_cur,
            "send_to_arrival_median_ms": s_arr_med,
            "send_to_group_ready_ms": s_ready,
            "group_ready_minus_arrival_median_ms": gr_minus_arr_med,
            "latest_only_enabled": bool(rec.get("latest_only_enabled", False)),
            "latest_only_skipped_groups": _safe_int(rec.get("latest_only_skipped_groups")),
            "latest_only_skipped_groups_total": _safe_int(rec.get("latest_only_skipped_groups_total")),
            "mvs_assembler_dropped_groups": _safe_int(rec.get("mvs_assembler_dropped_groups")),
            "mvs_assembler_pending_groups": _safe_int(rec.get("mvs_assembler_pending_groups")),
        }
        if isinstance(dbc, dict):
            for kk in cam_keys_seen:
                # 先填 None，后面统一补齐
                if f"det_cam_ms_{kk}" not in row:
                    row[f"det_cam_ms_{kk}"] = None
            for k, v in dbc.items():
                fv = _safe_float(v)
                if fv is None:
                    continue
                row[f"det_cam_ms_{str(k)}"] = fv

        # 采集侧列（若存在则填；不存在保持 None）
        if isinstance(arrival_by_cam, dict) and arrival_median is not None:
            for kk in cam_keys_seen:
                row.setdefault(f"arrival_delta_ms_{kk}", None)
                row.setdefault(f"ready_minus_arrival_ms_{kk}", None)
            for k, v in arrival_by_cam.items():
                av = _safe_float(v)
                if av is None:
                    continue
                kk = str(k)
                row[f"arrival_delta_ms_{kk}"] = (av - arrival_median) * 1000.0
                if group_ready is not None:
                    row[f"ready_minus_arrival_ms_{kk}"] = (group_ready - av) * 1000.0

        if isinstance(map_delta, dict):
            for kk in cam_keys_seen:
                row.setdefault(f"time_mapping_delta_ms_{kk}", None)
            for k, v in map_delta.items():
                fv = _safe_float(v)
                if fv is None:
                    continue
                row[f"time_mapping_delta_ms_{str(k)}"] = fv

        # 软触发列（逐相机 send->arrival）
        if isinstance(s_by_cam, dict):
            for kk in cam_keys_seen:
                row.setdefault(f"send_to_arrival_ms_{kk}", None)
            for k, v in s_by_cam.items():
                fv = _safe_float(v)
                if fv is None:
                    continue
                row[f"send_to_arrival_ms_{str(k)}"] = fv

        rows.append(row)

        processed += 1
        if int(args.limit or 0) > 0 and processed >= int(args.limit):
            break

    print(f"input: {in_path}")
    print(f"records_processed: {processed}")

    print(_fmt_ms_stat("align_ms", _summarize(align_ms)))
    print(_fmt_ms_stat("detect_ms", _summarize(detect_ms)))
    print(_fmt_ms_stat("localize_ms", _summarize(localize_ms)))
    print(_fmt_ms_stat("total_ms", _summarize(total_ms)))
    print(_fmt_ms_stat("lag_ms", _summarize(lag_ms)))
    print(_fmt_ms_stat("write_ms", _summarize(write_ms)))

    print(_fmt_ms_stat("send_to_arrival_median_ms", _summarize(send_to_arrival_median_ms)))
    print(_fmt_ms_stat("send_to_group_ready_ms", _summarize(send_to_group_ready_ms)))
    print(_fmt_ms_stat("group_ready_minus_arrival_median_ms", _summarize(group_ready_minus_arrival_median_ms)))

    # latest-only 汇总（若启用）：
    # - skipped_groups：每条 record 跳过了多少个已就绪的组
    # - skipped_total_max：总跳组（来自 latest_only_skipped_groups_total 的最大值）
    if latest_only_skipped_groups:
        print(_fmt_count_stat("latest_only_skipped_groups", _summarize(latest_only_skipped_groups)))
        if latest_only_skipped_total_max is not None:
            processed_plus_skipped = int(processed) + int(latest_only_skipped_total_max)
            skip_ratio = (
                float(latest_only_skipped_total_max) / float(processed_plus_skipped)
                if processed_plus_skipped > 0
                else 0.0
            )
            print(
                "latest_only_summary: "
                f"skipped_total={int(latest_only_skipped_total_max)} "
                f"processed={int(processed)} "
                f"skip_ratio={skip_ratio:.3f}"
            )

    if assembler_dropped_groups:
        print(_fmt_count_stat("mvs_assembler_dropped_groups", _summarize(assembler_dropped_groups)))
    if assembler_pending_groups:
        print(_fmt_count_stat("mvs_assembler_pending_groups", _summarize(assembler_pending_groups)))

    if det_by_cam_ms:
        print("detect_ms_by_camera:")
        for cam in sorted(det_by_cam_ms.keys()):
            stat = _summarize(det_by_cam_ms[cam])
            print("  " + _fmt_ms_stat(cam, stat))

    if arrival_delta_by_cam_ms:
        print("capture_arrival_delta_ms_by_camera (arrival - median):")
        for cam in sorted(arrival_delta_by_cam_ms.keys()):
            stat = _summarize(arrival_delta_by_cam_ms[cam])
            print("  " + _fmt_ms_stat(cam, stat))

    if ready_minus_arrival_by_cam_ms:
        print("capture_ready_minus_arrival_ms_by_camera (group_ready - arrival):")
        for cam in sorted(ready_minus_arrival_by_cam_ms.keys()):
            stat = _summarize(ready_minus_arrival_by_cam_ms[cam])
            print("  " + _fmt_ms_stat(cam, stat))

    if time_mapping_delta_by_cam_ms:
        print("time_mapping_delta_ms_by_camera (mapped_host_ms - median):")
        for cam in sorted(time_mapping_delta_by_cam_ms.keys()):
            stat = _summarize(time_mapping_delta_by_cam_ms[cam])
            print("  " + _fmt_ms_stat(cam, stat))

    if send_to_arrival_by_cam_ms:
        print("soft_trigger_send_to_arrival_ms_by_camera:")
        for cam in sorted(send_to_arrival_by_cam_ms.keys()):
            stat = _summarize(send_to_arrival_by_cam_ms[cam])
            print("  " + _fmt_ms_stat(cam, stat))

    if out_csv_path is not None:
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # 补齐所有行的相机列，保证 CSV 列一致
        cams_sorted = sorted(cam_keys_seen)
        cam_cols = [f"det_cam_ms_{k}" for k in cams_sorted]
        cam_cols += [f"arrival_delta_ms_{k}" for k in cams_sorted]
        cam_cols += [f"ready_minus_arrival_ms_{k}" for k in cams_sorted]
        cam_cols += [f"time_mapping_delta_ms_{k}" for k in cams_sorted]
        cam_cols += [f"send_to_arrival_ms_{k}" for k in cams_sorted]
        fieldnames = [
            "group_index",
            "balls",
            "align_ms",
            "detect_ms",
            "localize_ms",
            "total_ms",
            "lag_ms",
            "write_ms",
            "send_to_arrival_median_ms",
            "send_to_group_ready_ms",
            "group_ready_minus_arrival_median_ms",
            "latest_only_enabled",
            "latest_only_skipped_groups",
            "latest_only_skipped_groups_total",
            "mvs_assembler_dropped_groups",
            "mvs_assembler_pending_groups",
        ] + cam_cols

        for r in rows:
            for c in cam_cols:
                r.setdefault(c, None)

        with out_csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

        print(f"csv_written: {out_csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
