"""在线模式：records 消费与输出循环。"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from typing import Any

from .jsonl_writer import _JsonlBufferedWriter
from .spec import OnlineRunSpec
from .terminal_format import _format_all_balls_lines, _format_best_ball_line


def run_output_loop(
    *,
    records: Iterable[dict[str, Any]],
    jsonl_writer: _JsonlBufferedWriter | None,
    spec: OnlineRunSpec,
    get_groups_done: Callable[[], int],
) -> tuple[int, int]:
    """消费 records 并执行：计数、状态输出、JSONL 写盘、逐组打印。"""

    def _format_timing_line(
        *,
        out_rec: dict[str, Any],
        loop_wait_ms: float | None,
        loop_out_ms: float | None,
        loop_total_ms: float | None,
        out_write_ms: float | None,
        out_print_ms: float | None,
        out_timing_print_ms: float | None,
        loop_start_monotonic: float | None,
        record_ready_monotonic: float | None,
    ) -> str:
        """格式化单条 timing 行。

        说明：
            - 该输出用于在线排障/性能观察，默认关闭（--terminal-timing）。
            - pipeline 的分解耗时来自 out_rec['latency_host']。
            - output_loop 自己统计 write/print 的耗时，帮助区分“算得慢”还是“输出慢”。
        """

        gi = out_rec.get("group_index")
        balls = out_rec.get("balls") or []
        balls_n = len(balls) if isinstance(balls, list) else 0

        lat = out_rec.get("latency_host")
        align_ms = detect_ms = localize_ms = total_ms = None
        pipe_start = pipe_end = None
        det_by_cam: dict[str, float] | None = None
        if isinstance(lat, dict):
            raw_align = lat.get("align_ms")
            if raw_align is not None:
                try:
                    align_ms = float(raw_align)
                except Exception:
                    align_ms = None

            raw_det = lat.get("detect_ms")
            if raw_det is not None:
                try:
                    detect_ms = float(raw_det)
                except Exception:
                    detect_ms = None

            raw_loc = lat.get("localize_ms")
            if raw_loc is not None:
                try:
                    localize_ms = float(raw_loc)
                except Exception:
                    localize_ms = None

            raw_total = lat.get("total_ms")
            if raw_total is not None:
                try:
                    total_ms = float(raw_total)
                except Exception:
                    total_ms = None

            raw_pipe_start = lat.get("pipe_start_monotonic")
            if raw_pipe_start is not None:
                try:
                    pipe_start = float(raw_pipe_start)
                except Exception:
                    pipe_start = None

            raw_pipe_end = lat.get("pipe_end_monotonic")
            if raw_pipe_end is not None:
                try:
                    pipe_end = float(raw_pipe_end)
                except Exception:
                    pipe_end = None

            dbc = lat.get("detect_ms_by_camera")
            if isinstance(dbc, dict) and dbc:
                det_by_cam = {}
                for k, v in dbc.items():
                    try:
                        det_by_cam[str(k)] = float(v)
                    except Exception:
                        continue

        parts: list[str] = [f"timing: group={gi} balls={balls_n}"]

        # 闭环循环耗时：严格可加和。
        # - wait：output_loop 等待下一条 record（next(records) 的阻塞时间）
        # - out：output_loop 处理当前 record（写盘/打印/格式化/计数等）
        # - total = wait + out
        if loop_wait_ms is not None or loop_out_ms is not None or loop_total_ms is not None:
            w = float(loop_wait_ms) if loop_wait_ms is not None else 0.0
            o = float(loop_out_ms) if loop_out_ms is not None else 0.0
            t = float(loop_total_ms) if loop_total_ms is not None else (w + o)
            parts.append(f"loop_ms{{wait={w:.1f},out={o:.1f},total={t:.1f}}}")

        # pipeline 内部分解耗时
        pipe_parts: list[str] = []
        if align_ms is not None:
            pipe_parts.append(f"align={align_ms:.1f}")
        if detect_ms is not None:
            pipe_parts.append(f"det={detect_ms:.1f}")
        if localize_ms is not None:
            pipe_parts.append(f"loc={localize_ms:.1f}")
        if total_ms is not None:
            pipe_parts.append(f"total={total_ms:.1f}")
        if pipe_parts:
            parts.append("pipe_ms{" + ",".join(pipe_parts) + "}")

        # wait 进一步分解：pre_pipe + pipe_total + post_pipe（仅当时序落在同一等待区间时才有意义）。
        # 说明：
        # - 该分解用于解释：等待是卡在采集/齐组，还是卡在模型推理，还是卡在 curve stage 等后处理。
        # - 若出现 backlog（record 早已算好），pipe_end 可能早于 loop_start，此时 pre/post 会变为负值；
        #   这里做 max(0, ...) 夹紧，只用于“近实时”场景下的诊断。
        pre_pipe_ms = post_pipe_ms = None
        if (
            loop_start_monotonic is not None
            and record_ready_monotonic is not None
            and pipe_start is not None
            and pipe_end is not None
        ):
            try:
                pre_pipe_ms = 1000.0 * max(0.0, float(pipe_start) - float(loop_start_monotonic))
                post_pipe_ms = 1000.0 * max(0.0, float(record_ready_monotonic) - float(pipe_end))
            except Exception:
                pre_pipe_ms = None
                post_pipe_ms = None

        if pre_pipe_ms is not None or post_pipe_ms is not None:
            wait_inner: list[str] = []
            if pre_pipe_ms is not None:
                wait_inner.append(f"pre={float(pre_pipe_ms):.1f}")
            if total_ms is not None:
                wait_inner.append(f"pipe={float(total_ms):.1f}")
            if post_pipe_ms is not None:
                wait_inner.append(f"post={float(post_pipe_ms):.1f}")
            if wait_inner:
                parts.append("wait_split_ms{" + ",".join(wait_inner) + "}")

        # source 侧耗时（可选）：用于解释 wait_split_ms.pre 的来源。
        # 说明：
        # - source_get_group_ms：等待下一组完整组包（latest-only queue / get_next_group）的阻塞耗时
        # - source_decode_ms：像素格式转换/解码（frame_to_bgr）耗时
        src_inner: list[str] = []
        raw_get_ms = out_rec.get("source_get_group_ms")
        if raw_get_ms is not None:
            try:
                src_inner.append(f"get={float(raw_get_ms):.1f}")
            except Exception:
                pass
        raw_dec_ms = out_rec.get("source_decode_ms")
        if raw_dec_ms is not None:
            try:
                src_inner.append(f"decode={float(raw_dec_ms):.1f}")
            except Exception:
                pass
        if src_inner:
            parts.append("src_ms{" + ",".join(src_inner) + "}")

        # 软触发（可选）：从“下发 TriggerSoftware”到“帧到达应用层/组包 ready”的耗时。
        # 注意：
        # - 仅在启用了软触发发送事件且 source 将其并入 record 时才会出现。
        # - send->arrival 更接近“相机曝光+读出+传输+SDK 到达”的总延迟；
        #   send->group_ready 还会包含主机侧排队/积压（backlog）。
        send_seq = out_rec.get("soft_trigger_send_seq")
        send_to_arr_med_ms = out_rec.get("soft_trigger_send_to_arrival_median_ms")
        send_to_ready_ms = out_rec.get("soft_trigger_send_to_group_ready_ms")
        if send_seq is not None or send_to_arr_med_ms is not None or send_to_ready_ms is not None:
            send_inner: list[str] = []
            if send_seq is not None:
                try:
                    send_inner.append(f"seq={int(send_seq)}")
                except Exception:
                    pass
            if send_to_arr_med_ms is not None:
                try:
                    send_inner.append(f"arr_med={float(send_to_arr_med_ms):.1f}")
                except Exception:
                    pass
            if send_to_ready_ms is not None:
                try:
                    send_inner.append(f"ready={float(send_to_ready_ms):.1f}")
                except Exception:
                    pass
            if send_inner:
                parts.append("send_ms{" + ",".join(send_inner) + "}")

        # latest-only（可选）：量化“为了追新而跳过了多少组”。
        # 说明：
        # - skipped_groups 是“本次迭代 drain 丢弃的组数”；skipped_total 是累计值。
        # - 这些字段由在线 source 写入 meta，并在 pipeline 输出中被展开保留。
        if bool(out_rec.get("latest_only_enabled", False)):
            lo_inner: list[str] = []
            raw_sk = out_rec.get("latest_only_skipped_groups")
            if raw_sk is not None:
                try:
                    lo_inner.append(f"sk={int(raw_sk)}")
                except Exception:
                    pass
            raw_tot = out_rec.get("latest_only_skipped_groups_total")
            if raw_tot is not None:
                try:
                    lo_inner.append(f"tot={int(raw_tot)}")
                except Exception:
                    pass
            if lo_inner:
                parts.append("latest_only{" + ",".join(lo_inner) + "}")

        # 输出耗时（写盘/打印）
        out_parts: list[str] = []
        if out_write_ms is not None:
            out_parts.append(f"write={float(out_write_ms):.1f}")
        if out_print_ms is not None:
            out_parts.append(f"print={float(out_print_ms):.1f}")
        if out_timing_print_ms is not None:
            out_parts.append(f"timing={float(out_timing_print_ms):.1f}")
        if out_parts:
            parts.append("out_ms{" + ",".join(out_parts) + "}")

        if det_by_cam:
            # 说明：按 key 排序，保持输出稳定，便于 grep/对比。
            det_inner = ",".join(f"{k}:{det_by_cam[k]:.1f}" for k in sorted(det_by_cam.keys()))
            parts.append("det_cam_ms{" + det_inner + "}")

        # detector 输出规模（诊断）：用于区分 balls=0 的两类常见原因。
        # - det_n: detector 有输出，但几何/门控导致无法三角化
        # - det_n=0: 纯粹没有检出（像素格式/曝光/ROI/模型等问题）
        raw_tot = out_rec.get("detections_n_total")
        raw_tot_ms = out_rec.get("detections_n_total_min_score")
        raw_by_cam_ms = out_rec.get("detections_n_by_camera_min_score")
        detn_inner: list[str] = []
        if raw_tot is not None:
            try:
                detn_inner.append(f"raw={int(raw_tot)}")
            except Exception:
                pass
        if raw_tot_ms is not None:
            try:
                detn_inner.append(f"ms={int(raw_tot_ms)}")
            except Exception:
                pass
        if detn_inner:
            parts.append("det_n{" + ",".join(detn_inner) + "}")

        if isinstance(raw_by_cam_ms, dict) and raw_by_cam_ms:
            # 只输出 min_score 过滤后的数量（更接近几何模块的有效输入）。
            detn_cam = {}
            for k, v in raw_by_cam_ms.items():
                try:
                    detn_cam[str(k)] = int(v)
                except Exception:
                    continue
            if detn_cam:
                detn_cam_inner = ",".join(f"{k}:{detn_cam[k]}" for k in sorted(detn_cam.keys()))
                parts.append("det_n_cam{" + detn_cam_inner + "}")

        return " ".join(parts)

    records_done = 0
    balls_done = 0

    if str(spec.terminal_print_mode) != "none" or float(spec.terminal_status_interval_s) > 0:
        print("Waiting for first ball observation...")

    last_status_t = time.monotonic()
    terminal_print_interval_s = float(getattr(spec, "terminal_print_interval_s", 0.0) or 0.0)
    last_record_print_t: float | None = None
    last_status_records = 0
    last_status_groups = 0
    last_status_capture_host_ms: int | None = None
    last_status_group_index: int | None = None
    last_status_sk_tot: int | None = None

    # status 区间统计：用于回答“哪部分影响 proc_fps / cap_fps”。
    # 说明：
    # - 这些统计是按 status 心跳窗口（terminal_status_interval_s）聚合的“均值”。
    # - 目的不是严格 profiling，而是快速判断瓶颈属于：source 等待/解码、pipeline 推理、curve 后处理、输出写盘等。
    sum_loop_wait_ms = 0.0
    sum_src_get_ms = 0.0
    sum_src_decode_ms = 0.0
    sum_pipe_total_ms = 0.0
    sum_pipe_det_ms = 0.0
    sum_curve_ms = 0.0
    it = iter(records)
    prev_write_ms: float | None = None

    while True:
        # 关键：把“等待下一条 record 的阻塞时间”显式计入闭环。
        loop_start = time.monotonic()
        try:
            out_rec = next(it)
        except StopIteration:
            break
        record_ready = time.monotonic()

        loop_wait_ms = 1000.0 * max(0.0, float(record_ready) - float(loop_start))
        sum_loop_wait_ms += float(loop_wait_ms)

        # 从 record 中提取 source/pipeline/curve 的诊断耗时（若存在）。
        try:
            v = out_rec.get("source_get_group_ms")
            if v is not None:
                sum_src_get_ms += float(v)
        except Exception:
            pass

        try:
            v = out_rec.get("source_decode_ms")
            if v is not None:
                sum_src_decode_ms += float(v)
        except Exception:
            pass

        lat = out_rec.get("latency_host")
        if isinstance(lat, dict):
            try:
                v = lat.get("total_ms")
                if v is not None:
                    sum_pipe_total_ms += float(v)
            except Exception:
                pass
            try:
                v = lat.get("detect_ms")
                if v is not None:
                    sum_pipe_det_ms += float(v)
            except Exception:
                pass

        tm = out_rec.get("timing_ms")
        if isinstance(tm, dict):
            try:
                v = tm.get("curve_stage_ms")
                if v is not None:
                    sum_curve_ms += float(v)
            except Exception:
                pass

        records_done += 1
        balls = out_rec.get("balls") or []
        if isinstance(balls, list):
            balls_done += int(len(balls))

        if float(spec.terminal_status_interval_s) > 0:
            now = time.monotonic()
            if (now - last_status_t) >= float(spec.terminal_status_interval_s):
                dt_s = max(now - last_status_t, 1e-9)
                rec_delta = records_done - last_status_records

                groups_done = int(get_groups_done())
                grp_delta = groups_done - last_status_groups

                cap_host_ms = out_rec.get("capture_host_timestamp")
                cap_fps = None
                if cap_host_ms is not None:
                    try:
                        cap_host_ms_i = int(cap_host_ms)
                        if last_status_capture_host_ms is not None:
                            dms = cap_host_ms_i - last_status_capture_host_ms
                            if dms > 0:
                                # cap_fps 口径说明：
                                # - 非 latest-only：按 groups_done 与 capture_host_timestamp 估算，基本等同于“完整组包产出率”。
                                # - latest-only：record 的 group_index 会跳变（因为跳组），
                                #   直接用 group_index / capture_host_timestamp 容易出现“看起来 100fps”的错觉。
                                #   这里优先用 wall-clock dt_s 与（处理组 + 跳过组）的数量来估计真实采集吞吐。
                                if bool(out_rec.get("latest_only_enabled", False)):
                                    sk_tot = out_rec.get("latest_only_skipped_groups_total")
                                    sk_tot_i = int(sk_tot) if sk_tot is not None else 0
                                    prev_sk = int(last_status_sk_tot) if last_status_sk_tot is not None else 0
                                    cap_groups_delta = int(rec_delta) + int(max(0, sk_tot_i - prev_sk))
                                    if dt_s > 1e-9:
                                        cap_fps = float(cap_groups_delta) / float(dt_s)
                                    last_status_sk_tot = int(sk_tot_i)
                                else:
                                    cap_fps = 1000.0 * float(grp_delta) / float(dms)
                        last_status_capture_host_ms = cap_host_ms_i
                    except Exception:
                        pass

                proc_fps = float(rec_delta) / dt_s

                loop_ms_avg = None
                if rec_delta > 0:
                    loop_ms_avg = 1000.0 * dt_s / float(rec_delta)

                loop_ms_part = ""
                if loop_ms_avg is not None:
                    loop_ms_part += f" loop_avg~{loop_ms_avg:.1f}ms"
                # 说明：loop_wait 是“等待下一条 record”的耗时；对用户更有意义的是窗口均值。
                avg_loop_wait_ms = float(sum_loop_wait_ms) / float(max(1, rec_delta))
                loop_ms_part += f" loop_wait~{avg_loop_wait_ms:.1f}ms"

                lag_ms = None
                try:
                    ca = out_rec.get("created_at")
                    ta = out_rec.get("capture_t_abs")
                    if ca is not None and ta is not None:
                        lag_ms = (float(ca) - float(ta)) * 1000.0
                except Exception:
                    lag_ms = None

                cap_part = f" cap_fps~{cap_fps:.2f}" if cap_fps is not None else ""
                lag_part = f" lag~{lag_ms:.0f}ms" if lag_ms is not None else ""

                # latest-only 量化：累计跳组与占比（用于在线验收，避免“看起来实时但其实丢很多组”）。
                lo_part = ""
                if bool(out_rec.get("latest_only_enabled", False)):
                    try:
                        sk_tot = out_rec.get("latest_only_skipped_groups_total")
                        sk_tot_i = int(sk_tot) if sk_tot is not None else 0
                        denom = float(int(records_done) + int(sk_tot_i))
                        ratio = float(sk_tot_i) / denom if denom > 0 else 0.0
                        lo_part = f" latest_only{{sk_tot={sk_tot_i},skip_ratio~{ratio:.3f}}}"
                    except Exception:
                        lo_part = " latest_only{enabled}"

                # 瓶颈拆分：按 status 窗口输出均值，帮助理解 proc_fps 被哪些部分限制。
                # 注意：latest-only 预解码场景下，decode 可能与上一组 detect 跨组重叠，因此这些均值
                # 更适合用于“哪里花得多”，不应机械相加当作理论下限。
                breakdown = ""
                if rec_delta > 0:
                    src_get_avg = float(sum_src_get_ms) / float(rec_delta)
                    src_dec_avg = float(sum_src_decode_ms) / float(rec_delta)
                    pipe_tot_avg = float(sum_pipe_total_ms) / float(rec_delta)
                    pipe_det_avg = float(sum_pipe_det_ms) / float(rec_delta)
                    curve_avg = float(sum_curve_ms) / float(rec_delta)
                    # 仅在有数据时才输出，避免噪声。
                    parts_ms: list[str] = []
                    if src_get_avg > 0.0 or src_dec_avg > 0.0:
                        parts_ms.append(f"src(get={src_get_avg:.1f},dec={src_dec_avg:.1f})")
                    if pipe_tot_avg > 0.0 or pipe_det_avg > 0.0:
                        parts_ms.append(f"pipe(det={pipe_det_avg:.1f},tot={pipe_tot_avg:.1f})")
                    if curve_avg > 0.0:
                        parts_ms.append(f"curve={curve_avg:.1f}")
                    if parts_ms:
                        breakdown = " breakdown_ms{" + ",".join(parts_ms) + "}"

                print(
                    f"status: groups={groups_done} records={records_done} balls={balls_done} "
                    f"proc_fps~{proc_fps:.2f}{cap_part}{lag_part}{loop_ms_part}{lo_part}{breakdown}"
                )

                last_status_t = now
                last_status_records = records_done
                last_status_groups = groups_done

                # 重置窗口累计（下一次 status 从 0 重新积）。
                sum_loop_wait_ms = 0.0
                sum_src_get_ms = 0.0
                sum_src_decode_ms = 0.0
                sum_pipe_total_ms = 0.0
                sum_pipe_det_ms = 0.0
                sum_curve_ms = 0.0

        # 逐组终端输出节流：限制 best/all 逐组打印与 timing 行的频率，避免刷屏。
        # 说明：
        # - 使用 record_ready（已在本循环中采样的 monotonic 时间戳）作为节流时钟，避免额外 time.monotonic 调用。
        # - 仅影响逐组打印与 timing；不影响 status 心跳行。
        allow_record_print = True
        if terminal_print_interval_s > 0:
            wants_record_print = str(spec.terminal_print_mode) != "none" or bool(
                getattr(spec, "terminal_timing", False)
            )
            if wants_record_print and last_record_print_t is not None:
                if (float(record_ready) - float(last_record_print_t)) < float(terminal_print_interval_s):
                    allow_record_print = False

        # 说明：若开启 --terminal-timing，则额外统计 output_loop 自身耗时。
        # - write_ms：jsonl_writer.write_line 的耗时
        # - print_ms：终端格式化 + print 的耗时
        out_write_ms: float | None = None
        out_print_ms: float | None = None
        loop_out_ms: float | None = None
        loop_total_ms: float | None = None

        if jsonl_writer is not None:
            if not (spec.out_jsonl_only_when_balls and (not isinstance(balls, list) or not balls)):
                timing_ms = out_rec.get("timing_ms")
                if not isinstance(timing_ms, dict):
                    timing_ms = {}
                    out_rec["timing_ms"] = timing_ms

                # 说明：write_ms 表示“上一条成功写入 JSONL 的序列化+写入+flush 耗时”。
                # 由于 JSONL 是追加写入，无法在写完当前行后再回填当前行的 write_ms；
                # 因此这里采用 1 条记录的延迟（上一条写盘耗时写到下一条记录里）。
                if "write_ms" not in timing_ms:
                    timing_ms["write_ms"] = prev_write_ms

                t0 = time.monotonic()
                jsonl_writer.write_line(json.dumps(out_rec, ensure_ascii=False))
                dt_ms = 1000.0 * max(0.0, time.monotonic() - t0)
                prev_write_ms = dt_ms
                if bool(getattr(spec, "terminal_timing", False)):
                    out_write_ms = dt_ms

        did_print_record = False

        if allow_record_print and str(spec.terminal_print_mode) != "none":
            t0 = time.monotonic() if bool(getattr(spec, "terminal_timing", False)) else None
            if str(spec.terminal_print_mode) == "all":
                lines = _format_all_balls_lines(out_rec)
                if lines:
                    for ln in lines:
                        print(ln)
                    did_print_record = True
            else:
                line = _format_best_ball_line(out_rec)
                if line is not None:
                    print(line)
                    did_print_record = True
            if t0 is not None:
                out_print_ms = 1000.0 * max(0.0, time.monotonic() - t0)

        if allow_record_print and bool(getattr(spec, "terminal_timing", False)):
            # timing 行本身的 print 不纳入 loop_out_ms（否则需要“打印完再回填”，会导致双行输出）。
            # 这样 loop_ms{wait,out,total} 始终严格可加和，且一条 record 只输出一行 timing。
            out_end = time.monotonic()
            loop_out_ms = 1000.0 * max(0.0, float(out_end) - float(record_ready))
            loop_total_ms = 1000.0 * max(0.0, float(out_end) - float(loop_start))

            print(
                _format_timing_line(
                    out_rec=out_rec,
                    loop_wait_ms=loop_wait_ms,
                    loop_out_ms=loop_out_ms,
                    loop_total_ms=loop_total_ms,
                    out_write_ms=out_write_ms,
                    out_print_ms=out_print_ms,
                    out_timing_print_ms=None,
                    loop_start_monotonic=float(loop_start),
                    record_ready_monotonic=float(record_ready),
                )
            )
            did_print_record = True

        if did_print_record and terminal_print_interval_s > 0:
            last_record_print_t = float(record_ready)

    return records_done, balls_done
