"""在线输入源：从 MVS `QuadCapture` 产出每组图像。

说明：
- 该模块属于在线 app/adapter 层，因此放在 `tennis3d_online` 包内。
- `tennis3d-core` 目标是“纯算法库”，不应再直接依赖 `mvs`。

职责：
- 从 `cap.get_next_group()` 持续读取同步组包。
- 将每帧转换为 OpenCV BGR 图像。
- 计算组级别的时间轴信息（capture_t_abs / capture_host_timestamp）。
- Optional：在线滑窗拟合 dev_timestamp -> host_ms（time_sync_mode=dev_timestamp_mapping）。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import queue
import threading
import time
from typing import Any, Iterator

import numpy as np

from mvs import MvsBinding, OnlineDevToHostMapper, QuadCapture
from mvs.capture.image import frame_to_bgr

from tennis3d.pipeline.time_utils import (
    delta_to_median_by_camera,
    host_timestamp_to_ms_epoch,
    host_timestamp_to_seconds,
    median_float,
    median_int,
    spread_ms,
)

__all__ = [
    "OnlineGroupWaitTimeout",
    "iter_mvs_image_groups",
]


class OnlineGroupWaitTimeout(RuntimeError):
    """在线采集在限定时间内未收到任何完整组包。"""


@dataclass(frozen=True)
class _RawQueuedGroup:
    """采集线程写入的 raw group（未解码）。

    说明：
    - group_index 表示“cap 侧看到的完整组序号”（0-based，包含被丢弃的组）。
    - group_ready_monotonic 在采集线程选定（drain 后的）最终 group 时记录。
      它不包含消费端/下游 backlog。
    """

    group_index: int
    group: list[Any]
    group_ready_monotonic: float


@dataclass(frozen=True)
class _DecodedQueuedGroup:
    """decode 线程写入的 decoded group（已转为 BGR）。"""

    group_index: int
    group: list[Any]
    group_ready_monotonic: float
    images_by_camera: dict[str, np.ndarray]
    source_decode_ms: float


class _LatestOnlyGroupPipeline:
    """latest-only：三段式流水线（capture -> decode -> consume）。

    设计目标：
    - capture 线程：只做 get_next_group + drain（尽量轻量），并把“最新 raw group”写入容量=1队列。
    - decode 线程：从 raw 队列取最新 raw group，解码为 BGR，再写入容量=1队列。
    - 主线程：从 decoded 队列取最新 decoded group 跑 pipeline。

    关键收益：
    - capture 不被 decode 拖慢，更及时地 drain QuadCapture 内部 backlog（避免 lag 线性增长）。
    - decode 与 detect/curve 跨组重叠并行，吞吐接近 max(decode,detect)。

    注意：
    - 为保持可退出性，所有阻塞点都使用小正 timeout 轮询。
    - decode 线程会在开始解码前尽力 drain raw 队列，只解码最新 raw group，避免“解码旧组”。
    """

    def __init__(self, *, cap: QuadCapture, binding: MvsBinding, poll_timeout_s: float) -> None:
        self._cap = cap
        self._binding = binding
        self._poll_timeout_s = float(poll_timeout_s)

        self._raw_q: "queue.Queue[_RawQueuedGroup]" = queue.Queue(maxsize=1)
        self._dec_q: "queue.Queue[_DecodedQueuedGroup]" = queue.Queue(maxsize=1)

        self._stop = threading.Event()
        self._err: BaseException | None = None

        self._capture_thread = threading.Thread(
            target=self._capture_run, name="latest_only_capture", daemon=True
        )
        self._decode_thread = threading.Thread(
            target=self._decode_run, name="latest_only_decode", daemon=True
        )

        self._next_group_index = 0

    def __enter__(self) -> "_LatestOnlyGroupPipeline":
        self._capture_thread.start()
        self._decode_thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    def close(self) -> None:
        self._stop.set()
        self._capture_thread.join(timeout=2.0)
        self._decode_thread.join(timeout=2.0)

    def _set_err(self, exc: BaseException) -> None:
        # 说明：记录首个异常即可；后续异常不覆盖，避免丢失根因。
        if self._err is None:
            self._err = exc

    @staticmethod
    def _put_overwrite(q: "queue.Queue[Any]", item: Any) -> None:
        """容量=1 覆盖写入：若满则丢弃旧 item，只保留最新。"""

        try:
            q.put_nowait(item)
            return
        except queue.Full:
            pass

        try:
            _ = q.get_nowait()
        except queue.Empty:
            pass

        try:
            q.put_nowait(item)
        except queue.Full:
            # 极端竞争条件下直接放弃：下一轮会再写入最新。
            pass

    def _capture_run(self) -> None:
        try:
            while not self._stop.is_set():
                cap_stop = getattr(self._cap, "stop_event", None)
                if cap_stop is not None:
                    try:
                        if bool(cap_stop.is_set()):
                            break
                    except Exception:
                        pass

                # 阻塞等到至少一个完整组。
                g0 = self._cap.get_next_group(timeout_s=float(self._poll_timeout_s))
                if g0 is None:
                    continue

                # 关键：尽力 drain QuadCapture 内部 backlog，只保留最后一个（最新）group。
                drain_timeout_s = float(min(self._poll_timeout_s, 0.01))
                if drain_timeout_s <= 0:
                    drain_timeout_s = 0.01

                chosen_group = g0
                chosen_ready = float(time.monotonic())
                chosen_index = int(self._next_group_index)
                self._next_group_index += 1

                drained = 0
                max_drain = 500
                while drained < max_drain and not self._stop.is_set():
                    g2 = self._cap.get_next_group(timeout_s=float(drain_timeout_s))
                    if g2 is None:
                        break
                    chosen_group = g2
                    chosen_ready = float(time.monotonic())
                    chosen_index = int(self._next_group_index)
                    self._next_group_index += 1
                    drained += 1

                item = _RawQueuedGroup(
                    group_index=int(chosen_index),
                    group=list(chosen_group),
                    group_ready_monotonic=float(chosen_ready),
                )
                self._put_overwrite(self._raw_q, item)
        except BaseException as exc:  # noqa: BLE001
            self._set_err(exc)

    def _decode_run(self) -> None:
        try:
            while not self._stop.is_set():
                if self._err is not None:
                    return

                # 轮询拿 raw item；这里用小 timeout 便于退出与传播异常。
                try:
                    raw_item = self._raw_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                # decode 前再 drain raw 队列：只解码最新 raw group，避免浪费在旧组上。
                latest = raw_item
                while True:
                    try:
                        latest = self._raw_q.get_nowait()
                    except queue.Empty:
                        break

                t_dec0 = time.monotonic()
                images_by_camera: dict[str, np.ndarray] = {}
                for fr in latest.group:
                    serial = str(getattr(fr, "serial", ""))
                    if not serial:
                        continue
                    cam_idx = int(getattr(fr, "cam_index"))
                    cam = self._cap.cameras[cam_idx].cam
                    images_by_camera[serial] = frame_to_bgr(binding=self._binding, cam=cam, frame=fr)
                dec_ms = float(1000.0 * max(0.0, time.monotonic() - t_dec0))

                dec_item = _DecodedQueuedGroup(
                    group_index=int(latest.group_index),
                    group=list(latest.group),
                    group_ready_monotonic=float(latest.group_ready_monotonic),
                    images_by_camera=images_by_camera,
                    source_decode_ms=float(dec_ms),
                )
                self._put_overwrite(self._dec_q, dec_item)
        except BaseException as exc:  # noqa: BLE001
            self._set_err(exc)

    def get_next(self, *, timeout_s: float) -> _DecodedQueuedGroup | None:
        """从 decoded 队列取一组。

        说明：
        - 返回的 item 已完成解码，可直接用于后续 detector。
        - 若 decode 跟不上采集，capacity=1 会导致中间组被跳过（符合 latest-only 语义）。
        """

        if self._err is not None:
            raise RuntimeError("latest-only 采集/解码线程异常") from self._err

        t = float(timeout_s)
        if t <= 0:
            t = 0.1

        try:
            item = self._dec_q.get(timeout=float(t))
            return item
        except queue.Empty:
            if self._err is not None:
                raise RuntimeError("latest-only 采集/解码线程异常") from self._err
            return None


def iter_mvs_image_groups(
    *,
    cap: QuadCapture,
    binding: MvsBinding,
    max_groups: int = 0,
    timeout_s: float = 0.5,
    max_wait_seconds: float = 0.0,
    latest_only: bool = False,
    time_sync_mode: str = "frame_host_timestamp",
    time_mapping_warmup_groups: int = 20,
    time_mapping_window_groups: int = 200,
    time_mapping_update_every_groups: int = 5,
    time_mapping_min_points: int = 20,
    time_mapping_hard_outlier_ms: float = 50.0,
) -> Iterator[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    """从在线 MVS `QuadCapture` 迭代读取每组图像。

    Args:
        cap: 已打开的 QuadCapture。
        binding: 已加载的 MVS binding（用于像素格式解码）。
        max_groups: 最多处理的组数（0 表示不限）。
        timeout_s: 等待 cap.get_next_group 的超时时间。
        max_wait_seconds: 超过该时长仍无任何组包则抛出 OnlineGroupWaitTimeout（0 表示不限）。
        latest_only: 是否启用“只处理最新完整组”（latest-only）。
            - 启用后：启动一个采集线程持续取完整组，并写入容量=1 的队列；消费端每次只会看到最新完整组。
            - 代价：会跳组/丢帧；输出 fps 仍受 pipeline 吞吐限制。
            - 收益：当处理慢于采集时，可显著抑制 backlog（保持新鲜度）。
        time_sync_mode: 时间轴策略（frame_host_timestamp / dev_timestamp_mapping）。

    Yields:
        (meta, images_by_camera_serial)
    """

    groups_yielded = 0

    # latest-only：
    # - last_yielded_group_index 记录“上一次交给 pipeline 的 cap 侧组序号”。
    # - skipped 的语义是“本次 yield 相比上次 yield 中间跳过了多少组”。
    latest_only_skipped_total = 0
    last_yielded_group_index = -1
    last_progress = time.monotonic()

    # 软触发发送事件（host_monotonic）缓存：
    # - 事件由 `mvs.capture.soft_trigger.SoftwareTriggerLoop` 写入 QuadCapture.event_queue。
    # - 在线 record 默认是“按 group 产出”，不会单独写事件流；这里把 send 事件尽力并入每个 group 的 meta。
    # - 匹配策略：对每个 group，选择“最后一个 host_monotonic <= arrival_median”的 send 事件。
    #   在近实时无 backlog 时，该策略能稳定对齐 send -> 该帧组。
    pending_soft_sends: deque[dict[str, Any]] = deque()

    time_sync_mode_norm = str(time_sync_mode).strip()

    mapper: OnlineDevToHostMapper | None = None
    if time_sync_mode_norm == "dev_timestamp_mapping":
        mapper = OnlineDevToHostMapper(
            warmup_groups=int(time_mapping_warmup_groups),
            window_groups=int(time_mapping_window_groups),
            update_every_groups=int(time_mapping_update_every_groups),
            min_points=int(time_mapping_min_points),
            hard_outlier_ms=float(time_mapping_hard_outlier_ms),
        )

    def _drain_soft_trigger_events() -> None:
        # 先尽力吸收一批事件，避免 event_queue 无界增长。
        # 说明：在线默认未订阅 camera_event，因此这里通常只会看到 soft_trigger_send。
        try:
            for ev in cap.drain_events(max_items=2000):
                if not isinstance(ev, dict):
                    continue
                if str(ev.get("type")) != "soft_trigger_send":
                    continue
                hm = ev.get("host_monotonic")
                if hm is None:
                    continue
                pending_soft_sends.append(dict(ev))
        except Exception:
            # 事件通道仅用于诊断；读取失败不影响主流程。
            pass

    def _convert_one_group(
        *,
        group: list[Any],
        current_group_index: int,
        group_ready_monotonic: float,
        prefetched_images_by_camera: dict[str, np.ndarray] | None,
        prefetched_decode_ms: float | None,
        latest_only_enabled: bool,
        latest_only_skipped_groups: int,
        latest_only_skipped_groups_total: int,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """把一组原始 frame 转换为 (meta, images_by_camera)。

        说明：
        - 这里不做 latest-only 的 skipped 计算；调用方应先算好并传入。
        - 返回 tuple，而不是在此处 yield，减少控制流复杂度。
        """
        # 说明：这里的耗时属于“source 侧开销”，不计入 pipeline 的 detect/localize。
        # 之所以要单独记录，是为了区分：
        # - 组包等待（get_next_group / latest-only queue wait）
        # - 像素格式转换/解码（frame_to_bgr）
        images_by_camera: dict[str, np.ndarray] = dict(prefetched_images_by_camera or {})
        host_ts_list: list[int] = []
        host_ms_by_camera: dict[str, float] = {}
        serials_in_group: list[str] = []

        # 采集侧诊断字段（主机单调时间 + 原始帧时间戳）。
        arrival_monotonic_by_camera: dict[str, float] = {}
        dev_ts_by_camera: dict[str, int] = {}
        host_ts_by_camera: dict[str, int] = {}

        # decode 统计口径：
        # - latest-only 下通常来自采集线程预解码（prefetched_decode_ms）。
        # - 若预解码缺失（例如极端竞争/单测 stub），会在此处补解码并把耗时累加。
        decode_ms_acc = float(prefetched_decode_ms) if prefetched_decode_ms is not None else 0.0
        t_decode0 = time.monotonic() if prefetched_decode_ms is None else None
        for fr in group:
            serial = str(fr.serial)
            serials_in_group.append(serial)

            # 优先使用采集线程预解码结果；缺失则回退到本线程解码。
            if serial not in images_by_camera:
                t0 = time.monotonic()
                bgr = frame_to_bgr(binding=binding, cam=cap.cameras[fr.cam_index].cam, frame=fr)
                images_by_camera[serial] = bgr
                decode_ms_acc += float(1000.0 * max(0.0, time.monotonic() - t0))

            try:
                arrival_monotonic_by_camera[serial] = float(getattr(fr, "arrival_monotonic"))
            except Exception:
                pass

            try:
                dev_ts_by_camera[serial] = int(getattr(fr, "dev_timestamp"))
            except Exception:
                pass

            try:
                host_ts_list.append(int(fr.host_timestamp))
            except Exception:
                pass

            try:
                host_ts_by_camera[serial] = int(getattr(fr, "host_timestamp"))
            except Exception:
                pass

            hm = host_timestamp_to_ms_epoch(getattr(fr, "host_timestamp", None))
            if hm is not None:
                host_ms_by_camera[serial] = float(hm)

            if mapper is not None:
                mapper.observe_pair(serial=serial, dev_ts=int(fr.dev_timestamp), host_ms=int(fr.host_timestamp))

        host_ts_med = median_int(host_ts_list)
        capture_t_abs = host_timestamp_to_seconds(host_ts_med) if host_ts_med is not None else None
        capture_t_source: str | None = "frame_host_timestamp" if capture_t_abs is not None else None

        mapped_ms_by_camera: dict[str, float] = {}

        if mapper is not None:
            mapper.on_group_end()

            mapped_ms_list: list[float] = []
            for fr in group:
                m = mapper.get_mapping(str(fr.serial))
                if m is None:
                    continue
                try:
                    ms = float(m.map_dev_to_host_ms(int(fr.dev_timestamp)))
                except Exception:
                    continue
                mapped_ms_list.append(ms)
                mapped_ms_by_camera[str(fr.serial)] = float(ms)

            mapped_ms_med = median_float(mapped_ms_list)
            if mapped_ms_med is not None:
                capture_t_abs = float(mapped_ms_med) / 1000.0
                capture_t_source = "dev_timestamp_mapping"

        meta: dict[str, Any] = {
            "group_index": int(current_group_index),
            "capture_t_abs": float(capture_t_abs) if capture_t_abs is not None else None,
            "capture_t_source": str(capture_t_source) if capture_t_source is not None else None,
            "capture_host_timestamp": int(host_ts_med) if host_ts_med is not None else None,
            "time_sync_mode": time_sync_mode_norm or None,
            # 主机侧诊断：组包完成的单调时间戳（秒）。
            "capture_group_ready_monotonic": float(group_ready_monotonic),
            # source 侧像素格式转换/解码耗时（毫秒）：
            # - latest-only 场景下通常来自采集线程预解码，但仍按“本组 decode 耗时”记录，便于诊断。
            "source_decode_ms": float(
                decode_ms_acc
                if t_decode0 is None
                else (1000.0 * max(0.0, time.monotonic() - float(t_decode0)))
            ),
        }

        # latest-only 量化字段：用于后处理统计“跳组/丢帧”程度。
        if latest_only_enabled:
            meta["latest_only_enabled"] = True
            meta["latest_only_skipped_groups"] = int(latest_only_skipped_groups)
            meta["latest_only_skipped_groups_total"] = int(latest_only_skipped_groups_total)

        # MVS 组包器诊断字段：便于区分“latest-only 主动跳组”与“assembler 因超时/上限丢组”。
        try:
            meta["mvs_assembler_dropped_groups"] = int(getattr(cap.assembler, "dropped_groups"))
        except Exception:
            pass
        try:
            meta["mvs_assembler_pending_groups"] = int(getattr(cap.assembler, "pending_groups"))
        except Exception:
            pass

        if arrival_monotonic_by_camera:
            meta["capture_arrival_monotonic_by_camera"] = dict(arrival_monotonic_by_camera)
            try:
                arr = list(arrival_monotonic_by_camera.values())
                if arr:
                    arr_sorted = sorted(arr)
                    arr_med = float(arr_sorted[len(arr_sorted) // 2])
                    meta["capture_arrival_monotonic_median"] = float(arr_med)
                    meta["capture_arrival_monotonic_spread_ms"] = float(1000.0 * (max(arr) - min(arr)))
                    meta["capture_group_ready_minus_arrival_median_ms"] = float(
                        1000.0 * (float(group_ready_monotonic) - float(arr_med))
                    )

                    # 软触发：把“下发时刻”并入该组 meta，用于端到端诊断。
                    send_ev: dict[str, Any] | None = None
                    try:
                        while pending_soft_sends:
                            hm = pending_soft_sends[0].get("host_monotonic")
                            if hm is None:
                                pending_soft_sends.popleft()
                                continue
                            if float(hm) <= float(arr_med):
                                send_ev = pending_soft_sends.popleft()
                                continue
                            break
                    except Exception:
                        send_ev = None

                    if send_ev is not None:
                        raw_hm = None
                        try:
                            raw_hm = send_ev.get("host_monotonic")
                        except Exception:
                            raw_hm = None

                        send_mon: float | None
                        if raw_hm is None:
                            send_mon = None
                        else:
                            try:
                                send_mon = float(raw_hm)
                            except Exception:
                                send_mon = None

                        if send_mon is not None:
                            meta["soft_trigger_send_monotonic"] = float(send_mon)
                            raw_seq = None
                            try:
                                raw_seq = send_ev.get("seq")
                            except Exception:
                                raw_seq = None
                            if raw_seq is not None:
                                try:
                                    meta["soft_trigger_send_seq"] = int(raw_seq)
                                except Exception:
                                    pass
                            try:
                                targets = send_ev.get("targets")
                                if isinstance(targets, list):
                                    meta["soft_trigger_send_targets"] = [str(x) for x in targets]
                            except Exception:
                                pass

                            # send -> arrival（逐相机 + 中位数）
                            try:
                                d_by_cam = {
                                    str(k): float(1000.0 * (float(v) - float(send_mon)))
                                    for k, v in arrival_monotonic_by_camera.items()
                                }
                                meta["soft_trigger_send_to_arrival_ms_by_camera"] = d_by_cam
                                d_list = list(d_by_cam.values())
                                if d_list:
                                    d_sorted = sorted(d_list)
                                    meta["soft_trigger_send_to_arrival_median_ms"] = float(
                                        d_sorted[len(d_sorted) // 2]
                                    )
                            except Exception:
                                pass

                            # send -> 组包 ready
                            try:
                                meta["soft_trigger_send_to_group_ready_ms"] = float(
                                    1000.0 * (float(group_ready_monotonic) - float(send_mon))
                                )
                            except Exception:
                                pass

                            # 可选：在 epoch 时间轴上给一个粗略差值（仅用于肉眼 sanity check）。
                            try:
                                send_wall = send_ev.get("created_at")
                                if send_wall is not None and host_ts_med is not None:
                                    meta["soft_trigger_send_to_capture_host_timestamp_ms"] = float(
                                        float(host_ts_med) - 1000.0 * float(send_wall)
                                    )
                            except Exception:
                                pass
            except Exception:
                pass

        if dev_ts_by_camera:
            meta["capture_frame_dev_timestamp_by_camera"] = {str(k): int(v) for k, v in dev_ts_by_camera.items()}

        if host_ts_by_camera:
            meta["capture_frame_host_timestamp_by_camera"] = {str(k): int(v) for k, v in host_ts_by_camera.items()}

        if host_ms_by_camera:
            meta["time_mapping_host_ms_by_camera"] = dict(host_ms_by_camera)
            meta["time_mapping_host_ms_spread_ms"] = spread_ms(list(host_ms_by_camera.values()))
            meta["time_mapping_host_ms_delta_to_median_by_camera"] = delta_to_median_by_camera(host_ms_by_camera)

        if mapper is not None:
            meta["time_mapping_groups_seen"] = int(mapper.groups_seen)
            meta["time_mapping_ready_count"] = int(mapper.ready_count(serials_in_group))
            worst_p95 = mapper.worst_p95_ms(serials_in_group)
            worst_rms = mapper.worst_rms_ms(serials_in_group)
            meta["time_mapping_worst_p95_ms"] = float(worst_p95) if worst_p95 is not None else None
            meta["time_mapping_worst_rms_ms"] = float(worst_rms) if worst_rms is not None else None

            if mapped_ms_by_camera:
                meta["time_mapping_mapped_host_ms_by_camera"] = dict(mapped_ms_by_camera)
                meta["time_mapping_mapped_host_ms_spread_ms"] = spread_ms(list(mapped_ms_by_camera.values()))
                meta["time_mapping_mapped_host_ms_delta_to_median_by_camera"] = delta_to_median_by_camera(
                    mapped_ms_by_camera
                )

        return meta, images_by_camera

    if latest_only:
        # 线程轮询 timeout：即使 timeout_s<=0，也使用一个小正数，保证线程可退出。
        poll_timeout_s = float(timeout_s)
        if poll_timeout_s <= 0:
            poll_timeout_s = 0.1

        with _LatestOnlyGroupPipeline(cap=cap, binding=binding, poll_timeout_s=poll_timeout_s) as latest_q:
            while True:
                if int(max_groups) > 0 and groups_yielded >= int(max_groups):
                    break

                _drain_soft_trigger_events()

                # 统计 source 等待“下一组完整组包”的耗时（毫秒）。
                t_get0 = time.monotonic()
                item = latest_q.get_next(timeout_s=float(timeout_s))
                t_get1 = time.monotonic()
                if item is None:
                    max_wait = float(max_wait_seconds)
                    if max_wait > 0 and (time.monotonic() - last_progress) > max_wait:
                        raise OnlineGroupWaitTimeout(
                            f"在线采集等待超时：超过 {max_wait:.3f}s 未收到任何完整组包。"
                        )
                    continue

                last_progress = time.monotonic()
                # 只处理最新完整组：队列保证 item 永远是“当下最新”。
                current_group_index = int(item.group_index)
                skipped = int(current_group_index - int(last_yielded_group_index) - 1)
                if last_yielded_group_index < 0:
                    skipped = int(current_group_index)
                if skipped < 0:
                    skipped = 0
                latest_only_skipped_total += int(skipped)
                last_yielded_group_index = int(current_group_index)

                meta, images_by_camera = _convert_one_group(
                    group=item.group,
                    current_group_index=int(current_group_index),
                    group_ready_monotonic=float(item.group_ready_monotonic),
                    prefetched_images_by_camera=dict(item.images_by_camera),
                    prefetched_decode_ms=float(item.source_decode_ms),
                    latest_only_enabled=True,
                    latest_only_skipped_groups=int(skipped),
                    latest_only_skipped_groups_total=int(latest_only_skipped_total),
                )
                meta["source_get_group_ms"] = float(1000.0 * max(0.0, float(t_get1) - float(t_get0)))
                yield meta, images_by_camera
                groups_yielded += 1
    else:
        # 不开 latest-only：直接按顺序取组（无后台线程）。
        group_index = 0
        while True:
            if int(max_groups) > 0 and groups_yielded >= int(max_groups):
                break

            _drain_soft_trigger_events()

            # 统计 source 等待“下一组完整组包”的耗时（毫秒）。
            t_get0 = time.monotonic()
            group = cap.get_next_group(timeout_s=float(timeout_s))
            t_get1 = time.monotonic()
            if group is None:
                max_wait = float(max_wait_seconds)
                if max_wait > 0 and (time.monotonic() - last_progress) > max_wait:
                    raise OnlineGroupWaitTimeout(f"在线采集等待超时：超过 {max_wait:.3f}s 未收到任何完整组包。")
                continue

            last_progress = time.monotonic()
            current_group_index = int(group_index)
            group_index += 1

            meta, images_by_camera = _convert_one_group(
                group=list(group),
                current_group_index=int(current_group_index),
                group_ready_monotonic=float(time.monotonic()),
                prefetched_images_by_camera=None,
                prefetched_decode_ms=None,
                latest_only_enabled=False,
                latest_only_skipped_groups=0,
                latest_only_skipped_groups_total=0,
            )
            meta["source_get_group_ms"] = float(1000.0 * max(0.0, float(t_get1) - float(t_get0)))
            yield meta, images_by_camera
            groups_yielded += 1
