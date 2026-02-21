"""ROS2 下发脚本：在同一进程内运行 online，并按“新观测”连续发布 targets_v1。

特点：
- 低延迟：不经 JSONL 写盘/读盘，直接订阅在线 records 流。
- 依赖隔离：仅在运行时延迟导入 ROS2（rclpy/std_msgs）。

发布模式：
- 仅支持 continuous：仅当 `episode.active=true` 且 track 的 `n_obs` 增加时发布一次 targets_v1。
    当 corridor 暂不可用时，会回退到基于最近观测的轻量拟合。

接口对齐（重要）：
- 对齐 `host_ros2_bridge_demo.py` 的话题与 time_sync 交互：
    - 发布 time_sync_req_v1（默认 /time_sync/req）
    - 订阅 time_sync_resp_v1（默认 /time_sync/resp）并打印 offset/delay
    - 订阅 car_state_v1（默认 /hit/car_state，可选打印）
- `/hit/targets` 只发布 `targets_v1`。

前置条件：
- online 配置需要开启 curve，并启用 episode：
    - continuous 模式需要 `episode.active` 字段（仅在 episode enabled 时输出）。
- hit_targets 当前从 curve_v3 的 corridor_on_planes_y 选点；通常建议将 curve.corridor 配置为 y_min==y_max。

用法（示例）：
- 在已 source ROS2 的 Python 环境中：
    - python -m hit.ros2_uplink.host_ros2_bridge_demo2 --online-config configs/online/pt_windows_cpu_software_trigger_interception.yaml

说明：
- 本脚本只负责“发布”；targets_v1 的构造与命中点选择逻辑在 `hit_targets` 包。
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import statistics
import threading
import time
from pathlib import Path
from typing import Any

from hit_targets import ArmOffset, ContinuousHitTargetEmitter, build_targets_v1
from tennis3d.config import load_online_app_config
from tennis3d_online.api import build_spec_from_config, run_online


def _json_text(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _calc_offset_delay(*, t1: float, t2: float, t3: float, t4: float) -> tuple[float, float]:
    """按 NTP 四时间戳估计 offset/delay。

    定义与 `host_ros2_bridge_demo.py` 保持一致：
        offset = ((t2 - t1) + (t3 - t4)) / 2
        delay  = (t4 - t1) - (t3 - t2)
    """

    offset_s = ((t2 - t1) + (t3 - t4)) / 2.0
    delay_s = (t4 - t1) - (t3 - t2)
    return offset_s, delay_s


def _ms(x_s: float) -> float:
    return x_s * 1000.0


def _import_rclpy():
    """延迟导入 ROS2 依赖。

    说明：
        在非 ROS2 环境（例如 CI/本机纯 Python 环境）安装与跑单测时，
        rclpy 通常不可用；该函数用于把 ImportError 推迟到真正运行脚本的时刻。
    """

    try:
        import rclpy  # type: ignore
        from rclpy.node import Node  # type: ignore
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy  # type: ignore
        from std_msgs.msg import String  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "未检测到 ROS2 Python 环境（rclpy/std_msgs 导入失败）。\n"
            f"原始错误：{e!r}"
        ) from e

    return rclpy, Node, QoSProfile, HistoryPolicy, ReliabilityPolicy, String


def _run(args: argparse.Namespace) -> int:
    cfg = load_online_app_config(Path(args.online_config).resolve())
    spec = build_spec_from_config(cfg)

    # 约束：continuous 依赖 episode.active 字段（只在 episode enabled 时输出）。
    curve = spec.curve_cfg
    if not getattr(curve, "enabled", False):
        raise RuntimeError("online config 需要启用 curve.enabled=true")
    if not getattr(curve, "episode_enabled", False):
        raise RuntimeError("online config 需要启用 curve.episode.enabled=true（用于输出 episode.active）")

    # hit 参数：优先 CLI，其次读 online config 的 hit 段。
    if args.arm_offset is None:
        arm_offset = ArmOffset(
            x=float(cfg.hit.arm_offset_x),
            y=float(cfg.hit.arm_offset_y),
            z=float(cfg.hit.arm_offset_z),
        )
    else:
        arm_offset = ArmOffset(
            x=float(args.arm_offset[0]),
            y=float(args.arm_offset[1]),
            z=float(args.arm_offset[2]),
        )

    chassis_yaw = float(cfg.hit.chassis_yaw) if args.chassis_yaw is None else float(args.chassis_yaw)

    rclpy, Node, QoSProfile, HistoryPolicy, ReliabilityPolicy, String = _import_rclpy()

    rclpy.init()
    # 对齐 host_ros2_bridge_demo.py：固定 node 名称，便于运维统一。
    node = Node("host_ros2_bridge_demo")

    rel = args.qos.strip().lower()
    if rel not in ("reliable", "best_effort"):
        rel = "reliable"
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=max(1, args.qos_depth),
        reliability=ReliabilityPolicy.RELIABLE if rel == "reliable" else ReliabilityPolicy.BEST_EFFORT,
    )
    pub_targets = node.create_publisher(String, args.targets_topic, qos)
    pub_time_sync_req = node.create_publisher(String, args.time_sync_req_topic, qos)

    cont_emitter = ContinuousHitTargetEmitter()

    seq_targets = 0
    seq_sync = 0

    pending_sync: dict[int, float] = {}

    # time_sync 的滑动窗口样本。
    # 说明：样本无限增长会造成常驻进程内存持续上升，这里做上限。
    # 约定：offset 定义为 car_mono = host_mono + offset（见 docs/serial_link_architecture.md）。
    ntp_samples: deque[tuple[float, float]] = deque(maxlen=512)  # (delay_s, offset_s)
    offset_est_s: float | None = None
    best_delay_s: float | None = None

    last_tx_sync_mono_s = 0.0
    last_diag_mono_s = time.perf_counter()

    # 用于避免"未 time_sync 仍生成 arm 目标"的警告刷屏。
    warned_no_time_sync_for_arm = False

    node.get_logger().info(
        "ros2 hit targets online bridge started "
        f"targets_topic={args.targets_topic} "
        f"car_state_topic={args.car_state_topic} "
        f"time_sync_req_topic={args.time_sync_req_topic} time_sync_resp_topic={args.time_sync_resp_topic} "
        f"qos={args.qos} depth={args.qos_depth} "
        f"config={args.online_config} "
        f"time_sync_hz={args.time_sync_hz} max_inflight={args.time_sync_max_inflight} "
        f"timeout_s={args.time_sync_timeout_s:.3f} "
        f"require_time_sync_for_arm={args.require_time_sync_for_arm}"
    )

    def _on_time_sync_resp(msg: Any) -> None:
        nonlocal offset_est_s, best_delay_s

        t4 = time.perf_counter()

        try:
            obj = json.loads(str(msg.data))
        except Exception:
            return
        if not isinstance(obj, dict):
            return
        if obj.get("type") != "time_sync_resp_v1":
            return

        try:
            seq_raw = obj.get("seq")
            if seq_raw is None:
                return
            seq = int(seq_raw)
            t2 = float(obj["t2_car_mono_s"])
            t3 = float(obj["t3_car_mono_s"])
            t1_echo = float(obj["t1_host_mono_s"])
        except Exception:
            return

        t1 = pending_sync.pop(seq, None)
        if t1 is None:
            return

        # 基本一致性校验：避免多源请求导致错配
        if abs(t1_echo - t1) > 1e-3:
            return

        offset_s, delay_s = _calc_offset_delay(t1=t1, t2=t2, t3=t3, t4=t4)
        ntp_samples.append((delay_s, offset_s))

        # 用"最小 delay"的样本估计 offset：对称链路更可能接近真实偏移。
        try:
            best_delay_s, offset_est_s = min(ntp_samples, key=lambda x: x[0])
        except Exception:
            best_delay_s = None
            offset_est_s = None

        node.get_logger().info(
            "[time_sync] "
            f"seq={seq} "
            f"offset_ms={_ms(offset_s):.3f} "
            f"delay_ms={_ms(delay_s):.3f} "
            f"car_proc_ms={_ms(t3 - t2):.3f}"
        )

    def _on_car_state(msg: Any) -> None:
        if not args.print_car_state:
            return
        node.get_logger().info(f"[car_state] {msg.data[:200]}")

    node.create_subscription(String, args.time_sync_resp_topic, _on_time_sync_resp, qos)
    node.create_subscription(String, args.car_state_topic, _on_car_state, qos)

    def _tick() -> None:
        nonlocal seq_sync, last_tx_sync_mono_s, last_diag_mono_s

        now = time.perf_counter()

        # 丢包/无回包时：回收超时 pending，避免 max_inflight=1 永远卡死。
        timeout_s = args.time_sync_timeout_s
        if timeout_s > 1e-6 and pending_sync:
            expired = [seq for seq, t1 in pending_sync.items() if (now - t1) > timeout_s]
            for seq in expired:
                pending_sync.pop(seq, None)

        # time_sync_req_v1
        if args.time_sync_hz > 1e-6:
            min_interval = 1.0 / args.time_sync_hz
            if now - last_tx_sync_mono_s >= min_interval:
                # 单飞/限流：避免 pending_sync 无界增长，并降低错配概率。
                if len(pending_sync) < max(1, args.time_sync_max_inflight):
                    last_tx_sync_mono_s = now
                    seq_sync += 1
                    t1 = time.perf_counter()
                    req = {"type": "time_sync_req_v1", "seq": seq_sync, "t1_host_mono_s": t1}
                    out = String()
                    out.data = _json_text(req)
                    pending_sync[seq_sync] = t1
                    pub_time_sync_req.publish(out)

        # 简单诊断汇总（每 5s 一次），保持与 host_ros2_bridge_demo.py 类似。
        if now - last_diag_mono_s >= 5.0:
            last_diag_mono_s = now
            pending_n = len(pending_sync)
            if not ntp_samples or best_delay_s is None:
                node.get_logger().info(f"[diag] pending_sync={pending_n} samples=0")
            else:
                delays = [d for d, _ in ntp_samples]
                med = statistics.median(delays) if delays else None
                node.get_logger().info(
                    "[diag] "
                    f"pending_sync={pending_n} "
                    f"samples={len(ntp_samples)} "
                    f"best_delay_ms={_ms(best_delay_s):.3f}"
                )
                if med is not None:
                    node.get_logger().info(f"[diag] median_delay_ms={_ms(med):.3f}")
                if offset_est_s is not None:
                    node.get_logger().info(f"[diag] offset_est_ms={_ms(offset_est_s):+.3f}")

    node.create_timer(0.01, _tick)

    def _on_record(record: dict[str, Any]) -> None:
        nonlocal seq_targets, warned_no_time_sync_for_arm

        # 说明：record hook 运行在 online 线程里。为了避免 hook 抛异常导致 online 中断，
        # 这里用兜底 try/except 把问题记录到日志。
        try:
            # continuous：仅在 episode.active=true 且 n_obs 增加时尝试提取；每次新观测都发布一次。
            hit, _status = cont_emitter.process_record(record)
        except Exception as e:
            node.get_logger().warning(f"[on_record] cont_emitter.process_record 异常，已忽略：{e!r}")
            return

        if hit is None:
            return

        seq_targets += 1
        now_unix_s = time.time()
        targets = build_targets_v1(
            seq=seq_targets,
            hit=hit,
            arm_offset=arm_offset,
            chassis_yaw=chassis_yaw,
            now_unix_s=now_unix_s,
            mode="curve_episode_continuous_hit_v1",
        )
        # 坐标变换：对 chassis 做场地坐标系映射
        chassis = targets.get("chassis")
        if isinstance(chassis, dict):
            pose = chassis.get("target_pose")
            # FIXME 删了，或者封装，或者替换，后续再说吧
            if isinstance(pose, list) and len(pose) >= 2:
                pose[0] = -2.5 - pose[0] + 0.645 - 0.3
                pose[1] = pose[1] - 0.3
        # 关键：arm.hit_time_mono_s 必须在“车端 perf_counter 时间域”。
        # time_sync 的 offset 定义为：car_mono = host_mono + offset。
        # 这里把 hit_targets 输出的 hit_time（若存在）从 host 域换算到 car 域。
        used_offset_s = None
        arm = targets.get("arm")
        has_arm_hit_time = isinstance(arm, dict) and arm.get("hit_time_mono_s") is not None
        if has_arm_hit_time and offset_est_s is None and args.require_time_sync_for_arm:
            # 安全策略：未 time_sync 时，arm.hit_time_mono_s 的时间域无法保证正确，宁可不发 arm。
            targets["arm"] = None
            arm = None
            if not warned_no_time_sync_for_arm:
                warned_no_time_sync_for_arm = True
                node.get_logger().warning(
                    "[time_sync] 尚未获得 offset 估计，已丢弃 arm 目标以避免时间域错误；"
                    "请先确保 time_sync_resp 正常回包，或用 --no-require-time-sync-for-arm 关闭该保护。"
                )
        elif offset_est_s is not None and has_arm_hit_time:
            assert isinstance(arm, dict)
            arm["hit_time_mono_s"] = arm["hit_time_mono_s"] + offset_est_s
            used_offset_s = offset_est_s

        out = String()
        out.data = _json_text(targets)
        pub_targets.publish(out)

        # 日志：提取关键字段（保持简洁）
        t_until_text = "n/a"
        if isinstance(chassis, dict):
            v = chassis.get("time_until_arrival_s")
            if v is not None:
                t_until_text = f"{v:.3f}s"

        offset_text = "n/a" if used_offset_s is None else f"{_ms(used_offset_s):+.3f}ms"

        chassis_pose_text = "n/a"
        if isinstance(chassis, dict):
            pose = chassis.get("target_pose")
            if isinstance(pose, list) and len(pose) >= 3:
                chassis_pose_text = f"[{pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}]"

        arm_text = "none"
        if isinstance(arm, dict):
            p_hit = arm.get("p_hit")
            if isinstance(p_hit, list) and len(p_hit) >= 3:
                arm_text = f"p_hit=[{p_hit[0]:.3f}, {p_hit[1]:.3f}, {p_hit[2]:.3f}]"
            else:
                arm_text = "present"
            hit_time = arm.get("hit_time_mono_s")
            if hit_time is not None:
                arm_text += f" hit_time={hit_time:.3f}s"

        node.get_logger().info(
            "published targets_v1 "
            f"mode=continuous seq={seq_targets} track_id={hit.track_id} episode_id={hit.episode_id} "
            f"t_until={t_until_text} chassis_pose={chassis_pose_text} arm={arm_text} time_sync_offset={offset_text}"
        )

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        return int(run_online(spec, record_hooks=[_on_record]))
    except Exception as e:
        node.get_logger().error(f"ros2 hit targets online bridge 异常退出：{e!r}")
        return 0
    finally:
        # 先 shutdown 让 spin 线程退出，再销毁 node，避免竞态。
        try:
            rclpy.shutdown()
        except Exception:
            pass
        try:
            spin_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run tennis3d_online in-process and publish targets_v1 to ROS2")

    p.add_argument(
        "--online-config",
        default="configs/online/master_slave_line0.yaml",
        help="在线配置文件路径（.json/.yaml/.yml），将用于启动 tennis3d_online",
    )

    p.add_argument("--targets-topic", default="/hit/targets", help="发布 targets_v1 的 topic")
    p.add_argument("--car-state-topic", default="/hit/car_state", help="订阅 car_state_v1 的 topic（可选打印）")
    p.add_argument("--time-sync-req-topic", default="/time_sync/req", help="发布 time_sync_req_v1 的 topic")
    p.add_argument("--time-sync-resp-topic", default="/time_sync/resp", help="订阅 time_sync_resp_v1 的 topic")
    p.add_argument("--qos", choices=("reliable", "best_effort"), default="reliable")
    p.add_argument("--qos-depth", type=int, default=1)

    p.add_argument("--time-sync-hz", type=float, default=1.0, help="time_sync_req_v1 发送频率，设为 0 关闭")
    p.add_argument(
        "--time-sync-max-inflight",
        type=int,
        default=1,
        help="time_sync 请求最大并发数（默认 1；避免 pending 无界增长与错配）",
    )
    p.add_argument(
        "--time-sync-timeout-s",
        type=float,
        default=0.5,
        help=(
            "time_sync pending 超时回收时间（秒）。"
            "当 --time-sync-max-inflight=1 且回包丢失时，用于解除永远卡住的问题；"
            "设为 0 可关闭超时回收。"
        ),
    )
    p.add_argument(
        "--require-time-sync-for-arm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="若尚未得到 time_sync offset，则丢弃 arm 目标（默认开启，避免击球时刻时间域错误）",
    )
    p.add_argument("--print-car-state", action="store_true", help="是否把 car_state_v1 打到日志")
    p.add_argument(
        "--arm-offset",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "机械臂相对小车基座的固定 offset[x,y,z]（world 坐标系，米）。"
            "若不提供，将读取 online config 的 hit.arm_offset_xyz/arm_offset。"
        ),
    )
    p.add_argument(
        "--chassis-yaw",
        type=float,
        default=None,
        help="小车目标 yaw（弧度；若不提供，将读取 online config 的 hit.chassis_yaw）",
    )

    args = p.parse_args(argv)
    return int(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
