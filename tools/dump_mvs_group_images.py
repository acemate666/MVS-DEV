# -*- coding: utf-8 -*-

"""从在线配置抓取少量同步组并落盘每相机图像（用于排障）。

动机：
- 当 online 出现 det_n=0 / balls=0 时，最快的分流方式是“把 detector 输入长什么样”直接保存下来目视确认。
- 本脚本复用 online 配置（.yaml/.yml/.json）里的：serials、触发计划、ROI、pixel_format、曝光/增益。

输出内容：
- 每组一个目录：group_000000/...
- 每相机至少输出一张 SDK BMP：<serial>.bmp（由 MVS SDK 保存，通常最可信）
- 可选输出一张 BGR PNG：<serial>.png（通过 frame_to_bgr 解码，便于快速预览；若解码有误也能暴露问题）
- 组元信息：meta.json

用法示例：
- uv run python tools/dump_mvs_group_images.py --config configs/online/master_slave_line0.yaml --max-groups 2

注意：
- 这是调试脚本（tools），不应被包代码 import。
- 采集依赖真实硬件与 MVS SDK 环境；CI/单测不会运行它。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Literal, cast

import cv2

from mvs import load_mvs_binding, open_quad_capture
from mvs.capture.image import frame_to_bgr
from mvs.capture.save import save_frame_as_bmp
from mvs.capture.triggering import build_trigger_plan

from tennis3d.config import load_online_app_config


def _now_compact() -> str:
    # 用于文件夹名，避免 Windows 不支持 ':'
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def main() -> int:
    parser = argparse.ArgumentParser(description="从 online 配置抓取同步组并落盘图像（排障用）")
    parser.add_argument(
        "--config",
        required=True,
        help="在线配置文件路径（.yaml/.yml/.json），例如 configs/online/master_slave_line0.yaml",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help=(
            "输出目录（可选）。留空则写入 data/tools_output/mvs_dump/<timestamp>/"
        ),
    )
    parser.add_argument("--max-groups", type=int, default=1, help="最多抓取多少个同步组（默认 1）")
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=1.0,
        help="等待完整组包的超时时间（秒；默认 1.0）",
    )
    parser.add_argument(
        "--max-wait-seconds",
        type=float,
        default=10.0,
        help="连续这么久拿不到任何完整组包就退出（秒；默认 10）",
    )
    parser.add_argument(
        "--bayer-method",
        type=int,
        default=2,
        help="SDK 保存 BMP 时的 Bayer 插值方法：0=快速 1=均衡 2=最优 3=最优+（默认 2）",
    )
    parser.add_argument(
        "--save-bgr-png",
        action="store_true",
        help="额外保存通过 frame_to_bgr 解码后的 PNG（便于预览；默认关闭）",
    )

    args = parser.parse_args()

    cfg_path = Path(str(args.config)).resolve()
    cfg = load_online_app_config(cfg_path)

    ts = _now_compact()
    if str(args.out_dir).strip():
        base_out = Path(str(args.out_dir)).resolve()
    else:
        base_out = (Path("data") / "tools_output" / "mvs_dump" / ts).resolve()

    base_out.mkdir(parents=True, exist_ok=True)

    binding = load_mvs_binding(
        mvimport_dir=(str(cfg.mvimport_dir) if cfg.mvimport_dir is not None else None),
        dll_dir=(str(cfg.dll_dir) if cfg.dll_dir is not None else None),
    )

    trig = cfg.trigger
    plan = build_trigger_plan(
        serials=cfg.serials,
        trigger_source=str(trig.trigger_source),
        master_serial=str(trig.master_serial),
        soft_trigger_fps=float(trig.soft_trigger_fps),
    )

    group_by_raw = str(cfg.group_by).strip()
    if group_by_raw not in {"frame_num", "sequence"}:
        raise RuntimeError(f"不支持的 group_by={group_by_raw!r}（仅支持 frame_num/sequence）")
    group_by = cast(Literal["frame_num", "sequence"], group_by_raw)

    # 说明：此脚本只负责抓取与落盘，不做 online 的 detect/localize。
    cap = open_quad_capture(
        binding=binding,
        serials=cfg.serials,
        trigger_sources=plan.trigger_sources,
        trigger_activation=str(trig.trigger_activation),
        trigger_cache_enable=bool(trig.trigger_cache_enable),
        timeout_ms=int(cfg.timeout_ms),
        group_timeout_ms=int(cfg.group_timeout_ms),
        max_pending_groups=int(cfg.max_pending_groups),
        group_by=group_by,
        enable_soft_trigger_fps=float(plan.enable_soft_trigger_fps),
        soft_trigger_serials=list(plan.soft_trigger_serials),
        master_serial=str(trig.master_serial),
        master_line_output=str(trig.master_line_out),
        master_line_source=str(trig.master_line_source),
        master_line_mode=str(trig.master_line_mode or "Output"),
        pixel_format=str(cfg.pixel_format or ""),
        image_width=cfg.image_width,
        image_height=cfg.image_height,
        image_offset_x=int(cfg.image_offset_x),
        image_offset_y=int(cfg.image_offset_y),
        exposure_auto=str(cfg.exposure_auto or ""),
        exposure_time_us=(float(cfg.exposure_time_us) if cfg.exposure_time_us is not None else None),
        gain_auto=str(cfg.gain_auto or ""),
        gain=(float(cfg.gain) if cfg.gain is not None else None),
    )

    last_got = time.monotonic()
    got_any = False
    try:
        for gi in range(max(0, int(args.max_groups))):
            group = cap.get_next_group(timeout_s=float(args.timeout_s))
            if group is None:
                if got_any:
                    continue

                if float(args.max_wait_seconds) > 0 and (time.monotonic() - last_got) > float(
                    args.max_wait_seconds
                ):
                    print(f"超过 {args.max_wait_seconds}s 未收到任何完整组包，退出。")
                    return 2
                continue

            got_any = True
            last_got = time.monotonic()

            out_dir = base_out / f"group_{gi:06d}"
            out_dir.mkdir(parents=True, exist_ok=True)

            meta: dict[str, Any] = {
                "config": str(cfg_path),
                "group_index": int(gi),
                "saved_at_unix": float(time.time()),
                "serials": list(cfg.serials),
                "trigger_sources": list(plan.trigger_sources),
                "pixel_format_requested": str(cfg.pixel_format or ""),
                "roi": {
                    "width": (int(cfg.image_width) if cfg.image_width is not None else None),
                    "height": (int(cfg.image_height) if cfg.image_height is not None else None),
                    "offset_x": int(cfg.image_offset_x),
                    "offset_y": int(cfg.image_offset_y),
                },
                "frames": [],
            }

            for fr in group:
                serial = str(getattr(fr, "serial", ""))
                cam_idx = int(getattr(fr, "cam_index"))
                cam = cap.cameras[cam_idx].cam

                bmp_path = out_dir / f"{serial}.bmp"
                save_frame_as_bmp(
                    binding=binding,
                    cam=cam,
                    out_path=bmp_path,
                    frame=fr,
                    bayer_method=int(args.bayer_method),
                )

                png_path = out_dir / f"{serial}.png"
                if bool(args.save_bgr_png):
                    # 说明：这里保存的是“解码后图像”，方便直接用图像浏览器打开。
                    # 若怀疑解码流程有问题，应优先以 SDK BMP 为准。
                    bgr = frame_to_bgr(binding=binding, cam=cam, frame=fr)
                    cv2.imwrite(str(png_path), bgr)

                meta["frames"].append(
                    {
                        "serial": serial,
                        "cam_index": int(cam_idx),
                        "frame_num": int(getattr(fr, "frame_num")),
                        "width": int(getattr(fr, "width")),
                        "height": int(getattr(fr, "height")),
                        "pixel_type": int(getattr(fr, "pixel_type")),
                        "frame_len": int(getattr(fr, "frame_len")),
                        "lost_packet": int(getattr(fr, "lost_packet")),
                        "dev_timestamp": int(getattr(fr, "dev_timestamp")),
                        "host_timestamp": int(getattr(fr, "host_timestamp")),
                        "arrival_monotonic": float(getattr(fr, "arrival_monotonic")),
                        "bmp": str(bmp_path.name),
                        "png": (str(png_path.name) if bool(args.save_bgr_png) else None),
                    }
                )

            (out_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            print(f"已保存：{out_dir}")

        print(f"完成。输出根目录：{base_out}")
        return 0
    finally:
        cap.close()


if __name__ == "__main__":
    raise SystemExit(main())
