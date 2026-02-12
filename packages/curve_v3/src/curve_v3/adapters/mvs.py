"""多目定位（MVS）输出记录到 curve_v3 输入的适配。

本模块适配的上游数据形态来自仓库的 `protocols/mvs_ball_localization_types.py`：
- `FusedLocalizationRecord`：一帧输出，包含 0..N 个 `FusedBall`。
- `FusedBall`：包含世界系 3D 点、质量分、以及可选的每相机 2D 观测与协方差。

说明：
    - `curve_v3` 的核心输入是 `curve_v3.types.BallObservation`。
    - 像素域闭环需要每相机 2D 观测，因此当上游提供 `obs_2d_by_camera` 时，
      这里会把它转成 `BallObservation.obs_2d_by_camera`。

注意：
    - 该适配层只做字段映射与轻量清洗，不做三角化/轨迹拟合。
    - record 的时间戳字段因项目不同可能透传为不同 key；这里提供一个
      可复现的优先级策略（见 `extract_time_abs`）。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from curve_v3.types import BallObservation, Obs2D


def extract_time_abs(record: Mapping[str, Any]) -> float:
    """从 MVS record 中提取绝对时间戳（秒）。

    优先级（从高到低）：
        1) capture_t_abs
        2) t_abs
        3) created_at

    Raises:
        KeyError: 如果上述字段都不存在。
        ValueError: 如果时间戳不可转换为 float。
    """

    for k in ("capture_t_abs", "t_abs", "created_at"):
        if k in record:
            return float(record[k])
    raise KeyError("record 中未找到时间戳字段：capture_t_abs/t_abs/created_at")


def _clamp01(x: float) -> float:
    return float(min(max(float(x), 0.0), 1.0))


def _parse_cov_uv(obs: Mapping[str, Any]) -> np.ndarray:
    """把上游 cov_uv（2x2 list）解析成 ndarray。

    若 cov_uv 缺失或形状非法：退化为 sigma_px^2 * I。
    若 sigma_px 也缺失：退化为 1px^2 * I。
    """

    sigma_px = float(obs.get("sigma_px", 1.0))
    sigma_px = float(max(sigma_px, 1e-6))

    cov = obs.get("cov_uv", None)
    if cov is None:
        return (sigma_px * sigma_px) * np.eye(2, dtype=float)

    try:
        c = np.asarray(cov, dtype=float)
        if c.shape != (2, 2) or not np.all(np.isfinite(c)):
            raise ValueError("bad shape")
        return c
    except Exception:
        return (sigma_px * sigma_px) * np.eye(2, dtype=float)


def _parse_uv(obs: Mapping[str, Any]) -> np.ndarray:
    uv = np.asarray(obs.get("uv", None), dtype=float).reshape(2)
    return uv


def ball_observation_from_fused_ball(
    ball: Mapping[str, Any],
    *,
    t_abs: float,
) -> BallObservation:
    """把单个 FusedBall 映射为 curve_v3 的 BallObservation。"""

    xyz = np.asarray(ball["ball_3d_world"], dtype=float).reshape(3)

    # 质量分 -> conf（约定 0..1）。
    conf = ball.get("quality", None)
    conf_val = _clamp01(float(conf)) if conf is not None else None

    obs_2d_by_camera_raw = ball.get("obs_2d_by_camera", None)
    obs_2d_by_camera: dict[str, Obs2D] | None = None
    if isinstance(obs_2d_by_camera_raw, Mapping):
        tmp: dict[str, Obs2D] = {}
        for cam, obs in obs_2d_by_camera_raw.items():
            if obs is None or not isinstance(obs, Mapping):
                continue
            try:
                uv = _parse_uv(obs)
            except Exception:
                continue
            cov_uv = _parse_cov_uv(obs)
            sigma_px = float(obs.get("sigma_px", float(np.sqrt(float(cov_uv[0, 0])))))
            cov_source = str(obs.get("cov_source", ""))
            reproj_error_px = obs.get("reproj_error_px", None)

            tmp[str(cam)] = Obs2D(
                uv=uv,
                cov_uv=cov_uv,
                sigma_px=sigma_px,
                cov_source=cov_source,
                reproj_error_px=(float(reproj_error_px) if reproj_error_px is not None else None),
            )

        if tmp:
            obs_2d_by_camera = tmp

    return BallObservation(
        x=float(xyz[0]),
        y=float(xyz[1]),
        z=float(xyz[2]),
        t=float(t_abs),
        conf=conf_val,
        obs_2d_by_camera=obs_2d_by_camera,
    )


def select_ball(record: Mapping[str, Any], *, ball_id: int | None) -> Mapping[str, Any] | None:
    """从 record 中选择要处理的 ball。

    规则：
        - ball_id 为 None：若 balls 只有 1 个，选它；否则返回 None。
        - ball_id 非 None：在 balls 中按字段 ball_id 精确匹配。

    Returns:
        ball dict 或 None。
    """

    balls = record.get("balls", None)
    if not isinstance(balls, list):
        return None

    if ball_id is None:
        if len(balls) == 1 and isinstance(balls[0], Mapping):
            return balls[0]
        return None

    for b in balls:
        if not isinstance(b, Mapping):
            continue
        if int(b.get("ball_id", -1)) == int(ball_id):
            return b
    return None
