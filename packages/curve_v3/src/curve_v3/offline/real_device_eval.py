"""curve_v3 的“实机/回放”离线评测工具。

定位：
    - 该模块用于把“球轨迹 JSON 回放 -> curve_v3 拟合/预测 -> 结果保存/可视化”的流程
      收敛成一条可复用的离线链路。
    - 这里不属于核心在线算法路径（见 `curve_v3.core.CurvePredictorV3`），因此放在
      `curve_v3.offline` 命名空间下。

坐标约定：
    - x：向右为正（m）
    - y：向上为正（m）
    - z：向前为正（m）

JSON 输入约定（建议）：
    {
      "meta": { ... },
      "observations": [
        {"t": 100.0, "x": 0.1, "y": 1.1, "z": 0.2, "conf": 1.0},
        ...
      ]
    }

注意：
    - 该模块刻意保持依赖极简（标准库 + numpy + curve_v3）。
    - 画图（matplotlib）属于可选能力：不作为硬依赖，避免影响库本体的安装体积。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from curve_v3.configs import CurveV3Config
from curve_v3.core import CurvePredictorV3
from curve_v3.types import BallObservation


@dataclass(frozen=True)
class PlaneCrossing:
    """一条轨迹与平面 y==target_y 的一次穿越。"""

    target_y: float
    t_rel: float
    x: float
    z: float
    direction: str  # "up" | "down" | "flat"


def load_observations_from_json(path: str | Path) -> tuple[dict[str, Any], list[BallObservation]]:
    """从 JSON 文件加载观测序列。

    Args:
        path: JSON 路径。

    Returns:
        (meta, observations)
    """

    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    meta = dict(data.get("meta", {}))
    items = data.get("observations")
    if not isinstance(items, list) or not items:
        raise ValueError(f"JSON 中 observations 字段缺失或为空：{p}")

    obs: list[BallObservation] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise TypeError(f"observations[{i}] 不是对象：{type(it)}")

        t = float(it["t"])
        x = float(it["x"])
        y = float(it["y"])
        z = float(it["z"])
        conf = it.get("conf", None)
        conf_val = None if conf is None else float(conf)

        obs.append(BallObservation(x=x, y=y, z=z, t=t, conf=conf_val))

    # 排序：实机回放有时可能会乱序。
    obs.sort(key=lambda o: float(o.t))

    return meta, obs


def _as_rel_arrays(
    observations: list[BallObservation],
    *,
    time_base_abs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_abs = np.array([float(o.t) for o in observations], dtype=float)
    xs = np.array([float(o.x) for o in observations], dtype=float)
    ys = np.array([float(o.y) for o in observations], dtype=float)
    zs = np.array([float(o.z) for o in observations], dtype=float)

    ts = t_abs - float(time_base_abs)
    return ts, xs, ys, zs


def split_pre_post_by_first_bounce(
    observations: list[BallObservation],
    *,
    cfg: CurveV3Config,
    eps_y_m: float = 0.06,
    min_pre_points: int = 6,
) -> tuple[list[BallObservation], list[BallObservation], int]:
    """用“第一次触地附近的局部最低点”把观测切成 pre/post。

    工程假设：
        - 输入序列只包含一次反弹（或至少我们只关心第一次反弹）。
        - 触地时刻球心高度约为 cfg.bounce_contact_y()。

    Args:
        observations: 时间升序的观测。
        cfg: curve_v3 配置，用于获取 y_contact 口径。
        eps_y_m: 允许的触地邻域高度冗余（m）。
        min_pre_points: 最少 pre 点数；用于避免过早切分。

    Returns:
        (pre, post, bounce_index)
    """

    if len(observations) < max(2, int(min_pre_points)):
        raise ValueError("observations 太短，无法做分段")

    y_contact = float(cfg.bounce_contact_y())
    ys = np.array([float(o.y) for o in observations], dtype=float)

    # 1) 找到第一次进入“触地邻域”的索引。
    near = np.where(ys <= (y_contact + float(eps_y_m)))[0]
    if near.size == 0:
        # 兜底：没有近地面点，则用全局最低点。
        idx0 = int(np.argmin(ys))
    else:
        idx0 = int(near[0])

    # 2) 在 idx0 附近取一个窗口，找局部最低点作为 bounce。
    w = 5
    lo = max(0, idx0 - w)
    hi = min(len(observations) - 1, idx0 + w)
    idx_min = int(lo + np.argmin(ys[lo : hi + 1]))

    # 3) 避免切得太早：如果 idx_min 太小，则扩展到 min_pre_points。
    idx_min = max(idx_min, int(min_pre_points) - 1)

    pre = list(observations[: idx_min + 1])
    post = list(observations[idx_min + 1 :])
    return pre, post, idx_min


def make_post_point_counts(num_post_points: int) -> list[int]:
    """生成 N=0,3,5,7,...,all 的 post 点数序列。"""

    m = int(max(num_post_points, 0))
    if m <= 0:
        return [0]

    ns: list[int] = [0]

    # 用户指定的起始序列。
    for n in (3, 5, 7):
        if n <= m:
            ns.append(int(n))

    # 后续按 2 递增（... 9, 11, 13, ...），最后补 all。
    n0 = 9
    if ns:
        n0 = max(n0, ns[-1] + 2)

    for n in range(n0, m + 1, 2):
        ns.append(int(n))

    if ns[-1] != m:
        ns.append(m)

    # 去重 + 保序。
    out: list[int] = []
    seen: set[int] = set()
    for n in ns:
        if n not in seen:
            out.append(int(n))
            seen.add(int(n))
    return out


def _find_plane_crossings_from_samples(
    *,
    ts_rel: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    target_y: float,
    t_rel_min: float | None,
    eps: float = 1e-12,
) -> list[PlaneCrossing]:
    """从采样点序列中找平面穿越（线性插值）。"""

    target_y = float(target_y)
    ts_rel = np.asarray(ts_rel, dtype=float)
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    zs = np.asarray(zs, dtype=float)

    if ts_rel.size < 2:
        return []

    t_min = -math.inf if t_rel_min is None else float(t_rel_min)

    out: list[PlaneCrossing] = []
    for i in range(ts_rel.size - 1):
        t1, t2 = float(ts_rel[i]), float(ts_rel[i + 1])
        if t2 < t_min:
            continue

        y1 = float(ys[i]) - target_y
        y2 = float(ys[i + 1]) - target_y

        # 同侧且都不近似为 0：不穿越。
        if abs(y1) > eps and abs(y2) > eps and (y1 > 0) == (y2 > 0):
            continue

        # 线性插值：y = y1 + a*(y2-y1) == 0
        denom = (y2 - y1)
        if abs(denom) < 1e-15:
            alpha = 0.0
            direction = "flat"
        else:
            alpha = float((-y1) / denom)
            alpha = min(max(alpha, 0.0), 1.0)
            direction = "up" if (ys[i + 1] > ys[i]) else "down"

        t = t1 + alpha * (t2 - t1)
        x = float(xs[i]) + alpha * float(xs[i + 1] - xs[i])
        z = float(zs[i]) + alpha * float(zs[i + 1] - zs[i])

        if t < t_min:
            continue

        out.append(PlaneCrossing(target_y=target_y, t_rel=float(t), x=float(x), z=float(z), direction=direction))

    # 去重（避免 y==target 恰好连续出现导致重复点）。
    dedup: list[PlaneCrossing] = []
    last_t: float | None = None
    for c in out:
        if last_t is None or abs(float(c.t_rel) - float(last_t)) > 1e-9:
            dedup.append(c)
            last_t = float(c.t_rel)
    return dedup


def find_plane_crossings_from_observations(
    observations: list[BallObservation],
    *,
    target_y: float,
    time_base_abs: float,
    t_rel_min: float | None,
) -> list[PlaneCrossing]:
    """从观测点中查找 y==target_y 的穿越点（线性插值）。"""

    ts_rel, xs, ys, zs = _as_rel_arrays(observations, time_base_abs=float(time_base_abs))
    return _find_plane_crossings_from_samples(
        ts_rel=ts_rel,
        xs=xs,
        ys=ys,
        zs=zs,
        target_y=float(target_y),
        t_rel_min=t_rel_min,
    )


def _sample_nominal_trajectory(
    *,
    engine: CurvePredictorV3,
    t_rel_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_rel_grid = np.asarray(t_rel_grid, dtype=float)

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    ts: list[float] = []

    for t_rel in t_rel_grid.tolist():
        p = engine.point_at_time_rel(float(t_rel))
        if p is None:
            xs.append(float("nan"))
            ys.append(float("nan"))
            zs.append(float("nan"))
        else:
            xs.append(float(p[0]))
            ys.append(float(p[1]))
            zs.append(float(p[2]))
        ts.append(float(t_rel))

    return (
        np.asarray(ts, dtype=float),
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(zs, dtype=float),
    )


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], *, header: list[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def run_post_bounce_subsampling_eval(
    *,
    observations: list[BallObservation],
    cfg: CurveV3Config,
    target_plane_y_m: float,
    out_dir: str | Path,
    ns_post_points: list[int] | None = None,
    nominal_dt_s: float = 0.005,
    make_plots: bool = True,
) -> dict[str, Any]:
    """按不同 post 点数 N 对比 curve_v3 的拟合/预测输出，并保存结果。

    Args:
        observations: 完整观测序列（必须按时间升序）。
        cfg: curve_v3 配置。
        target_plane_y_m: 目标水平平面 y==const（米）。
        out_dir: 输出目录。
        ns_post_points: 需要评估的 N 序列；None 则自动生成 N=0,3,5,7,...,all。
        nominal_dt_s: 名义轨迹采样间隔（用于近似求平面穿越）。
        make_plots: 是否尝试生成图（matplotlib 可用时）。

    Returns:
        报告 dict（同时会落盘）。
    """

    if not observations:
        raise ValueError("observations 为空")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # time_base_abs：与 CurvePredictorV3 内部定义一致：第一帧时间戳。
    time_base_abs = float(observations[0].t)

    pre, post, bounce_idx = split_pre_post_by_first_bounce(observations, cfg=cfg)
    ns = ns_post_points or make_post_point_counts(len(post))

    # ground-truth：用观测插值得到“反弹后第一次穿越”。
    # 这里把 pre[-1] 当作反弹附近点，因此 t_rel_min 取其时间。
    t_rel_min = float(pre[-1].t) - time_base_abs
    gt_crossings = find_plane_crossings_from_observations(
        observations,
        target_y=float(target_plane_y_m),
        time_base_abs=time_base_abs,
        t_rel_min=t_rel_min,
    )
    gt_first = gt_crossings[0] if gt_crossings else None

    rows_csv: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []

    for n in ns:
        n = int(n)
        engine = CurvePredictorV3(config=cfg)

        # feed：pre + 前 n 个 post。
        feed = list(pre) + list(post[:n])
        for o in feed:
            engine.add_observation(o)

        bounce = engine.get_bounce_event()
        bounce_t_rel = None if bounce is None else float(bounce.t_rel)

        corridor = engine.corridor_on_plane_y(float(target_plane_y_m))

        # 名义轨迹：从 0 到 当前最后一帧 + 一点裕度。
        t_end_rel = float(feed[-1].t) - time_base_abs
        if bounce_t_rel is not None:
            t_end_rel = max(t_end_rel, bounce_t_rel + 1.0)

        dt = float(max(nominal_dt_s, 1e-4))
        t_grid = np.arange(0.0, t_end_rel + 1e-9, dt, dtype=float)

        t_nom, x_nom, y_nom, z_nom = _sample_nominal_trajectory(engine=engine, t_rel_grid=t_grid)

        # 名义穿越：只取反弹后的穿越。
        nom_crossings = _find_plane_crossings_from_samples(
            ts_rel=t_nom,
            xs=x_nom,
            ys=y_nom,
            zs=z_nom,
            target_y=float(target_plane_y_m),
            t_rel_min=(bounce_t_rel if bounce_t_rel is not None else t_rel_min),
        )
        nom_first = nom_crossings[0] if nom_crossings else None

        case: dict[str, Any] = {
            "n_post_points": int(n),
            "feed_points": int(len(feed)),
            "bounce": None
            if bounce is None
            else {
                "t_rel": float(bounce.t_rel),
                "x": float(bounce.x),
                "z": float(bounce.z),
                "v_minus": [float(x) for x in np.asarray(bounce.v_minus, dtype=float).tolist()],
            },
            "plane_y_m": float(target_plane_y_m),
            "corridor_on_plane": None
            if corridor is None
            else {
                "target_y": float(corridor.target_y),
                "mu_xz": [float(x) for x in np.asarray(corridor.mu_xz, dtype=float).tolist()],
                "cov_xz": [
                    [float(x) for x in np.asarray(corridor.cov_xz, dtype=float)[0].tolist()],
                    [float(x) for x in np.asarray(corridor.cov_xz, dtype=float)[1].tolist()],
                ],
                "t_rel_mu": float(corridor.t_rel_mu),
                "t_rel_var": float(corridor.t_rel_var),
                "valid_ratio": float(corridor.valid_ratio),
                "crossing_prob": float(corridor.crossing_prob),
                "is_valid": bool(corridor.is_valid),
                "quantile_levels": None
                if corridor.quantile_levels is None
                else [float(x) for x in np.asarray(corridor.quantile_levels, dtype=float).tolist()],
                "quantiles_xz": None
                if corridor.quantiles_xz is None
                else [[float(a), float(b)] for a, b in np.asarray(corridor.quantiles_xz, dtype=float).tolist()],
                "quantiles_t_rel": None
                if corridor.quantiles_t_rel is None
                else [float(x) for x in np.asarray(corridor.quantiles_t_rel, dtype=float).tolist()],
            },
            "nominal_first_crossing": None
            if nom_first is None
            else {
                "t_rel": float(nom_first.t_rel),
                "x": float(nom_first.x),
                "z": float(nom_first.z),
                "direction": str(nom_first.direction),
            },
            "gt_first_crossing": None
            if gt_first is None
            else {
                "t_rel": float(gt_first.t_rel),
                "x": float(gt_first.x),
                "z": float(gt_first.z),
                "direction": str(gt_first.direction),
            },
        }

        # 误差：以 ground-truth 第一穿越为基准。
        if gt_first is not None and nom_first is not None:
            dx = float(nom_first.x - gt_first.x)
            dz = float(nom_first.z - gt_first.z)
            dt_rel = float(nom_first.t_rel - gt_first.t_rel)
            case["errors"] = {
                "dx_m": dx,
                "dz_m": dz,
                "dt_rel_s": dt_rel,
                "d_xz_m": float(math.sqrt(dx * dx + dz * dz)),
            }
        else:
            case["errors"] = None

        cases.append(case)

        rows_csv.append(
            {
                "n_post": int(n),
                "bounce_t_rel": "" if bounce_t_rel is None else float(bounce_t_rel),
                "corr_crossing_prob": "" if corridor is None else float(corridor.crossing_prob),
                "corr_t_rel_mu": "" if corridor is None else float(corridor.t_rel_mu),
                "corr_x_mu": "" if corridor is None else float(np.asarray(corridor.mu_xz, dtype=float)[0]),
                "corr_z_mu": "" if corridor is None else float(np.asarray(corridor.mu_xz, dtype=float)[1]),
                "nom_t_rel": "" if nom_first is None else float(nom_first.t_rel),
                "nom_x": "" if nom_first is None else float(nom_first.x),
                "nom_z": "" if nom_first is None else float(nom_first.z),
                "gt_t_rel": "" if gt_first is None else float(gt_first.t_rel),
                "gt_x": "" if gt_first is None else float(gt_first.x),
                "gt_z": "" if gt_first is None else float(gt_first.z),
                "err_dx_m": "" if case["errors"] is None else float(case["errors"]["dx_m"]),
                "err_dz_m": "" if case["errors"] is None else float(case["errors"]["dz_m"]),
                "err_dt_rel_s": "" if case["errors"] is None else float(case["errors"]["dt_rel_s"]),
                "err_d_xz_m": "" if case["errors"] is None else float(case["errors"]["d_xz_m"]),
            }
        )

        _write_json(out_root / "cases" / f"case_n={int(n):03d}.json", case)

    report: dict[str, Any] = {
        "meta": {
            "time_base_abs": float(time_base_abs),
            "num_observations": int(len(observations)),
            "bounce_index": int(bounce_idx),
            "num_pre": int(len(pre)),
            "num_post": int(len(post)),
        },
        "target_plane_y_m": float(target_plane_y_m),
        "ns_post_points": [int(n) for n in ns],
        "gt_first_crossing": None
        if gt_first is None
        else {
            "t_rel": float(gt_first.t_rel),
            "x": float(gt_first.x),
            "z": float(gt_first.z),
            "direction": str(gt_first.direction),
        },
        "cases": cases,
    }

    _write_json(out_root / "summary.json", report)
    _write_csv(
        out_root / "summary.csv",
        rows_csv,
        header=[
            "n_post",
            "bounce_t_rel",
            "corr_crossing_prob",
            "corr_t_rel_mu",
            "corr_x_mu",
            "corr_z_mu",
            "nom_t_rel",
            "nom_x",
            "nom_z",
            "gt_t_rel",
            "gt_x",
            "gt_z",
            "err_dx_m",
            "err_dz_m",
            "err_dt_rel_s",
            "err_d_xz_m",
        ],
    )

    if make_plots:
        try:
            _try_make_plots(
                observations=observations,
                cfg=cfg,
                target_plane_y_m=float(target_plane_y_m),
                report=report,
                out_dir=out_root,
            )
        except ModuleNotFoundError:
            # 不强依赖 matplotlib：缺失时不算失败。
            _write_json(
                out_root / "plotting_skipped.json",
                {
                    "reason": "matplotlib 未安装，已跳过绘图。",
                    "hint": "如需图像输出，可在开发环境安装 matplotlib（建议放在本地 dev 依赖中）。",
                },
            )

    return report


def _try_make_plots(
    *,
    observations: list[BallObservation],
    cfg: CurveV3Config,
    target_plane_y_m: float,
    report: dict[str, Any],
    out_dir: Path,
) -> None:
    """绘图（可选依赖 matplotlib）。

    说明：
        - 这里的 import 放在函数内，以避免把 matplotlib 变成硬依赖。
        - 图按 case（每个 N）单独输出，满足“图可以分开画”。
    """

    # 说明：这里用动态 import，避免把 matplotlib 变成静态/硬依赖。
    # 同时也减少 type checker 在未安装时的误报。
    import importlib

    matplotlib = importlib.import_module("matplotlib")
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")

    time_base_abs = float(report["meta"]["time_base_abs"])
    pre, post, bounce_idx = split_pre_post_by_first_bounce(observations, cfg=cfg)

    # 观测（用于背景）。
    t_obs_rel, x_obs, y_obs, z_obs = _as_rel_arrays(observations, time_base_abs=time_base_abs)

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ground-truth crossings：反弹后在 y==target_plane_y_m 平面上通常有两个交点：
    # - 上升段（up）：反弹后向上穿越
    # - 下降段（down）：到达峰值后向下穿越
    # 这里在图中都画出来，便于做误差分析。
    t_rel_min = float(pre[-1].t) - time_base_abs
    gt_crossings = find_plane_crossings_from_observations(
        observations,
        target_y=float(target_plane_y_m),
        time_base_abs=float(time_base_abs),
        t_rel_min=float(t_rel_min),
    )
    gt_up = next((c for c in gt_crossings if c.direction == "up"), None)
    gt_down = next((c for c in gt_crossings if c.direction == "down"), None)

    for case in report["cases"]:
        n = int(case["n_post_points"])

        # 重跑一次 engine，用于得到名义轨迹采样（避免 report 变得过大）。
        engine = CurvePredictorV3(config=cfg)
        feed = list(pre) + list(post[:n])
        for o in feed:
            engine.add_observation(o)

        bounce = engine.get_bounce_event()
        t_b = None if bounce is None else float(bounce.t_rel)

        # 采样名义轨迹
        t_end_rel = float(feed[-1].t) - time_base_abs
        if t_b is not None:
            t_end_rel = max(t_end_rel, t_b + 1.0)
        t_grid = np.arange(0.0, t_end_rel + 1e-9, 0.005, dtype=float)
        t_nom, x_nom, y_nom, z_nom = _sample_nominal_trajectory(engine=engine, t_rel_grid=t_grid)

        nom_crossings = _find_plane_crossings_from_samples(
            ts_rel=t_nom,
            xs=x_nom,
            ys=y_nom,
            zs=z_nom,
            target_y=float(target_plane_y_m),
            t_rel_min=(float(t_b) if t_b is not None else float(t_rel_min)),
        )
        nom_up = next((c for c in nom_crossings if c.direction == "up"), None)
        nom_down = next((c for c in nom_crossings if c.direction == "down"), None)

        # ---------- 图1：y(t) ----------
        fig1 = plt.figure(figsize=(8, 4.5))
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.plot(t_obs_rel, y_obs, ".-", label="obs y")
        ax1.plot(t_nom, y_nom, "-", label="nominal y")
        ax1.axhline(float(target_plane_y_m), color="k", linestyle="--", linewidth=1.0, label=f"y={target_plane_y_m:.3f}m")
        if t_b is not None:
            ax1.axvline(float(t_b), color="C3", linestyle=":", linewidth=1.0, label="bounce(t_rel)")
        if gt_up is not None:
            ax1.plot([float(gt_up.t_rel)], [float(target_plane_y_m)], "^", color="C2", label="gt crossing (up)")
        if gt_down is not None:
            ax1.plot([float(gt_down.t_rel)], [float(target_plane_y_m)], "v", color="C2", label="gt crossing (down)")
        if nom_up is not None:
            ax1.plot([float(nom_up.t_rel)], [float(target_plane_y_m)], "^", color="C1", label="nom crossing (up)")
        if nom_down is not None:
            ax1.plot([float(nom_down.t_rel)], [float(target_plane_y_m)], "v", color="C1", label="nom crossing (down)")
        ax1.set_title(f"Y-T (N_post={n})")
        ax1.set_xlabel("t_rel (s)")
        ax1.set_ylabel("y (m)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best")
        fig1.tight_layout()
        fig1.savefig(str(plot_dir / f"y_t_n={n:03d}.png"), dpi=150)
        plt.close(fig1)

        # ---------- 图2：x-z 俯视 ----------
        fig2 = plt.figure(figsize=(6, 6))
        ax2 = fig2.add_subplot(1, 1, 1)
        ax2.plot(z_obs, x_obs, ".", label="obs (z,x)")
        ax2.plot(z_nom, x_nom, "-", label="nominal (z,x)")
        if gt_up is not None:
            ax2.plot([float(gt_up.z)], [float(gt_up.x)], "^", color="C2", label="gt crossing (up)")
        if gt_down is not None:
            ax2.plot([float(gt_down.z)], [float(gt_down.x)], "v", color="C2", label="gt crossing (down)")
        if nom_up is not None:
            ax2.plot([float(nom_up.z)], [float(nom_up.x)], "^", color="C1", label="nom crossing (up)")
        if nom_down is not None:
            ax2.plot([float(nom_down.z)], [float(nom_down.x)], "v", color="C1", label="nom crossing (down)")
        ax2.set_title(f"Top-down XZ (N_post={n})")
        ax2.set_xlabel("z (m)")
        ax2.set_ylabel("x (m)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best")
        fig2.tight_layout()
        fig2.savefig(str(plot_dir / f"xz_topdown_n={n:03d}.png"), dpi=150)
        plt.close(fig2)


__all__ = [
    "PlaneCrossing",
    "load_observations_from_json",
    "split_pre_post_by_first_bounce",
    "make_post_point_counts",
    "find_plane_crossings_from_observations",
    "run_post_bounce_subsampling_eval",
]
