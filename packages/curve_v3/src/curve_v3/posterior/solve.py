"""posterior 求解相关的底层小工具（内部模块）。

说明：
    该模块聚焦在“线性系统求解 + MAP/RLS 信息形式递推”的共用实现，
    供 `fit_map/fit_ls` 复用。
"""

from __future__ import annotations

import numpy as np

from curve_v3.configs import CurveV3Config


def safe_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """求解小规模线性方程组，并在病态时回退到最小二乘。"""

    try:
        return np.asarray(np.linalg.solve(A, b), dtype=float)
    except np.linalg.LinAlgError:
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return np.asarray(x, dtype=float)


def posterior_obs_sigma_m(cfg: CurveV3Config) -> float:
    """获取后验观测噪声尺度（米）。"""

    sigma = cfg.posterior.posterior_obs_sigma_m
    if sigma is None:
        sigma = cfg.posterior.weight_sigma_m
    return float(max(float(sigma), 1e-6))


def solve_map_with_prior(
    *,
    H: np.ndarray,
    y_vec: np.ndarray,
    theta0: np.ndarray,
    Q: np.ndarray,
    sigma_m: float,
    fit_mode: str,
    rls_lambda: float,
) -> np.ndarray:
    """求解带高斯先验的 MAP（正则最小二乘）问题。

    目标函数（与 `docs/curve.md` 对齐）：
        (1/σ^2) * ||Hθ - y||^2 + (θ-θ0)^T Q (θ-θ0)

    说明：
        - 仅保留“信息形式递推”（RLS）这一条代码路径。
        - 当历史遗留传入 "ls" 时，这里等价为 "rls" 且 λ=1。
    """

    inv_sigma2 = 1.0 / max(float(sigma_m) * float(sigma_m), 1e-12)

    mode = str(fit_mode).strip().lower()
    lam = float(rls_lambda) if mode == "rls" else 1.0
    lam = float(min(max(lam, 1e-6), 1.0))

    A = np.asarray(Q, dtype=float).copy()
    b = (A @ np.asarray(theta0, dtype=float)).astype(float)

    H2 = np.asarray(H, dtype=float)
    y2 = np.asarray(y_vec, dtype=float).reshape(-1)

    for i in range(int(H2.shape[0])):
        hi = H2[i, :].reshape(-1, 1)
        yi = float(y2[i])
        A = lam * A + inv_sigma2 * (hi @ hi.T)
        b = lam * b + inv_sigma2 * (hi.reshape(-1) * yi)

    return safe_solve(A, b)


__all__ = [
    "posterior_obs_sigma_m",
    "safe_solve",
    "solve_map_with_prior",
]
