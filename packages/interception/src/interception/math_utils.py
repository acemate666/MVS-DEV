"""interception 的轻量数学工具。

动机：
    `interception` 是一个独立算法库。为避免跨包依赖 `curve_v3` 的内部实现路径
    （例如 `curve_v3.utils.*`），这里在本包内提供 selector 所需的最小数学工具。

设计约束：
    - 仅依赖 NumPy（不引入额外第三方库）。
    - 以“足够用、易读、可复现”为优先，避免过度通用化。

注意：
    - 这些函数当前主要服务于 `interception.selector`。
    - 若未来需要对外暴露，请优先在 `interception.__init__` 明确导出，避免形成隐式公共 API。
"""

from __future__ import annotations
import numpy as np


def real_roots_of_quadratic(coeffs: np.ndarray) -> list[float]:
    """返回二次多项式的实根（升序）。

    Args:
        coeffs: 系数数组，期望形如 [a, b, c]，表示 a*x^2 + b*x + c。

    Returns:
        实根列表（升序）。若无实根或输入非法，返回空列表。

    说明：
        - 当 a≈0 时退化为一次方程。
        - 输入包含 NaN/Inf 时直接返回空。
    """

    c = np.asarray(coeffs, dtype=float).reshape(-1)
    if c.size != 3:
        raise ValueError(f"Expected 3 coefficients [a,b,c], got size={int(c.size)}")
    if not bool(np.all(np.isfinite(c))):
        return []

    a, b, cc = float(c[0]), float(c[1]), float(c[2])

    # 退化：线性/常数。
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return []
        return [float(-cc / b)]

    d = b * b - 4.0 * a * cc
    if not np.isfinite(d) or d < 0.0:
        return []

    sqrt_d = float(np.sqrt(d))
    r1 = (-b - sqrt_d) / (2.0 * a)
    r2 = (-b + sqrt_d) / (2.0 * a)

    roots = [float(r1), float(r2)]
    roots.sort()
    return roots


def weighted_quantile_1d(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """计算一维样本的加权分位数。

    Args:
        values: 样本值，shape=(N,)。
        weights: 样本权重（非负），shape=(N,)。
        q: 分位数水平，取值范围 [0, 1]。

    Returns:
        加权分位数。

    说明：
        - 当权重和为 0 时，会回退到普通（非加权）分位数。
        - 使用加权 CDF 上的分段线性插值。
        - 该实现刻意保持简单，便于与论文/文档中的定义对应。
    """

    v = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if v.size == 0:
        return float("nan")

    q = float(min(max(float(q), 0.0), 1.0))

    w = np.maximum(w, 0.0)
    sw = float(np.sum(w))
    if sw <= 0.0:
        return float(np.quantile(v, q))

    idx = np.argsort(v)
    v_sorted = v[idx]
    w_sorted = w[idx]

    cdf = np.cumsum(w_sorted, dtype=float) / sw

    k = int(np.searchsorted(cdf, q, side="left"))
    if k <= 0:
        return float(v_sorted[0])
    if k >= v_sorted.size:
        return float(v_sorted[-1])

    c0 = float(cdf[k - 1])
    c1 = float(cdf[k])
    if abs(c1 - c0) < 1e-12:
        return float(v_sorted[k])

    alpha = (q - c0) / (c1 - c0)
    return float(v_sorted[k - 1] + alpha * (v_sorted[k] - v_sorted[k - 1]))


def weighted_quantiles_1d(values: np.ndarray, weights: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """对多个分位点 qs 的便捷封装（内部逐个计算）。"""

    q_arr = np.asarray(qs, dtype=float).reshape(-1)
    out = np.zeros((q_arr.size,), dtype=float)
    for i, q in enumerate(q_arr.tolist()):
        out[i] = weighted_quantile_1d(values, weights, float(q))
    return out
