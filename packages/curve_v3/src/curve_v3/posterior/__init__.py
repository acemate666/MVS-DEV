"""curve_v3 第二阶段（posterior）算法实现命名空间。

说明：
    本包已经完成“职责拆分”：核心实现位于子模块中，`__init__.py` 不再聚合/导出
    任何实现函数，避免形成 package-level 的“巨型公共 API 面”。

    新代码请直接从子模块导入需要的能力，例如：
        - `curve_v3.posterior.fit_map.fit_posterior_map_for_candidate`
        - `curve_v3.posterior.fit_fused.fit_posterior_fused_map`
        - `curve_v3.posterior.fit_ls.fit_posterior_ls`
        - `curve_v3.posterior.anchor.inject_posterior_anchor`
        - `curve_v3.posterior.anchor.prior_nominal_state`

breaking=1：仓库内已同步修复所有调用点，禁止依赖“posterior 包级聚合导出”的写法。
"""

from __future__ import annotations

__all__: list[str] = []
