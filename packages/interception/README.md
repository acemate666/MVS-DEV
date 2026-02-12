# interception：单一击球目标点选择（基于 curve_v3）

`interception` 用于把 `curve_v3` 输出的“反弹后二段离散候选分布（prior / posterior）”压缩成单一的击球目标点：

- 输出目标：$(x^*, y^*, z^*, t^*)$（球心在高度平面 $y=y^*$ 的下降穿越点与时刻）。
- 关键特性：
  - 多峰分布不取均值/中位数：用命中半径 $r_{hit}$ 做“命中概率最大化”的离散选点。
  - 支持 $N=0$（仅 prefit/prior）与 $1\le N\le 5$（prefit + post points）。
  - 提供跨帧迟滞：`HitTargetStabilizer`。

文档：

- 直观解释：`docs/interception.md`
- 技术规格：`docs/interception_tech_spec.md`

## 快速开始

### 作为库调用

```python
import numpy as np

from curve_v3.configs import CurveV3Config
from curve_v3.types import BounceEvent, Candidate
from interception import InterceptionConfig, select_hit_target_prefit_only

curve_cfg = CurveV3Config(gravity=9.8, fit_params="v_only")

bounce = BounceEvent(
    t_rel=0.0,
    x=0.0,
    z=0.0,
    v_minus=np.zeros((3,), dtype=float),
    y=float(curve_cfg.bounce_contact_y()),
)

candidates = [
    Candidate(e=0.7, kt=0.65, weight=1.0, v_plus=np.array([1.0, 6.0, 0.0], dtype=float)),
]

cfg = InterceptionConfig(y_min=0.6, y_max=0.9, num_heights=5)

out = select_hit_target_prefit_only(
    bounce=bounce,
    candidates=candidates,
    time_base_abs=0.0,
    t_now_abs=0.0,
    cfg=cfg,
    curve_cfg=curve_cfg,
)

print(out.valid, out.target)
```

## 安装（推荐 uv）

在你的目标工程里以本仓库子模块/子树方式引入后：

- `uv add --editable ./packages/curve_v3`
- `uv add --editable ./packages/interception`

说明：

- `interception` 依赖 `curve_v3`（见 `pyproject.toml` 的 dependencies）。
- 本仓库不提供 CLI 入口；`interception` 仅作为纯算法库使用。

## 合成带噪数据测试

如果你希望用“虚拟生成的带噪声数据”验证 `interception` 的输出合法性（不崩溃、输出有限、时序合理），可以直接参考并运行：

- `tests/test_interception_synthetic_noisy.py`

该测试使用 `curve_v3` 提供的合成数据工具生成 pre/post 观测（含高斯噪声），并调用：

- `interception.selector.select_hit_target_with_post(...)`

合成数据生成工具在：

    - `packages/curve_v3/src/curve_v3/offline/testing/synthetic.py`

运行方式：

- 在 `packages/interception` 目录下执行：`uv run python -m pytest`
