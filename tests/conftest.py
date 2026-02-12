"""pytest 运行期配置。

本仓库为 uv workspace（monorepo）。代码位于 `packages/*/src/`，测试运行
应基于已安装到当前环境的 workspace members（例如使用 `uv sync --group dev`
后再执行 `uv run python -m pytest`）。

注意：请不要在测试侧把仓库根目录的 `./src` 注入 sys.path。
一旦出现“根目录 src + workspace 安装包”双来源，会导致 `import tennis3d`
等导入出现歧义，进而引入难以排查的不一致问题。
"""

from __future__ import annotations
