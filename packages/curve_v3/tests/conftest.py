"""pytest 配置。

说明：
    - 本包采用 src-layout（`packages/curve_v3/src/curve_v3`）。
    - 测试运行环境应当“已安装本包”（例如使用 uv 的项目环境）。
    - 不再使用 `sys.path` 注入 `src` 路径，避免出现“本地能跑但安装/打包后失败”的环境差异。
"""
