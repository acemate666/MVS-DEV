# MVS_Deployment

## 快速开始

### 1) 安装依赖（uv workspace）

在仓库根目录执行：

- `uv sync --group dev`
- `uv sync --all-groups`

说明：
- `pyproject.toml` 的 workspace members 已显式列出（避免误装历史/废弃内容）。

### 2) 跑单元测试

- `uv run pytest -q`

## 运行方式（在线/离线）

常见在线运行示例：

- `uv run python -m tennis3d_online --config configs/online/master_slave_line0.yaml`

常见离线运行示例：

- `uv run tennis3d-offline --config <your_offline_config.yaml>`
