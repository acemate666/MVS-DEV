"""单测：约束跨包依赖必须走“稳定 Public API”。

目标：
- 业务包/集成代码在依赖 `curve_v3`、`interception` 时，只允许从其“稳定入口”导入。
- 防止下游直接依赖内部模块路径（例如 `curve_v3.core` / `curve_v3.types` / `interception.selector`），
  否则一旦上游重构内部目录结构，集成侧会被非预期破坏。

说明：
- 该约束只针对“跨包”依赖：
  - `curve_v3` 包自身可以导入其内部模块（例如在 `curve_v3/__init__.py` 中做重导出）。
  - `interception` 包自身同理。
- 对于少量 IO 边界模块（例如 `curve_v3.config_yaml` / `interception.config_yaml`），
  若被明确写入文档为稳定入口，可在 allowlist 中放行。
"""

from __future__ import annotations

from pathlib import Path
import re


_CURVE_V3_ALLOWED_SUBMODULES = {"config_yaml"}
_INTERCEPTION_ALLOWED_SUBMODULES = {"config_yaml"}


def _iter_python_files(repo_root: Path) -> list[Path]:
    """收集需要检查的 .py 文件。

    说明：
    - 只扫描源码与测试/脚本目录，避免遍历 data/、.venv/ 等无关内容。
    """

    # 说明：这里刻意只检查“发布/集成面”的代码：
    # - packages/**/src：各包真正会被其它包 import 的实现代码
    # - tests/**：仓库根集成测试（代表上游真实依赖面）
    #
    # 不扫描 packages/**/tests：
    # - 各包的内部单测可以为了覆盖内部细节而导入内部模块，这是合理的；
    # - 我们要锁住的是“跨包运行时代码”的稳定依赖边界。
    patterns = [
        "packages/**/src/**/*.py",
        "tests/**/*.py",
    ]

    files: list[Path] = []
    for pat in patterns:
        files.extend(repo_root.glob(pat))

    # 去重 + 排序，保证失败输出稳定。
    uniq = sorted({p.resolve() for p in files})

    # 过滤缓存与 egg-info。
    out: list[Path] = []
    for p in uniq:
        parts = set(p.parts)
        if "__pycache__" in parts:
            continue
        if any(part.endswith(".egg-info") for part in p.parts):
            continue
        out.append(p)

    return out


def _find_disallowed_imports(
    *,
    file_path: Path,
    package_name: str,
    allowed_submodules: set[str],
) -> list[str]:
    """返回该文件中命中的“禁止导入”行（原样文本，便于定位）。"""

    # 仅做简单行级匹配：我们的目标是 repo hygiene，而非完整解析 Python AST。
    # 如果未来出现更复杂的多行 import，可再升级为 ast.parse。
    from_re = re.compile(rf"^\s*from\s+{re.escape(package_name)}\.(?P<seg>[A-Za-z_][A-Za-z0-9_]*)\b")
    import_re = re.compile(rf"^\s*import\s+{re.escape(package_name)}\.(?P<seg>[A-Za-z_][A-Za-z0-9_]*)\b")

    bad: list[str] = []
    text = file_path.read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        m = from_re.match(raw)
        if m:
            seg = m.group("seg")
            if seg not in allowed_submodules:
                bad.append(raw)
            continue

        m = import_re.match(raw)
        if m:
            seg = m.group("seg")
            if seg not in allowed_submodules:
                bad.append(raw)
            continue

    return bad


def test_public_api_boundary_for_curve_v3_and_interception() -> None:
    """跨包只允许依赖稳定 Public API。"""

    repo_root = Path(__file__).resolve().parents[1]
    files = _iter_python_files(repo_root)

    bad_msgs: list[str] = []

    for p in files:
        rel = p.relative_to(repo_root)
        rel_posix = rel.as_posix()

        # 说明：curve_v3 包内允许使用内部导入（例如顶层重导出）。
        if not rel_posix.startswith("packages/curve_v3/"):
            bad = _find_disallowed_imports(
                file_path=p,
                package_name="curve_v3",
                allowed_submodules=_CURVE_V3_ALLOWED_SUBMODULES,
            )
            if bad:
                bad_msgs.append(
                    "\n".join(
                        [
                            f"- {rel_posix}",
                            *[f"    {x}" for x in bad],
                        ]
                    )
                )

        # 说明：interception 包内允许使用内部导入（它本身就是实现）。
        if not rel_posix.startswith("packages/interception/"):
            bad = _find_disallowed_imports(
                file_path=p,
                package_name="interception",
                allowed_submodules=_INTERCEPTION_ALLOWED_SUBMODULES,
            )
            if bad:
                bad_msgs.append(
                    "\n".join(
                        [
                            f"- {rel_posix}",
                            *[f"    {x}" for x in bad],
                        ]
                    )
                )

    assert bad_msgs == [], (
        "发现跨包依赖使用了内部模块路径；请改为从包顶层稳定入口导入：\n"
        + "\n".join(bad_msgs)
    )
