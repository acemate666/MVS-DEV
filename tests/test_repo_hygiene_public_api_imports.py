"""单测：约束跨包依赖边界只使用“稳定 Public API”。

动机：
- `packages/curve_v3` 与 `packages/interception` 作为独立算法库演进时，内部模块结构很可能重构；
  若上游（其它包）直接 `from curve_v3.core/types/... import ...`，会导致非必要的耦合与脆弱性。

本测试只检查“生产代码”（packages/*/src），不检查 packages/*/tests 或仓库根 tests：
- 生产代码的 import 边界必须更严格；
- 单测/脚本可按需要使用内部模块（若你希望也收紧测试导入边界，可再加一条更严格的测试）。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class _BadImport:
    file: str
    lineno: int
    module: str


def _iter_production_py_files(repo_root: Path) -> list[Path]:
    """收集 packages/*/src 下的 .py 文件。"""

    packages_dir = repo_root / "packages"
    if not packages_dir.exists():
        return []

    out: list[Path] = []
    for p in packages_dir.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        # 说明：只检查生产代码目录，避免把测试/脚本也绑死。
        if "src" not in p.parts:
            continue
        out.append(p)

    return out


def _is_under(rel_posix: str, prefix_posix: str) -> bool:
    return rel_posix == prefix_posix or rel_posix.startswith(prefix_posix + "/")


def _check_import_boundary(*, repo_root: Path) -> list[_BadImport]:
    """扫描 import AST，找出跨包使用内部模块路径的地方。"""

    curve_v3_src_prefix = "packages/curve_v3/src/curve_v3"
    interception_src_prefix = "packages/interception/src/interception"

    bad: list[_BadImport] = []

    for p in _iter_production_py_files(repo_root):
        rel = p.relative_to(repo_root).as_posix()

        is_curve_v3_src = _is_under(rel, curve_v3_src_prefix)
        is_interception_src = _is_under(rel, interception_src_prefix)

        try:
            src = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # 说明：源码文件应为 UTF-8；遇到异常时直接报错更利于定位问题。
            raise

        try:
            tree = ast.parse(src, filename=rel)
        except SyntaxError as e:
            raise AssertionError(f"无法解析 Python 语法：{rel}:{e.lineno}:{e.offset}") from e

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name.startswith("curve_v3.") and not is_curve_v3_src:
                        if not name.startswith("curve_v3.configs"):
                            bad.append(_BadImport(rel, node.lineno, name))

                    if name.startswith("interception.") and not is_interception_src:
                        if not name.startswith("interception.config_yaml"):
                            bad.append(_BadImport(rel, node.lineno, name))

            elif isinstance(node, ast.ImportFrom):
                # 说明：相对导入只在包内使用；此处不参与跨包边界约束。
                if node.level and node.level > 0:
                    continue

                mod = node.module
                if not mod:
                    continue

                if mod.startswith("curve_v3.") and not is_curve_v3_src:
                    if not mod.startswith("curve_v3.configs"):
                        bad.append(_BadImport(rel, node.lineno, mod))

                if mod.startswith("interception.") and not is_interception_src:
                    if not mod.startswith("interception.config_yaml"):
                        bad.append(_BadImport(rel, node.lineno, mod))

    return bad


def test_production_code_imports_use_stable_public_api_only() -> None:
    """确保生产代码跨包导入不耦合内部模块路径。"""

    repo_root = Path(__file__).resolve().parents[1]
    bad = _check_import_boundary(repo_root=repo_root)

    assert bad == [], (
        "发现生产代码跨包导入了内部模块路径（请改为从包顶层稳定导出导入；"
        "curve_v3 可额外允许 curve_v3.configs）：\n"
        + "\n".join(f"- {b.file}:{b.lineno} import {b.module}" for b in bad)
    )
