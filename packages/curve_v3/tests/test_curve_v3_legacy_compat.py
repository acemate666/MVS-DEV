import importlib
import unittest


class TestCurveV3LegacyCompat(unittest.TestCase):
    def test_legacy_adapter_removed(self):
        """旧版 legacy/curve2 兼容层已移除。

        约束：仓库不再提供任何向下兼容的转发/适配模块，因此旧路径应当不可导入。
        """

        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("curve_v3.legacy")

        # 包入口不再导出 Curve 兼容适配器。
        pkg = importlib.import_module("curve_v3")
        self.assertFalse(hasattr(pkg, "Curve"))

