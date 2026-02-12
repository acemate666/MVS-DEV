import unittest

import numpy as np

from curve_v3.utils import BounceTransitionDetector
from curve_v3.configs import CurveV3Config


class TestBounceDetector(unittest.TestCase):
    def test_triggers_and_returns_local_min_cut(self):
        """验证分段检测能在“先下降后上升且近地”时触发，并返回局部最小点索引。

        说明：
            - 这里用合成的分段线性 y(t)：反弹前稳定下降，反弹后稳定上升。
            - 检测器内部带状态（去抖累计），因此需要逐步喂入数据。
        """

        cfg = CurveV3Config()
        det = BounceTransitionDetector(cfg=cfg)

        dt = 0.01
        t_b = 0.10
        y_contact = float(cfg.bounce_contact_y())

        # 构造：在 t_b 处达到最小值 y_contact。
        y0 = y_contact + 1.0 * t_b

        ts = np.arange(0.0, 0.25, dt, dtype=float)
        ys: list[float] = []
        for t in ts:
            if t <= t_b:
                ys.append(float(y0 - 1.0 * t))
            else:
                ys.append(float(y_contact + 1.0 * (t - t_b)))

        cut: int | None = None
        reason: str | None = None
        for i in range(len(ts)):
            t_i = ts[: i + 1]
            y_i = np.asarray(ys[: i + 1], dtype=float)
            cut, reason = det.find_cut_index(ts=t_i, ys=y_i, y_contact=y_contact)
            if cut is not None:
                # 触发时应当切到当前窗口内的最小点（即反弹点附近）。
                self.assertEqual(int(cut), int(np.argmin(y_i)))
                self.assertEqual(str(reason), "vy_flip_and_near_ground")
                break

        self.assertIsNotNone(cut)
        assert cut is not None

        # 反弹点就是全局最小值点，cut 应该非常接近 t_b 的索引。
        expected = int(np.argmin(np.asarray(ys, dtype=float)))
        self.assertLessEqual(abs(int(cut) - expected), 1)

    def test_does_not_trigger_when_not_near_ground(self):
        """验证在速度翻转但不近地时，不会误触发分段。"""

        cfg = CurveV3Config()
        det = BounceTransitionDetector(cfg=cfg)

        dt = 0.01
        t_b = 0.10

        # y_contact 设置为 0，序列最小也在 0.9 左右，远离 eps_y。
        y_contact = 0.0
        y0 = 1.0

        ts = np.arange(0.0, 0.25, dt, dtype=float)
        ys: list[float] = []
        for t in ts:
            if t <= t_b:
                ys.append(float(y0 - 1.0 * t))
            else:
                ys.append(float((y0 - 1.0 * t_b) + 1.0 * (t - t_b)))

        cut: int | None = None
        for i in range(len(ts)):
            t_i = ts[: i + 1]
            y_i = np.asarray(ys[: i + 1], dtype=float)
            cut, _ = det.find_cut_index(ts=t_i, ys=y_i, y_contact=y_contact)
            self.assertIsNone(cut)

    def test_triggers_gap_freeze_when_bounce_in_invisible_gap(self):
        """验证在反弹附近不可见（形成时间缺口）时，能用 gap-freeze 做安全切分。

        说明：
            - 模拟规则：y<0.2m 时网球不可见，因此观测序列在反弹附近出现 gap。
            - 在这种场景下 near_ground 通常永远不成立；若不额外处理，prefit 会被
              gap 右侧的点（可能已 post）污染。
            - 该测试期望检测器返回 cut_index=gap 左侧最后一点，并给出 reason。
        """

        cfg = CurveV3Config()
        det = BounceTransitionDetector(cfg=cfg)

        g = float(cfg.physics.gravity)
        y_contact = float(cfg.bounce_contact_y())
        y_visible_min = 0.2

        dt = 0.01
        t_b = 0.10

        # 生成一个简单的“重力抛体 + 反弹后向上”的 y(t)，保证反弹发生在 y_contact。
        y_init = 0.60
        vy0 = (y_contact - y_init + 0.5 * g * t_b * t_b) / t_b
        vy_up = 2.0

        ts_full = np.arange(0.0, 0.26, dt, dtype=float)
        ys_full: list[float] = []
        for t in ts_full:
            if t <= t_b:
                ys_full.append(float(y_init + vy0 * t - 0.5 * g * t * t))
            else:
                tau = float(t - t_b)
                ys_full.append(float(y_contact + vy_up * tau - 0.5 * g * tau * tau))

        ys_full_arr = np.asarray(ys_full, dtype=float)
        mask = ys_full_arr >= float(y_visible_min)
        ts_obs = ts_full[mask]
        ys_obs = ys_full_arr[mask]

        # 观测两侧都应有点，否则测试没有意义。
        self.assertGreaterEqual(int(ts_obs.size), 10)

        cut: int | None = None
        reason: str | None = None
        for i in range(len(ts_obs)):
            t_i = ts_obs[: i + 1]
            y_i = ys_obs[: i + 1]
            cut, reason = det.find_cut_index(ts=t_i, ys=y_i, y_contact=y_contact)
            if cut is not None:
                # gap-freeze 触发点：当前窗口里应当存在一个显著的时间缺口。
                dts = np.diff(t_i)
                self.assertGreater(float(np.max(dts)), float(2.0 * np.median(dts)))
                expected_k = int(np.argmax(dts))
                self.assertEqual(int(cut), expected_k)
                self.assertEqual(str(reason), "visibility_gap_freeze")
                break

        self.assertIsNotNone(cut)
        self.assertEqual(str(reason), "visibility_gap_freeze")


if __name__ == "__main__":
    unittest.main()
