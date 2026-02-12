import math
import unittest

from curve_v3.types import BallObservation
from curve_v3.offline.vl11.filtering import extract_best_inlier_run, run_quality_score
from curve_v3.offline.vl11.split import find_bounce_index, split_points_into_shots
from curve_v3.offline.vl11.types import TrajectoryFilterConfig


def _make_ballistic_segment(*, n: int, dt: float, t0: float) -> list[BallObservation]:
    """构造一个简单抛体段（y 轴受重力），用于验证过滤不会误删主段。"""

    g = 9.8
    x0, y0, z0 = 0.2, 1.0, 1.0
    vx, vy, vz = 0.8, 4.0, 8.0

    pts: list[BallObservation] = []
    for i in range(int(n)):
        t = float(t0 + i * dt)
        x = float(x0 + vx * (t - t0))
        y = float(y0 + vy * (t - t0) - 0.5 * g * (t - t0) * (t - t0))
        z = float(z0 + vz * (t - t0))
        pts.append(BallObservation(x=x, y=y, z=z, t=t))

    return pts


def _insert_fast_noise(points: list[BallObservation], *, every: int = 5) -> list[BallObservation]:
    """在抛体点中插入一些“高速跳变”的噪点。"""

    out: list[BallObservation] = []
    for i, p in enumerate(points):
        out.append(p)
        if (i + 1) % int(every) == 0:
            t_noise = float(p.t + 0.5 * (points[1].t - points[0].t))
            # 构造一个会导致速度非常大的跳变点（典型手晃/误识别）。
            out.append(
                BallObservation(
                    x=float(p.x + (2.0 if (i % 2 == 0) else -2.0)),
                    y=float(max(0.2, p.y - 0.3)),
                    z=float(p.z + (1.0 if (i % 2 == 0) else -1.0)),
                    t=t_noise,
                )
            )

    out.sort(key=lambda q: float(q.t))
    return out


def _make_hand_waving_segment(*, n: int, dt: float, t0: float) -> list[BallObservation]:
    """构造一个“手里乱挥”段：z 方向来回，y 不符合重力抛物线。"""

    pts: list[BallObservation] = []
    x, y, z = 0.0, 0.8, 0.0
    for i in range(int(n)):
        t = float(t0 + i * dt)
        x += 0.02 * (1.0 if (i % 2 == 0) else -1.0)
        z += 0.08 * (1.0 if (i % 3 == 0) else -1.0)
        y = 0.8 + 0.15 * math.sin(6.0 * (t - t0))
        pts.append(BallObservation(x=float(x), y=float(y), z=float(z), t=t))
    return pts


class TestVl11NoiseFilter(unittest.TestCase):
    def test_find_bounce_index_rejects_shallow_valley_far_from_low_y(self):
        # 构造一个“中途小凹陷”，但该凹陷离 y 的低值区域很远。
        # 期望：不要把它误判成 bounce。
        pts: list[BallObservation] = []
        t0 = 0.0
        dt = 0.03
        ys = [
            0.62,
            0.70,
            0.78,
            0.86,
            0.94,
            1.02,
            1.10,
            1.18,
            1.26,
            1.34,
            1.30,
            1.22,  # shallow valley (candidate) but far above low-y region
            1.28,
            1.36,
            1.44,
            1.52,
            1.60,
            1.68,
            1.76,
            1.84,
            1.92,
        ]
        for i, y in enumerate(ys):
            t = float(t0 + i * dt)
            pts.append(BallObservation(x=0.0, y=float(y), z=1.0, t=t))

        bidx = find_bounce_index(pts)
        self.assertIsNone(bidx)

    def test_split_points_into_shots_auto_by_gap(self):
        pts: list[BallObservation] = []
        # 第一段
        for i in range(6):
            t = 1.0 + 0.03 * i
            pts.append(BallObservation(x=0.0 + 0.1 * i, y=1.0, z=0.0, t=t))
        # gap
        for i in range(6):
            t = 10.0 + 0.03 * i
            pts.append(BallObservation(x=1.0 + 0.1 * i, y=1.0, z=0.0, t=t))

        groups = split_points_into_shots(
            pts,
            expected_num_shots=None,
            gap_s=2.0,
            min_shot_points=3,
        )
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]), 6)
        self.assertEqual(len(groups[1]), 6)

    def test_extract_best_inlier_run_prefers_ballistic(self):
        ballistic = _make_ballistic_segment(n=45, dt=0.02, t0=0.0)
        mixed = _insert_fast_noise(ballistic, every=4)

        waving = _make_hand_waving_segment(n=60, dt=0.02, t0=10.0)

        cfg = TrajectoryFilterConfig(
            enabled=True,
            max_speed_m_s=40.0,
            min_forward_ratio=0.70,
            max_gravity_mad_ratio=1.2,
            min_run_points=12,
        )

        run_ball = extract_best_inlier_run(mixed, cfg)
        self.assertGreaterEqual(len(run_ball), 35)

        score_ball = float(run_quality_score(run_ball, cfg))
        self.assertGreater(score_ball, 0.0)

        run_noise = extract_best_inlier_run(waving, cfg)
        score_noise = float(run_quality_score(run_noise, cfg))

        # 过滤应更偏好抛体段。
        self.assertGreater(score_ball, score_noise)

    def test_extract_best_inlier_run_prefers_quality_over_length(self):
        # 前段：速度/时间都“合理”，但 y 近似线性（a_y ~ 0），且 z 方向来回（forward_ratio 低）。
        # 后段：标准抛体（a_y ~ -g）且 z 方向一致。
        # 期望：过滤结果应更偏好后段（更像球在飞），而不是把前段一起保留下来。

        dt = 0.02
        t0 = 0.0
        n_prefix = 60
        n_ball = 30

        pts: list[BallObservation] = []
        for i in range(n_prefix):
            t = float(t0 + i * dt)
            y = float(1.0 + 0.05 * (t - t0))
            z = 0.0 if (i % 2 == 0) else float(0.02)  # vz 交替正负，但 |vz| 仍可过门禁
            pts.append(BallObservation(x=0.0, y=y, z=z, t=t))

        # 让后段在 z 上连续衔接，避免边界速度超阈值。
        z_last = float(pts[-1].z)
        t_start = float(pts[-1].t + dt)
        g = 9.8
        vy0 = 3.5
        vz = 4.0
        y_start = float(pts[-1].y)
        for j in range(n_ball):
            t = float(t_start + j * dt)
            tau = float(t - t_start)
            y = float(y_start + vy0 * tau - 0.5 * g * tau * tau)
            z = float(z_last + vz * tau)
            pts.append(BallObservation(x=0.0, y=y, z=z, t=t))

        cfg = TrajectoryFilterConfig(
            enabled=True,
            min_dt_s=0.005,
            max_dt_s=0.200,
            max_speed_m_s=40.0,
            z_speed_range=(0.5, 27.0),
            min_forward_ratio=0.70,
            min_downward_acc_m_s2=2.0,
            max_gravity_mad_ratio=1.2,
            min_run_points=12,
            max_skip_points=3,
        )

        run = extract_best_inlier_run(pts, cfg)
        self.assertGreaterEqual(len(run), 20)
        # 应当把前面那段“低质量前缀”大幅裁掉：run 的起点时间应明显晚于 t0。
        self.assertGreater(float(run[0].t), 0.5)


if __name__ == "__main__":
    unittest.main()
