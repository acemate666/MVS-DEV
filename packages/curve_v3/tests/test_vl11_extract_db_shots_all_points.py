import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from curve_v3.types import BallObservation
from curve_v3.offline.vl11 import TrajectoryFilterConfig, extract_db_shots


def _insert_points(conn: sqlite3.Connection, *, table: str, points: list[BallObservation]) -> None:
    cur = conn.cursor()
    cur.execute(f"CREATE TABLE {table} (ts REAL, abs_loc TEXT)")
    for p in points:
        abs_loc = json.dumps([float(p.x), float(p.y), float(p.z), float(p.t)])
        cur.execute(f"INSERT INTO {table} (ts, abs_loc) VALUES (?, ?)", (float(p.t), abs_loc))
    conn.commit()


def _make_bounce_shot_with_noise() -> list[BallObservation]:
    """构造一段带 bounce 的 shot，并插入明显不物理的噪点。

    目标：
        让 extract_db_shots 在开启 filter_config 时，points 变少，但 all_points 仍保留全量。
    """

    pts: list[BallObservation] = []

    # 让 z 持续递减（符合离线过滤默认 forward_ratio 的期望）。
    dt = 0.02
    t0 = 100.0
    z0 = 10.0

    # y 先降后升，制造一个明显 valley 作为 bounce。
    # 需要满足 find_bounce_index 的默认 min_pre/min_post 等约束。
    n_pre = 22
    n_post = 14
    n = n_pre + n_post
    valley_i = 18

    for i in range(n):
        t = float(t0 + i * dt)
        x = float(0.1 * i)
        z = float(z0 - 0.20 * i)

        if i <= valley_i:
            y = float(2.2 - 0.07 * i)  # 下降
        else:
            y = float((2.2 - 0.07 * valley_i) + 0.10 * (i - valley_i))  # 上升

        pts.append(BallObservation(x=x, y=y, z=z, t=t))

        # 每隔几帧插入一个“高速跳变”噪点，应该被过滤逻辑剔除。
        if i % 5 == 3:
            t_noise = float(t + 0.5 * dt)
            pts.append(
                BallObservation(
                    x=float(x + 5.0),
                    y=float(max(0.2, y - 0.4)),
                    z=float(z + 5.0),
                    t=t_noise,
                )
            )

    pts.sort(key=lambda p: float(p.t))
    return pts


class TestVl11ExtractDbShotsAllPoints(unittest.TestCase):
    def test_extract_db_shots_keeps_all_points_when_filter_enabled(self):
        points = _make_bounce_shot_with_noise()

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "tmp_vl11.db"
            conn = sqlite3.connect(str(db_path))
            try:
                _insert_points(conn, table="ball_info", points=points)
            finally:
                conn.close()

            cfg = TrajectoryFilterConfig(
                enabled=True,
                max_speed_m_s=40.0,
                min_forward_ratio=0.70,
                min_downward_acc_m_s2=2.0,
                max_gravity_mad_ratio=1.2,
                min_run_points=10,
                max_skip_points=2,
            )

            shots = extract_db_shots(
                str(db_path),
                table="ball_info",
                abs_loc_col="abs_loc",
                order_by_col="ts",
                expected_num_shots=None,
                gap_s=1.0,
                min_shot_points=5,
                min_post_points=5,
                y_threshold_m=None,
                filter_config=cfg,
                return_start_config=None,
                only_bounce=False,
            )

        self.assertEqual(len(shots), 1)
        s = shots[0]

        self.assertGreater(len(s.all_points), 0)
        self.assertEqual(len(s.all_points), len(points))

        # 开启过滤后，points 应该不大于 all_points；在该构造下应明显变少。
        self.assertLess(len(s.points), len(s.all_points))

        self.assertAlmostEqual(float(s.all_points[0].t), float(points[0].t))
        self.assertAlmostEqual(float(s.all_points[-1].t), float(points[-1].t))

        # all_bounce_index 应该指向 all_points 内部。
        idx = s.all_bounce_index
        self.assertIsNotNone(idx)
        # 说明：unittest 的 assertIsNotNone 不会被静态类型检查器用于类型收窄。
        # 这里用显式断言保证 idx 为 int，避免 IDE 误报。
        assert idx is not None
        self.assertGreaterEqual(int(idx), 0)
        self.assertLess(int(idx), len(s.all_points))


if __name__ == "__main__":
    unittest.main()
