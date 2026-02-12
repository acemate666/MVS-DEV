from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

import tennis3d_online.sources as sources_mod


@dataclass(frozen=True)
class _FakeFrame:
    """最小帧对象：用于单测 sources.iter_mvs_image_groups 的 latest-only 行为。

    说明：
    - 这里不依赖真实 MVS SDK。
    - 字段名与 `iter_mvs_image_groups()` 内部 getattr/属性访问对齐。
    """

    cam_index: int
    serial: str
    host_timestamp: int
    dev_timestamp: int
    arrival_monotonic: float


class _FakeQuadCapture:
    """最小 QuadCapture stub。

    说明：
    - 仅实现本单测会触达的属性/方法：get_next_group/drain_events/cameras/assembler。
    - get_next_group 的 timeout_s 参数在 stub 中不会生效，但用于保持接口一致。
    """

    def __init__(self, groups: list[list[_FakeFrame]]) -> None:
        self._groups = list(groups)
        self.cameras = [SimpleNamespace(cam=object())]
        self.assembler = SimpleNamespace(dropped_groups=7, pending_groups=9)

    def drain_events(self, max_items: int = 2000):  # noqa: ANN001
        return []

    def get_next_group(self, timeout_s: float = 0.0):  # noqa: ANN001
        if self._groups:
            return self._groups.pop(0)
        return None


class _StrictTimeoutQuadCapture(_FakeQuadCapture):
    """用于回归：确保 latest-only drain 不会用 timeout_s<=0 导致真实实现无限阻塞。"""

    def __init__(self, groups: list[list[_FakeFrame]]) -> None:
        super().__init__(groups)
        self.seen_timeouts: list[float] = []

    def get_next_group(self, timeout_s: float = 0.0):  # noqa: ANN001
        self.seen_timeouts.append(float(timeout_s))
        # 在真实 QuadCapture 中，timeout_s<=0 会导致“无限等待”。
        # 这里用断言锁死该行为，避免 sources.latest-only 再次引入卡死。
        assert float(timeout_s) > 0.0
        return super().get_next_group(timeout_s=timeout_s)


def _fake_frame_to_bgr(*, binding, cam, frame):  # noqa: ANN001
    # 说明：绕过真实像素格式解码；只要能返回一个 BGR 数组即可。
    return np.zeros((2, 2, 3), dtype=np.uint8)


def test_iter_mvs_image_groups_latest_only_drains_to_latest(monkeypatch) -> None:
    # 说明：latest-only 应该在一次迭代中 drain 掉已就绪的旧组，只处理最新完整组。
    monkeypatch.setattr(sources_mod, "frame_to_bgr", _fake_frame_to_bgr)

    g0 = [_FakeFrame(0, "A", 100, 10, 1.00)]
    g1 = [_FakeFrame(0, "B", 200, 20, 2.00)]
    g2 = [_FakeFrame(0, "C", 300, 30, 3.00)]

    cap = _FakeQuadCapture([g0, g1, g2])

    out = list(
        sources_mod.iter_mvs_image_groups(
            cap=cast(Any, cap),
            binding=cast(Any, object()),
            max_groups=1,
            timeout_s=0.0,
            latest_only=True,
            max_wait_seconds=0.0,
        )
    )

    assert len(out) == 1
    meta, images = out[0]

    # 3 组中只处理最后一组：group_index 应跳到 2（0-based）。
    gi = meta.get("group_index")
    assert gi is not None
    assert int(gi) == 2

    assert meta.get("latest_only_enabled") is True
    sk = meta.get("latest_only_skipped_groups")
    assert sk is not None
    assert int(sk) == 2

    sk_tot = meta.get("latest_only_skipped_groups_total")
    assert sk_tot is not None
    assert int(sk_tot) == 2

    # 组包器诊断字段应尽力透传（便于区分主动跳组 vs assembler 丢组）。
    dropped = meta.get("mvs_assembler_dropped_groups")
    pending = meta.get("mvs_assembler_pending_groups")
    assert dropped is not None
    assert pending is not None
    assert int(dropped) == 7
    assert int(pending) == 9

    # 确认确实处理了最后一组（serial=C）。
    assert set(images.keys()) == {"C"}


def test_iter_mvs_image_groups_latest_only_drain_never_blocks(monkeypatch) -> None:
    monkeypatch.setattr(sources_mod, "frame_to_bgr", _fake_frame_to_bgr)

    g0 = [_FakeFrame(0, "A", 100, 10, 1.00)]
    g1 = [_FakeFrame(0, "B", 200, 20, 2.00)]
    g2 = [_FakeFrame(0, "C", 300, 30, 3.00)]

    cap = _StrictTimeoutQuadCapture([g0, g1, g2])

    out = list(
        sources_mod.iter_mvs_image_groups(
            cap=cast(Any, cap),
            binding=cast(Any, object()),
            max_groups=1,
            timeout_s=0.01,
            latest_only=True,
            max_wait_seconds=0.0,
        )
    )

    assert len(out) == 1
    meta, images = out[0]
    gi = meta.get("group_index")
    assert gi is not None
    assert int(gi) == 2
    assert set(images.keys()) == {"C"}


def test_iter_mvs_image_groups_without_latest_only_processes_in_order(monkeypatch) -> None:
    # 说明：不开 latest-only 时，应按顺序处理第一组（group_index=0），且不应写入 latest-only 字段。
    monkeypatch.setattr(sources_mod, "frame_to_bgr", _fake_frame_to_bgr)

    g0 = [_FakeFrame(0, "A", 100, 10, 1.00)]
    g1 = [_FakeFrame(0, "B", 200, 20, 2.00)]

    cap = _FakeQuadCapture([g0, g1])

    out = list(
        sources_mod.iter_mvs_image_groups(
            cap=cast(Any, cap),
            binding=cast(Any, object()),
            max_groups=1,
            timeout_s=0.0,
            latest_only=False,
            max_wait_seconds=0.0,
        )
    )

    assert len(out) == 1
    meta, images = out[0]

    gi = meta.get("group_index")
    assert gi is not None
    assert int(gi) == 0
    assert "latest_only_enabled" not in meta
    assert "latest_only_skipped_groups" not in meta
    assert "latest_only_skipped_groups_total" not in meta

    assert set(images.keys()) == {"A"}
