import logging
import numpy as np

from curve_v3.configs import CurveV3Config, OnlinePriorConfig, PriorConfig
from curve_v3.prior import OnlinePriorWeights, build_prior_candidates, maybe_init_online_prior, maybe_update_online_prior
from curve_v3.types import BounceEvent, Candidate


def test_online_prior_applies_to_candidate_weights(tmp_path):
    """启用在线沉淀后，prior 候选权重应按 JSON 权重池偏置。"""

    e_bins = (0.60, 0.80)
    kt_bins = (0.50,)
    ang_bins = (0.0, 0.1)

    path = tmp_path / "online_prior.json"

    # M = 2 * 1 * 2 = 4，顺序需与 `curve_v3.prior.candidates.build_prior_candidates` 一致：
    # for e in e_bins:
    #   for kt in kt_bins:
    #     for ang in ang_bins:
    expected = np.array([0.7, 0.1, 0.1, 0.1], dtype=float)
    OnlinePriorWeights(
        e_bins=e_bins,
        kt_bins=kt_bins,
        angle_bins_rad=ang_bins,
        weights=expected,
        ema_alpha=0.05,
        eps=1e-8,
    ).save_json(path)

    cfg = CurveV3Config(
        prior=PriorConfig(e_bins=e_bins, kt_bins=kt_bins, kt_angle_bins_rad=ang_bins),
        online_prior=OnlinePriorConfig(
            online_prior_enabled=True,
            online_prior_path=str(path),
            online_prior_autosave=False,
        ),
    )

    online_prior = maybe_init_online_prior(cfg=cfg, logger=logging.getLogger("test"))
    bounce = BounceEvent(
        t_rel=0.2,
        x=0.0,
        z=0.0,
        v_minus=np.array([1.0, -2.0, 3.0], dtype=float),
    )

    candidates = build_prior_candidates(
        bounce=bounce,
        cfg=cfg,
        prior_model=None,
        online_prior=online_prior,
    )
    assert len(candidates) == int(expected.size)

    got = np.array([float(c.weight) for c in candidates], dtype=float)
    assert np.allclose(got, expected, atol=1e-12)


def test_online_prior_updates_and_saves(tmp_path):
    """融合后更新应写回 JSON（默认不影响主流程，失败也不应抛异常）。"""

    e_bins = (0.60, 0.80)
    kt_bins = (0.50,)
    ang_bins = (0.0, 0.1)

    path = tmp_path / "online_prior.json"

    cfg = CurveV3Config(
        prior=PriorConfig(e_bins=e_bins, kt_bins=kt_bins, kt_angle_bins_rad=ang_bins),
        online_prior=OnlinePriorConfig(
            online_prior_enabled=True,
            online_prior_path=str(path),
            online_prior_ema_alpha=1.0,
            online_prior_autosave=True,
        ),
    )

    online_prior = maybe_init_online_prior(cfg=cfg, logger=logging.getLogger("test"))
    bounce = BounceEvent(
        t_rel=0.2,
        x=0.0,
        z=0.0,
        v_minus=np.array([1.0, -2.0, 3.0], dtype=float),
    )

    base = build_prior_candidates(
        bounce=bounce,
        cfg=cfg,
        prior_model=None,
        online_prior=online_prior,
    )
    assert len(base) == 4

    # 构造一个“极端后验”：把全部权重压到第 0 个候选。
    updated: list[Candidate] = []
    for i, c in enumerate(base):
        updated.append(
            Candidate(
                e=float(c.e),
                kt=float(c.kt),
                weight=1.0 if i == 0 else 0.0,
                v_plus=np.asarray(c.v_plus, dtype=float),
                kt_angle_rad=float(getattr(c, "kt_angle_rad", 0.0)),
                ax=float(c.ax),
                az=float(c.az),
            )
        )

    maybe_update_online_prior(
        online_prior=online_prior,
        cfg=cfg,
        candidates=updated,
        logger=logging.getLogger("test"),
    )

    assert path.exists()

    inst = OnlinePriorWeights.load_json(path)
    assert int(inst.num_updates) >= 1

    # ema_alpha=1.0 => 应该“近似”学成 one-hot。
    # 注意：OnlinePriorWeights 会对权重加 eps 下限，避免永远学不回来，因此不会出现严格 0。
    got = np.asarray(inst.weights, dtype=float).reshape(-1)
    assert float(got[0]) > 0.999999
    assert float(np.sum(got[1:])) < 1e-6
