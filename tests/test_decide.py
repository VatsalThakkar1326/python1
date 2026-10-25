import numpy as np

from mark1.decide.engine import DecisionConfig, DecisionEngine, moments_from_quantiles

LEVELS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
Z = np.array([-1.645, -1.2816, -0.6745, 0.0, 0.6745, 1.2816, 1.645])
TAUS = np.array([15, 60, 240, 1440])


def curve(means, stds):
    """Normal quantiles for one coin: means/stds per horizon -> [K, 1, Q]."""
    return (np.asarray(means)[:, None] + np.asarray(stds)[:, None] * Z[None, :])[:, None, :]


def run(q, held=0.0, cost=0.0045, cfg=None):
    eng = DecisionEngine(cfg or DecisionConfig())
    return eng.decide(q, TAUS, LEVELS, np.array([held]), np.array([cost]), np.array([True]), np.array([100.0]), 1.0)


def test_moments_recover_normal():
    mean, std = moments_from_quantiles(curve([0.01], [0.02])[:, 0], LEVELS)
    assert abs(mean[0] - 0.01) < 1e-9 and abs(std[0] - 0.02) < 1e-3


def test_no_entry_when_edge_below_round_trip():
    q = curve([0.001, 0.003, 0.006, 0.008], [0.004, 0.008, 0.015, 0.03])   # best 0.8% < 2 x 0.45% + margin
    w, info = run(q)
    assert w[0] == 0 and info.action == ["flat"]


def test_entry_when_edge_beats_costs():
    q = curve([0.002, 0.006, 0.015, 0.03], [0.004, 0.008, 0.015, 0.03])
    w, info = run(q)
    assert w[0] > 0 and info.action == ["enter"]


def test_hold_through_small_dip_but_exit_big_dip():
    # Down 0.5% in an hour, up 1.3% tomorrow: the dip is smaller than a 0.9% round trip -> hold.
    q = curve([-0.002, -0.005, 0.002, 0.013], [0.004, 0.008, 0.015, 0.03])
    w, info = run(q, held=0.1)
    assert info.action == ["hold"] and w[0] > 0
    # Down 1.5% in an hour: bigger than the round trip -> sell now, buy back later.
    q = curve([-0.004, -0.015, 0.002, 0.013], [0.004, 0.008, 0.015, 0.03])
    w, info = run(q, held=0.1)
    assert info.action == ["dip_exit"] and w[0] == 0


def test_exit_when_nothing_positive():
    q = curve([-0.001, -0.002, -0.003, -0.004], [0.004, 0.008, 0.015, 0.03])
    w, info = run(q, held=0.1)
    assert info.action == ["exit"] and w[0] == 0


def test_caps_and_drawdown_brake():
    cfg = DecisionConfig(max_weight=0.2)
    eng = DecisionEngine(cfg)
    q = np.repeat(curve([0.01, 0.03, 0.06, 0.1], [0.004, 0.008, 0.015, 0.03]), 8, axis=1)
    args = (TAUS, LEVELS, np.zeros(8), np.full(8, 0.0045), np.ones(8, bool), np.full(8, 100.0))
    w, _ = eng.decide(q, *args, equity=1.0)
    assert np.all(w <= 0.2 + 1e-12) and w.sum() <= 1.0 + 1e-12
    w2, _ = eng.decide(q, *args, equity=0.6)        # 40% drawdown -> floor multiplier
    assert np.isclose(w2.sum(), w.sum() * cfg.drawdown_floor)


def test_gross_cap_holds_after_keeping_small_changes():
    # Held coins slightly above their new targets are kept (small change), which used to push
    # the total over 100%.
    eng = DecisionEngine(DecisionConfig(max_weight=0.2, min_trade_weight=0.01))
    q = np.repeat(curve([0.01, 0.03, 0.06, 0.1], [0.004, 0.008, 0.015, 0.03]), 6, axis=1)
    cur = np.full(6, 0.175)          # 105% invested before the decision
    w, _ = eng.decide(q, TAUS, LEVELS, cur, np.full(6, 0.0045), np.ones(6, bool), np.full(6, 100.0), 1.0)
    assert w.sum() <= 1.0 + 1e-12
