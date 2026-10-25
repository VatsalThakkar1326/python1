import numpy as np

from mark1.data.panel import synthetic_panel
from mark1.features.build import WARMUP, build_features
from mark1.features.targets import forward_targets, invalid_cumsum


def _perturb_future(p, t0, seed=1):
    q = p.slice(0, p.n_times)
    rng = np.random.default_rng(seed)
    shock = np.exp(rng.normal(0, 0.05, size=q.close[t0 + 1:].shape))
    for arr in (q.open, q.high, q.low, q.close):
        arr[t0 + 1:] *= shock
    q.volume[t0 + 1:] *= 3.0
    q.quote_volume[t0 + 1:] *= 3.0
    q.taker_buy_volume[t0 + 1:] *= 0.1
    q.trades[t0 + 1:] += 7
    return q


def test_features_have_no_lookahead():
    p = synthetic_panel(n_times=WARMUP + 600, seed=3)
    t0 = WARMUP + 300
    a = build_features(p)
    b = build_features(_perturb_future(p, t0))
    np.testing.assert_array_equal(a.asset[: t0 + 1], b.asset[: t0 + 1])
    np.testing.assert_array_equal(a.market[: t0 + 1], b.market[: t0 + 1])
    np.testing.assert_array_equal(a.sigma[: t0 + 1], b.sigma[: t0 + 1])
    # and the future really did change, so the test is not vacuous
    assert not np.array_equal(a.asset[t0 + 1:], b.asset[t0 + 1:])


def test_ready_mask_respects_warmup_and_listing():
    p = synthetic_panel(n_times=WARMUP + 200, seed=4)
    p.valid[:100, 1] = False  # asset 1 lists 100 minutes late
    fs = build_features(p)
    assert not fs.ready[: WARMUP, 0].any() and fs.ready[WARMUP:, 0].all()
    assert not fs.ready[: WARMUP + 100, 1].any() and fs.ready[WARMUP + 100:, 1].all()


def test_forward_targets_values_and_masks():
    T, N = 50, 2
    log_close = np.cumsum(np.full((T, N), 0.001), axis=0)
    sigma = np.full((T, N), 0.001, dtype=np.float32)
    valid = np.ones((T, N), dtype=bool)
    valid[30, 1] = False
    inv = invalid_cumsum(valid)
    t = np.array([10, 45])
    tau = np.array([[4, 25], [1, 9]])
    y, mask = forward_targets(log_close, sigma, inv, t, tau)
    # constant drift of one sigma per minute => y = tau / sqrt(tau) = sqrt(tau)
    assert np.isclose(y[0, 0, 0], 2.0) and mask[0, 0].all()
    assert mask[0, 1, 0] and not mask[0, 1, 1]       # row 30 invalid for asset 1 inside (10, 35]
    assert mask[1, 0].all() and not mask[1, 1].any()  # 45 + 9 runs past the end
