"""Config defaults, cost inputs (volatility, spread, impact) and performance metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pytest import approx
from test_sim_helpers import FREE, bounce_panel, flat_panel, make_panel

from mark1.config import MINUTES_PER_YEAR
from mark1.data.panel import synthetic_panel
from mark1.sim import (Order, Side, SimConfig, Simulator, compute_metrics, cost_inputs, format_metrics,
                       sqrt_impact_bps)
from mark1.sim.costs import abdi_ranaldo_spread, corwin_schultz_spread, rolling_vol_bps
from mark1.sim.metrics import drawdown_series


# --------------------------------------------------------------------------- config
def test_config_defaults_match_the_spec():
    c = SimConfig()
    assert (c.maker_fee_bps, c.taker_fee_bps, c.half_spread_bps) == (25.0, 40.0, 2.0)
    assert (c.impact_coef, c.participation_cap, c.latency_bars, c.min_notional) == (0.5, 0.10, 0, 10.0)
    assert (c.initial_cash, c.long_only, c.allow_leverage, c.spread_estimator) == (100_000.0, True, False, None)
    assert c.replace(latency_bars=3).latency_bars == 3 and c.latency_bars == 0


@pytest.mark.parametrize("kw", [dict(participation_cap=0.0), dict(participation_cap=1.5), dict(latency_bars=-1),
                                dict(taker_fee_bps=-1), dict(spread_estimator="nope"), dict(vol_window=1)])
def test_config_validation(kw):
    with pytest.raises(ValueError):
        SimConfig(**kw)


def test_per_asset_half_spread():
    panel = make_panel(np.full((6, 2), 100.0))
    sim = Simulator(panel, FREE.replace(half_spread_bps=[1.0, 10.0]))
    sim.reset(0)
    sim.submit([Order.market(0, Side.BUY, 1), Order.market(1, Side.BUY, 1)])
    sim.step()
    assert sim.fills_frame().set_index("asset")["spread_bps"].to_dict() == {0: 1.0, 1: 10.0}
    with pytest.raises(ValueError):
        Simulator(panel, FREE.replace(half_spread_bps=[1.0, 2.0, 3.0]))


# --------------------------------------------------------------------------- impact
def test_sqrt_impact_formula_and_cap():
    assert sqrt_impact_bps(1.0, 100.0, 20.0, 0.5, 50.0) == approx(0.5 * 20 * 0.1)
    assert sqrt_impact_bps(100.0, 1.0, 20.0, 0.5, 50.0) == 50.0
    assert sqrt_impact_bps(1.0, 0.0, 20.0, 0.5, 50.0) == 50.0


def test_engine_impact_scales_with_sqrt_of_participation_and_is_capped():
    panel = flat_panel(8)
    panel.volume[:] = 1000.0
    # flat prices: sigma stays at default_sigma_bps (10) until enough returns exist (they are all 0 here)
    cfg = SimConfig(half_spread_bps=0, taker_fee_bps=0, participation_cap=1.0, min_notional=0,
                    default_sigma_bps=40.0, vol_min_periods=100)
    out = {}
    for qty in (1.0, 4.0, 400.0):
        sim = Simulator(panel, cfg)
        sim.reset(0)
        sim.submit(Order.market(0, Side.BUY, qty))
        (f,) = sim.step().fills
        out[qty] = f.impact_bps
        assert f.price == approx(100.0 * (1 + f.impact_bps / 1e4))
    assert out[1.0] == approx(0.5 * 40 * math.sqrt(1 / 1000))
    assert out[4.0] == approx(2 * out[1.0])
    assert out[400.0] == approx(0.5 * 40 * math.sqrt(0.4))
    # a much bigger order against thin volume hits the cap
    panel.volume[:] = 1.0
    sim = Simulator(panel, cfg.replace(impact_cap_bps=7.0))
    sim.reset(0)
    sim.submit(Order.market(0, Side.BUY, 1.0))
    assert sim.step().fills[0].impact_bps == 7.0


# --------------------------------------------------------------------------- volatility
def test_rolling_vol_matches_loop_and_skips_invalid_bars():
    p = synthetic_panel(300, seed=1, vol_per_minute=0.002)
    p.valid[100:110, 1] = False
    out = rolling_vol_bps(p.close, p.valid, window=30, min_periods=5)
    for k in (4, 5, 29, 30, 120, 299):
        for i in (0, 1):
            rets = []
            for j in range(max(1, k - 29), k + 1):
                if p.valid[j, i] and p.valid[j - 1, i]:
                    rets.append(math.log(p.close[j, i] / p.close[j - 1, i]))
            if len(rets) >= 5:
                assert out[k, i] == approx(1e4 * math.sqrt(np.mean(np.square(rets))), rel=1e-9)
            else:
                assert np.isnan(out[k, i])


def test_cost_inputs_are_strictly_causal_and_block_independent():
    p = synthetic_panel(500, seed=2, vol_per_minute=0.002)
    other = synthetic_panel(500, seed=77, vol_per_minute=0.01)
    b = 300
    for est in (None, "corwin_schultz", "abdi_ranaldo"):
        cfg = SimConfig(spread_estimator=est, half_spread_bps=0.5)
        sig, hs = cost_inputs(p, cfg, 0, 500)
        q = p.slice(0, 500)                              # rows >= b replaced by unrelated data
        for name in ("open", "high", "low", "close", "volume"):
            getattr(q, name)[b:] = getattr(other, name)[b:]
        sig_q, hs_q = cost_inputs(q, cfg, 0, 500)
        np.testing.assert_array_equal(sig[: b + 1], sig_q[: b + 1])  # row b only sees rows < b
        np.testing.assert_array_equal(np.asarray(hs)[: b + 1], np.asarray(hs_q)[: b + 1])
        assert not np.array_equal(sig[b + 1:], sig_q[b + 1:])        # ... but later rows do change
        sig_blk, hs_blk = cost_inputs(p, cfg, 200, 400)               # same values from another block
        np.testing.assert_allclose(sig[200:400], sig_blk, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(hs)[200:400], np.asarray(hs_blk), rtol=1e-12)


def test_fills_are_bit_identical_whatever_the_episode_start():
    panel = synthetic_panel(9_000, seed=5, vol_per_minute=0.002)
    panel.volume *= 1e3
    prices = []
    for start in (0, 1_234, 5_000):
        sim = Simulator(panel, SimConfig(spread_estimator="corwin_schultz", participation_cap=1.0))
        sim.reset(start, 8_000)
        while sim.t < 7_000:
            sim.step()
        sim.submit(Order.market(0, Side.BUY, 5.0))
        prices.append(sim.step().fills[0].price)
    assert prices[0] == prices[1] == prices[2]


def test_sigma_falls_back_to_default_until_enough_returns():
    p = synthetic_panel(40, seed=3)
    cfg = SimConfig(default_sigma_bps=12.5, vol_min_periods=10)
    sig, _ = cost_inputs(p, cfg, 0, 40)
    assert (sig[:11] == 12.5).all()           # row b uses returns up to b-1: first real value at b=11
    assert sig[11, 0] != 12.5


# --------------------------------------------------------------------------- spread estimators
def test_spread_estimators_recover_a_known_spread():
    p = bounce_panel(1500, spread_bps=10.0, vol_bps=0.0)
    cs = corwin_schultz_spread(p.high, p.low, p.valid, 60)[-1, 0] * 1e4
    ar = abdi_ranaldo_spread(p.high, p.low, p.close, p.valid, 60)[-1, 0] * 1e4
    assert cs == approx(10.0, rel=0.01) and ar == approx(10.0, rel=0.01)
    p = bounce_panel(1500, spread_bps=10.0, vol_bps=1.0)  # a little mid movement: still close for AR
    assert abdi_ranaldo_spread(p.high, p.low, p.close, p.valid, 60)[-1, 0] * 1e4 == approx(10.0, rel=0.1)


def test_spread_estimators_are_nonnegative_and_nan_during_warmup():
    p = synthetic_panel(400, seed=6, vol_per_minute=0.003)
    for full in (corwin_schultz_spread(p.high, p.low, p.valid, 60),
                 abdi_ranaldo_spread(p.high, p.low, p.close, p.valid, 60)):
        assert np.isnan(full[:5]).all() and np.nanmin(full) >= 0


@pytest.mark.parametrize("est", ["corwin_schultz", "abdi_ranaldo"])
def test_estimated_half_spread_uses_floor_and_cap(est):
    p = bounce_panel(600, spread_bps=40.0, vol_bps=0.0)         # true half-spread 20 bps
    _, hs = cost_inputs(p, SimConfig(half_spread_bps=0.0, spread_estimator=est), 0, 600)
    assert hs[-1, 0] == approx(20.0, rel=0.01) and hs[0, 0] == 0.0  # estimate only after warm-up
    _, hs = cost_inputs(p, SimConfig(half_spread_bps=0.0, spread_estimator=est, max_half_spread_bps=8.0), 0, 600)
    assert hs[-1, 0] == approx(8.0)                                   # clipped
    _, hs = cost_inputs(p, SimConfig(half_spread_bps=30.0, spread_estimator=est), 0, 600)
    assert (hs == 30.0).all()                                         # the configured floor wins
    _, hs = cost_inputs(p, SimConfig(half_spread_bps=1.0), 0, 600)
    assert (hs == 1.0).all()                                          # no estimator: just the floor


def test_engine_uses_estimated_spread_in_fill_price():
    panel = bounce_panel(300, spread_bps=20.0, vol_bps=0.0)
    cfg = SimConfig(half_spread_bps=1.0, spread_estimator="abdi_ranaldo", impact_coef=0, taker_fee_bps=0,
                    participation_cap=1.0, min_notional=0)
    sim = Simulator(panel, cfg)
    sim.reset(0)
    while sim.t < 200:
        sim.step()
    sim.submit(Order.market(0, Side.BUY, 1.0))
    (f,) = sim.step().fills
    assert f.spread_bps == approx(10.0, rel=0.01)
    assert f.price == approx(panel.open[201, 0] * (1 + f.spread_bps / 1e4))


# --------------------------------------------------------------------------- metrics
def test_constant_growth_curve():
    n = MINUTES_PER_YEAR + 1
    g = 1.5 ** (1 / MINUTES_PER_YEAR) - 1
    eq = 100_000 * (1 + g) ** np.arange(n)
    m = compute_metrics(eq)
    assert m["years"] == approx(1.0) and m["total_return"] == approx(0.5, rel=1e-9)
    assert m["cagr"] == approx(0.5, rel=1e-9)
    assert m["max_drawdown"] == 0.0 and m["max_dd_duration_bars"] == 0 and np.isnan(m["calmar"])
    assert m["ann_vol"] == approx(0.0, abs=1e-9) and np.isnan(m["sharpe"])
    half = compute_metrics(eq[: n // 2 + 1])  # half a year of the same growth rate: CAGR is unchanged
    assert half["years"] == approx(0.5) and half["cagr"] == approx(0.5, rel=1e-6)


def test_cagr_uses_elapsed_time():
    n = MINUTES_PER_YEAR // 4 + 1                 # a quarter of a year, doubling in that time
    eq = 100.0 * 2.0 ** (np.arange(n) / (n - 1))
    m = compute_metrics(eq)
    assert m["total_return"] == approx(1.0) and m["cagr"] == approx(2.0**4 - 1, rel=1e-9)


def test_drawdown_and_duration():
    eq = np.array([100, 120, 90, 95, 130, 104, 131.0])
    m = compute_metrics(eq)
    assert m["max_drawdown"] == approx(-0.25) and m["max_dd_duration_bars"] == 2
    assert m["max_dd_duration_days"] == approx(2 / 1440)
    dd = drawdown_series(eq)
    assert dd[3] == approx(95 / 120 - 1) and dd[5] == approx(104 / 130 - 1) and dd[-1] == 0.0
    assert math.isinf(m["cagr"])  # a 31% gain in 6 minutes annualises past float range, no exception


def test_calmar_and_long_drawdown_duration():
    n = MINUTES_PER_YEAR + 1
    eq = 100.0 * 1.5 ** (np.arange(n) / (n - 1))
    eq[1000:2000] *= 0.8                                  # a 20% dip that lasts 1000 bars
    m = compute_metrics(eq)
    assert m["max_drawdown"] == approx(-0.2, abs=1e-4) and m["max_dd_duration_bars"] == 1000
    assert m["cagr"] == approx(0.5, rel=1e-6) and m["calmar"] == approx(0.5 / 0.2, rel=1e-3)


def test_sharpe_sortino_match_independent_formula():
    rng = np.random.default_rng(0)
    r = rng.normal(2e-6, 1e-3, 5000)
    eq = 1e5 * np.cumprod(np.concatenate([[1.0], 1 + r]))
    m = compute_metrics(eq)
    rr = eq[1:] / eq[:-1] - 1
    assert m["sharpe"] == approx(rr.mean() / rr.std(ddof=1) * math.sqrt(MINUTES_PER_YEAR), rel=1e-9)
    down = np.sqrt(np.mean(np.minimum(rr, 0.0) ** 2))
    assert m["sortino"] == approx(rr.mean() / down * math.sqrt(MINUTES_PER_YEAR), rel=1e-9)
    assert m["ann_vol"] == approx(rr.std(ddof=1) * math.sqrt(MINUTES_PER_YEAR), rel=1e-9)
    rf = compute_metrics(eq, risk_free=0.05)
    assert rf["sharpe"] < m["sharpe"]


def test_turnover_costs_trades_exposure_and_win_rate():
    eq = np.full(MINUTES_PER_YEAR + 1, 100_000.0)
    m = compute_metrics(eq, exposure=np.full(len(eq), 0.6), fees_paid=300.0, slippage_cost=120.0,
                        traded_notional=250_000.0, n_trades=12, round_trip_pnls=[5.0, -2.0, 7.0, 0.0])
    assert m["turnover_annual"] == approx(2.5) and m["total_costs"] == approx(420.0)
    assert m["n_trades"] == 12 and m["avg_exposure"] == approx(0.6)
    assert m["n_round_trips"] == 4 and m["win_rate"] == approx(0.5)
    assert np.isnan(compute_metrics(eq)["win_rate"])


def test_format_metrics_pretty_print():
    text = format_metrics(compute_metrics(np.array([100.0, 101.0, 99.0, 103.0])), title="demo")
    lines = text.splitlines()
    assert lines[0] == "demo" and any(l.startswith("Sharpe") for l in lines)
    assert "%" in next(l for l in lines if l.startswith("Total return"))
    assert "n/a" in format_metrics({"win_rate": float("nan")})
    with pytest.raises(ValueError):
        compute_metrics(np.array([1.0]))
