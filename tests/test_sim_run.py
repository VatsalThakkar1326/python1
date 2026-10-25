"""Run loop, target-weight rebalancing, results object, benchmarks and a speed smoke test."""

from __future__ import annotations

import time

import numpy as np
import pytest
from pytest import approx
from test_sim_helpers import FREE, flat_panel, make_panel

from mark1.config import MINUTES_PER_YEAR
from mark1.data.panel import synthetic_panel
from mark1.sim import (BacktestResult, Decision, EqualWeightBuyAndHold, FixedWeights, Order, Side,
                       SimConfig, Simulator, buy_and_hold_benchmark, cash_benchmark, run_backtest)


def liquid_panel(n_times=500, n_assets=3, seed=0, **kw):
    p = synthetic_panel(n_times=n_times, symbols=[f"A{i}" for i in range(n_assets)], seed=seed, **kw)
    p.volume *= 1e4
    return p


# --------------------------------------------------------------------------- target weights
def test_target_weights_become_market_orders_filled_next_bar():
    panel = make_panel(np.column_stack([np.full(6, 100.0), np.full(6, 50.0)]))
    res = run_backtest(panel, lambda t, m, p: np.array([0.5, 0.25]) if t == 0 else None, FREE)
    f = res.fills
    assert f["t"].tolist() == [1, 1] and set(f["tag"]) == {"rebalance"}
    assert f.set_index("asset")["qty"].to_dict() == approx({0: 500.0, 1: 500.0})
    assert res.positions.iloc[0].tolist() == [0.0, 0.0]          # weights before the first trade
    assert res.targets.iloc[0].tolist() == [0.5, 0.25]
    assert res.equity.iloc[-1] == approx(100_000)


def test_buy_and_hold_with_costs_scales_buys_to_fit_cash():
    panel = liquid_panel()
    res = run_backtest(panel, FixedWeights([1 / 3] * 3), SimConfig(), decision_every=10_000)
    f = res.fills
    assert len(f) == 3 and (res.orders["status"] == "FILLED").sum() >= 2
    # fees + spread + impact cost money but the account is nearly fully invested, never overspent
    assert 0.99 < res.exposure.iloc[1:].mean() <= 1.0 + 1e-12
    cash = 100_000 - (f["qty"] * f["price"]).sum() - f["fee"].sum()
    assert -1e-6 <= cash < 400


def test_rebalance_threshold_and_min_notional_skip_small_trades():
    panel = liquid_panel(600, 2, seed=1)
    w = FixedWeights([0.5, 0.5])
    base = run_backtest(panel, w, SimConfig(), decision_every=1)
    coarse = run_backtest(panel, w, SimConfig(), decision_every=1, rebalance_threshold=0.002)
    assert len(coarse.fills) < len(base.fills)
    assert coarse.fills["notional"].iloc[2:].min() >= 0.002 * 90_000
    huge_min = run_backtest(panel, w, SimConfig(min_notional=1e9), decision_every=1)
    assert huge_min.fills.empty and huge_min.equity.iloc[-1] == 100_000


def test_decision_every_and_range():
    panel = flat_panel(100)
    seen = []
    res = run_backtest(panel, lambda t, m, p: seen.append((t, m.close.shape[0])), FREE, start=10, end=50,
                       decision_every=7)
    assert [t for t, _ in seen] == [10, 17, 24, 31, 38, 45]
    assert all(rows == t + 1 for t, rows in seen)
    assert len(res.equity) == 40 and res.equity.index[0] == panel.times[10]
    assert len(res.positions) == len(seen) and res.positions.index[0] == panel.times[10]
    assert (res.start, res.end, res.decision_every) == (10, 50, 7)


def test_pending_rebalance_orders_are_cancelled_not_stacked():
    # cap lets only 10 units per bar through; target needs 500 units -> orders stay pending
    panel = make_panel(np.full(40, 100.0), v=100.0)
    cfg = FREE.replace(participation_cap=0.10)
    res = run_backtest(panel, lambda t, m, p: np.array([0.5]), cfg, rebalance_threshold=0.0)
    orders = res.orders
    assert (orders["tag"] == "rebalance").all()
    # each decision cancelled its predecessor (reason user) except the last, which stays open
    assert (orders["status"] == "CANCELLED").sum() >= len(orders) - 2
    assert res.fills["qty"].max() <= 10 + 1e-9
    held = res.fills["qty"].sum()
    assert held == approx(10 * len(res.fills))                       # 10 per bar, no more
    assert held < 500                                                # never overshoots the target


def test_latency_with_rebalancing_each_bar_does_not_overshoot():
    panel = liquid_panel(200, 2, seed=3)
    cfg = FREE.replace(latency_bars=3)
    res = run_backtest(panel, FixedWeights([0.4, 0.4]), cfg, decision_every=1)
    pos = res.positions.iloc[-1]
    assert abs(pos.sum() - 0.8) < 0.01 and (pos < 0.41).all()       # converged on target, no stacking
    assert res.fills["t"].min() == 4                                  # first fill at t + 1 + latency


def test_strategy_return_types_and_explicit_orders_and_cancel():
    panel = flat_panel(12)
    seen = {}

    def strat(t, market, pf):
        if t == 0:
            return Decision(orders=[Order.limit(0, Side.BUY, 1, 50.0, tag="lim")])
        if t == 1:
            (o,) = pf.open_orders
            seen["open"] = (o.id, o.tag, o.remaining)
            return Decision(cancel=[o.id], orders=[Order.market(0, Side.BUY, 2)])
        if t == 3:
            return [0.0]  # bare weights: sell everything
        return None

    res = run_backtest(panel, strat, FREE)
    assert seen["open"][1:] == ("lim", 1.0)
    o = res.orders.set_index("order_id")
    assert o.loc[seen["open"][0], "status"] == "CANCELLED"
    assert res.fills["side"].tolist() == ["buy", "sell"] and res.fills["qty"].tolist() == approx([2, 2])
    with pytest.raises(TypeError):
        run_backtest(panel, lambda t, m, p: "buy", FREE)


@pytest.mark.parametrize("bad", [[-0.1, 0.5], [0.7, 0.7], [np.nan, 0.1], [0.1]])
def test_invalid_target_weights_raise(bad):
    with pytest.raises(ValueError):
        run_backtest(make_panel(np.full((5, 2), 100.0)), lambda t, m, p: np.array(bad), FREE)


def test_rotation_sells_first_so_the_buys_are_funded():
    panel = make_panel(np.column_stack([np.full(8, 100.0), np.full(8, 50.0)]))
    cfg = SimConfig(half_spread_bps=0, impact_coef=0, participation_cap=1.0)

    def rotate(t, m, p):
        return np.array([0.95, 0.0]) if t == 0 else (np.array([0.0, 0.95]) if t == 3 else None)

    res = run_backtest(panel, rotate, cfg)
    second = res.fills[res.fills["t"] == 4]
    assert second["side"].tolist() == ["sell", "buy"]            # the sale is processed first
    assert res.positions.iloc[-1].tolist() == approx([0.0, 0.95], abs=0.01)
    assert res.orders["reason"].eq("insufficient_funds").sum() <= 1  # at most the last buy is trimmed


def test_target_weights_with_shorts_and_lot_size():
    panel = flat_panel(8, 100.0)
    cfg = FREE.replace(long_only=False, qty_step=0.5)
    res = run_backtest(panel, lambda t, m, p: np.array([-0.5]) if t == 0 else None, cfg)
    assert res.fills["side"].tolist() == ["sell"] and res.fills["qty"].iloc[0] == approx(500.0)
    res = run_backtest(panel, lambda t, m, p: np.array([0.0123456]), FREE.replace(qty_step=0.5))
    assert res.fills["qty"].iloc[0] % 0.5 == 0 and res.fills["qty"].iloc[0] == approx(12.0)


def test_result_object_contents():
    panel = liquid_panel(300)
    cfg = SimConfig()
    res = run_backtest(panel, FixedWeights([0.3, 0.3, 0.3]), cfg, start=50, end=250, decision_every=20)
    assert isinstance(res, BacktestResult) and res.config is cfg and res.symbols == panel.symbols
    assert len(res.equity) == 200 and len(res.exposure) == 200
    assert res.positions.shape == (10, 3) and res.targets.shape == (10, 3)
    assert {"time", "symbol", "side", "qty", "price", "fee", "slippage", "order_type", "tag"} <= set(res.fills)
    assert {"order_id", "status", "reason", "filled_qty", "created_time"} <= set(res.orders)
    assert res.metrics["n_trades"] == len(res.fills) and res.metrics["fees_paid"] == approx(res.fills["fee"].sum())
    assert res.equity.iloc[0] == 100_000
    text = res.summary("test")
    assert "Sharpe" in text and "Max drawdown" in text and repr(res).startswith("BacktestResult(")


def test_runs_are_deterministic_and_record_flags():
    panel = liquid_panel(300, seed=7)
    a = run_backtest(panel, FixedWeights([0.2, 0.2, 0.2]), SimConfig(), decision_every=5)
    b = run_backtest(panel, FixedWeights([0.2, 0.2, 0.2]), SimConfig(), decision_every=5)
    assert a.equity.equals(b.equity) and a.fills.equals(b.fills)
    lean = run_backtest(panel, FixedWeights([0.2, 0.2, 0.2]), SimConfig(log_fills=False, log_orders=False),
                        decision_every=5, record_positions=False)
    assert lean.fills.empty and lean.orders.empty
    assert lean.equity.equals(a.equity) and lean.metrics["n_trades"] == a.metrics["n_trades"]


# --------------------------------------------------------------------------- benchmarks
def test_buy_and_hold_benchmark_equal_weight_over_valid_assets():
    n = 50
    o = np.column_stack([np.linspace(100, 150, n), np.linspace(10, 5, n), np.linspace(20, 40, n)])
    valid = np.ones((n, 3), dtype=bool)
    valid[:20, 2] = False  # third asset lists late: not part of the basket
    panel = make_panel(o, valid=valid)
    res = buy_and_hold_benchmark(panel, FREE)
    assert set(res.fills["asset"]) == {0, 1}
    qty = 50_000 / o[0, :2]                     # sized at the decision-time close, filled at bar 1's open
    expected = qty @ o[-1, :2] + 100_000 - qty @ o[1, :2]
    assert res.fills["qty"].tolist() == approx(qty.tolist())
    assert res.equity.iloc[-1] == approx(expected, rel=1e-12)
    assert len(res.fills) == 2  # a single purchase, no later trades


def test_cash_benchmark_flat_and_with_interest():
    panel = flat_panel(MINUTES_PER_YEAR // 100)
    c = cash_benchmark(panel, SimConfig())
    assert (c.equity == 100_000).all() and c.metrics["max_drawdown"] == 0 and c.fills.empty
    r = cash_benchmark(panel, SimConfig(), annual_rate=0.05)
    years = (len(r.equity) - 1) / MINUTES_PER_YEAR
    assert r.equity.iloc[-1] == approx(100_000 * 1.05**years)
    assert r.metrics["cagr"] == approx(0.05, rel=1e-6)


def test_equal_weight_strategy_is_stateless():
    panel = liquid_panel(60)
    strat = EqualWeightBuyAndHold()
    a = run_backtest(panel, strat, SimConfig(), decision_every=10_000)
    b = run_backtest(panel, strat, SimConfig(), decision_every=10_000)
    assert a.equity.equals(b.equity)


def test_stepping_loop_matches_run_backtest():
    panel = liquid_panel(200, seed=9)
    cfg = SimConfig()
    res = run_backtest(panel, FixedWeights([0.3, 0.3, 0.3]), cfg, decision_every=10)
    sim = Simulator(panel, cfg)
    sim.reset(0, 200)
    while not sim.done:
        if sim.t % 10 == 0:
            sim.rebalance(np.array([0.3, 0.3, 0.3]))
        sim.step()
    np.testing.assert_array_equal(sim.equity_curve(), res.equity.to_numpy())


# --------------------------------------------------------------------------- speed
def _speed(panel, strategy, cfg, **kw):
    t0 = time.perf_counter()
    res = run_backtest(panel, strategy, cfg, **kw)
    dt = time.perf_counter() - t0
    return res, (panel.n_times - 1) / dt


def test_speed_smoke(capsys):
    n_t, n_a = 50_000, 20
    panel = synthetic_panel(n_times=n_t, symbols=[f"C{i}" for i in range(n_a)], seed=0)
    panel.volume *= 1e3
    w = np.full(n_a, 1.0 / n_a)
    cases = {
        "buy&hold (decide once)": (EqualWeightBuyAndHold(), dict(decision_every=n_t)),
        "fixed weights, decide every bar, 0.5% threshold": (FixedWeights(w), dict(rebalance_threshold=0.005)),
        "fixed weights, decide every bar, no threshold": (FixedWeights(w), {}),
        "fixed weights, decide hourly": (FixedWeights(w), dict(decision_every=60)),
    }
    lines = []
    for name, (strat, kw) in cases.items():
        res, speed = _speed(panel, strat, SimConfig(), **kw)
        assert len(res.equity) == n_t and np.isfinite(res.equity.to_numpy()).all()
        lines.append(f"{name:<50s} {speed:>10,.0f} bars/s   fills={len(res.fills):>7,d}")
    with capsys.disabled():
        print(f"\nsimulator speed, {n_t:,} bars x {n_a} assets:\n  " + "\n  ".join(lines))
