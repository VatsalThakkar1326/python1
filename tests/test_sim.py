"""Execution-model tests for the minute-bar simulator (orders, fills, costs, constraints)."""

from __future__ import annotations

import numpy as np
import pytest
from pytest import approx
from test_sim_helpers import FREE, bounce_panel, flat_panel, make_panel

from mark1.data.panel import synthetic_panel
from mark1.sim import (Decision, MarketView, Order, OrderStatus, OrderType, Side, SimConfig,
                       Simulator, TimeInForce, run_backtest)

BUY, SELL = Side.BUY, Side.SELL


def run_out(sim: Simulator) -> None:
    while not sim.done:
        sim.step()


def hold(panel, qty, cfg=FREE, asset=0):
    """Simulator that already holds `qty` of `asset` (bought at step 0, filled on bar 1)."""
    sim = Simulator(panel, cfg)
    sim.reset(0)
    sim.submit(Order.market(asset, BUY, qty))
    sim.step()
    assert sim.positions[asset] == approx(qty)
    return sim


# --------------------------------------------------------------------------- basic timing / costs
def test_stepping_api_and_range():
    sim = Simulator(flat_panel(30), FREE)
    info = sim.reset(start=10, end=15)
    assert sim.t == 10 and info.t == 10 and not sim.done
    seen = []
    while not sim.done:
        seen.append(sim.step().t)
    assert seen == [11, 12, 13, 14] and sim.done
    with pytest.raises(RuntimeError):
        sim.step()
    assert len(sim.equity_curve()) == 5
    with pytest.raises(ValueError):
        sim.reset(5, 6)  # needs at least two bars


def test_market_buy_fills_at_next_open_with_spread_impact_and_taker_fee():
    panel = synthetic_panel(n_times=40, seed=5)
    panel.volume *= 1e3
    cfg = SimConfig(participation_cap=1.0)  # default costs: 40 bps taker, 2 bps half-spread, 0.5 impact
    sim = Simulator(panel, cfg)
    sim.reset(0)
    qty = 30.0
    sim.submit(Order.market(0, BUY, qty, tag="x"))
    info = sim.step()
    (f,) = info.fills
    assert f.t == 1 and f.qty == qty and f.tag == "x" and not f.maker
    # no returns exist yet at t=0, so sigma falls back to default_sigma_bps (10)
    impact = 0.5 * 10.0 * np.sqrt(qty / panel.volume[1, 0])
    price = panel.open[1, 0] * (1 + (2.0 + impact) / 1e4)
    assert f.price == approx(price, rel=1e-12)
    assert f.fee == approx(0.0040 * qty * price, rel=1e-12)
    assert f.spread_bps == 2.0 and f.impact_bps == approx(impact)
    assert f.slippage == approx((price - panel.open[1, 0]) * qty)
    assert sim.cash == approx(100_000 - qty * price - f.fee)


def test_buy_and_hold_final_equity_matches_hand_calculation():
    panel = synthetic_panel(n_times=300, seed=3)
    panel.volume *= 1e3
    sim = Simulator(panel, SimConfig(participation_cap=1.0))
    sim.reset(0, 300)
    while sim.t < 50:
        sim.step()
    qty = 50_000.0 / panel.close[50, 0]
    sim.submit(Order.market(0, BUY, qty))
    sim.step()
    run_out(sim)

    # independent hand calculation; sigma = rms of the 50 returns seen up to bar 50
    rets = np.diff(np.log(panel.close[:51, 0]))
    sigma = 1e4 * np.sqrt(np.mean(rets**2))
    impact = 0.5 * sigma * np.sqrt(qty / panel.volume[51, 0])
    px = panel.open[51, 0] * (1 + (2.0 + impact) / 1e4)
    fee = 0.004 * qty * px
    expected = 100_000 - qty * px - fee + qty * panel.close[-1, 0]
    assert sim.equity == approx(expected, rel=1e-10)
    assert sim.fills_frame().iloc[0]["impact_bps"] == approx(impact)


def test_no_fill_before_next_bar_and_equity_marked_at_close():
    panel = make_panel([100, 100, 100, 100], c=[100, 110, 120, 130])
    sim = Simulator(panel, FREE)
    sim.reset(0)
    sim.submit(Order.market(0, BUY, 10))
    assert sim.positions[0] == 0 and sim.cash == 100_000  # nothing fills at submission
    sim.step()
    assert sim.positions[0] == 10
    assert sim.equity == approx(100_000 - 1000 + 10 * 110)  # bought at open 100, marked at close 110


# --------------------------------------------------------------------------- latency
def test_latency_shifts_first_fill():
    panel = flat_panel(12)
    for lat in (0, 1, 2, 3):
        sim = Simulator(panel, FREE.replace(latency_bars=lat))
        sim.reset(0)
        sim.submit(Order.market(0, BUY, 1))
        run_out(sim)
        assert sim.fills_frame()["t"].tolist() == [1 + lat]


def test_latency_delays_cancel_too():
    sim = Simulator(flat_panel(12), FREE.replace(latency_bars=2))
    sim.reset(0)
    (oid,) = sim.submit(Order.market(0, BUY, 1))
    sim.step()                       # bar 1: not yet active
    assert sim.cancel(oid)           # the cancel reaches the exchange at bar 4
    run_out(sim)
    assert sim.fills_frame()["t"].tolist() == [3]  # the order was live on bar 3, before the cancel


# --------------------------------------------------------------------------- participation cap
def test_participation_cap_splits_big_market_order_across_bars():
    vol = [100, 100, 50, 200, 100, 100, 100, 100]
    panel = make_panel(np.full(8, 10.0), v=np.array(vol, dtype=float))
    cfg = FREE.replace(participation_cap=0.10, initial_cash=1e9)
    sim = Simulator(panel, cfg)
    sim.reset(0)
    sim.submit(Order.market(0, BUY, 35))
    run_out(sim)
    f = sim.fills_frame()
    assert f["t"].tolist() == [1, 2, 3] and f["qty"].tolist() == approx([10.0, 5.0, 20.0])
    assert (f["qty"].to_numpy() <= 0.10 * panel.volume[f["t"].to_numpy(), 0] + 1e-12).all()
    assert sim.open_orders == []


def test_participation_cap_is_shared_by_all_orders_in_a_bar():
    panel = make_panel(np.full(5, 10.0), v=100.0)
    sim = Simulator(panel, FREE.replace(participation_cap=0.10, initial_cash=1e9))
    sim.reset(0)
    sim.submit([Order.market(0, BUY, 8), Order.market(0, BUY, 8)])
    sim.step()
    assert sim.positions[0] == approx(10.0)  # 10% of 100 across both orders
    run_out(sim)
    assert sim.positions[0] == approx(16.0)


def test_ioc_cancels_remainder_after_first_bar():
    panel = make_panel(np.full(6, 10.0), v=100.0)
    sim = Simulator(panel, FREE.replace(participation_cap=0.10, initial_cash=1e9))
    sim.reset(0)
    sim.submit(Order.market(0, BUY, 35, tif=TimeInForce.IOC))
    run_out(sim)
    f = sim.fills_frame()
    assert f["qty"].tolist() == approx([10.0])
    o = sim.orders_frame().iloc[0]
    assert o["status"] == "CANCELLED" and o["reason"] == "ioc_remainder" and o["filled_qty"] == approx(10)


def test_ioc_limit_that_does_not_trade_through_is_cancelled():
    panel = make_panel([100] * 5, l=[100, 99.5, 99.5, 99.5, 99.5])
    sim = Simulator(panel, FREE)
    sim.reset(0)
    sim.submit(Order.limit(0, BUY, 1, 99.0, tif=TimeInForce.IOC))
    run_out(sim)
    assert sim.fills_frame().empty
    assert sim.orders_frame().iloc[0]["reason"] == "ioc_no_fill"


# --------------------------------------------------------------------------- limit orders
def limit_case(side, price, o, h, l, cfg=FREE, qty=2.0):
    """One decision at t=0, one bar (bar 1) with the given OHLC. Returns the simulator."""
    panel = make_panel([100, o], h=[100, h], l=[100, l], c=[100, o])
    sim = Simulator(panel, cfg)
    sim.reset(0)
    sim.submit(Order.limit(0, side, qty, price))
    sim.step()
    return sim


def test_limit_buy_not_filled_when_low_only_touches():
    sim = limit_case(BUY, 99.0, o=100, h=101, l=99.0)
    assert sim.fills_frame().empty and len(sim.open_orders) == 1


def test_limit_buy_fills_at_limit_when_traded_through():
    sim = limit_case(BUY, 99.0, o=100, h=101, l=98.9, cfg=FREE.replace(maker_fee_bps=25))
    (f,) = sim.fills_frame().itertuples()
    assert f.price == 99.0 and f.liquidity == "maker" and f.spread_bps == 0 and f.impact_bps == 0
    assert f.fee == approx(0.0025 * 2 * 99.0) and f.slippage == 0.0


def test_limit_buy_arriving_through_the_market_fills_as_taker_at_open():
    sim = limit_case(BUY, 99.0, o=98.0, h=100, l=97.0)
    (f,) = sim.fills_frame().itertuples()
    assert f.price == 98.0 and f.liquidity == "taker"   # crosses the spread like a market order


def test_resting_limit_buy_gapped_through_fills_at_its_own_price():
    panel = make_panel([100, 100, 98], h=[100, 100, 99], l=[100, 99.5, 97], c=[100, 100, 98])
    sim = Simulator(panel, FREE)
    sim.reset(0)
    sim.submit(Order.limit(0, BUY, 2, 99.0))
    sim.step()                      # bar 1: low 99.5, no fill, the order rests
    assert sim.fills_frame().empty
    sim.step()                      # bar 2 opens at 98, through the resting limit
    (f,) = sim.fills_frame().itertuples()
    assert f.price == 99.0 and f.liquidity == "maker"


def test_limit_sell_mirror_cases():
    sim = limit_case(SELL, 101.0, o=100, h=101.0, l=99)  # touch only: no fill (and no position)
    assert sim.fills_frame().empty
    panel = make_panel([100, 100, 100], h=[100, 100, 101.5], l=[100, 100, 99], c=[100, 100, 100])
    sim = Simulator(panel, FREE)
    sim.reset(0)
    sim.submit(Order.market(0, BUY, 2))
    sim.step()
    sim.submit(Order.limit(0, SELL, 2, 101.0))
    sim.step()
    f = sim.fills_frame().iloc[-1]
    assert f["price"] == 101.0 and f["side"] == "sell" and f["realized_pnl"] == approx(2.0)
    # arrives with the market already above the limit: fills at the open as a taker
    panel = make_panel([100, 100, 103], h=[100, 100, 104], l=[100, 100, 102], c=[100, 100, 103])
    sim = hold(panel, 2)
    sim.submit(Order.limit(0, SELL, 2, 101.0))
    sim.step()
    assert sim.fills_frame().iloc[-1]["price"] == 103.0


def test_limit_order_waits_then_fills_and_obeys_participation_cap():
    panel = make_panel([100] * 6, l=[100, 100, 100, 98, 98, 98], v=100.0)
    sim = Simulator(panel, FREE.replace(participation_cap=0.10))
    sim.reset(0)
    sim.submit(Order.limit(0, BUY, 15, 99.0))
    run_out(sim)
    f = sim.fills_frame()
    assert f["t"].tolist() == [3, 4] and f["qty"].tolist() == approx([10.0, 5.0])


def test_take_profit_uses_limit_rule():
    panel = make_panel([100, 100, 100], h=[100, 100, 106], l=[100, 100, 99], c=[100, 100, 100])
    sim = hold(panel, 1)
    sim.submit(Order.take_profit(0, SELL, 1, 105.0))
    sim.step()
    f = sim.fills_frame().iloc[-1]
    assert f["order_type"] == "TAKE_PROFIT" and f["price"] == 105.0 and f["liquidity"] == "maker"


# --------------------------------------------------------------------------- stop orders
STOP_COST = SimConfig(taker_fee_bps=40, half_spread_bps=2.0, impact_coef=0, min_notional=0,
                      participation_cap=1.0)


def test_stop_sell_triggers_on_touch_and_pays_spread_and_taker_fee():
    panel = make_panel([100, 100, 100], h=[100, 100, 101], l=[100, 100, 95.0], c=[100, 100, 99])
    sim = hold(panel, 3, STOP_COST)
    sim.submit(Order.stop(0, SELL, 3, 95.0))
    sim.step()
    f = sim.fills_frame().iloc[-1]
    assert f["order_type"] == "STOP" and f["liquidity"] == "taker"
    assert f["price"] == approx(95.0 * (1 - 2e-4)) and f["fee"] == approx(0.004 * 3 * f["price"])
    assert f["slippage"] == approx((95.0 - f["price"]) * 3)


def test_stop_sell_not_triggered_above_low():
    panel = make_panel([100, 100, 100], l=[100, 100, 95.01])
    sim = hold(panel, 1, STOP_COST)
    sim.submit(Order.stop(0, SELL, 1, 95.0))
    sim.step()
    assert len(sim.fills_frame()) == 1 and len(sim.open_orders) == 1  # only the entry buy


def test_stop_sell_gap_through_fills_at_open():
    panel = make_panel([100, 100, 90], h=[100, 100, 91], l=[100, 100, 88], c=[100, 100, 89])
    sim = hold(panel, 1, STOP_COST)
    sim.submit(Order.stop(0, SELL, 1, 95.0))
    sim.step()
    f = sim.fills_frame().iloc[-1]
    assert f["price"] == approx(90.0 * (1 - 2e-4))
    assert f["slippage"] == approx((95.0 - f["price"]) * 1)  # the gap shows up as slippage vs the stop


def test_stop_buy_symmetric():
    panel = make_panel([100, 100, 100, 108], h=[100, 100, 105, 110], l=[100, 100, 99, 107])
    sim = Simulator(panel, STOP_COST)
    sim.reset(0)
    sim.submit([Order.stop(0, BUY, 1, 105.0)])
    sim.step()  # bar 1: high 100 -> no trigger
    assert sim.fills_frame().empty
    sim.step()  # bar 2: high 105 touches the stop
    f = sim.fills_frame().iloc[-1]
    assert f["price"] == approx(105.0 * (1 + 2e-4))
    sim = Simulator(make_panel([100, 100, 108], h=[100, 100, 110]), STOP_COST)  # gap up through it
    sim.reset(0)
    sim.submit(Order.stop(0, BUY, 1, 105.0))
    sim.step()
    sim.step()
    assert sim.fills_frame().iloc[-1]["price"] == approx(108.0 * (1 + 2e-4))


def test_triggered_stop_remainder_continues_as_market_order():
    panel = make_panel([100, 100, 100, 100], h=[100, 100, 100, 100], l=[100, 100, 94, 94], v=10.0)
    cfg = STOP_COST.replace()
    sim = hold(panel, 1.0, cfg)
    cfg.participation_cap = 0.05
    sim.submit(Order.stop(0, SELL, 1.0, 95.0))
    sim.step()  # bar 2: cap 0.5 units
    sim.step()  # bar 3: rest, at the open now
    f = sim.fills_frame()
    assert f["qty"].iloc[1:].tolist() == approx([0.5, 0.5])
    assert f["price"].iloc[1] == approx(95.0 * (1 - 2e-4)) and f["price"].iloc[2] == approx(100.0 * (1 - 2e-4))


def test_stop_and_take_profit_in_same_bar_assume_worst_case_stop_first():
    # one bar spans both the stop (95) and the take-profit (105)
    panel = make_panel([100, 100, 100], h=[100, 100, 106], l=[100, 100, 94], c=[100, 100, 100])
    sim = hold(panel, 10, STOP_COST)
    sim.submit([Order.take_profit(0, SELL, 10, 105.0), Order.stop(0, SELL, 10, 95.0)])
    sim.step()
    f = sim.fills_frame()
    assert f["order_type"].tolist() == ["MARKET", "STOP"]  # the stop got the position
    o = sim.orders_frame().set_index("order_type")
    assert o.loc["TAKE_PROFIT", "status"] == "CANCELLED" and o.loc["TAKE_PROFIT", "reason"] == "no_position"


def test_gap_above_take_profit_fills_at_open_before_the_stop_is_considered():
    panel = make_panel([100, 100, 106], h=[100, 100, 107], l=[100, 100, 94], c=[100, 100, 100])
    sim = hold(panel, 10, STOP_COST)
    sim.submit([Order.take_profit(0, SELL, 10, 105.0), Order.stop(0, SELL, 10, 95.0)])
    sim.step()
    f = sim.fills_frame().iloc[-1]
    assert f["order_type"] == "TAKE_PROFIT" and f["liquidity"] == "taker"
    assert f["price"] == approx(106.0 * (1 - 2e-4))   # open minus the 2 bps half-spread


def test_oco_cancels_sibling_on_fill():
    panel = make_panel([100, 100, 100], h=[100, 100, 106], l=[100, 100, 99], c=[100, 100, 100])
    sim = hold(panel, 10, STOP_COST)
    sim.submit([Order.take_profit(0, SELL, 10, 105.0, oco=7), Order.stop(0, SELL, 10, 90.0, oco=7)])
    sim.step()
    o = sim.orders_frame().set_index("order_type")
    assert o.loc["STOP", "reason"] == "oco" and o.loc["TAKE_PROFIT", "status"] == "FILLED"
    assert sim.open_orders == []


# --------------------------------------------------------------------------- expiry / cancel
def test_expiry_and_cancel():
    panel = make_panel([100] * 8, l=[100] * 8)
    sim = Simulator(panel, FREE)
    sim.reset(0)
    ids = sim.submit([Order.limit(0, BUY, 1, 90.0, expire_bar=3), Order.limit(0, BUY, 1, 90.0)])
    sim.step()
    assert sim.cancel(ids[1]) and not sim.cancel(ids[1])  # second cancel is a no-op
    run_out(sim)
    o = sim.orders_frame().set_index("order_id")
    assert o.loc[ids[0], "status"] == "EXPIRED" and o.loc[ids[0], "end_t"] == 4
    assert o.loc[ids[1], "status"] == "CANCELLED"
    # expiry before the order can ever be active is rejected
    (rid,) = sim.submit(Order.limit(0, BUY, 1, 90.0, expire_bar=sim.t))
    assert sim.orders_frame().set_index("order_id").loc[rid, "status"] == "REJECTED"


# --------------------------------------------------------------------------- invalid bars
def test_no_fills_on_invalid_bars_and_orders_wait():
    valid = np.array([1, 0, 0, 0, 1, 1], dtype=bool)
    panel = make_panel([100, 1, 1, 1, 105, 105], c=[100, 1, 1, 1, 105, 105], valid=valid)
    sim = Simulator(panel, FREE)
    sim.reset(0)
    sim.submit(Order.market(0, BUY, 2))
    for _ in range(3):
        info = sim.step()
        assert info.fills == [] and sim.positions[0] == 0
    sim.step()
    assert sim.fills_frame()["t"].tolist() == [4] and sim.fills_frame()["price"].iloc[0] == 105.0


def test_invalid_assets_keep_last_valid_close_in_mark_to_market():
    valid = np.array([1, 1, 0, 0, 0], dtype=bool)
    panel = make_panel([100, 100, 5, 5, 5], c=[100, 120, 5, 5, 5], valid=valid)
    sim = hold(panel, 10)
    assert sim.equity == approx(100_000 - 1000 + 1200)
    run_out(sim)
    assert sim.equity == approx(100_000 - 1000 + 1200)  # junk prices on invalid bars are ignored
    assert sim.prices[0] == 120.0


def test_stop_does_not_trigger_on_invalid_bar():
    valid = np.array([1, 1, 0, 1], dtype=bool)
    panel = make_panel([100, 100, 100, 100], l=[100, 100, 50, 99], valid=valid)
    sim = hold(panel, 1, STOP_COST)
    sim.submit(Order.stop(0, SELL, 1, 95.0))
    sim.step()
    assert len(sim.fills_frame()) == 1
    sim.step()
    assert len(sim.fills_frame()) == 1 and len(sim.open_orders) == 1


# --------------------------------------------------------------------------- constraints
def test_long_only_cannot_oversell_or_sell_flat():
    panel = flat_panel(8)
    sim = hold(panel, 5)
    sim.submit(Order.market(0, SELL, 8))
    sim.step()
    assert sim.positions[0] == 0
    f = sim.fills_frame()
    assert f["qty"].tolist() == approx([5, 5])
    o = sim.orders_frame().iloc[-1]
    assert o["status"] == "CANCELLED" and o["reason"] == "no_position" and o["filled_qty"] == approx(5)
    sim.submit(Order.market(0, SELL, 1))
    sim.step()
    assert len(sim.fills_frame()) == 2 and sim.positions[0] == 0  # flat: nothing sold


def test_cash_constraint_scales_buys_down():
    cfg = SimConfig(initial_cash=1_000.0, half_spread_bps=0, impact_coef=0, participation_cap=1.0)
    sim = Simulator(flat_panel(5, 100.0), cfg)
    sim.reset(0)
    sim.submit(Order.market(0, BUY, 50))  # would cost 5,000
    sim.step()
    assert sim.positions[0] == approx(1000 / (100 * 1.004), rel=1e-9)
    assert 0 <= sim.cash < 1e-6
    o = sim.orders_frame().iloc[0]
    assert o["status"] == "CANCELLED" and o["reason"] == "insufficient_funds"


def test_leverage_flag_allows_negative_cash():
    cfg = FREE.replace(initial_cash=1_000.0, allow_leverage=True)
    sim = Simulator(flat_panel(5, 100.0), cfg)
    sim.reset(0)
    sim.submit(Order.market(0, BUY, 50))
    sim.step()
    assert sim.positions[0] == 50 and sim.cash == approx(-4_000)


def test_min_notional_rejects_small_orders_and_tiny_scaled_buys():
    cfg = FREE.replace(min_notional=10.0, initial_cash=5.0)
    sim = Simulator(flat_panel(5, 100.0), cfg)
    sim.reset(0)
    a, b = sim.submit([Order.market(0, BUY, 0.05), Order.limit(0, BUY, 1.0, 99.0)])  # $5 and $99
    assert sim.orders_frame().set_index("order_id").loc[a, "reason"] == "min_notional"
    sim.step()
    o = sim.orders_frame().set_index("order_id").loc[b]
    assert o["reason"] in ("open", "insufficient_funds") and sim.positions[0] == 0
    # a market buy that cash can only partly fund, below min notional after scaling: cancelled
    sim.reset(0)
    sim.submit(Order.market(0, BUY, 1.0))
    sim.step()
    assert sim.positions[0] == 0 and sim.orders_frame().iloc[0]["reason"] == "insufficient_funds"


def test_qty_step_floors_quantities():
    cfg = FREE.replace(qty_step=0.1)
    sim = Simulator(flat_panel(6), cfg)
    sim.reset(0)
    (a, b) = sim.submit([Order.market(0, BUY, 1.234), Order.market(0, BUY, 0.05)])
    sim.step()
    assert sim.positions[0] == approx(1.2)
    assert sim.orders_frame().set_index("order_id").loc[b, "reason"] == "lot_size"


def test_short_selling_when_long_only_disabled():
    panel = make_panel([100, 100, 90, 90, 80], c=[100, 100, 90, 90, 80])
    sim = Simulator(panel, FREE.replace(long_only=False))
    sim.reset(0)
    sim.submit(Order.market(0, SELL, 100))
    sim.step()
    assert sim.positions[0] == -100 and sim.cash == approx(110_000)
    run_out(sim)  # price falls from 100 to 80: the short gains 100 * 20
    assert sim.equity == approx(102_000)
    # without leverage the short cannot exceed equity (1000 units at 100)
    sim.reset(0)
    sim.submit(Order.market(0, SELL, 5_000))
    sim.step()
    assert sim.positions[0] == approx(-1000, rel=1e-9)
    # flipping through zero: short 50, then buy 80 -> long 30, covering profit realised
    sim.reset(0)
    sim.submit(Order.market(0, SELL, 50))
    sim.step()
    sim.submit(Order.market(0, BUY, 80))
    sim.step()
    assert sim.positions[0] == approx(30) and sim.pf.realized_pnl == approx(50 * (100 - 90))


# --------------------------------------------------------------------------- accounting
def random_trader(seed=0):
    rng = np.random.default_rng(seed)

    def strat(t, market, pf):
        orders = []
        for i in range(market.n_assets):
            if rng.random() < 0.06:
                side = BUY if rng.random() < 0.55 else SELL
                px = market.close[-1, i]
                qty = rng.uniform(0.2, 1.0) * 20_000 / px
                kind = rng.integers(0, 3)
                if kind == 0:
                    orders.append(Order.market(i, side, qty, tag="m"))
                elif kind == 1:
                    off = 1 - 0.002 if side == BUY else 1 + 0.002
                    orders.append(Order.limit(i, side, qty, px * off, tag="l"))
                else:
                    off = 1 - 0.003 if side == SELL else 1 + 0.003
                    orders.append(Order.stop(i, side, qty, px * off, tag="s"))
        return Decision(orders=orders)

    return strat


def test_accounting_reconciles_price_pnl_fees_and_costs():
    panel = synthetic_panel(n_times=3000, symbols=["A", "B", "C"], seed=11, vol_per_minute=0.002)
    panel.volume *= 50
    cfg = SimConfig(spread_estimator="corwin_schultz")
    res = run_backtest(panel, random_trader(1), cfg, decision_every=3)
    f = res.fills
    assert len(f) > 50 and set(f["order_type"]) >= {"MARKET", "LIMIT", "STOP"}
    sgn = np.where(f["side"] == "buy", 1.0, -1.0)
    signed_qty = sgn * f["qty"].to_numpy()

    # cash and positions rebuilt from the fill log
    cash = 100_000 - (signed_qty * f["price"]).sum() - f["fee"].sum()
    pos = np.zeros(3)
    np.add.at(pos, f["asset"].to_numpy(), signed_qty)
    final_equity = res.equity.iloc[-1]
    assert final_equity == approx(cash + pos @ panel.close[2999], rel=1e-10)
    assert (pos >= -1e-9).all()  # long-only

    # equity = start + price pnl at the reference prices - slippage - fees
    price_pnl = (signed_qty * (panel.close[2999][f["asset"].to_numpy()] - f["ref_price"])).sum()
    assert final_equity == approx(100_000 + price_pnl - f["slippage"].sum() - f["fee"].sum(), rel=1e-10)
    assert res.metrics["fees_paid"] == approx(f["fee"].sum())
    assert res.metrics["slippage_cost"] == approx(f["slippage"].sum())
    assert res.metrics["total_costs"] == approx(f["fee"].sum() + f["slippage"].sum())

    # realized + unrealized - fees identity (portfolio level)
    sim = Simulator(panel, cfg)
    sim.reset(0)
    strat, view = random_trader(1), sim.portfolio
    mv = MarketView(panel)
    while not sim.done:
        if sim.t % 3 == 0:
            mv.t = sim.t
            sim.submit(strat(sim.t, mv, view).orders)
        sim.step()
    pf = sim.pf
    assert sim.equity == approx(100_000 + pf.realized_pnl + pf.unrealized_pnl() - pf.fees_paid, rel=1e-12)
    assert sim.equity == approx(final_equity, rel=1e-12)


def test_round_trip_win_rate_and_realized_pnl():
    panel = make_panel([100, 100, 110, 110, 90, 90, 90], c=[100, 100, 110, 110, 90, 90, 90])
    sim = Simulator(panel, FREE.replace(taker_fee_bps=10))
    sim.reset(0)
    sim.submit(Order.market(0, BUY, 10))
    sim.step()
    sim.step()
    sim.submit(Order.market(0, SELL, 10))
    sim.step()
    rt = sim.round_trips_frame()
    assert len(rt) == 1 and rt["pnl"].iloc[0] == approx(100 - 0.001 * 1000 - 0.001 * 1100)
    f = sim.fills_frame()
    assert f["realized_pnl"].tolist() == approx([0.0, 100.0])


# --------------------------------------------------------------------------- no look-ahead
def momentum_strategy(log):
    def strat(t, market, pf):
        close = market.close
        log.append((t, close.shape, close[-5:].copy(), market.valid.shape, market.time))
        if t < 30:
            return None
        orders = []
        for i in range(market.n_assets):
            fast, slow = close[-5:, i].mean(), close[-30:, i].mean()
            if fast > slow and pf.positions[i] == 0:
                orders.append(Order.market(i, BUY, 5_000 / close[-1, i]))
            elif fast < slow and pf.positions[i] > 0:
                orders.append(Order.market(i, SELL, pf.positions[i]))
            elif fast > slow:
                orders.append(Order.limit(i, BUY, 1.0, close[-1, i] * 0.999, expire_bar=t + 5))
        return Decision(orders=orders)

    return strat


@pytest.mark.parametrize("poison", ["random", "nan"])
def test_no_lookahead_future_rows_cannot_change_decisions_or_fills(poison):
    t0 = 400
    a = synthetic_panel(n_times=900, seed=2, vol_per_minute=0.002)
    b = synthetic_panel(n_times=900, seed=2, vol_per_minute=0.002)
    other = synthetic_panel(n_times=900, seed=99)
    cut = t0 + 2  # rows >= cut differ; fills up to and including bar t0 + 1 must not notice
    for name in ("open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_volume"):
        arr = getattr(b, name)
        arr[cut:] = getattr(other, name)[cut:] if poison == "random" else np.nan
    if poison == "random":
        b.valid[cut:] = other.valid[cut:]
    cfg = SimConfig(spread_estimator="abdi_ranaldo", participation_cap=0.5)
    logs = ([], [])
    ra = run_backtest(a, momentum_strategy(logs[0]), cfg)
    rb = run_backtest(b, momentum_strategy(logs[1]), cfg)

    seen_a = {x[0]: x for x in logs[0] if x[0] <= t0 + 1}
    seen_b = {x[0]: x for x in logs[1] if x[0] <= t0 + 1}
    assert seen_a.keys() == seen_b.keys() and len(seen_a) > 300
    for t in seen_a:
        assert seen_a[t][1] == seen_b[t][1] == (t + 1, 3)
        np.testing.assert_array_equal(seen_a[t][2], seen_b[t][2])
    fa = ra.fills[ra.fills["t"] <= t0 + 1].drop(columns=["time"])
    fb = rb.fills[rb.fills["t"] <= t0 + 1].drop(columns=["time"])
    assert len(fa) > 5
    assert fa.reset_index(drop=True).equals(fb.reset_index(drop=True))
    np.testing.assert_array_equal(ra.equity.to_numpy()[: t0 + 2], rb.equity.to_numpy()[: t0 + 2])


@pytest.mark.parametrize("estimator", ["corwin_schultz", "abdi_ranaldo"])
def test_fill_costs_do_not_depend_on_the_fill_bars_close_or_range(estimator):
    # A market fill on bar b may use open[b] and volume[b], but sigma and the spread estimate
    # must come from bars < b: widening bar b's range and moving its close cannot change them.
    b = 150
    a, p2 = bounce_panel(300, spread_bps=20.0, vol_bps=5.0), bounce_panel(300, spread_bps=20.0, vol_bps=5.0)
    p2.close[b] *= 1.04
    p2.high[b] *= 1.08
    p2.low[b] *= 0.92
    cfg = SimConfig(half_spread_bps=0.0, spread_estimator=estimator, participation_cap=1.0)
    fills = []
    for panel in (a, p2):
        sim = Simulator(panel, cfg)
        sim.reset(0)
        while sim.t < b - 1:
            sim.step()
        sim.submit(Order.market(0, BUY, 20.0))
        fills.append(sim.step().fills[0])
    fa, fb = fills
    assert fa.t == b and fa.spread_bps > 1.0 and fa.impact_bps > 0  # the check is not vacuous
    assert (fa.price, fa.spread_bps, fa.impact_bps, fa.fee) == (fb.price, fb.spread_bps, fb.impact_bps, fb.fee)


def test_market_view_is_read_only_and_limited():
    panel = synthetic_panel(n_times=50)
    mv = MarketView(panel, t=10)
    assert mv.close.shape == (11, 3) and mv.times.shape == (11,) and mv.window(5).n_times == 5
    with pytest.raises(ValueError):
        mv.close[0, 0] = 1.0
    assert mv.time == panel.times[10]


def test_order_validation():
    with pytest.raises(ValueError):
        Order(0, BUY, 0.0)
    with pytest.raises(ValueError):
        Order(0, BUY, 1.0, OrderType.LIMIT)  # no price
    sim = Simulator(flat_panel(5), FREE)
    with pytest.raises(ValueError):
        sim.submit(Order.market(7, BUY, 1))
    spec = Order.market(0, BUY, 1)
    sim.submit(spec)
    assert spec.id == -1 and spec.status == OrderStatus.NEW  # caller's request is never mutated
