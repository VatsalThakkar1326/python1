"""Backtest run loop: strategy protocol, causal market view, results, and benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Protocol, Union

import numpy as np
import pandas as pd

from mark1.config import MINUTES_PER_YEAR
from mark1.data.panel import Panel
from mark1.sim.costs import SimConfig
from mark1.sim.engine import Simulator
from mark1.sim.metrics import compute_metrics, format_metrics
from mark1.sim.orders import Decision
from mark1.sim.portfolio import PortfolioView


class MarketView:
    """The panel as seen at the close of bar `t`: every array is a read-only slice of rows <= t.

    Shapes are [t + 1, N], so `view.close[-1]` is the close of the bar that just finished and
    nothing after it is reachable through the view.
    """

    __slots__ = ("_panel", "t")

    def __init__(self, panel: Panel, t: int = 0):
        self._panel = panel
        self.t = t

    def _rows(self, a: np.ndarray) -> np.ndarray:
        v = a[: self.t + 1]
        v.flags.writeable = False
        return v

    @property
    def symbols(self) -> list[str]:
        return self._panel.symbols

    @property
    def n_assets(self) -> int:
        return self._panel.n_assets

    @property
    def time(self) -> np.datetime64:
        return self._panel.times[self.t]

    @property
    def times(self) -> np.ndarray:
        return self._rows(self._panel.times)

    @property
    def open(self) -> np.ndarray:
        return self._rows(self._panel.open)

    @property
    def high(self) -> np.ndarray:
        return self._rows(self._panel.high)

    @property
    def low(self) -> np.ndarray:
        return self._rows(self._panel.low)

    @property
    def close(self) -> np.ndarray:
        return self._rows(self._panel.close)

    @property
    def volume(self) -> np.ndarray:
        return self._rows(self._panel.volume)

    @property
    def quote_volume(self) -> np.ndarray:
        return self._rows(self._panel.quote_volume)

    @property
    def trades(self) -> np.ndarray:
        return self._rows(self._panel.trades)

    @property
    def taker_buy_volume(self) -> np.ndarray:
        return self._rows(self._panel.taker_buy_volume)

    @property
    def valid(self) -> np.ndarray:
        return self._rows(self._panel.valid)

    def window(self, n: int) -> Panel:
        """Copy of the last `n` bars (rows t-n+1..t) as a Panel."""
        return self._panel.slice(max(0, self.t + 1 - n), self.t + 1)


StrategyReturn = Union[Decision, np.ndarray, None]


class Strategy(Protocol):
    """Anything callable as `strategy(t, market, portfolio)`.

    It is invoked at decision steps with the current row index, a `MarketView` limited to rows
    <= t and a read-only `PortfolioView`. It returns a `Decision`, a bare weight vector (shorthand
    for `Decision(target_weights=...)`), or None to do nothing.
    """

    def __call__(self, t: int, market: MarketView, portfolio: PortfolioView) -> StrategyReturn: ...


@dataclass
class BacktestResult:
    """Everything a run produced. Per-bar series are plain arrays wrapped in pandas at the end.

    equity / exposure: one point per bar from `start` to `end - 1` (point 0 = initial cash).
    positions: actual portfolio weights (value / equity, float32) at each decision step, before
        that step's orders; `targets` holds the requested target weights (nan = none requested).
    fills / orders / round_trips: DataFrames built from compact logs (see Simulator).
    """

    equity: pd.Series
    exposure: pd.Series
    positions: pd.DataFrame
    targets: pd.DataFrame
    fills: pd.DataFrame
    orders: pd.DataFrame
    round_trips: pd.DataFrame
    metrics: dict[str, float]
    config: SimConfig
    symbols: list[str]
    start: int
    end: int
    decision_every: int
    stats: dict[str, float] = field(default_factory=dict)

    def summary(self, title: str | None = None) -> str:
        return format_metrics(self.metrics, title)

    def __repr__(self) -> str:
        m = self.metrics
        return (f"BacktestResult(bars={m['n_bars']}, return={m['total_return']:.2%}, "
                f"sharpe={m['sharpe']:.2f}, max_dd={m['max_drawdown']:.2%}, fills={m['n_trades']})")


def _as_decision(x: StrategyReturn) -> Decision | None:
    if x is None or isinstance(x, Decision):
        return x
    if isinstance(x, (np.ndarray, list, tuple)):
        return Decision(target_weights=np.asarray(x, dtype=np.float64))
    raise TypeError(f"strategy must return Decision, weight array or None, got {type(x).__name__}")


def run_backtest(
    panel: Panel,
    strategy: Callable[[int, MarketView, PortfolioView], StrategyReturn],
    config: SimConfig | None = None,
    start: int = 0,
    end: int | None = None,
    decision_every: int = 1,
    *,
    rebalance_threshold: float = 0.0,
    record_positions: bool = True,
) -> BacktestResult:
    """Run `strategy` over rows [start, end) of `panel`.

    At t = start, start + decision_every, ... the strategy is called with data up to and including
    row t. Its decision is executed through the simulator (orders become active on bar t + 1 +
    latency). Order of handling: explicit `cancel`, then `target_weights` (cancel the strategy's
    still-pending rebalance orders, submit market orders for the difference versus holdings,
    skipping trades below max(`rebalance_threshold` * equity, min_notional)), then explicit
    `orders`. The simulator advances one bar per iteration until the last bar of the range.
    """
    if decision_every < 1:
        raise ValueError("decision_every must be >= 1")
    cfg = config or SimConfig()
    sim = Simulator(panel, cfg)
    sim.reset(start, end)
    start, end = sim.start, sim.end
    market, view = MarketView(panel), sim.portfolio
    n_dec = (end - 2 - start) // decision_every + 1
    n = panel.n_assets
    dec_t = np.empty(n_dec, dtype=np.int64)
    if record_positions:
        weights_hist = np.empty((n_dec, n), dtype=np.float32)
        target_hist = np.full((n_dec, n), np.nan, dtype=np.float32)
    k = 0
    while not sim.done:
        t = sim.t
        if (t - start) % decision_every == 0:
            market.t = t
            dec = _as_decision(strategy(t, market, view))
            dec_t[k] = t
            if record_positions:
                weights_hist[k] = sim.weights
            if dec is not None:
                for oid in dec.cancel:
                    sim.cancel(oid)
                if dec.target_weights is not None:
                    sim.rebalance(dec.target_weights, rebalance_threshold)  # validates the weights
                    if record_positions:
                        target_hist[k] = dec.target_weights
                if len(dec.orders):
                    sim.submit(dec.orders)
            k += 1
        sim.step()

    dec_t = dec_t[:k]
    idx = pd.DatetimeIndex(panel.times[dec_t], name="time")
    if record_positions:
        positions = pd.DataFrame(weights_hist[:k], index=idx, columns=panel.symbols)
        targets = pd.DataFrame(target_hist[:k], index=idx, columns=panel.symbols)
    else:
        positions = targets = pd.DataFrame(index=idx, columns=panel.symbols, dtype=np.float32)
    round_trips = sim.round_trips_frame()
    stats = sim.stats()
    metrics = compute_metrics(
        sim.equity_curve(), exposure=sim.exposure_curve(), fees_paid=stats["fees_paid"],
        slippage_cost=stats["slippage_cost"], traded_notional=stats["traded_notional"],
        n_trades=int(stats["n_fills"]), round_trip_pnls=round_trips["pnl"].to_numpy(),
    )
    return BacktestResult(
        equity=sim.equity_series(), exposure=sim.exposure_series(), positions=positions,
        targets=targets, fills=sim.fills_frame(), orders=sim.orders_frame(), round_trips=round_trips,
        metrics=metrics, config=cfg, symbols=list(panel.symbols), start=start, end=end,
        decision_every=decision_every, stats=stats,
    )


def buy_and_hold_benchmark(panel: Panel, config: SimConfig | None = None, start: int = 0,
                           end: int | None = None) -> BacktestResult:
    """Equal weight across the assets valid at `start`, bought once and held.

    Runs through the same simulator and cost model as the strategy, so it pays the same fees,
    spread, impact and participation limits and is directly comparable.
    """
    from mark1.sim.strategies import EqualWeightBuyAndHold

    end = panel.n_times if end is None else end
    return run_backtest(panel, EqualWeightBuyAndHold(), config, start, end,
                        decision_every=max(1, end - start))


def cash_benchmark(panel: Panel, config: SimConfig | None = None, start: int = 0,
                   end: int | None = None, annual_rate: float = 0.0) -> BacktestResult:
    """Hold the initial cash, optionally earning a constant `annual_rate` (compounded per bar)."""
    end = panel.n_times if end is None else end
    res = run_backtest(panel, lambda t, m, p: None, config, start, end,
                       decision_every=max(1, end - start))
    if annual_rate:
        growth = (1.0 + annual_rate) ** (np.arange(len(res.equity)) / MINUTES_PER_YEAR)
        equity = res.equity * growth
        res = replace(res, equity=equity, metrics=compute_metrics(equity.to_numpy(),
                                                                  exposure=res.exposure.to_numpy()))
    return res
