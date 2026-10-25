"""The aligned minute-bar panel shared by the data pipeline, features, and simulator.

Every array is indexed [time, asset] on a gap-free 1-minute UTC grid. Row t is the bar that
opens at times[t] and closes one minute later. Minutes where an asset had no data are
forward-filled for prices, get zero volume, and are flagged in `valid`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Panel:
    times: np.ndarray            # datetime64[ns], shape [T], bar open times (UTC)
    symbols: list[str]           # length N
    open: np.ndarray             # float64 [T, N]
    high: np.ndarray             # float64 [T, N]
    low: np.ndarray              # float64 [T, N]
    close: np.ndarray            # float64 [T, N]
    volume: np.ndarray           # float64 [T, N], base-asset units
    quote_volume: np.ndarray     # float64 [T, N], quote-asset units (USDT)
    trades: np.ndarray           # float64 [T, N], number of trades in the bar
    taker_buy_volume: np.ndarray # float64 [T, N], base volume bought by aggressive buyers
    valid: np.ndarray            # bool [T, N], False before listing, during long gaps, or missing
    meta: dict = field(default_factory=dict)

    @property
    def n_times(self) -> int:
        return len(self.times)

    @property
    def n_assets(self) -> int:
        return len(self.symbols)

    def slice(self, start: int, stop: int) -> "Panel":
        """Time slice [start, stop) as a new Panel sharing no memory with this one."""
        sl = slice(start, stop)
        return Panel(
            times=self.times[sl].copy(),
            symbols=list(self.symbols),
            open=self.open[sl].copy(),
            high=self.high[sl].copy(),
            low=self.low[sl].copy(),
            close=self.close[sl].copy(),
            volume=self.volume[sl].copy(),
            quote_volume=self.quote_volume[sl].copy(),
            trades=self.trades[sl].copy(),
            taker_buy_volume=self.taker_buy_volume[sl].copy(),
            valid=self.valid[sl].copy(),
            meta=dict(self.meta),
        )

    def select(self, symbols: list[str]) -> "Panel":
        """Subset of assets, in the given order."""
        idx = [self.symbols.index(s) for s in symbols]
        return Panel(
            times=self.times.copy(),
            symbols=list(symbols),
            open=self.open[:, idx].copy(),
            high=self.high[:, idx].copy(),
            low=self.low[:, idx].copy(),
            close=self.close[:, idx].copy(),
            volume=self.volume[:, idx].copy(),
            quote_volume=self.quote_volume[:, idx].copy(),
            trades=self.trades[:, idx].copy(),
            taker_buy_volume=self.taker_buy_volume[:, idx].copy(),
            valid=self.valid[:, idx].copy(),
            meta=dict(self.meta),
        )


def synthetic_panel(
    n_times: int = 2_000,
    symbols: list[str] | None = None,
    seed: int = 0,
    start: str = "2024-01-01T00:00:00",
    vol_per_minute: float = 0.001,
) -> Panel:
    """Random-walk panel with a shared market factor, for tests and smoke runs."""
    symbols = symbols or ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    rng = np.random.default_rng(seed)
    n = len(symbols)
    times = np.datetime64(start, "ns") + np.arange(n_times).astype("timedelta64[m]")
    market = rng.normal(0, vol_per_minute * 0.7, size=(n_times, 1))
    idio = rng.normal(0, vol_per_minute * 0.7, size=(n_times, n))
    log_ret = market + idio
    start_prices = rng.uniform(10, 1000, size=n)
    log_close = np.log(start_prices) + np.cumsum(log_ret, axis=0)
    close = np.exp(log_close)
    open_ = np.vstack([start_prices[None, :], close[:-1]])
    wiggle = np.abs(rng.normal(0, vol_per_minute * 0.5, size=(n_times, n)))
    high = np.maximum(open_, close) * (1 + wiggle)
    low = np.minimum(open_, close) * (1 - wiggle)
    volume = rng.lognormal(mean=3.0, sigma=0.6, size=(n_times, n))
    taker_buy = volume * rng.uniform(0.3, 0.7, size=(n_times, n))
    return Panel(
        times=times,
        symbols=list(symbols),
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        quote_volume=volume * close,
        trades=np.round(volume * 3),
        taker_buy_volume=taker_buy,
        valid=np.ones((n_times, n), dtype=bool),
        meta={"source": "synthetic", "seed": seed},
    )
