"""Simulator configuration and the per-bar cost inputs (volatility, spread, impact).

Everything that feeds a fill price at bar b (rolling volatility, estimated spread) is computed
from bars <= b - 1 only, so a fill can never depend on the bar it happens in or anything later.
The only same-bar quantity is the bar's traded volume, which is the liquidity the order trades
against (used for the participation cap and the impact denominator).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from mark1.data.panel import Panel

SPREAD_ESTIMATORS = (None, "corwin_schultz", "abdi_ranaldo")


@dataclass
class SimConfig:
    """All simulator settings. Defaults describe a long-only spot account.

    Fees: the defaults (maker 25 bps, taker 40 bps) are roughly the lowest retail tier on
    Kraken Pro. They are placeholders: VERIFY them against the venue's current fee schedule and
    your own 30-day volume tier before trusting any backtest. Fees are charged in quote currency.

    Spread: `half_spread_bps` is a floor (scalar or one value per asset). With a
    `spread_estimator` the per-bar half-spread is max(floor, min(estimate, max_half_spread_bps)),
    where the estimate is a trailing mean over `spread_window` bars of the Corwin-Schultz (2012)
    or Abdi-Ranaldo (2017) estimator. These are noisy at one-minute resolution, hence the cap.

    Impact: impact_bps = impact_coef * sigma_bps * sqrt(fill_qty / bar_volume), capped at
    `impact_cap_bps`; sigma is the trailing per-minute close-to-close volatility (rms of log
    returns over `vol_window` bars, in bps) using bars up to the previous one. Until
    `vol_min_periods` returns exist, `default_sigma_bps` is used.

    Liquidity: at most `participation_cap` of a bar's volume can be filled per asset and bar,
    across all orders. `latency_bars` delays both new orders and cancels.
    """

    initial_cash: float = 100_000.0
    maker_fee_bps: float = 25.0
    taker_fee_bps: float = 40.0
    half_spread_bps: float | Sequence[float] | np.ndarray = 2.0
    spread_estimator: str | None = None
    spread_window: int = 60
    max_half_spread_bps: float = 25.0
    impact_coef: float = 0.5
    impact_cap_bps: float = 50.0
    vol_window: int = 60
    vol_min_periods: int = 10
    default_sigma_bps: float = 10.0
    participation_cap: float = 0.10
    latency_bars: int = 0
    min_notional: float = 10.0
    qty_step: float | Sequence[float] | np.ndarray | None = None
    long_only: bool = True
    allow_leverage: bool = False
    log_fills: bool = True
    log_orders: bool = True

    def __post_init__(self) -> None:
        if self.spread_estimator == "none":
            self.spread_estimator = None
        if self.spread_estimator not in SPREAD_ESTIMATORS:
            raise ValueError(f"spread_estimator must be one of {SPREAD_ESTIMATORS}")
        if not (0.0 < self.participation_cap <= 1.0):
            raise ValueError("participation_cap must be in (0, 1]")
        if self.latency_bars < 0 or int(self.latency_bars) != self.latency_bars:
            raise ValueError("latency_bars must be a non-negative integer")
        for name in ("initial_cash", "maker_fee_bps", "taker_fee_bps", "impact_coef",
                     "impact_cap_bps", "min_notional", "max_half_spread_bps", "default_sigma_bps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.spread_window < 2 or self.vol_window < 2 or self.vol_min_periods < 1:
            raise ValueError("spread_window and vol_window must be >= 2, vol_min_periods >= 1")

    def replace(self, **changes) -> "SimConfig":
        """Copy with some fields changed."""
        return replace(self, **changes)

    def half_spread_array(self, n_assets: int) -> np.ndarray:
        arr = np.broadcast_to(np.asarray(self.half_spread_bps, dtype=np.float64), (n_assets,))
        if (arr < 0).any():
            raise ValueError("half_spread_bps must be >= 0")
        return arr.copy()

    def qty_step_array(self, n_assets: int) -> np.ndarray | None:
        if self.qty_step is None:
            return None
        arr = np.broadcast_to(np.asarray(self.qty_step, dtype=np.float64), (n_assets,))
        if (arr < 0).any():
            raise ValueError("qty_step must be >= 0")
        return arr.copy()


def sqrt_impact_bps(qty, bar_volume, sigma_bps, coef: float, cap_bps: float):
    """Square-root temporary impact in bps (vectorised twin of the formula inlined in the engine)."""
    qty = np.asarray(qty, dtype=np.float64)
    vol = np.asarray(bar_volume, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(vol > 0, qty / vol, np.inf)
    return np.minimum(coef * np.asarray(sigma_bps) * np.sqrt(frac), cap_bps)


# ---------------------------------------------------------------------------
# Trailing estimators. Row k of each result only uses rows <= k.
# ---------------------------------------------------------------------------

def _trailing_sum(x: np.ndarray, w: int) -> np.ndarray:
    cs = np.cumsum(x, axis=0)
    out = cs.copy()
    if w < len(cs):
        out[w:] -= cs[:-w]
    return out


def _trailing_mean(x: np.ndarray, ok: np.ndarray, w: int, min_periods: int) -> np.ndarray:
    s = _trailing_sum(np.where(ok, x, 0.0), w)
    c = _trailing_sum(ok.astype(np.float64), w)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = s / c
    m[c < min_periods] = np.nan
    return m


def rolling_vol_bps(close: np.ndarray, valid: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Trailing rms of one-bar log returns, in bps (nan until `min_periods` returns exist).

    A return counts only if both of its bars are valid. Row k uses closes of rows <= k.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        lc = np.log(close)
    r = np.zeros_like(lc)
    r[1:] = lc[1:] - lc[:-1]
    ok = np.zeros(close.shape, dtype=bool)
    ok[1:] = valid[1:] & valid[:-1] & np.isfinite(r[1:])
    return 1e4 * np.sqrt(_trailing_mean(r * r, ok, window, min_periods))


def corwin_schultz_spread(high: np.ndarray, low: np.ndarray, valid: np.ndarray,
                          window: int, min_periods: int = 5) -> np.ndarray:
    """Corwin-Schultz (2012) high-low spread estimate as a fraction of price (full spread).

    Uses consecutive bar pairs; negative pair estimates are floored at zero, then averaged over
    the trailing window. Row k uses bars <= k. nan until `min_periods` pairs exist.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        hl = np.log(high / low) ** 2
        beta = hl[1:] + hl[:-1]
        gamma = np.log(np.maximum(high[1:], high[:-1]) / np.minimum(low[1:], low[:-1])) ** 2
        k = 3.0 - 2.0 * math.sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = np.maximum(s, 0.0)
    ok = np.zeros(high.shape, dtype=bool)
    ok[1:] = valid[1:] & valid[:-1] & np.isfinite(s) & (low[1:] > 0) & (low[:-1] > 0)
    full = np.zeros(high.shape)
    full[1:] = s
    return _trailing_mean(full, ok, window, min_periods)


def abdi_ranaldo_spread(high: np.ndarray, low: np.ndarray, close: np.ndarray, valid: np.ndarray,
                        window: int, min_periods: int = 5) -> np.ndarray:
    """Abdi-Ranaldo (2017) close-high-low spread estimate as a fraction of price (full spread).

    Per pair: 4 * (c_t - eta_t) * (c_t - eta_{t+1}) with eta the log mid-range; the window mean is
    floored at zero before the square root. Row k uses bars <= k. nan until `min_periods` pairs.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = 0.5 * (np.log(high) + np.log(low))
        c = np.log(close)
        x = 4.0 * (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])
    ok = np.zeros(high.shape, dtype=bool)
    ok[1:] = valid[1:] & valid[:-1] & np.isfinite(x)
    full = np.zeros(high.shape)
    full[1:] = x
    m = _trailing_mean(full, ok, window, min_periods)
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.maximum(m, 0.0))


def _lag1(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    out[0] = np.nan
    out[1:] = x[:-1]
    return out


def cost_inputs(panel: Panel, cfg: SimConfig, lo: int, hi: int,
                hs_floor: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar (sigma_bps, half_spread_bps) for rows [lo, hi), each shaped [hi - lo, N].

    Row b is computed from bars <= b - 1 only (strictly causal), independent of the block bounds.
    """
    n = panel.n_assets
    if hs_floor is None:
        hs_floor = cfg.half_spread_array(n)
    pad = max(cfg.vol_window, cfg.spread_window) + 2
    a = max(0, lo - pad)
    sl = slice(a, hi)
    valid = panel.valid[sl]
    sigma = _lag1(rolling_vol_bps(panel.close[sl], valid, cfg.vol_window, cfg.vol_min_periods))
    sigma = np.where(np.isfinite(sigma), sigma, cfg.default_sigma_bps)[lo - a:]
    if cfg.spread_estimator is None:
        return sigma, np.broadcast_to(hs_floor, sigma.shape)
    if cfg.spread_estimator == "corwin_schultz":
        full = corwin_schultz_spread(panel.high[sl], panel.low[sl], valid, cfg.spread_window)
    else:
        full = abdi_ranaldo_spread(panel.high[sl], panel.low[sl], panel.close[sl], valid,
                                   cfg.spread_window)
    est = 0.5e4 * _lag1(full)[lo - a:]
    est = np.where(np.isfinite(est), np.minimum(est, cfg.max_half_spread_bps), 0.0)
    return sigma, np.maximum(hs_floor, est)


class MarketCosts:
    """Lazily computed cost inputs, served one row at a time from cached fixed-grid blocks.

    Blocks are aligned to multiples of `block` rows, so a bar gets bit-identical inputs no
    matter where an episode starts or ends.
    """

    def __init__(self, panel: Panel, cfg: SimConfig, hs_floor: np.ndarray, block: int = 4_096):
        self._panel, self._cfg, self._floor, self._block = panel, cfg, hs_floor, block
        self._lo = self._hi = 0
        self._sigma = self._hs = np.empty((0, panel.n_assets))

    def prepare(self) -> None:
        """Drop the cached block (called when an episode starts)."""
        self._lo = self._hi = 0

    def row(self, b: int) -> tuple[np.ndarray, np.ndarray]:
        if not (self._lo <= b < self._hi):
            self._lo = (b // self._block) * self._block
            self._hi = min(self._panel.n_times, self._lo + self._block)
            self._sigma, self._hs = cost_inputs(self._panel, self._cfg, self._lo, self._hi, self._floor)
        k = b - self._lo
        return self._sigma[k], self._hs[k]
