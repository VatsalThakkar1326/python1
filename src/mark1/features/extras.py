"""Extra input dimensions, all causal (every value at row t was public by the open of bar t).

  #12 positioning (per coin, from USD-M perpetual futures)
      funding rate and its last change, open-interest change over 1 h and 1 day,
      top-trader and taker long/short ratios, futures premium over spot and its 1 h average
  #11 options market (market-wide): Binance BTC and ETH implied-volatility indices (BVOL)
  #7  macro (market-wide): gold (PAXG) and EUR/USD minute returns, US 10-year yield,
      VIX and the broad dollar index (daily, from FRED)
  #9  scheduled events (market-wide): hours to the next / since the last Fed decision and US
      CPI release, and a +-2 h event window flag

Timing rules:
  * Snapshots (funding, open interest, BVOL) count from their own timestamp: a value stamped
    at time c is used at row t only if c <= times[t].
  * Minute bars (premium index, gold, EUR/USD) follow the same convention as the main bars:
    row t uses the bar that opens at times[t].
  * FRED daily values are used from 00:00 UTC two days after their observation date. The
    10-year yield is published with a one-business-day lag and VIX closes after the US
    session, so a two-day lag is a safe upper bound.
  * Event calendar: scheduled dates are public in advance, so "hours to next" is legitimate;
    unscheduled actions only count once they happen (handled by mark1.data.events).

Missing data (before a source starts, or a coin without futures) becomes 0 with a
`has_*` flag, so the model can tell "no data" from "neutral".
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FRED_LAG_DAYS = 2
CLIP = 10.0

POSITIONING_NAMES = ["funding_bps", "funding_chg_bps", "oi_chg_60", "oi_chg_1440", "toptrader_ls", "taker_ls",
                     "premium_bps", "premium_60_bps", "has_futures"]
MARKET_EXTRA_NAMES = ["btc_bvol", "btc_bvol_chg_1d", "eth_bvol", "eth_bvol_chg_1d", "has_bvol",
                      "gold_ret_60", "gold_ret_1440", "eur_ret_60", "eur_ret_1440", "has_fx",
                      "us10y_level", "us10y_chg_5d", "vix_level", "vix_chg_5d", "usd_chg_5d", "has_macro",
                      "fomc_to_next", "fomc_since", "fomc_window", "cpi_to_next", "cpi_since", "cpi_window"]


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def asof(grid: np.ndarray, stamps: np.ndarray, values: np.ndarray, max_age: np.timedelta64 | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Latest value with stamp <= grid time. Returns (values, available mask)."""
    stamps = np.asarray(stamps, dtype="datetime64[ns]")
    idx = np.searchsorted(stamps, grid, side="right") - 1
    ok = idx >= 0
    out = np.where(ok, np.asarray(values, dtype=float)[np.clip(idx, 0, None)], np.nan)
    if max_age is not None:
        age = grid - stamps[np.clip(idx, 0, None)]
        ok &= age <= max_age
        out = np.where(ok, out, np.nan)
    return out, ok


def _bars_close(grid: np.ndarray, df: pd.DataFrame, max_gap_minutes: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Close of the bar opening at each grid time, forward-filled across short gaps."""
    s = df["close"].astype(float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    aligned = s.reindex(pd.DatetimeIndex(grid))
    filled = aligned.ffill(limit=max_gap_minutes)
    return filled.to_numpy(), filled.notna().to_numpy()


def _lagdiff(x: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    out[k:] = x[k:] - x[:-k]
    return out


def _vol_scaled_returns(logp: np.ndarray, windows: list[int], halflife: int = 1440) -> list[np.ndarray]:
    r1 = np.nan_to_num(_lagdiff(logp, 1))
    sigma = np.maximum(np.sqrt(pd.Series(r1 ** 2).ewm(halflife=halflife, adjust=False).mean().to_numpy()), 1e-6)
    return [_lagdiff(logp, k) / (sigma * np.sqrt(k)) for k in windows]


def positioning_features(grid: np.ndarray, symbols: list[str], extras_dir: Path) -> np.ndarray:
    """[T, N, len(POSITIONING_NAMES)] float32 per-coin futures positioning features."""
    T, N = len(grid), len(symbols)
    out = np.zeros((T, N, len(POSITIONING_NAMES)), dtype=np.float32)
    day = np.timedelta64(1, "D")
    for j, sym in enumerate(symbols):
        cols = {}
        fund = _read(extras_dir / "funding" / f"{sym}.parquet")
        if fund is not None and len(fund):
            fund = fund.sort_index()
            rate, ok_f = asof(grid, fund.index.to_numpy(), fund["funding_rate"].to_numpy(), max_age=2 * day)
            prev = fund["funding_rate"].shift(1).to_numpy()
            prev_v, _ = asof(grid, fund.index.to_numpy(), prev, max_age=2 * day)
            cols["funding_bps"] = rate * 1e4
            cols["funding_chg_bps"] = (rate - prev_v) * 1e4
        else:
            ok_f = np.zeros(T, dtype=bool)
        met = _read(extras_dir / "metrics" / f"{sym}.parquet")
        if met is not None and len(met):
            met = met.sort_index()
            st = met.index.to_numpy()
            oi, ok_m = asof(grid, st, np.log(np.maximum(met["open_interest"].to_numpy(dtype=float), 1e-9)), max_age=np.timedelta64(30, "m"))
            cols["oi_chg_60"] = _lagdiff(oi, 60) * 100
            cols["oi_chg_1440"] = _lagdiff(oi, 1440) * 100
            tt, _ = asof(grid, st, met["toptrader_ls_sum"].to_numpy(dtype=float), max_age=np.timedelta64(30, "m"))
            tk, _ = asof(grid, st, met["taker_ls_vol"].to_numpy(dtype=float), max_age=np.timedelta64(30, "m"))
            cols["toptrader_ls"] = np.log(np.maximum(tt, 1e-6))
            cols["taker_ls"] = np.log(np.maximum(tk, 1e-6))
        else:
            ok_m = np.zeros(T, dtype=bool)
        prem = _read(extras_dir / "premium" / f"{sym}.parquet")
        if prem is not None and len(prem):
            pc, ok_p = _bars_close(grid, prem)
            cols["premium_bps"] = pc * 1e4
            cols["premium_60_bps"] = pd.Series(pc * 1e4).rolling(60, min_periods=10).mean().to_numpy()
        else:
            ok_p = np.zeros(T, dtype=bool)
        has = ok_f | ok_m | ok_p
        for k, name in enumerate(POSITIONING_NAMES[:-1]):
            if name in cols:
                out[:, j, k] = np.clip(np.nan_to_num(cols[name]), -CLIP * 10, CLIP * 10)
        out[:, j, -1] = has
        if not has.any():
            log.info("%s: no futures positioning data", sym)
    return out


def market_extra_features(grid: np.ndarray, extras_dir: Path) -> np.ndarray:
    """[T, len(MARKET_EXTRA_NAMES)] float32 market-wide options, macro and event features."""
    T = len(grid)
    cols: dict[str, np.ndarray] = {}
    day = np.timedelta64(1, "D")

    has_bvol = np.zeros(T, dtype=bool)
    for name, key in (("BTCBVOLUSDT", "btc"), ("ETHBVOLUSDT", "eth")):
        df = _read(extras_dir / "bvol" / f"{name}.parquet")
        if df is not None and len(df):
            df = df.sort_index()
            v, ok = asof(grid, df.index.to_numpy(), df["value"].to_numpy(dtype=float), max_age=day)
            lv = np.log(np.maximum(v, 1e-6) / 50.0)
            cols[f"{key}_bvol"] = lv
            cols[f"{key}_bvol_chg_1d"] = _lagdiff(lv, 1440)
            has_bvol |= ok
    cols["has_bvol"] = has_bvol.astype(float)

    has_fx = np.zeros(T, dtype=bool)
    for sym, key in (("PAXGUSDT", "gold"), ("EURUSDT", "eur")):
        df = _read(extras_dir / "spot" / f"{sym}.parquet")
        if df is not None and len(df):
            c, ok = _bars_close(grid, df)
            r60, r1440 = _vol_scaled_returns(np.log(np.maximum(c, 1e-12)), [60, 1440])
            cols[f"{key}_ret_60"], cols[f"{key}_ret_1440"] = r60, r1440
            has_fx |= ok
    cols["has_fx"] = has_fx.astype(float)

    has_macro = np.zeros(T, dtype=bool)
    lag = np.timedelta64(FRED_LAG_DAYS, "D")
    fred = {s: _read(extras_dir / "fred" / f"{s}.parquet") for s in ("DGS10", "VIXCLS", "DTWEXBGS")}

    def daily(series: str):
        df = fred[series]
        if df is None or not len(df):
            return None, None
        df = df.sort_index()
        stamps = df.index.to_numpy(dtype="datetime64[ns]") + lag            # first moment the value may be used
        vals = df["value"].to_numpy(dtype=float)
        v, ok = asof(grid, stamps, vals, max_age=10 * day)
        v5, _ = asof(grid, stamps, pd.Series(vals).shift(5).to_numpy(), max_age=10 * day)   # five observations earlier
        return (v, v5), ok

    r, ok = daily("DGS10")
    if r is not None:
        cols["us10y_level"] = (r[0] - 3.0) / 1.5
        cols["us10y_chg_5d"] = (r[0] - r[1]) * 100 / 25          # basis points / 25
        has_macro |= ok
    r, ok = daily("VIXCLS")
    if r is not None:
        cols["vix_level"] = np.log(np.maximum(r[0], 1e-6) / 20.0)
        cols["vix_chg_5d"] = np.log(np.maximum(r[0], 1e-6) / np.maximum(r[1], 1e-6))
        has_macro |= ok
    r, ok = daily("DTWEXBGS")
    if r is not None:
        cols["usd_chg_5d"] = np.log(np.maximum(r[0], 1e-6) / np.maximum(r[1], 1e-6)) * 100
        has_macro |= ok
    cols["has_macro"] = has_macro.astype(float)

    try:
        from mark1.data.events import event_features, load_events
        ev = event_features(grid, load_events())
        cap = np.log1p(24 * 14)
        for kind, key in (("FOMC_DECISION", "fomc"), ("CPI_RELEASE", "cpi")):
            cols[f"{key}_to_next"] = np.log1p(ev[f"{kind}_hours_to_next"]) / cap
            cols[f"{key}_since"] = np.log1p(ev[f"{kind}_hours_since_last"]) / cap
            cols[f"{key}_window"] = ev[f"{kind}_window"]
    except (ImportError, FileNotFoundError) as exc:
        log.warning("event calendar unavailable (%s); event features set to 0", exc)

    out = np.zeros((T, len(MARKET_EXTRA_NAMES)), dtype=np.float32)
    for k, name in enumerate(MARKET_EXTRA_NAMES):
        if name in cols:
            out[:, k] = np.clip(np.nan_to_num(cols[name]), -CLIP, CLIP)
    return out


def has_extras(extras_dir: Path) -> bool:
    return extras_dir.exists() and any(extras_dir.iterdir())
