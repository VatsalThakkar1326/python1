"""Load per-symbol Parquet bars into one aligned minute Panel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mark1 import config
from mark1.data.panel import Panel

FIELDS = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_volume"]


def load_panel(
    symbols: list[str] | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    bars_dir: Path = config.BARS_DIR,
    max_gap_minutes: int = 60,
) -> Panel:
    """Align symbols onto a gap-free UTC minute grid.

    Missing minutes get the previous close as open/high/low/close and zero volume. An asset
    is valid from its first bar onward, except while it has gone more than `max_gap_minutes`
    without a real bar (exchange outage or delisting), so long gaps are never traded.
    The grid spans [start, end) or, by default, the union of all symbols' data.
    """
    symbols = symbols or sorted(p.stem for p in bars_dir.glob("*.parquet"))
    if not symbols:
        raise FileNotFoundError(f"No bars found in {bars_dir}. Run `mark1 download` first.")
    frames = {}
    for sym in symbols:
        path = bars_dir / f"{sym}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run `mark1 download --symbols {sym}`.")
        frames[sym] = pd.read_parquet(path, columns=FIELDS)

    lo = pd.Timestamp(start) if start is not None else min(f.index[0] for f in frames.values())
    hi = pd.Timestamp(end) if end is not None else max(f.index[-1] for f in frames.values()) + pd.Timedelta(minutes=1)
    grid = pd.date_range(lo.floor("min"), hi.floor("min"), freq="min", inclusive="left")
    T, N = len(grid), len(symbols)

    arrays = {f: np.zeros((T, N), dtype=np.float64) for f in FIELDS}
    valid = np.zeros((T, N), dtype=bool)
    for j, sym in enumerate(symbols):
        df = frames[sym]
        df = df[(df.index >= grid[0]) & (df.index <= grid[-1])]
        aligned = df.reindex(grid)
        present = aligned["close"].notna().to_numpy()
        close = aligned["close"].ffill()
        # Before the first bar there is nothing to forward-fill from; backfill keeps prices
        # finite, and valid=False keeps them out of training and trading.
        close = close.bfill().to_numpy()
        for f in ("open", "high", "low"):
            arrays[f][:, j] = np.where(present, aligned[f].to_numpy(), close)
        arrays["close"][:, j] = close
        for f in ("volume", "quote_volume", "trades", "taker_buy_volume"):
            arrays[f][:, j] = np.nan_to_num(aligned[f].to_numpy(), nan=0.0)
        valid[:, j] = _validity(present, max_gap_minutes)

    return Panel(
        times=grid.to_numpy(dtype="datetime64[ns]"),
        symbols=list(symbols),
        valid=valid,
        meta={"source": "binance_archive", "max_gap_minutes": max_gap_minutes},
        **arrays,
    )


def _validity(present: np.ndarray, max_gap: int) -> np.ndarray:
    """True from the first real bar on, except inside runs of more than max_gap missing bars."""
    T = len(present)
    if not present.any():
        return np.zeros(T, dtype=bool)
    idx = np.arange(T)
    last_seen = np.where(present, idx, -1)
    last_seen = np.maximum.accumulate(last_seen)
    since = np.where(last_seen >= 0, idx - last_seen, np.iinfo(np.int64).max)
    valid = since <= max_gap
    # A long gap is invalid for its whole length, not just the part after max_gap minutes,
    # so the asset is not traded on stale prices at the start of an outage either.
    next_seen = np.where(present, idx, T)
    next_seen = np.minimum.accumulate(next_seen[::-1])[::-1]
    gap_len = next_seen - last_seen - 1
    in_long_gap = (~present) & (last_seen >= 0) & (gap_len > max_gap)
    return valid & ~in_long_gap
