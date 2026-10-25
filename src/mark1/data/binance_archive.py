"""Download 1-minute spot klines from the public Binance archive (data.binance.vision).

The archive needs no account or API key. Complete months come as one zip per month, the
current month as one zip per day. Every zip has a published SHA-256 checksum, which we
verify before using the file. Verified zips are cached in RAW_DIR so re-runs only fetch
what is missing.

Binance changed spot timestamps from milliseconds to microseconds on 2025-01-01; the
parser detects the unit per file.
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from mark1 import config

log = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/spot"
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]
BAR_COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_volume"]


@dataclass(frozen=True)
class ArchiveFile:
    symbol: str
    period: str          # "monthly" or "daily"
    stamp: str           # "2026-08" or "2026-10-23"
    interval: str = "1m"

    @property
    def name(self) -> str:
        return f"{self.symbol}-{self.interval}-{self.stamp}.zip"

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.period}/klines/{self.symbol}/{self.interval}/{self.name}"

    def local_path(self, raw_dir: Path) -> Path:
        return raw_dir / self.symbol / self.interval / self.name


def plan_files(symbol: str, start: date, end: date, today: date | None = None, interval: str = "1m") -> list[ArchiveFile]:
    """Archive files covering [start, end]: monthly zips for finished months, daily zips otherwise."""
    today = today or datetime.now(timezone.utc).date()
    files: list[ArchiveFile] = []
    month = date(start.year, start.month, 1)
    while month <= end:
        next_month = date(month.year + (month.month == 12), month.month % 12 + 1, 1)
        month_finished = next_month <= today
        if month_finished:
            files.append(ArchiveFile(symbol, "monthly", f"{month:%Y-%m}", interval))
        else:
            day = max(month, start)
            last = min(end, today - timedelta(days=1), next_month - timedelta(days=1))
            while day <= last:
                files.append(ArchiveFile(symbol, "daily", f"{day:%Y-%m-%d}", interval))
                day += timedelta(days=1)
        month = next_month
    return files


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _get(session: requests.Session, url: str, retries: int = 4) -> requests.Response | None:
    """GET with exponential backoff. Returns None on 404 (file does not exist)."""
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, timeout=60)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            log.warning("GET %s failed (%s), retrying in %.0fs", url, exc, delay)
            time.sleep(delay)
            delay *= 2
    return None


def fetch_file(f: ArchiveFile, raw_dir: Path, session: requests.Session) -> Path | None:
    """Download and checksum-verify one archive file. Returns its path, or None if missing upstream."""
    path = f.local_path(raw_dir)
    if path.exists():
        return path
    resp = _get(session, f.url)
    if resp is None:
        return None
    checksum = _get(session, f.url + ".CHECKSUM")
    if checksum is None:
        raise RuntimeError(f"Missing checksum for {f.url}")
    expected = checksum.text.split()[0].strip().lower()
    actual = _sha256(resp.content)
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch for {f.name}: expected {expected}, got {actual}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".zip.part")
    tmp.write_bytes(resp.content)
    tmp.replace(path)  # atomic, so a crash never leaves a half-written zip that looks valid
    return path


def parse_kline_zip(path: Path) -> pd.DataFrame:
    """Read one archive zip into a DataFrame indexed by bar open time (UTC, datetime64[ns])."""
    with zipfile.ZipFile(path) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        raw = zf.read(csv_name)
    df = pd.read_csv(io.BytesIO(raw), header=None, names=KLINE_COLUMNS)
    if not df.empty and isinstance(df.iloc[0, 0], str):  # a few old files ship a header row
        df = df.iloc[1:]
    ts = df["open_time"].astype("int64").to_numpy()
    # Microsecond epochs are ~1e15 in this era, millisecond epochs ~1e12.
    unit = "us" if ts.max() > 1e14 else "ms"
    index = pd.to_datetime(ts, unit=unit, utc=True).tz_convert(None).astype("datetime64[ns]")
    out = df[BAR_COLUMNS].astype("float64")
    out.index = pd.DatetimeIndex(index, name="open_time")
    return out


def download_symbol(
    symbol: str,
    start: date,
    end: date,
    raw_dir: Path = config.RAW_DIR,
    bars_dir: Path = config.BARS_DIR,
    workers: int = 8,
    session: requests.Session | None = None,
) -> Path | None:
    """Fetch all archive files for one symbol and write a single Parquet of 1-minute bars."""
    session = session or requests.Session()
    files = plan_files(symbol, start, end)
    paths: list[Path] = []
    missing = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_file, f, raw_dir, session): f for f in files}
        for fut in as_completed(futures):
            p = fut.result()
            if p is None:
                missing += 1
            else:
                paths.append(p)
    if missing:
        log.info("%s: %d of %d archive files not available (before listing or not yet published)", symbol, missing, len(files))
    if not paths:
        log.warning("%s: no data downloaded", symbol)
        return None
    frames = [parse_kline_zip(p) for p in sorted(paths)]
    bars = pd.concat(frames)
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    lo, hi = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    bars = bars[(bars.index >= lo) & (bars.index < hi)]
    bars_dir.mkdir(parents=True, exist_ok=True)
    out = bars_dir / f"{symbol}.parquet"
    bars.to_parquet(out)
    log.info("%s: %d bars %s -> %s written to %s", symbol, len(bars), bars.index[0], bars.index[-1], out)
    return out


def download_universe(
    symbols: list[str],
    start: date,
    end: date,
    raw_dir: Path = config.RAW_DIR,
    bars_dir: Path = config.BARS_DIR,
    workers: int = 8,
) -> dict[str, Path | None]:
    session = requests.Session()
    results = {}
    for i, sym in enumerate(symbols, 1):
        log.info("[%d/%d] %s", i, len(symbols), sym)
        results[sym] = download_symbol(sym, start, end, raw_dir, bars_dir, workers, session)
    return results


def coverage(bars_dir: Path = config.BARS_DIR) -> pd.DataFrame:
    """Summary of what is on disk: first/last bar, row count, and gap share per symbol."""
    rows = []
    for p in sorted(bars_dir.glob("*.parquet")):
        idx = pd.read_parquet(p, columns=["close"]).index
        expected = int((idx[-1] - idx[0]) / pd.Timedelta(minutes=1)) + 1
        rows.append({
            "symbol": p.stem,
            "first": idx[0],
            "last": idx[-1],
            "bars": len(idx),
            "missing_pct": round(100 * (1 - len(idx) / expected), 3),
        })
    return pd.DataFrame(rows)


__all__ = [
    "ArchiveFile", "plan_files", "fetch_file", "parse_kline_zip",
    "download_symbol", "download_universe", "coverage", "BAR_COLUMNS",
]
