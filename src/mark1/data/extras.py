"""Extra market-data "dimensions" from free public sources, saved as Parquet under config.EXTRAS_DIR.

Six kinds, all with a fixed schema (the feature code relies on it):

    funding/{SYMBOL}.parquet   index `time`       funding_rate, interval_hours       USD-M perpetual funding, one row per funding event (8h)
    metrics/{SYMBOL}.parquet   index `time`       open_interest, open_interest_value, toptrader_ls_count,
                                                  toptrader_ls_sum, ls_count, taker_ls_vol      futures positioning, 5-minute snapshots
    premium/{SYMBOL}.parquet   index `open_time`  open, high, low, close             futures premium-index 1-minute klines
    bvol/{NAME}.parquet        index `time`       value                              Binance options volatility index (BTCBVOLUSDT, ETHBVOLUSDT)
    spot/{SYMBOL}.parquet      index `open_time`  same as the 1-minute bars          context spot pairs (PAXGUSDT = gold, EURUSDT = EUR/USD)
    fred/{SERIES}.parquet      index `date`       value                              DGS10, VIXCLS, DTWEXBGS (daily, raw observation dates)

All indexes are UTC, tz-naive datetime64[ns]; all columns are float64.

Sources and availability (data.binance.vision, no key; FRED public CSV):

* funding: monthly zips only. A month is published a day or so after it ends, so the CURRENT month is missing
  until month end (there are no daily funding files). Timestamps are the exact calc time (a few ms of jitter).
* metrics: DAILY zips only (no monthly files exist), ~365 per symbol per year, each ~12 KB. Rows inside a file
  are NOT in time order and early files contain duplicate rows; we sort and drop duplicates. BTCUSDT starts
  2020-09-01, the other coins 2021-12-01. Series are located with a binary search over days (HEAD requests)
  instead of requesting thousands of 404s.
* premium: monthly zips for finished months and daily zips for the current month (a recently finished month
  whose monthly zip is not published yet falls back to daily zips). Early files (<= 2021) have no header row.
* bvol: daily zips only, from 2023-06-20, 1 Hz raw (86 400 rows/day, ~430 KB). Stored on a 1-minute grid:
  `time` is the END of the minute and `value` the last level published before that instant, so a row never
  contains information from after its own timestamp. Minutes without any update are absent (no forward fill).
* fred: https://fred.stlouisfed.org/graph/fredgraph.csv, full history is tiny. Missing values ('.' or empty)
  are dropped. NO publication lag is applied: `date` is the raw observation date. We keep FRED_LOOKBACK_DAYS
  before `start` so the first days of a run can still be forward-filled.

Timestamps may be milliseconds or microseconds (spot changed on 2025-01-01); the unit is detected per value.
Verified zips are cached under RAW_DIR/extras/{kind}/{SYMBOL}/ so re-runs only fetch what is missing.
Symbols without a same-named USD-M perpetual (e.g. SHIBUSDT trades as 1000SHIBUSDT), or whose futures data
starts after `start`, are skipped or trimmed with an INFO log, never an error.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from mark1 import config
from mark1.data import binance_archive as ba

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://data.binance.vision/data"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

BVOL_INDICES = ("BTCBVOLUSDT", "ETHBVOLUSDT")
CONTEXT_SPOT = ("PAXGUSDT", "EURUSDT")
FRED_SERIES = ("DGS10", "VIXCLS", "DTWEXBGS")
FRED_LOOKBACK_DAYS = 30

FUTURES_KINDS = ("funding", "metrics", "premium")
ALL_KINDS = ("funding", "metrics", "premium", "bvol", "spot", "fred")

FUNDING_COLUMNS = ["funding_rate", "interval_hours"]
METRICS_COLUMNS = ["open_interest", "open_interest_value", "toptrader_ls_count", "toptrader_ls_sum", "ls_count", "taker_ls_vol"]
PREMIUM_COLUMNS = ["open", "high", "low", "close"]
# CSV header -> our column name
_METRICS_RENAME = {
    "sum_open_interest": "open_interest",
    "sum_open_interest_value": "open_interest_value",
    "count_toptrader_long_short_ratio": "toptrader_ls_count",
    "sum_toptrader_long_short_ratio": "toptrader_ls_sum",
    "count_long_short_ratio": "ls_count",
    "sum_taker_long_short_vol_ratio": "taker_ls_vol",
}

# Index name used in each kind's Parquet files, and the nominal row spacing (for the coverage gap share).
INDEX_NAME = {"funding": "time", "metrics": "time", "premium": "open_time", "bvol": "time", "spot": "open_time", "fred": "date"}
_CADENCE = {"metrics": pd.Timedelta(minutes=5), "premium": pd.Timedelta(minutes=1), "spot": pd.Timedelta(minutes=1)}
DEFAULT_BVOL_FREQ = "1min"
SMALL_FILE_KINDS = ("funding", "metrics")   # ~1-15 KB per zip: request latency dominates, so use more threads
SMALL_FILE_THREADS = 4                       # multiplier on `workers` for those kinds
UNIT_THREADS = 4                             # (kind, symbol) series processed concurrently


# --------------------------------------------------------------------------------------------------------------------
# Archive file planning
# --------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Layout:
    path: str                    # directory under ARCHIVE_URL; {period} and {symbol} are filled in
    name: str                    # file name; {symbol} and {stamp} are filled in
    periods: tuple[str, ...]     # which granularities the archive publishes


_LAYOUTS = {
    "funding": _Layout("futures/um/{period}/fundingRate/{symbol}", "{symbol}-fundingRate-{stamp}.zip", ("monthly",)),
    "metrics": _Layout("futures/um/{period}/metrics/{symbol}", "{symbol}-metrics-{stamp}.zip", ("daily",)),
    "premium": _Layout("futures/um/{period}/premiumIndexKlines/{symbol}/1m", "{symbol}-1m-{stamp}.zip", ("monthly", "daily")),
    "bvol": _Layout("option/{period}/BVOLIndex/{symbol}", "{symbol}-BVOLIndex-{stamp}.zip", ("daily",)),
}


@dataclass(frozen=True)
class ExtraFile:
    """One archive zip. Same interface as binance_archive.ArchiveFile, so ba.fetch_file works on it."""
    kind: str            # "funding", "metrics", "premium" or "bvol"
    symbol: str
    period: str          # "monthly" or "daily"
    stamp: str           # "2026-08" or "2026-10-23"

    @property
    def name(self) -> str:
        return _LAYOUTS[self.kind].name.format(symbol=self.symbol, stamp=self.stamp)

    @property
    def url(self) -> str:
        directory = _LAYOUTS[self.kind].path.format(period=self.period, symbol=self.symbol)
        return f"{ARCHIVE_URL}/{directory}/{self.name}"

    def local_path(self, raw_dir: Path) -> Path:
        return raw_dir / "extras" / self.kind / self.symbol / self.name


def _next_month(month: date) -> date:
    return date(month.year + (month.month == 12), month.month % 12 + 1, 1)


def plan_extra_files(kind: str, symbol: str, start: date, end: date, today: date | None = None) -> list[ExtraFile]:
    """Archive files covering [start, end], oldest first.

    Monthly zips for finished months (funding has nothing else, so its current month is absent), daily zips for the
    current month when the archive has them, and daily zips throughout for daily-only kinds (metrics, bvol).
    """
    today = today or datetime.now(timezone.utc).date()
    periods = _LAYOUTS[kind].periods
    last_day = min(end, today - timedelta(days=1))   # the newest day that can have a published zip
    files: list[ExtraFile] = []
    if "monthly" not in periods:
        day = start
        while day <= last_day:
            files.append(ExtraFile(kind, symbol, "daily", f"{day:%Y-%m-%d}"))
            day += timedelta(days=1)
        return files
    month = date(start.year, start.month, 1)
    while month <= end:
        nxt = _next_month(month)
        if nxt <= today:
            files.append(ExtraFile(kind, symbol, "monthly", f"{month:%Y-%m}"))
        elif "daily" in periods:
            day = max(month, start)
            last = min(last_day, nxt - timedelta(days=1))
            while day <= last:
                files.append(ExtraFile(kind, symbol, "daily", f"{day:%Y-%m-%d}"))
                day += timedelta(days=1)
        month = nxt
    return files


def _daily_fallback(f: ExtraFile, start: date, end: date, today: date) -> list[ExtraFile]:
    """Daily zips standing in for a monthly zip that is not published yet (a month that ended in the last ~40 days)."""
    if f.period != "monthly" or "daily" not in _LAYOUTS[f.kind].periods:
        return []
    month = date.fromisoformat(f.stamp + "-01")
    nxt = _next_month(month)
    if nxt <= today - timedelta(days=40):
        return []
    day, last = max(month, start), min(end, nxt - timedelta(days=1), today - timedelta(days=1))
    out = []
    while day <= last:
        out.append(ExtraFile(f.kind, f.symbol, "daily", f"{day:%Y-%m-%d}"))
        day += timedelta(days=1)
    return out


def first_available(n: int, exists: Callable[[int], bool], tolerance: int = 2) -> int | None:
    """Index of the first file worth requesting among `n` chronologically ordered files, or None if none exist.

    Binary search, so a series that starts years after the requested start costs ~log2(n) probes, not n 404s. It
    assumes a series stays published once it starts. A probe at i counts as a hit when any of i..i+tolerance exists,
    so an isolated missing day does not send the search the wrong way. The result is at most `tolerance` positions
    before the true first file (that file is then simply skipped as a 404 when downloading).
    """
    if n <= 0:
        return None
    memo: dict[int, bool] = {}

    def probe(i: int) -> bool:
        if i not in memo:
            memo[i] = exists(i)
        return memo[i]

    def hit(i: int) -> bool:
        return any(probe(j) for j in range(i, min(i + tolerance + 1, n)))

    if hit(0):
        return 0
    hi = max(n - 1 - tolerance, 0)
    if not hit(hi):
        return None
    lo = 0   # invariant: hit(lo) is False, hit(hi) is True
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if hit(mid):
            hi = mid
        else:
            lo = mid
    return hi


# --------------------------------------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------------------------------------

def make_session(workers: int = 8) -> requests.Session:
    """A Session whose connection pool is big enough for `workers` threads (keep-alive matters for thousands of tiny files)."""
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=4, pool_maxsize=max(workers * 2, 10))
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _head_exists(session: requests.Session, url: str, retries: int = 3) -> bool:
    """HEAD probe: True if the archive has the file, False on 404. Other errors are retried, then raised."""
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            resp = session.head(url, timeout=30)
            if resp.status_code == 404:
                return False
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            log.warning("HEAD %s failed (%s), retrying in %.0fs", url, exc, delay)
            time.sleep(delay)
            delay *= 2
    return False


def file_exists(f: ExtraFile, raw_dir: Path, session: requests.Session) -> bool:
    return f.local_path(raw_dir).exists() or _head_exists(session, f.url)


# --------------------------------------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------------------------------------

def _zip_csv(path: Path) -> bytes:
    with zipfile.ZipFile(path) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        return zf.read(name)


def _read_csv(raw: bytes, columns: dict[str, int]) -> pd.DataFrame:
    """CSV bytes -> frame of the wanted columns, found by header name (or by position when the file has no header).

    `columns` maps source column name -> position in a header-less file. Some archive files (early premium klines) have
    no header row while newer ones do, and the header text is the only thing that tells them apart.
    """
    if raw.lstrip(b"\xef\xbb\xbf \r\n")[:1].isalpha():
        df = pd.read_csv(io.BytesIO(raw), usecols=list(columns))   # ValueError when a wanted column is missing
        return df[list(columns)]
    ordered = sorted(columns.items(), key=lambda kv: kv[1])
    df = pd.read_csv(io.BytesIO(raw), header=None, usecols=[pos for _, pos in ordered])
    df.columns = [name for name, _ in ordered]
    return df[list(columns)]


def _epoch_index(ts, name: str) -> pd.DatetimeIndex:
    """Integer epochs in ms, us or ns (detected per value) -> UTC tz-naive datetime64[ns] index."""
    ts = np.asarray(ts, dtype="int64")
    # Epochs around 1970-2100 are 1e9-4e9 s: ms ~1e12, us ~1e15, ns ~1e18.
    ns = np.where(ts > 10**17, ts, np.where(ts > 10**14, ts * 1_000, ts * 1_000_000))
    return pd.DatetimeIndex(ns.astype("datetime64[ns]"), name=name)


def _time_index(col: pd.Series, name: str) -> pd.DatetimeIndex:
    if pd.api.types.is_numeric_dtype(col):
        return _epoch_index(col.to_numpy(), name)
    idx = pd.DatetimeIndex(pd.to_datetime(col))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return pd.DatetimeIndex(idx.astype("datetime64[ns]"), name=name)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by time and drop duplicate timestamps (the last row for a timestamp wins)."""
    df = df.sort_index(kind="stable")
    return df[~df.index.duplicated(keep="last")]


def _num(col: pd.Series) -> np.ndarray:
    return pd.to_numeric(col, errors="coerce").to_numpy(dtype="float64")


def parse_funding_zip(path: Path) -> pd.DataFrame:
    """Funding zip -> index `time` (calc time), columns funding_rate, interval_hours."""
    df = _read_csv(_zip_csv(path), {"calc_time": 0, "funding_interval_hours": 1, "last_funding_rate": 2})
    out = pd.DataFrame({"funding_rate": _num(df["last_funding_rate"]), "interval_hours": _num(df["funding_interval_hours"])},
                       index=_epoch_index(df["calc_time"], "time"))
    return _finalize(out)


def parse_metrics_zip(path: Path) -> pd.DataFrame:
    """Daily futures-metrics zip -> index `time` (create_time), six float64 positioning columns, sorted and deduplicated."""
    positions = {"create_time": 0, **{c: i + 2 for i, c in enumerate(_METRICS_RENAME)}}
    df = _read_csv(_zip_csv(path), positions)
    out = pd.DataFrame({new: _num(df[old]) for old, new in _METRICS_RENAME.items()}, index=_time_index(df["create_time"], "time"))
    return _finalize(out)


def parse_premium_zip(path: Path) -> pd.DataFrame:
    """Premium-index kline zip (monthly or daily) -> index `open_time`, columns open, high, low, close."""
    df = _read_csv(_zip_csv(path), {"open_time": 0, "open": 1, "high": 2, "low": 3, "close": 4})
    out = pd.DataFrame({c: _num(df[c]) for c in PREMIUM_COLUMNS}, index=_epoch_index(df["open_time"], "open_time"))
    return _finalize(out)


def parse_bvol_zip(path: Path, freq: str | None = DEFAULT_BVOL_FREQ) -> pd.DataFrame:
    """Daily BVOL zip (1 Hz) -> index `time`, column `value`.

    With `freq` (default "1min") the series is thinned to the last level published in each bucket and stamped at the
    bucket END, i.e. `value` is the index level known at `time`. `freq=None` keeps every raw 1 Hz row.
    """
    df = _read_csv(_zip_csv(path), {"calc_time": 0, "index_value": 4})
    out = pd.DataFrame({"value": _num(df["index_value"])}, index=_epoch_index(df["calc_time"], "time"))
    out = _finalize(out.dropna())
    if freq is None or out.empty:
        return out
    bucket = out.index.floor(freq)
    last_in_bucket = ~bucket.duplicated(keep="last")
    out = out[last_in_bucket]
    out.index = pd.DatetimeIndex(bucket[last_in_bucket] + pd.Timedelta(freq), name="time")
    return out


def parse_fred_csv(text: str) -> pd.DataFrame:
    """FRED fredgraph.csv text -> index `date` (observation date), column `value`; missing values are dropped.

    The first column is the date (named observation_date, or DATE in older exports) and the second the series. Missing
    observations appear as '.' in old exports and as empty cells now; both are dropped.
    """
    raw = pd.read_csv(io.StringIO(text), dtype=str)
    if raw.shape[1] < 2:
        raise ValueError(f"unexpected FRED response: {text[:80]!r}")
    dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    values = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
    keep = dates.notna() & values.notna()
    out = pd.DataFrame({"value": values[keep].to_numpy(dtype="float64")},
                       index=pd.DatetimeIndex(dates[keep].astype("datetime64[ns]"), name="date"))
    return _finalize(out)


# --------------------------------------------------------------------------------------------------------------------
# Downloading
# --------------------------------------------------------------------------------------------------------------------

def _write_parquet(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    df.to_parquet(tmp)
    tmp.replace(out)  # atomic, like the zip cache


def _clip(df: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    return df[(df.index >= lo) & (df.index < hi)]


def _fetch_parse(f: ExtraFile, raw_dir: Path, session: requests.Session, parser: Callable[[Path], pd.DataFrame]) -> pd.DataFrame | None:
    """Download (checksum-verified, cached) and parse one file. A corrupt cached zip is dropped and fetched again once."""
    path = ba.fetch_file(f, raw_dir, session)
    if path is None:
        return None
    try:
        return parser(path)
    except (zipfile.BadZipFile, ValueError, KeyError, StopIteration, EOFError) as exc:
        log.warning("%s: cached file unreadable (%s: %s), downloading it again", f.name, type(exc).__name__, exc)
        path.unlink(missing_ok=True)
        path = ba.fetch_file(f, raw_dir, session)
        return None if path is None else parser(path)


def _fetch_all(files: list[ExtraFile], raw_dir: Path, session: requests.Session, pool: ThreadPoolExecutor,
               parser: Callable[[Path], pd.DataFrame]) -> tuple[list[pd.DataFrame], list[ExtraFile]]:
    """Fetch + parse files in the pool. Returns (frames in time order, files missing upstream)."""
    futures: list[Future] = [pool.submit(_fetch_parse, f, raw_dir, session, parser) for f in files]
    try:
        results = [fut.result() for fut in futures]
    except BaseException:
        for fut in futures:
            fut.cancel()
        raise
    frames = [r for r in results if r is not None]
    missing = [f for f, r in zip(files, results) if r is None]
    return frames, missing


def _download_archive_kind(
    kind: str, symbol: str, start: date, end: date, *, raw_dir: Path, out_dir: Path, session: requests.Session,
    pool: ThreadPoolExecutor, today: date, bvol_freq: str | None = DEFAULT_BVOL_FREQ,
) -> Path | None:
    """Download one (kind, symbol) series from the Binance archive and write its Parquet. None if nothing is available."""
    label = f"{kind}/{symbol}"
    files = plan_extra_files(kind, symbol, start, end, today)
    if kind == "funding" and end >= today.replace(day=1) and start <= today:
        log.info("%s: the current month (%s) has no published file yet (funding is monthly only); the series ends at the previous month end",
                 label, f"{today:%Y-%m}")
    tolerance = 1 if "monthly" in _LAYOUTS[kind].periods else 2   # months are rarely missing mid-series, days occasionally are

    def exists(i: int) -> bool:   # a recent month counts as there when its monthly zip is not up yet but its daily zips are
        return file_exists(files[i], raw_dir, session) or any(
            file_exists(d, raw_dir, session) for d in _daily_fallback(files[i], start, end, today)[:2])

    if not files:
        log.info("%s: no archive files fall inside %s..%s (today is %s); skipped", label, start, end, today)
        return None
    first = first_available(len(files), exists, tolerance)
    if first is None:
        log.info("%s: not in the archive between %s and %s (no such USD-M perpetual, or not listed yet); skipped", label, start, end)
        return None
    if first > 0:
        log.info("%s: archive data starts at %s, requested from %s", label, files[first].stamp, files[0].stamp)
    files = files[first:]

    parser = {"funding": parse_funding_zip, "metrics": parse_metrics_zip, "premium": parse_premium_zip,
              "bvol": partial(parse_bvol_zip, freq=bvol_freq)}[kind]
    frames, missing = _fetch_all(files, raw_dir, session, pool, parser)
    fallbacks = {f: _daily_fallback(f, start, end, today) for f in missing}
    retry = [d for days in fallbacks.values() for d in days]
    if retry:
        log.info("%s: %d monthly file(s) not published yet, falling back to %d daily files", label, sum(1 for d in fallbacks.values() if d), len(retry))
        more, missing_daily = _fetch_all(retry, raw_dir, session, pool, parser)
        frames += more
        missing = [f for f, days in fallbacks.items() if not days] + missing_daily
    if missing:
        log.info("%s: %d of %d archive files not available (%s ... %s)", label, len(missing), len(files),
                 missing[0].stamp, missing[-1].stamp)
    if not frames:
        log.info("%s: no data downloaded", label)
        return None
    df = _finalize(pd.concat(frames))
    df = _clip(df, pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1))
    if df.empty:
        log.info("%s: no rows inside %s..%s", label, start, end)
        return None
    out = out_dir / kind / f"{symbol}.parquet"
    _write_parquet(df, out)
    log.info("%s: %d rows %s -> %s written to %s", label, len(df), df.index[0], df.index[-1], out)
    return out


def download_fred(series: str, start: date, end: date, out_dir: Path = config.EXTRAS_DIR,
                  session: requests.Session | None = None) -> Path | None:
    """Fetch one FRED series (full history, a few hundred KB) and write fred/{series}.parquet for [start - lookback, end]."""
    session = session or requests.Session()
    resp = ba._get(session, f"{FRED_URL}?id={series}")
    if resp is None:
        log.info("fred/%s: not found (HTTP 404); skipped", series)
        return None
    df = parse_fred_csv(resp.text)
    lo = pd.Timestamp(start) - pd.Timedelta(days=FRED_LOOKBACK_DAYS)
    df = _clip(df, lo, pd.Timestamp(end) + pd.Timedelta(days=1))
    if df.empty:
        log.info("fred/%s: no observations inside %s..%s", series, start, end)
        return None
    out = out_dir / "fred" / f"{series}.parquet"
    _write_parquet(df, out)
    log.info("fred/%s: %d rows %s -> %s written to %s", series, len(df), df.index[0].date(), df.index[-1].date(), out)
    return out


def download_extras(
    symbols: Iterable[str],
    start: date,
    end: date,
    workers: int = 8,
    raw_dir: Path = config.RAW_DIR,
    extras_dir: Path = config.EXTRAS_DIR,
    kinds: Iterable[str] | None = None,
    session: requests.Session | None = None,
    bvol_freq: str | None = DEFAULT_BVOL_FREQ,
    today: date | None = None,
) -> dict[str, dict[str, Path | None]]:
    """Download every extra data kind for [start, end] and write the Parquet files under `extras_dir`.

    `symbols` are spot symbols (the trading universe); funding/metrics/premium are fetched for each one that has a
    same-named USD-M perpetual. BVOL (BTC, ETH), the context spot pairs (PAXGUSDT, EURUSDT) and the FRED series are
    always the same fixed lists. `kinds` restricts the run to a subset of ALL_KINDS. Returns {kind: {name: path or None}}
    where None means nothing was available.

    `workers` threads fetch the big files (premium, bvol, spot). Funding and metrics are many tiny files, where the
    time is all request latency, so they get 4x as many threads. Anything merely missing upstream (404) is logged and
    skipped; a real failure (network, checksum) is logged, the remaining work still runs, and a RuntimeError listing
    the failures is raised at the end. Re-running resumes from the zip cache.
    """
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    kinds = tuple(kinds) if kinds is not None else ALL_KINDS
    unknown = set(kinds) - set(ALL_KINDS)
    if unknown:
        raise ValueError(f"unknown kinds {sorted(unknown)}; choose from {list(ALL_KINDS)}")
    symbols = list(dict.fromkeys(symbols))
    today = today or datetime.now(timezone.utc).date()
    session = session or make_session(workers * SMALL_FILE_THREADS)
    names = {"funding": symbols, "metrics": symbols, "premium": symbols, "bvol": list(BVOL_INDICES),
             "spot": list(CONTEXT_SPOT), "fred": list(FRED_SERIES)}
    results: dict[str, dict[str, Path | None]] = {k: {} for k in kinds}
    failures: list[str] = []

    def run_unit(kind: str, name: str, i: int, pool: ThreadPoolExecutor) -> Path | None:
        log.info("[%s %d/%d] %s", kind, i, len(names[kind]), name)
        try:
            if kind == "spot":
                return ba.download_symbol(name, start, end, raw_dir, extras_dir / "spot", workers, session)
            if kind == "fred":
                return download_fred(name, start, end, extras_dir, session)
            return _download_archive_kind(kind, name, start, end, raw_dir=raw_dir, out_dir=extras_dir, session=session,
                                          pool=pool, today=today, bvol_freq=bvol_freq)
        except Exception as exc:  # keep going; the zip cache makes the re-run cheap
            log.exception("%s/%s failed", kind, name)
            failures.append(f"{kind}/{name}: {type(exc).__name__}: {exc}")
            return None

    # A few series at a time, so one series' probing/parsing/writing overlaps another's downloading.
    with ThreadPoolExecutor(workers) as big, ThreadPoolExecutor(workers * SMALL_FILE_THREADS) as small, \
            ThreadPoolExecutor(UNIT_THREADS) as units:
        for kind in kinds:
            pool = small if kind in SMALL_FILE_KINDS else big
            futures = {name: units.submit(run_unit, kind, name, i, pool) for i, name in enumerate(names[kind], 1)}
            for name, fut in futures.items():
                results[kind][name] = fut.result()
    if failures:
        raise RuntimeError(f"{len(failures)} extras download(s) failed (re-run to resume): " + "; ".join(sorted(failures)))
    return results


# --------------------------------------------------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------------------------------------------------

def extras_coverage(extras_dir: Path = config.EXTRAS_DIR) -> pd.DataFrame:
    """What is on disk: first/last timestamp, row count and (for fixed-cadence kinds) the share of missing rows per file."""
    import pyarrow.parquet as pq

    rows = []
    for kind in ALL_KINDS:
        for p in sorted((extras_dir / kind).glob("*.parquet")):
            pf = pq.ParquetFile(p)
            n = pf.metadata.num_rows
            col = INDEX_NAME[kind] if INDEX_NAME[kind] in pf.schema_arrow.names else pf.schema_arrow.names[0]
            idx = pf.read(columns=[col]).column(0).to_numpy()
            first, last = (pd.Timestamp(idx.min()), pd.Timestamp(idx.max())) if n else (pd.NaT, pd.NaT)
            step = _CADENCE.get(kind)
            if kind == "bvol" and n > 1:   # the grid spacing is a download option: use what the file actually has
                step = pd.Timedelta(np.median(np.diff(idx)))
            missing = np.nan
            if step is not None and n:
                missing = round(100 * (1 - n / (int((last - first) / step) + 1)), 3)
            rows.append({"kind": kind, "name": p.stem, "first": first, "last": last, "rows": n, "missing_pct": missing})
    return pd.DataFrame(rows, columns=["kind", "name", "first", "last", "rows", "missing_pct"])


__all__ = [
    "ExtraFile", "plan_extra_files", "first_available", "make_session", "file_exists",
    "parse_funding_zip", "parse_metrics_zip", "parse_premium_zip", "parse_bvol_zip", "parse_fred_csv",
    "download_fred", "download_extras", "extras_coverage",
    "ALL_KINDS", "BVOL_INDICES", "CONTEXT_SPOT", "FRED_SERIES",
    "FUNDING_COLUMNS", "METRICS_COLUMNS", "PREMIUM_COLUMNS",
]
