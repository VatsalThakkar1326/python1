"""Offline tests for the extra-data downloaders (funding, metrics, premium, BVOL, context spot, FRED).

No test touches the network: archive files are built in memory and served by FakeSession, which stands in for
requests.Session.
"""

import hashlib
import io
import logging
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest
import requests

from mark1.data import binance_archive as ba
from mark1.data import extras as ex

FUNDING_HEADER = "calc_time,funding_interval_hours,last_funding_rate"
METRICS_HEADER = ("create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,"
                  "sum_toptrader_long_short_ratio,count_long_short_ratio,sum_taker_long_short_vol_ratio")
PREMIUM_HEADER = "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"
BVOL_HEADER = "calc_time,symbol,base_asset,quote_asset,index_value"


def ms(ts: str) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000)


def us(ts: str) -> int:
    return int(pd.Timestamp(ts).value // 1_000)


def csv(header: str | None, rows) -> str:
    lines = [header] if header else []
    lines += [",".join(str(x) for x in r) for r in rows]
    return "\n".join(lines) + "\n"


def write_zip(path, text):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(path.stem + ".csv", text)
    return path


def zip_bytes(name: str, text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, text)
    return buf.getvalue()


def premium_row(t, o, h, lo, c):
    return [t, o, h, lo, c, 0, t + 59_999, 0, 12, 0, 0, 0]


class FakeResponse:
    def __init__(self, status: int, content: bytes = b""):
        self.status_code = status
        self.content = content

    @property
    def text(self) -> str:
        return self.content.decode()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """requests.Session look-alike: serves registered URLs (plus their .CHECKSUM), 404 for everything else."""

    def __init__(self):
        self.bodies: dict[str, bytes] = {}
        self.broken: set[str] = set()   # URLs whose GET/HEAD raise a connection error
        self.requests: list[tuple[str, str]] = []

    def add_zip(self, url: str, text: str):
        self.bodies[url] = zip_bytes(url.rsplit("/", 1)[1].replace(".zip", ".csv"), text)

    def add_text(self, url: str, text: str):
        self.bodies[url] = text.encode()

    def add_file(self, f, text: str):
        self.add_zip(f.url, text)

    def _lookup(self, method: str, url: str) -> FakeResponse:
        self.requests.append((method, url))
        if url in self.broken:
            raise requests.ConnectionError("boom")
        if url in self.bodies:
            return FakeResponse(200, self.bodies[url])
        if url.endswith(".CHECKSUM") and url[: -len(".CHECKSUM")] in self.bodies:
            data = self.bodies[url[: -len(".CHECKSUM")]]
            return FakeResponse(200, f"{hashlib.sha256(data).hexdigest()}  {url.rsplit('/', 1)[1][:-9]}\n".encode())
        return FakeResponse(404)

    def get(self, url, timeout=None, **kw):
        return self._lookup("GET", url)

    def head(self, url, timeout=None, **kw):
        resp = self._lookup("HEAD", url)
        return FakeResponse(resp.status_code)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    # retry/backoff sleeps in the HTTP helpers must not slow the tests down
    monkeypatch.setattr(ba.time, "sleep", lambda s: None)
    monkeypatch.setattr(ex.time, "sleep", lambda s: None)


def assert_schema(df: pd.DataFrame, index_name: str, columns: list[str]):
    assert df.index.name == index_name
    assert str(df.index.dtype) == "datetime64[ns]"
    assert df.index.tz is None
    assert list(df.columns) == columns
    assert all(str(t) == "float64" for t in df.dtypes)
    assert df.index.is_monotonic_increasing and df.index.is_unique


# --------------------------------------------------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------------------------------------------------

def test_parse_funding_schema_ms_and_us(tmp_path):
    rows_ms = [(ms("2026-09-01 00:00:00") + 5, 8, 0.00008482), (ms("2026-09-01 08:00:00"), 8, 0.0001)]
    rows_us = [(us("2026-09-01 16:00:00") + 2_000, 4, -0.00000244)]   # microsecond epochs, as spot did from 2025
    a = ex.parse_funding_zip(write_zip(tmp_path / "A-fundingRate-2026-09.zip", csv(FUNDING_HEADER, rows_ms)))
    b = ex.parse_funding_zip(write_zip(tmp_path / "B-fundingRate-2026-09.zip", csv(FUNDING_HEADER, rows_us)))
    assert_schema(a, "time", ["funding_rate", "interval_hours"])
    assert_schema(b, "time", ["funding_rate", "interval_hours"])
    assert a.index[0] == pd.Timestamp("2026-09-01 00:00:00.005") and a.index[1] == pd.Timestamp("2026-09-01 08:00:00")
    assert a["funding_rate"].tolist() == [0.00008482, 0.0001] and a["interval_hours"].tolist() == [8.0, 8.0]
    assert b.index[0] == pd.Timestamp("2026-09-01 16:00:00.002") and b["funding_rate"].iloc[0] == -0.00000244
    assert b["interval_hours"].iloc[0] == 4.0


def test_parse_funding_sorts_and_dedupes(tmp_path):
    rows = [(ms("2026-09-01 08:00:00"), 8, 0.0002), (ms("2026-09-01 00:00:00"), 8, 0.0001), (ms("2026-09-01 08:00:00"), 8, 0.0003)]
    df = ex.parse_funding_zip(write_zip(tmp_path / "F-fundingRate-2026-09.zip", csv(FUNDING_HEADER, rows)))
    assert_schema(df, "time", ["funding_rate", "interval_hours"])
    assert df["funding_rate"].tolist() == [0.0001, 0.0003]   # sorted; the last duplicate wins


def test_parse_metrics_maps_columns_sorts_and_dedupes(tmp_path):
    # a real daily file is not in time order and old ones repeat rows
    rows = [
        ("2026-10-23 00:10:00", "BTCUSDT", "95000.5000000000000000", "7.7e9", 2.2, 1.5, 2.1, 1.1),
        ("2026-10-23 00:05:00", "BTCUSDT", "95970.8040000000000000", "7797679539.1623660000000000", 2.23853141, 1.476975, 2.09535846, 1.006736),
        ("2026-10-23 00:05:00", "BTCUSDT", "95970.8040000000000000", "7797679539.1623660000000000", 2.23853141, 1.476975, 2.09535846, 1.006736),
        ("2026-10-23 00:00:00", "BTCUSDT", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    ]
    df = ex.parse_metrics_zip(write_zip(tmp_path / "BTCUSDT-metrics-2026-10-23.zip", csv(METRICS_HEADER, rows)))
    assert_schema(df, "time", ["open_interest", "open_interest_value", "toptrader_ls_count", "toptrader_ls_sum", "ls_count", "taker_ls_vol"])
    assert df.index.tolist() == [pd.Timestamp("2026-10-23 00:00"), pd.Timestamp("2026-10-23 00:05"), pd.Timestamp("2026-10-23 00:10")]
    assert df.iloc[0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    row = df.loc["2026-10-23 00:05"]
    assert row["open_interest"] == pytest.approx(95970.804) and row["open_interest_value"] == pytest.approx(7797679539.162366)
    assert row["toptrader_ls_count"] == pytest.approx(2.23853141) and row["toptrader_ls_sum"] == pytest.approx(1.476975)
    assert row["ls_count"] == pytest.approx(2.09535846) and row["taker_ls_vol"] == pytest.approx(1.006736)


def test_parse_metrics_epoch_create_time(tmp_path):
    rows = [(ms("2026-10-23 00:05:00"), "BTCUSDT", 1, 2, 3, 4, 5, 6)]
    df = ex.parse_metrics_zip(write_zip(tmp_path / "M-metrics-2026-10-23.zip", csv(METRICS_HEADER, rows)))
    assert df.index[0] == pd.Timestamp("2026-10-23 00:05:00") and str(df.index.dtype) == "datetime64[ns]"


@pytest.mark.parametrize("header", [PREMIUM_HEADER, None], ids=["header", "no_header"])
def test_parse_premium_ms_us_header_and_no_header(tmp_path, header):
    # early archive files (<= 2021) have no header row; recent ones do
    t0, t1 = ms("2026-09-01 00:00:00"), ms("2026-09-01 00:01:00")
    a = ex.parse_premium_zip(write_zip(tmp_path / "A-1m-2026-09.zip", csv(header, [premium_row(t1, 2, 3, 1, 2.5), premium_row(t0, -0.00045, -0.00044, -0.0005, -0.00048)])))
    b = ex.parse_premium_zip(write_zip(tmp_path / "B-1m-2026-09.zip", csv(header, [premium_row(t0 * 1000, 1, 2, 0.5, 1.5)])))
    assert_schema(a, "open_time", ["open", "high", "low", "close"])
    assert_schema(b, "open_time", ["open", "high", "low", "close"])
    assert a.index.tolist() == [pd.Timestamp("2026-09-01 00:00"), pd.Timestamp("2026-09-01 00:01")]   # sorted
    assert a.iloc[0].tolist() == [-0.00045, -0.00044, -0.0005, -0.00048] and a.iloc[1].tolist() == [2.0, 3.0, 1.0, 2.5]
    assert b.index[0] == pd.Timestamp("2026-09-01 00:00") and b.iloc[0].tolist() == [1.0, 2.0, 0.5, 1.5]


def test_parse_bvol_one_minute_grid_has_no_lookahead(tmp_path):
    base = ms("2026-10-23 00:00:00")
    rows = [(base + 6 + 1000 * k, "BTCBVOLUSDT", "BTCBVOL", "USDT", 40 + k) for k in range(150)]   # 1 Hz for 2.5 minutes, values 40..189
    rows.append(rows[10])                                                                          # duplicate timestamp
    path = write_zip(tmp_path / "BTCBVOLUSDT-BVOLIndex-2026-10-23.zip", csv(BVOL_HEADER, rows[::-1]))   # shuffled order
    df = ex.parse_bvol_zip(path)
    assert_schema(df, "time", ["value"])
    # stamped at the END of each minute; the value is the last one published inside that minute
    assert df.index.tolist() == [pd.Timestamp("2026-10-23 00:01"), pd.Timestamp("2026-10-23 00:02"), pd.Timestamp("2026-10-23 00:03")]
    assert df["value"].tolist() == [99.0, 159.0, 189.0]         # 40+59, 40+119, and 40+149 for the partial last minute
    raw = ex.parse_bvol_zip(path, freq=None)
    assert len(raw) == 150 and raw.index[0] == pd.Timestamp("2026-10-23 00:00:00.006")
    assert_schema(raw, "time", ["value"])
    for t, v in df["value"].items():                            # every grid value was published at or before its own timestamp
        assert raw.index[raw["value"] == v][0] <= t


def test_parse_bvol_microsecond_epochs(tmp_path):
    rows = [(us("2026-10-23 00:00:30"), "ETHBVOLUSDT", "ETHBVOL", "USDT", 75.5)]
    df = ex.parse_bvol_zip(write_zip(tmp_path / "ETHBVOLUSDT-BVOLIndex-2026-10-23.zip", csv(BVOL_HEADER, rows)))
    assert df.index.tolist() == [pd.Timestamp("2026-10-23 00:01")] and df["value"].tolist() == [75.5]


@pytest.mark.parametrize("header,missing", [("observation_date,DGS10", ""), ("DATE,DGS10", ".")], ids=["new_empty_cells", "old_dot"])
def test_parse_fred_drops_missing(header, missing):
    text = f"{header}\n2021-01-01,{missing}\n2021-01-04,0.93\n2021-01-05,{missing}\n2021-01-06,1.04\n2021-01-07,.\n"
    df = ex.parse_fred_csv(text)
    assert_schema(df, "date", ["value"])
    assert df.index.tolist() == [pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-06")]   # raw observation dates, no lag
    assert df["value"].tolist() == [0.93, 1.04]


def test_parse_fred_rejects_garbage():
    with pytest.raises(ValueError):
        ex.parse_fred_csv("<html>not a csv</html>")


# --------------------------------------------------------------------------------------------------------------------
# planning and the earliest-date search
# --------------------------------------------------------------------------------------------------------------------

def test_file_urls_match_the_archive_layout():
    base = "https://data.binance.vision/data"
    assert ex.ExtraFile("funding", "BTCUSDT", "monthly", "2026-09").url == f"{base}/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2026-09.zip"
    assert ex.ExtraFile("metrics", "BTCUSDT", "daily", "2026-10-23").url == f"{base}/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2026-10-23.zip"
    assert ex.ExtraFile("premium", "BTCUSDT", "monthly", "2026-09").url == f"{base}/futures/um/monthly/premiumIndexKlines/BTCUSDT/1m/BTCUSDT-1m-2026-09.zip"
    assert ex.ExtraFile("premium", "BTCUSDT", "daily", "2026-10-23").url == f"{base}/futures/um/daily/premiumIndexKlines/BTCUSDT/1m/BTCUSDT-1m-2026-10-23.zip"
    assert ex.ExtraFile("bvol", "BTCBVOLUSDT", "daily", "2026-10-23").url == f"{base}/option/daily/BVOLIndex/BTCBVOLUSDT/BTCBVOLUSDT-BVOLIndex-2026-10-23.zip"


def test_plan_funding_is_monthly_only_and_skips_the_current_month():
    files = ex.plan_extra_files("funding", "BTCUSDT", date(2026, 8, 15), date(2026, 10, 24), today=date(2026, 10, 25))
    assert [(f.period, f.stamp) for f in files] == [("monthly", "2026-08"), ("monthly", "2026-09")]


def test_plan_metrics_is_daily_only_up_to_yesterday():
    files = ex.plan_extra_files("metrics", "BTCUSDT", date(2026, 8, 30), date(2026, 10, 24), today=date(2026, 9, 2))
    assert [f.stamp for f in files] == ["2026-08-30", "2026-08-31", "2026-09-01"]
    assert {f.period for f in files} == {"daily"}


def test_plan_premium_monthly_then_daily_for_the_current_month():
    files = ex.plan_extra_files("premium", "BTCUSDT", date(2026, 8, 15), date(2026, 10, 3), today=date(2026, 10, 5))
    assert [(f.period, f.stamp) for f in files] == [
        ("monthly", "2026-08"), ("monthly", "2026-09"), ("daily", "2026-10-01"), ("daily", "2026-10-02"), ("daily", "2026-10-03")]


def test_first_available_binary_search_uses_few_probes():
    n, first = 2000, 1500
    probes = []

    def exists(i):
        probes.append(i)
        return i >= first

    got = ex.first_available(n, exists)
    assert first - 2 <= got <= first                  # at most `tolerance` positions early
    assert len(probes) < 40                           # not 1500 requests for 404s


def test_first_available_edge_cases():
    assert ex.first_available(0, lambda i: True) is None
    assert ex.first_available(100, lambda i: False) is None
    calls = []
    assert ex.first_available(100, lambda i: calls.append(i) or True) == 0 and calls == [0]


def test_first_available_survives_an_isolated_missing_day():
    first = 700
    missing = {first + 3}   # the search would otherwise be fooled when it lands exactly here

    def exists(i):
        return i >= first and i not in missing

    for n in (900, 1000, 1024, 1500):
        got = ex.first_available(n, exists)
        assert first - 2 <= got <= first


def test_file_exists_prefers_the_local_cache(tmp_path):
    f = ex.ExtraFile("metrics", "BTCUSDT", "daily", "2026-10-23")
    session = FakeSession()
    assert not ex.file_exists(f, tmp_path, session) and session.requests == [("HEAD", f.url)]
    f.local_path(tmp_path).parent.mkdir(parents=True)
    write_zip(f.local_path(tmp_path), "x")
    session.requests.clear()
    assert ex.file_exists(f, tmp_path, session) and session.requests == []


# --------------------------------------------------------------------------------------------------------------------
# download_extras end to end (fake HTTP)
# --------------------------------------------------------------------------------------------------------------------

START, END, TODAY = date(2026, 8, 1), date(2026, 9, 3), date(2026, 10, 25)


def populate(session: FakeSession):
    """A tiny archive: BTCUSDT everywhere; ETHUSDT metrics only from 2026-08-29 (listing later than the requested start)."""
    for sym in ("BTCUSDT", "ETHUSDT"):
        for month, day0 in (("2026-08", "2026-08-01"), ("2026-09", "2026-09-01")):
            rows = [(ms(f"{day0} 00:00:00") + 3, 8, 0.0001), (ms(f"{day0} 08:00:00"), 8, 0.0002), (ms(f"{day0} 16:00:00"), 8, 0.0003)]
            session.add_file(ex.ExtraFile("funding", sym, "monthly", month), csv(FUNDING_HEADER, rows))
            rows = [premium_row(ms(f"{day0} 00:01:00"), 2, 3, 1, 2.5), premium_row(ms(f"{day0} 00:00:00"), 1, 2, 0, 1.5)]
            session.add_file(ex.ExtraFile("premium", sym, "monthly", month), csv(PREMIUM_HEADER, rows))
    for k in range(34):
        day = (pd.Timestamp(START) + pd.Timedelta(days=k)).date()
        rows = [(f"{day} 00:05:00", "BTCUSDT", 10 + k, 20 + k, 1, 2, 3, 4), (f"{day} 00:00:00", "BTCUSDT", 9 + k, 19 + k, 1, 2, 3, 4)]
        session.add_file(ex.ExtraFile("metrics", "BTCUSDT", "daily", f"{day}"), csv(METRICS_HEADER, rows))
        if day >= date(2026, 8, 29):
            session.add_file(ex.ExtraFile("metrics", "ETHUSDT", "daily", f"{day}"), csv(METRICS_HEADER, rows))
        for name in ex.BVOL_INDICES:
            rows = [(ms(f"{day} 00:00:00") + 6, name, name[:-4], "USDT", 50.0), (ms(f"{day} 00:00:30"), name, name[:-4], "USDT", 51.0),
                    (ms(f"{day} 00:01:10"), name, name[:-4], "USDT", 52.0)]
            session.add_file(ex.ExtraFile("bvol", name, "daily", f"{day}"), csv(BVOL_HEADER, rows))
    # context spot: one monthly 1-minute-bar zip, PAXGUSDT only (EURUSDT stays a 404)
    for month, day0 in (("2026-08", "2026-08-01"), ("2026-09", "2026-09-01")):
        t = ms(f"{day0} 00:00:00")
        rows = [[t + 60_000 * k, 100, 101, 99, 100.5, 1, t + 60_000 * k + 59_999, 100, 10, 0.5, 50, 0] for k in range(3)]
        session.add_zip(ba.ArchiveFile("PAXGUSDT", "monthly", month).url, csv(None, rows))
    for series, text in (("DGS10", "observation_date,DGS10\n2026-07-31,\n2026-08-03,4.1\n2026-08-04,4.2\n2026-09-01,4.3\n2026-09-04,4.4\n"),
                         ("VIXCLS", "observation_date,VIXCLS\n2026-08-03,16.5\n2026-08-04,.\n"),
                         ("DTWEXBGS", "observation_date,DTWEXBGS\n2026-08-03,120.1\n")):
        session.add_text(f"{ex.FRED_URL}?id={series}", text)


@pytest.fixture
def downloaded(tmp_path):
    session = FakeSession()
    populate(session)
    raw, out = tmp_path / "raw", tmp_path / "extras"
    results = ex.download_extras(["BTCUSDT", "ETHUSDT", "NOPERPUSDT"], START, END, workers=4, raw_dir=raw, extras_dir=out,
                                 session=session, today=TODAY)
    return results, session, raw, out


def test_download_extras_writes_every_kind_with_the_fixed_schema(downloaded):
    results, _, _, out = downloaded
    assert set(results) == set(ex.ALL_KINDS)

    funding = pd.read_parquet(out / "funding" / "BTCUSDT.parquet")
    assert_schema(funding, "time", ["funding_rate", "interval_hours"])
    assert len(funding) == 6 and funding.index[0] == pd.Timestamp("2026-08-01 00:00:00.003")

    metrics = pd.read_parquet(out / "metrics" / "BTCUSDT.parquet")
    assert_schema(metrics, "time", ex.METRICS_COLUMNS)
    assert len(metrics) == 34 * 2 and metrics.index[0] == pd.Timestamp("2026-08-01 00:00:00")
    assert metrics.index[-1] == pd.Timestamp("2026-09-03 00:05:00")

    premium = pd.read_parquet(out / "premium" / "BTCUSDT.parquet")
    assert_schema(premium, "open_time", ["open", "high", "low", "close"])
    assert premium.index.tolist() == [pd.Timestamp("2026-08-01 00:00"), pd.Timestamp("2026-08-01 00:01"),
                                      pd.Timestamp("2026-09-01 00:00"), pd.Timestamp("2026-09-01 00:01")]

    for name in ex.BVOL_INDICES:
        bvol = pd.read_parquet(out / "bvol" / f"{name}.parquet")
        assert_schema(bvol, "time", ["value"])
        assert bvol.index[:2].tolist() == [pd.Timestamp("2026-08-01 00:01"), pd.Timestamp("2026-08-01 00:02")]
        assert bvol["value"].iloc[:2].tolist() == [51.0, 52.0]

    spot = pd.read_parquet(out / "spot" / "PAXGUSDT.parquet")
    assert list(spot.columns) == ba.BAR_COLUMNS and spot.index.name == "open_time" and len(spot) == 6
    assert str(spot.index.dtype) == "datetime64[ns]"

    dgs10 = pd.read_parquet(out / "fred" / "DGS10.parquet")
    assert_schema(dgs10, "date", ["value"])
    # kept: 30 days of lookback before START and everything up to END; '.'/empty dropped; 2026-09-04 is after END
    assert dgs10.index.tolist() == [pd.Timestamp("2026-08-03"), pd.Timestamp("2026-08-04"), pd.Timestamp("2026-09-01")]
    assert pd.read_parquet(out / "fred" / "VIXCLS.parquet")["value"].tolist() == [16.5]


def test_download_extras_skips_what_is_not_there(downloaded):
    results, session, _, out = downloaded
    assert results["funding"]["NOPERPUSDT"] is None and results["metrics"]["NOPERPUSDT"] is None
    assert results["premium"]["NOPERPUSDT"] is None
    assert results["spot"]["EURUSDT"] is None and results["spot"]["PAXGUSDT"] is not None
    assert not (out / "metrics" / "NOPERPUSDT.parquet").exists() and not (out / "spot" / "EURUSDT.parquet").exists()
    # ETHUSDT metrics start later than the requested start: found by searching, not by requesting every day
    eth = pd.read_parquet(out / "metrics" / "ETHUSDT.parquet")
    assert eth.index[0] == pd.Timestamp("2026-08-29 00:00:00")
    eth_gets = [u for m, u in session.requests if m == "GET" and "/metrics/ETHUSDT/" in u and not u.endswith(".CHECKSUM")]
    eth_404_gets = [u for u in eth_gets if not any(u.endswith(f"{d}.zip") for d in pd.date_range("2026-08-29", END).strftime("%Y-%m-%d"))]
    assert len(eth_404_gets) <= 2                      # at most the `tolerance` days before the first file
    eth_heads = [u for m, u in session.requests if m == "HEAD" and "/metrics/ETHUSDT/" in u]
    assert 0 < len(eth_heads) < 25


def test_download_extras_logs_skips_at_info(tmp_path, caplog):
    session = FakeSession()
    caplog.set_level(logging.INFO, logger=ex.log.name)
    ex.download_extras(["NOPERPUSDT"], START, END, workers=2, raw_dir=tmp_path / "raw", extras_dir=tmp_path / "e", session=session,
                       kinds=["funding", "metrics", "premium"], today=TODAY)
    skipped = [r for r in caplog.records if "skipped" in r.getMessage()]
    assert len(skipped) == 3 and all(r.levelno == logging.INFO for r in skipped)


def test_download_extras_range_with_no_published_files_is_skipped(tmp_path):
    session = FakeSession()
    # asking for "today" only: daily archive files exist up to yesterday, so there is nothing to request
    res = ex.download_extras(["BTCUSDT"], TODAY, TODAY, workers=2, raw_dir=tmp_path / "raw", extras_dir=tmp_path / "e", session=session,
                             today=TODAY, kinds=["metrics", "bvol"])
    assert res == {"metrics": {"BTCUSDT": None}, "bvol": {"BTCBVOLUSDT": None, "ETHBVOLUSDT": None}}
    assert session.requests == []


def test_download_extras_all_404_never_crashes(tmp_path):
    session = FakeSession()   # serves nothing at all, FRED included
    out = tmp_path / "extras"
    results = ex.download_extras(["BTCUSDT", "ETHUSDT"], START, END, workers=3, raw_dir=tmp_path / "raw", extras_dir=out,
                                 session=session, today=TODAY)
    assert all(path is None for per_kind in results.values() for path in per_kind.values())
    assert not list(out.rglob("*.parquet"))
    assert len(ex.extras_coverage(out)) == 0


def test_download_extras_cache_makes_reruns_free(downloaded, tmp_path):
    results, session, raw, out = downloaded
    before = sum(1 for m, u in session.requests if m == "GET" and u.endswith(".zip") and "/futures/" in u)
    assert before > 0
    session.requests.clear()
    ex.download_extras(["BTCUSDT"], START, END, workers=4, raw_dir=raw, extras_dir=out, session=session, today=TODAY,
                       kinds=["funding", "metrics", "premium"])
    zip_gets = [u for m, u in session.requests if m == "GET" and u.endswith(".zip")]
    assert zip_gets == []                              # every archive zip came from RAW_DIR; no HEAD probes needed either
    assert not [1 for m, u in session.requests if m == "HEAD"]


def test_download_extras_failure_is_reported_but_other_series_are_written(tmp_path):
    session = FakeSession()
    populate(session)
    broken = ex.ExtraFile("funding", "BTCUSDT", "monthly", "2026-09")
    session.broken.add(broken.url)
    out = tmp_path / "extras"
    with pytest.raises(RuntimeError, match=r"funding/BTCUSDT"):
        ex.download_extras(["BTCUSDT"], START, END, workers=2, raw_dir=tmp_path / "raw", extras_dir=out, session=session, today=TODAY,
                           kinds=["funding", "premium"])
    assert not (out / "funding" / "BTCUSDT.parquet").exists()     # never a silent hole
    assert (out / "premium" / "BTCUSDT.parquet").exists()


def test_checksum_mismatch_is_an_error_not_a_skip(tmp_path):
    session = FakeSession()
    f = ex.ExtraFile("funding", "BTCUSDT", "monthly", "2026-08")
    session.add_file(f, csv(FUNDING_HEADER, [(ms("2026-08-01 00:00:00"), 8, 0.0001)]))
    session.bodies[f.url + ".CHECKSUM"] = b"0" * 64 + b"  x.zip\n"
    with pytest.raises(RuntimeError, match="Checksum mismatch|failed"):
        ex.download_extras(["BTCUSDT"], date(2026, 8, 1), date(2026, 8, 31), workers=2, raw_dir=tmp_path / "raw",
                           extras_dir=tmp_path / "e", session=session, today=TODAY, kinds=["funding"])
    assert not f.local_path(tmp_path / "raw").exists()


def test_premium_falls_back_to_daily_files_when_the_monthly_zip_is_not_published_yet(tmp_path):
    session = FakeSession()
    today = date(2026, 10, 3)   # September just ended: its monthly zip is not up yet, the daily ones are
    for day in ("2026-09-01", "2026-09-02", "2026-09-03"):
        session.add_file(ex.ExtraFile("premium", "BTCUSDT", "daily", day), csv(PREMIUM_HEADER, [premium_row(ms(f"{day} 00:00:00"), 1, 2, 0, 1)]))
    out = tmp_path / "extras"
    res = ex.download_extras(["BTCUSDT"], date(2026, 9, 1), date(2026, 9, 3), workers=2, raw_dir=tmp_path / "raw", extras_dir=out,
                             session=session, today=today, kinds=["premium"])
    df = pd.read_parquet(res["premium"]["BTCUSDT"])
    assert df.index.tolist() == [pd.Timestamp("2026-09-01"), pd.Timestamp("2026-09-02"), pd.Timestamp("2026-09-03")]


def test_corrupt_cached_zip_is_replaced(tmp_path):
    session = FakeSession()
    f = ex.ExtraFile("funding", "BTCUSDT", "monthly", "2026-08")
    session.add_file(f, csv(FUNDING_HEADER, [(ms("2026-08-01 00:00:00"), 8, 0.0001)]))
    cached = f.local_path(tmp_path / "raw")
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"not a zip")
    res = ex.download_extras(["BTCUSDT"], date(2026, 8, 1), date(2026, 8, 31), workers=2, raw_dir=tmp_path / "raw",
                             extras_dir=tmp_path / "e", session=session, today=TODAY, kinds=["funding"])
    assert len(pd.read_parquet(res["funding"]["BTCUSDT"])) == 1


def test_download_extras_rejects_bad_arguments(tmp_path):
    with pytest.raises(ValueError):
        ex.download_extras(["BTCUSDT"], date(2026, 9, 2), date(2026, 9, 1), session=FakeSession())
    with pytest.raises(ValueError):
        ex.download_extras(["BTCUSDT"], START, END, kinds=["bogus"], session=FakeSession())


# --------------------------------------------------------------------------------------------------------------------
# coverage and CLI
# --------------------------------------------------------------------------------------------------------------------

def test_extras_coverage_summary(downloaded):
    _, _, _, out = downloaded
    cov = ex.extras_coverage(out)
    assert list(cov.columns) == ["kind", "name", "first", "last", "rows", "missing_pct"]
    row = cov[(cov.kind == "metrics") & (cov.name == "BTCUSDT")].iloc[0]
    assert row["first"] == pd.Timestamp("2026-08-01 00:00") and row["last"] == pd.Timestamp("2026-09-03 00:05") and row["rows"] == 68
    assert row["missing_pct"] > 99                    # two 5-minute snapshots per day: almost all of the grid is absent
    dgs = cov[(cov.kind == "fred") & (cov.name == "DGS10")].iloc[0]
    assert dgs["rows"] == 3 and np.isnan(dgs["missing_pct"])
    assert set(cov.kind) == {"funding", "metrics", "premium", "bvol", "spot", "fred"}


def test_extras_coverage_empty_dir(tmp_path):
    assert len(ex.extras_coverage(tmp_path / "nothing")) == 0


def test_cli_download_extras_wiring(monkeypatch, capsys):
    from mark1 import cli
    seen = {}

    def fake_download(symbols, start, end, workers=8, kinds=None, **kw):
        seen.update(symbols=symbols, start=start, end=end, workers=workers, kinds=kinds)

    monkeypatch.setattr(ex, "download_extras", fake_download)
    monkeypatch.setattr(ex, "extras_coverage", lambda *a, **k: pd.DataFrame({"kind": ["fred"], "name": ["DGS10"], "rows": [3]}))
    cli.main(["download-extras", "--symbols", "BTCUSDT", "ETHUSDT", "--start", "2026-01-01", "--end", "2026-01-31", "--workers", "3",
              "--kinds", "funding", "fred"])
    assert seen == {"symbols": ["BTCUSDT", "ETHUSDT"], "start": date(2026, 1, 1), "end": date(2026, 1, 31), "workers": 3,
                    "kinds": ["funding", "fred"]}
    assert "DGS10" in capsys.readouterr().out
