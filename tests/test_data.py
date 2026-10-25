import io
import zipfile
from datetime import date

import numpy as np
import pandas as pd

from mark1.data.binance_archive import parse_kline_zip, plan_files
from mark1.data.store import _validity, load_panel


def _write_zip(path, rows):
    buf = io.StringIO()
    for r in rows:
        buf.write(",".join(str(x) for x in r) + "\n")
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(path.stem + ".csv", buf.getvalue())


def _row(ts, price, vol=1.0):
    # open_time, o, h, l, c, volume, close_time, quote_vol, trades, taker_buy, taker_buy_quote, ignore
    return [ts, price, price + 1, price - 1, price, vol, ts + 59_999, vol * price, 10, vol / 2, vol * price / 2, 0]


def test_plan_files_monthly_then_daily():
    files = plan_files("BTCUSDT", date(2026, 8, 15), date(2026, 10, 3), today=date(2026, 10, 5))
    stamps = [(f.period, f.stamp) for f in files]
    assert stamps[:2] == [("monthly", "2026-08"), ("monthly", "2026-09")]
    assert stamps[2:] == [("daily", "2026-10-01"), ("daily", "2026-10-02"), ("daily", "2026-10-03")]


def test_parse_handles_ms_and_us(tmp_path):
    ms = int(pd.Timestamp("2021-01-01").value // 1_000_000)
    us = int(pd.Timestamp("2026-01-01").value // 1_000)
    _write_zip(tmp_path / "A-1m-2021-01.zip", [_row(ms, 100.0), _row(ms + 60_000, 101.0)])
    _write_zip(tmp_path / "A-1m-2026-01.zip", [_row(us, 200.0), _row(us + 60_000_000, 201.0)])
    a = parse_kline_zip(tmp_path / "A-1m-2021-01.zip")
    b = parse_kline_zip(tmp_path / "A-1m-2026-01.zip")
    assert a.index[0] == pd.Timestamp("2021-01-01") and a.index[1] == pd.Timestamp("2021-01-01 00:01")
    assert b.index[0] == pd.Timestamp("2026-01-01") and b.index[1] == pd.Timestamp("2026-01-01 00:01")
    assert list(a.columns) == ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_volume"]


def test_validity_short_and_long_gaps():
    present = np.array([0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1], dtype=bool)
    v = _validity(present, max_gap=2)
    # before listing: invalid; 1-bar gap: valid; 4-bar gap (> 2): invalid for its whole length
    assert v.tolist() == [False, False, True, True, True, True, False, False, False, False, True]


def test_load_panel_aligns_and_fills(tmp_path):
    bars_dir = tmp_path / "bars"
    bars_dir.mkdir()
    idx = pd.date_range("2026-01-01", periods=5, freq="min")
    base = dict(volume=1.0, quote_volume=1.0, trades=1.0, taker_buy_volume=0.5)
    a = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": [1, 2, 3, 4, 5.0], **base}, index=idx)
    b = pd.DataFrame({"open": 9.0, "high": 9.0, "low": 9.0, "close": [9.0, 9.5], **base}, index=idx[[1, 3]])
    a.to_parquet(bars_dir / "AUSDT.parquet")
    b.to_parquet(bars_dir / "BUSDT.parquet")
    p = load_panel(["AUSDT", "BUSDT"], bars_dir=bars_dir, max_gap_minutes=5)
    assert p.close.shape == (5, 2)
    assert p.close[:, 1].tolist() == [9.0, 9.0, 9.0, 9.5, 9.5]
    assert p.volume[:, 1].tolist() == [0, 1, 0, 1, 0]
    assert p.valid[:, 1].tolist() == [False, True, True, True, True]
