import numpy as np
import pandas as pd
import pytest

from mark1.data.panel import synthetic_panel
from mark1.features.build import WARMUP, build_features
from mark1.features.extras import (FRED_LAG_DAYS, MARKET_EXTRA_NAMES, POSITIONING_NAMES, asof,
                                   market_extra_features, positioning_features)

START = "2024-01-01T00:00:00"


def _write_extras(root, times, symbols, seed=0, shock_after=None):
    """Fake extras following the downloader's schema. `shock_after` scrambles everything after that time."""
    rng = np.random.default_rng(seed)
    shock_rng = np.random.default_rng(seed + 999)     # separate stream so scrambling never changes the past
    t = pd.DatetimeIndex(times)

    def scramble(df, cols):
        if shock_after is not None:
            m = df.index > shock_after
            for c in cols:
                df.loc[m, c] = df.loc[m, c] * shock_rng.uniform(0.2, 5.0, size=m.sum())
        return df

    for sym in symbols:
        for sub in ("funding", "metrics", "premium"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        f_idx = pd.date_range(t[0].floor("8h"), t[-1], freq="8h", name="time")
        fund = pd.DataFrame({"funding_rate": rng.normal(1e-4, 5e-5, len(f_idx)), "interval_hours": 8.0}, index=f_idx)
        scramble(fund, ["funding_rate"]).to_parquet(root / "funding" / f"{sym}.parquet")
        m_idx = pd.date_range(t[0], t[-1], freq="5min", name="time")
        met = pd.DataFrame({"open_interest": np.exp(np.cumsum(rng.normal(0, 0.001, len(m_idx)))) * 1e4,
                            "open_interest_value": 1e8, "toptrader_ls_count": 1.5,
                            "toptrader_ls_sum": rng.uniform(0.8, 2.5, len(m_idx)), "ls_count": 1.4,
                            "taker_ls_vol": rng.uniform(0.5, 2.0, len(m_idx))}, index=m_idx)
        scramble(met, ["open_interest", "toptrader_ls_sum", "taker_ls_vol"]).to_parquet(root / "metrics" / f"{sym}.parquet")
        p_idx = pd.DatetimeIndex(t, name="open_time")
        pr = rng.normal(2e-4, 1e-4, len(p_idx))
        prem = pd.DataFrame({"open": pr, "high": pr, "low": pr, "close": pr}, index=p_idx)
        scramble(prem, ["close"]).to_parquet(root / "premium" / f"{sym}.parquet")
    (root / "bvol").mkdir(exist_ok=True)
    for name in ("BTCBVOLUSDT", "ETHBVOLUSDT"):
        b_idx = pd.date_range(t[0], t[-1], freq="1min", name="time")
        scramble(pd.DataFrame({"value": rng.uniform(40, 70, len(b_idx))}, index=b_idx), ["value"]).to_parquet(root / "bvol" / f"{name}.parquet")
    (root / "spot").mkdir(exist_ok=True)
    for sym in ("PAXGUSDT", "EURUSDT"):
        c = np.exp(np.cumsum(rng.normal(0, 2e-4, len(t))))
        df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0, "quote_volume": 1.0,
                           "trades": 1.0, "taker_buy_volume": 0.5}, index=pd.DatetimeIndex(t, name="open_time"))
        scramble(df, ["close"]).to_parquet(root / "spot" / f"{sym}.parquet")
    (root / "fred").mkdir(exist_ok=True)
    d_idx = pd.date_range(t[0].normalize() - pd.Timedelta(days=20), t[-1].normalize(), freq="D", name="date")
    for series, base in (("DGS10", 4.0), ("VIXCLS", 18.0), ("DTWEXBGS", 120.0)):
        scramble(pd.DataFrame({"value": base + rng.normal(0, 0.5, len(d_idx))}, index=d_idx), ["value"]).to_parquet(root / "fred" / f"{series}.parquet")


def test_asof_uses_only_published_values():
    stamps = np.array(["2024-01-01T00:05", "2024-01-01T00:10"], dtype="datetime64[ns]")
    grid = np.array(["2024-01-01T00:04", "2024-01-01T00:05", "2024-01-01T00:09", "2024-01-01T00:30"], dtype="datetime64[ns]")
    v, ok = asof(grid, stamps, np.array([1.0, 2.0]), max_age=np.timedelta64(15, "m"))
    assert ok.tolist() == [False, True, True, False]          # before the first stamp, and stale after 15 min
    assert v[1] == 1.0 and v[2] == 1.0


def test_fred_values_wait_for_the_publication_lag(tmp_path):
    times = np.datetime64(START, "ns") + np.arange(6 * 1440).astype("timedelta64[m]")
    (tmp_path / "fred").mkdir()
    d = pd.date_range("2024-01-01", periods=6, freq="D", name="date")
    pd.DataFrame({"value": [10.0, 20, 30, 40, 50, 60]}, index=d).to_parquet(tmp_path / "fred" / "VIXCLS.parquet")
    m = market_extra_features(times, tmp_path)
    vix = m[:, MARKET_EXTRA_NAMES.index("vix_level")]
    day = 1440
    assert np.all(vix[: FRED_LAG_DAYS * day] == 0)                                   # nothing known before the lag
    assert np.isclose(vix[FRED_LAG_DAYS * day], np.log(10 / 20))                     # Jan 1 value usable from Jan 3 00:00
    assert np.isclose(vix[(FRED_LAG_DAYS + 1) * day], np.log(20 / 20))


def test_missing_sources_give_zeros_and_flags(tmp_path):
    times = np.datetime64(START, "ns") + np.arange(500).astype("timedelta64[m]")
    pos = positioning_features(times, ["AUSDT"], tmp_path)
    mkt = market_extra_features(times, tmp_path)
    assert pos.shape == (500, 1, len(POSITIONING_NAMES)) and not pos.any()
    for flag in ("has_bvol", "has_fx", "has_macro"):
        assert not mkt[:, MARKET_EXTRA_NAMES.index(flag)].any()


def test_extra_features_have_no_lookahead(tmp_path):
    p = synthetic_panel(n_times=WARMUP + 600, symbols=["AUSDT", "BUSDT"], seed=5, start=START)
    t0 = WARMUP + 300
    cut = pd.Timestamp(p.times[t0])
    _write_extras(tmp_path / "a", p.times, p.symbols, seed=1)
    _write_extras(tmp_path / "b", p.times, p.symbols, seed=1, shock_after=cut)
    a = build_features(p, extras_dir=tmp_path / "a")
    b = build_features(p, extras_dir=tmp_path / "b")
    np.testing.assert_array_equal(a.asset[: t0 + 1], b.asset[: t0 + 1])
    np.testing.assert_array_equal(a.market[: t0 + 1], b.market[: t0 + 1])
    assert not np.array_equal(a.asset[t0 + 1:], b.asset[t0 + 1:])      # the future really changed


def test_build_with_extras_adds_named_columns(tmp_path):
    p = synthetic_panel(n_times=WARMUP + 100, symbols=["AUSDT", "BUSDT"], seed=6, start=START)
    _write_extras(tmp_path, p.times, p.symbols, seed=2)
    base = build_features(p)
    fs = build_features(p, extras_dir=tmp_path)
    assert fs.asset.shape[2] == base.asset.shape[2] + len(POSITIONING_NAMES)
    assert fs.market.shape[1] == base.market.shape[1] + len(MARKET_EXTRA_NAMES)
    assert fs.asset_names[-len(POSITIONING_NAMES):] == POSITIONING_NAMES
    has = fs.asset[WARMUP:, :, fs.asset_names.index("has_futures")]
    assert has.all()
    assert np.isfinite(fs.asset).all() and np.isfinite(fs.market).all()
    funding = fs.asset[WARMUP:, 0, fs.asset_names.index("funding_bps")]
    assert 0 < np.abs(funding).mean() < 10                               # roughly 1 bp, in bps units
