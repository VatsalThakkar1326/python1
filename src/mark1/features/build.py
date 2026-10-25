"""Causal features for Mark I v1, built from the aligned minute Panel.

Every value at row t uses bars <= t only. Features are scale-free where possible (returns
divided by recent volatility, volumes relative to their own history) so one model can learn
across BTC and small alts. Final standardisation uses training-period statistics and
happens in the training dataset, not here, to avoid leaking test-period information.

Dimensions covered (see the design notes in README):
  1 price action, 2 volume, 3 volatility, 4 order flow (taker buy/sell imbalance),
  5 other coins (cross-sectional ranks; the model adds attention across coins),
  6 market mood (equal-weight market returns, breadth, dispersion, correlation),
  8 clock patterns (time of day, day of week, 8-hour funding cycle).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mark1.data.panel import Panel

SIGMA_HALFLIFE = 360           # minutes, the volatility used to normalise returns and targets
RET_WINDOWS = [1, 5, 15, 60, 240, 1440]
WARMUP = 2 * 1440              # rows after an asset's first valid bar before features are trusted
SIGMA_FLOOR = 1e-5
CLIP = 10.0


@dataclass
class FeatureSet:
    asset: np.ndarray          # float32 [T, N, Fa]
    market: np.ndarray         # float32 [T, Fm]
    log_close: np.ndarray      # float64 [T, N]
    sigma: np.ndarray          # float32 [T, N], per-minute volatility at row t (causal)
    valid: np.ndarray          # bool [T, N], tradable bar
    ready: np.ndarray          # bool [T, N], valid and past warmup, so features are trustworthy
    times: np.ndarray          # datetime64[ns] [T]
    symbols: list[str]
    asset_names: list[str]
    market_names: list[str]


def _ewm(x: np.ndarray, halflife: float) -> np.ndarray:
    return pd.Series(x).ewm(halflife=halflife, adjust=False).mean().to_numpy()


def _lagdiff(x: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(x)
    out[k:] = x[k:] - x[:-k]
    return out


def _roll(x: np.ndarray, w: int, how: str) -> np.ndarray:
    r = pd.Series(x).rolling(w, min_periods=1)
    return getattr(r, how)().to_numpy()


def asset_feature_names() -> list[str]:
    names = [f"ret_{k}" for k in RET_WINDOWS]
    names += ["vol_15_vs_360", "vol_60_vs_360", "vol_360_vs_1440", "log_vol_360"]
    names += ["range_5"]
    names += ["qvol_vs_60", "qvol_vs_1440"]
    names += ["imb_1", "imb_5", "imb_30"]
    names += ["trades_vs_1440"]
    names += ["dist_high_60", "dist_low_60", "dist_high_1440", "dist_low_1440"]
    names += ["vwap_dist_60", "vwap_dist_1440"]
    names += ["staleness"]
    names += ["xs_rank_ret_60", "xs_rank_ret_1440"]
    return names


def market_feature_names() -> list[str]:
    names = [f"mkt_ret_{k}" for k in RET_WINDOWS]
    names += ["mkt_log_vol", "mkt_vol_15_vs_360"]
    names += ["breadth_60", "breadth_1440", "dispersion_60", "corr_proxy"]
    names += ["mkt_qvol_vs_1440", "mkt_imb_5", "mkt_imb_30"]
    names += ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "fund_sin", "fund_cos"]
    return names


def _per_asset(p: Panel, j: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Features for one asset. Returns (features [T, Fa-2], log_close, sigma, r1)."""
    close, high, low = p.close[:, j], p.high[:, j], p.low[:, j]
    vol, qvol, trades, tbv = p.volume[:, j], p.quote_volume[:, j], p.trades[:, j], p.taker_buy_volume[:, j]
    lc = np.log(close)
    r1 = _lagdiff(lc, 1)
    r1[~p.valid[:, j]] = 0.0
    sigma = np.maximum(np.sqrt(_ewm(r1 ** 2, SIGMA_HALFLIFE)), SIGMA_FLOOR)

    cols = []
    for k in RET_WINDOWS:
        cols.append(_lagdiff(lc, k) / (sigma * np.sqrt(k)))

    vol15 = np.sqrt(_ewm(r1 ** 2, 15)) + SIGMA_FLOOR
    vol60 = np.sqrt(_ewm(r1 ** 2, 60)) + SIGMA_FLOOR
    vol1440 = np.sqrt(_ewm(r1 ** 2, 1440)) + SIGMA_FLOOR
    cols += [np.log(vol15 / sigma), np.log(vol60 / sigma), np.log(sigma / vol1440), np.log(sigma)]

    hl_range = np.log(np.maximum(high, close) / np.maximum(np.minimum(low, close), 1e-12))
    cols.append(_ewm(hl_range / sigma, 5))

    lq = np.log1p(qvol)
    cols += [lq - _ewm(lq, 60), lq - _ewm(lq, 1440)]

    imb = np.where(vol > 0, (2 * tbv - vol) / np.maximum(vol, 1e-12), 0.0)
    cols += [imb, _ewm(imb, 5), _ewm(imb, 30)]

    lt = np.log1p(trades)
    cols.append(lt - _ewm(lt, 1440))

    for w in (60, 1440):
        scale = sigma * np.sqrt(w)
        cols.append((lc - np.log(_roll(high, w, "max"))) / scale)
        cols.append((lc - np.log(_roll(low, w, "min"))) / scale)

    for w in (60, 1440):
        qsum, vsum = _roll(qvol, w, "sum"), _roll(vol, w, "sum")
        vwap = np.where(vsum > 0, qsum / np.maximum(vsum, 1e-12), close)
        cols.append((lc - np.log(vwap)) / (sigma * np.sqrt(w)))

    # minutes since the last bar with any volume, log-scaled
    idx = np.arange(len(vol))
    last = np.maximum.accumulate(np.where(vol > 0, idx, 0))
    cols.append(np.log1p(idx - last))

    feats = np.stack(cols, axis=1)
    return feats, lc, sigma, r1


def _xs_rank(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Cross-sectional rank at each row among valid assets, scaled to [-1, 1]."""
    filled = np.where(valid, x, np.inf)
    ranks = np.argsort(np.argsort(filled, axis=1), axis=1).astype(np.float64)
    n = valid.sum(axis=1, keepdims=True)
    scaled = np.where(n > 1, 2 * ranks / np.maximum(n - 1, 1) - 1, 0.0)
    return np.where(valid, scaled, 0.0)


def build_features(p: Panel) -> FeatureSet:
    T, N = p.n_times, p.n_assets
    a_names, m_names = asset_feature_names(), market_feature_names()
    asset = np.zeros((T, N, len(a_names)), dtype=np.float32)
    log_close = np.zeros((T, N), dtype=np.float64)
    sigma = np.zeros((T, N), dtype=np.float32)
    r1_all = np.zeros((T, N), dtype=np.float64)
    ret60 = np.zeros((T, N))
    ret1440 = np.zeros((T, N))

    n_local = len(a_names) - 2
    for j in range(N):
        feats, lc, sig, r1 = _per_asset(p, j)
        asset[:, j, :n_local] = np.clip(np.nan_to_num(feats), -CLIP, CLIP)
        log_close[:, j], sigma[:, j], r1_all[:, j] = lc, sig, r1
        ret60[:, j] = feats[:, RET_WINDOWS.index(60)]
        ret1440[:, j] = feats[:, RET_WINDOWS.index(1440)]

    valid = p.valid
    asset[:, :, n_local] = _xs_rank(ret60, valid)
    asset[:, :, n_local + 1] = _xs_rank(ret1440, valid)
    asset[~valid] = 0.0

    market = _market_features(p, r1_all, ret60, ret1440, valid)

    first_valid = np.where(valid.any(axis=0), valid.argmax(axis=0), T)
    age = np.arange(T)[:, None] - first_valid[None, :]
    ready = valid & (age >= WARMUP)

    return FeatureSet(
        asset=asset, market=market, log_close=log_close, sigma=sigma, valid=valid, ready=ready,
        times=p.times, symbols=list(p.symbols), asset_names=a_names, market_names=m_names,
    )


def _market_features(p: Panel, r1: np.ndarray, ret60: np.ndarray, ret1440: np.ndarray, valid: np.ndarray) -> np.ndarray:
    n_valid = np.maximum(valid.sum(axis=1), 1)
    mkt_r1 = np.where(valid, r1, 0.0).sum(axis=1) / n_valid
    mkt_sigma = np.maximum(np.sqrt(_ewm(mkt_r1 ** 2, SIGMA_HALFLIFE)), SIGMA_FLOOR)
    mkt_lc = np.cumsum(mkt_r1)

    cols = [_lagdiff(mkt_lc, k) / (mkt_sigma * np.sqrt(k)) for k in RET_WINDOWS]
    vol15 = np.sqrt(_ewm(mkt_r1 ** 2, 15)) + SIGMA_FLOOR
    cols += [np.log(mkt_sigma), np.log(vol15 / mkt_sigma)]

    cols.append(2 * (np.where(valid, ret60 > 0, False).sum(axis=1) / n_valid) - 1)
    cols.append(2 * (np.where(valid, ret1440 > 0, False).sum(axis=1) / n_valid) - 1)
    masked60 = np.where(valid, ret60, np.nan)
    with np.errstate(invalid="ignore"):
        disp = np.nanstd(masked60, axis=1)
    cols.append(np.nan_to_num(disp))
    # Average pairwise correlation proxy: variance of the mean return over mean variance.
    var_mean = _ewm(mkt_r1 ** 2, 240)
    mean_var = _ewm(np.where(valid, r1 ** 2, 0.0).sum(axis=1) / n_valid, 240)
    cols.append(np.clip(var_mean / np.maximum(mean_var, 1e-18), 0, 1))

    lq = np.log1p(np.where(valid, p.quote_volume, 0.0).sum(axis=1))
    cols.append(lq - _ewm(lq, 1440))
    vol_tot = np.where(valid, p.volume * p.close, 0.0).sum(axis=1)
    tb_tot = np.where(valid, p.taker_buy_volume * p.close, 0.0).sum(axis=1)
    imb = np.where(vol_tot > 0, (2 * tb_tot - vol_tot) / np.maximum(vol_tot, 1e-12), 0.0)
    cols += [_ewm(imb, 5), _ewm(imb, 30)]

    t = pd.DatetimeIndex(p.times)
    minute_of_day = (t.hour * 60 + t.minute).to_numpy()
    dow = t.day_of_week.to_numpy() + minute_of_day / 1440
    funding = (minute_of_day % 480) / 480  # 00:00, 08:00, 16:00 UTC funding cycle
    for phase in (minute_of_day / 1440, dow / 7, funding):
        cols += [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]

    market = np.stack(cols, axis=1)
    return np.clip(np.nan_to_num(market), -CLIP, CLIP).astype(np.float32)


def save_features(fs: FeatureSet, out_dir: Path) -> None:
    """Write a FeatureSet as .npy files (memory-mappable) plus a JSON manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "asset.npy", fs.asset)
    np.save(out_dir / "market.npy", fs.market)
    np.save(out_dir / "log_close.npy", fs.log_close)
    np.save(out_dir / "sigma.npy", fs.sigma)
    np.save(out_dir / "valid.npy", fs.valid)
    np.save(out_dir / "ready.npy", fs.ready)
    np.save(out_dir / "times.npy", fs.times.astype("datetime64[ns]").astype(np.int64))
    manifest = {
        "symbols": fs.symbols,
        "asset_names": fs.asset_names,
        "market_names": fs.market_names,
        "n_times": int(len(fs.times)),
        "start": str(pd.Timestamp(fs.times[0])),
        "end": str(pd.Timestamp(fs.times[-1])),
        "sigma_halflife": SIGMA_HALFLIFE,
        "warmup": WARMUP,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_features(in_dir: Path, mmap: bool = True) -> FeatureSet:
    mode = "r" if mmap else None
    m = json.loads((in_dir / "manifest.json").read_text())
    return FeatureSet(
        asset=np.load(in_dir / "asset.npy", mmap_mode=mode),
        market=np.load(in_dir / "market.npy", mmap_mode=mode),
        log_close=np.load(in_dir / "log_close.npy", mmap_mode=mode),
        sigma=np.load(in_dir / "sigma.npy", mmap_mode=mode),
        valid=np.load(in_dir / "valid.npy", mmap_mode=mode),
        ready=np.load(in_dir / "ready.npy", mmap_mode=mode),
        times=np.load(in_dir / "times.npy").astype("datetime64[ns]"),
        symbols=m["symbols"],
        asset_names=m["asset_names"],
        market_names=m["market_names"],
    )
