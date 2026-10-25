"""Training samples for Mark I: time splits, normalisation, and the window sampler.

A sample is one decision time t with every coin at once (the model is cross-sectional).
For each coin it gets three views of the feature history ending at t (1-min, 15-min,
2-hour strides), the same for market features, a set of horizons tau, and the targets.

Leakage rules:
  * Splits are contiguous in time. Labels of a split never reach past its end, and the
    next split starts `gap` minutes later, so no future return is shared across splits.
  * Normalisation statistics come from the training split only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mark1 import config
from mark1.features.build import FeatureSet, load_features
from mark1.features.targets import forward_targets, invalid_cumsum

NORM_CLIP = 8.0


@dataclass
class Split:
    name: str
    start: int   # first decision row
    end: int     # decision rows and label endpoints stay < end


def lookback_rows(views: tuple[tuple[int, int], ...]) -> int:
    return max((length - 1) * stride for length, stride in views)


def make_splits(
    n_times: int,
    views: tuple[tuple[int, int], ...],
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    gap: int = config.MAX_HORIZON,
    first_row: int = 0,
) -> dict[str, Split]:
    """Train / val / test, in time order, separated by `gap` minutes."""
    start = max(first_row, lookback_rows(views))
    usable = n_times - start - 2 * gap
    if usable <= 0:
        raise ValueError("Not enough data for the requested views and gap")
    n_train = int(usable * fractions[0])
    n_val = int(usable * fractions[1])
    train = Split("train", start, start + n_train)
    val = Split("val", train.end + gap, train.end + gap + n_val)
    test = Split("test", val.end + gap, n_times)
    return {"train": train, "val": val, "test": test}


def walk_forward_splits(
    n_times: int,
    views: tuple[tuple[int, int], ...],
    n_folds: int = 4,
    min_train_frac: float = 0.4,
    gap: int = config.MAX_HORIZON,
) -> list[dict[str, Split]]:
    """Expanding-window folds: train on everything up to a point, test on the next block."""
    start = lookback_rows(views)
    span = n_times - start
    first_test = start + int(span * min_train_frac)
    block = (n_times - first_test) // n_folds
    folds = []
    for k in range(n_folds):
        test_start = first_test + k * block
        test_end = n_times if k == n_folds - 1 else test_start + block
        train = Split("train", start, test_start - gap)
        test = Split("test", test_start, test_end)
        folds.append({"train": train, "test": test})
    return folds


@dataclass
class Normalizer:
    asset_mean: np.ndarray
    asset_std: np.ndarray
    market_mean: np.ndarray
    market_std: np.ndarray

    @classmethod
    def fit(cls, fs: FeatureSet, split: Split, max_rows: int = 20_000, seed: int = 0) -> "Normalizer":
        rng = np.random.default_rng(seed)
        rows = np.arange(split.start, split.end)
        rows = np.sort(rng.choice(rows, size=min(max_rows, len(rows)), replace=False))
        a = np.asarray(fs.asset[rows])            # [R, N, Fa]
        ready = np.asarray(fs.ready[rows])
        a = a[ready]
        m = np.asarray(fs.market[rows])
        return cls(
            asset_mean=a.mean(0).astype(np.float32),
            asset_std=np.maximum(a.std(0), 1e-6).astype(np.float32),
            market_mean=m.mean(0).astype(np.float32),
            market_std=np.maximum(m.std(0), 1e-6).astype(np.float32),
        )

    def to_dict(self) -> dict:
        return {k: v.tolist() for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(**{k: np.asarray(v, dtype=np.float32) for k, v in d.items()})


def assemble_inputs(
    fs: FeatureSet,
    norm: Normalizer,
    views: tuple[tuple[int, int], ...],
    t: np.ndarray,
) -> dict[str, np.ndarray | list[np.ndarray]]:
    """Model inputs for decision rows t (shape [B]). All rows used are <= t."""
    asset_views, market_views = [], []
    for length, stride in views:
        rows = t[:, None] - stride * np.arange(length - 1, -1, -1)[None, :]   # [B, L], oldest first
        a = np.asarray(fs.asset[rows])                                       # [B, L, N, Fa]
        a = (a - norm.asset_mean) / norm.asset_std
        a = np.clip(a, -NORM_CLIP, NORM_CLIP)
        a[~np.asarray(fs.valid[rows])] = 0.0
        asset_views.append(np.ascontiguousarray(a.transpose(0, 2, 1, 3), dtype=np.float32))
        m = (np.asarray(fs.market[rows]) - norm.market_mean) / norm.market_std
        market_views.append(np.clip(m, -NORM_CLIP, NORM_CLIP).astype(np.float32))
    return {
        "asset_views": asset_views,
        "market_views": market_views,
        "ready": np.asarray(fs.ready[t]).copy(),
        "sigma": np.asarray(fs.sigma[t]).astype(np.float32),
    }


def sample_taus(rng: np.random.Generator, shape: tuple[int, ...], lo: int = config.MIN_HORIZON, hi: int = config.MAX_HORIZON) -> np.ndarray:
    """Integer horizons, log-uniform in [lo, hi], so short and long horizons get equal attention."""
    u = rng.uniform(np.log(lo), np.log(hi + 1), size=shape)
    return np.clip(np.floor(np.exp(u)), lo, hi).astype(np.int64)


class WindowSampler:
    def __init__(
        self,
        fs: FeatureSet,
        norm: Normalizer,
        views: tuple[tuple[int, int], ...],
        split: Split,
        min_ready: int = 2,
        seed: int = 0,
    ):
        self.fs, self.norm, self.views, self.split = fs, norm, views, split
        self.rng = np.random.default_rng(seed)
        self.inv_cum = invalid_cumsum(np.asarray(fs.valid))
        lo = max(split.start, lookback_rows(views))
        rows = np.arange(lo, split.end - config.MIN_HORIZON)
        n_ready = np.asarray(fs.ready[lo : split.end - config.MIN_HORIZON]).sum(axis=1)
        self.rows = rows[n_ready >= min_ready]
        if len(self.rows) == 0:
            raise ValueError(f"No usable decision rows in split {split}")

    def batch(self, t: np.ndarray, tau: np.ndarray) -> dict:
        out = assemble_inputs(self.fs, self.norm, self.views, t)
        y, mask = forward_targets(np.asarray(self.fs.log_close), np.asarray(self.fs.sigma), self.inv_cum, t, tau)
        inside = (t[:, None] + tau) < self.split.end           # labels stay inside the split
        mask &= inside[:, :, None] & out["ready"][:, None, :]
        out.update({"t": t, "tau": tau, "y": y, "mask": mask})
        return out

    def sample(self, batch_size: int, n_tau: int) -> dict:
        t = self.rng.choice(self.rows, size=batch_size)
        return self.batch(t, sample_taus(self.rng, (batch_size, n_tau)))

    def fixed_batches(self, n_batches: int, batch_size: int, taus: list[int], seed: int = 1234) -> list[dict]:
        """Deterministic evaluation batches at fixed horizons, spread across the split."""
        rng = np.random.default_rng(seed)
        t_all = np.sort(rng.choice(self.rows, size=min(n_batches * batch_size, len(self.rows)), replace=False))
        tau = np.tile(np.asarray(taus, dtype=np.int64), (batch_size, 1))
        return [self.batch(t_all[i : i + batch_size], tau[: len(t_all[i : i + batch_size])]) for i in range(0, len(t_all), batch_size)]


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def open_dataset(dataset_dir: Path) -> FeatureSet:
    return load_features(dataset_dir, mmap=True)
