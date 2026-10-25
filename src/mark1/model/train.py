"""Train and evaluate Mark I.

The yardstick is a baseline that knows current volatility but nothing else: for each horizon
bucket it predicts the training-period quantiles of the volatility-scaled return. Skill
= 1 - model_loss / baseline_loss, so 0 means no better than knowing the volatility, and
anything positive is information the model found in the data.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from mark1 import config
from mark1.model.dataset import (
    Normalizer, Split, WindowSampler, make_splits, open_dataset, save_json,
)
from mark1.model.network import MarkOne, ModelConfig, pinball_loss

log = logging.getLogger(__name__)

BASELINE_EDGES = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1441])


@dataclass
class TrainConfig:
    dataset_dir: str = str(config.DATASET_DIR)
    run_dir: str = str(config.RUNS_DIR / "mark1")
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    n_experts: int = 4
    dropout: float = 0.1
    use_asset_embedding: bool = True
    batch_size: int = 64
    n_tau: int = 8
    steps: int = 20_000
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    grad_clip: float = 1.0
    eval_every: int = 500
    eval_batches: int = 40
    num_workers: int = 4
    seed: int = 0
    device: str = "auto"
    amp: bool = True
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)
    gap: int = config.MAX_HORIZON
    views: tuple[tuple[int, int], ...] = ((120, 1), (96, 15), (84, 120))
    report_horizons: list[int] = field(default_factory=lambda: list(config.REPORT_HORIZONS))


class Baseline:
    """Volatility-only forecast: empirical quantiles of scaled returns per horizon bucket."""

    def __init__(self, table: np.ndarray, quantiles: list[float]):
        self.table = table              # [n_buckets, Q]
        self.quantiles = quantiles

    @classmethod
    def fit(cls, sampler: WindowSampler, quantiles: list[float], n_batches: int = 60, batch_size: int = 128) -> "Baseline":
        ys = [[] for _ in range(len(BASELINE_EDGES) - 1)]
        for _ in range(n_batches):
            b = sampler.sample(batch_size, 8)
            bucket = np.digitize(b["tau"], BASELINE_EDGES) - 1
            for k in range(len(ys)):
                sel = (bucket == k)[:, :, None] & b["mask"]
                ys[k].append(b["y"][sel])
        table = np.zeros((len(ys), len(quantiles)), dtype=np.float32)
        for k, parts in enumerate(ys):
            v = np.concatenate(parts) if parts else np.zeros(0)
            table[k] = np.quantile(v, quantiles) if len(v) > 100 else np.quantile(np.random.default_rng(0).normal(size=10_000), quantiles)
        return cls(table, quantiles)

    def predict(self, tau: np.ndarray, n_assets: int) -> np.ndarray:
        """tau [B, K] -> quantiles [B, K, N, Q]."""
        bucket = np.clip(np.digitize(tau, BASELINE_EDGES) - 1, 0, len(self.table) - 1)
        q = self.table[bucket]                                     # [B, K, Q]
        return np.repeat(q[:, :, None, :], n_assets, axis=2)

    def to_dict(self) -> dict:
        return {"edges": BASELINE_EDGES.tolist(), "table": self.table.tolist(), "quantiles": self.quantiles}


class BatchStream(IterableDataset):
    """Endless random training batches; each DataLoader worker opens its own memory map."""

    def __init__(self, dataset_dir: str, norm: Normalizer, views, split: Split, batch_size: int, n_tau: int, seed: int):
        self.dataset_dir, self.norm, self.views, self.split = dataset_dir, norm, views, split
        self.batch_size, self.n_tau, self.seed = batch_size, n_tau, seed

    def __iter__(self):
        info = get_worker_info()
        seed = self.seed + (info.id + 1 if info else 0) * 1000
        sampler = WindowSampler(open_dataset(Path(self.dataset_dir)), self.norm, self.views, self.split, seed=seed)
        while True:
            yield sampler.sample(self.batch_size, self.n_tau)


def to_device(batch: dict, device: torch.device) -> dict:
    def conv(x):
        if isinstance(x, list):
            return [conv(v) for v in x]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x
    return {k: conv(v) for k, v in batch.items()}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def _xs_spearman(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    """Mean cross-sectional rank correlation over rows with at least 3 valid coins."""
    vals = []
    for p_row, y_row, m_row in zip(pred, y, mask):
        if m_row.sum() < 3:
            continue
        pr = np.argsort(np.argsort(p_row[m_row]))
        yr = np.argsort(np.argsort(y_row[m_row]))
        if pr.std() == 0 or yr.std() == 0:
            continue
        vals.append(np.corrcoef(pr, yr)[0, 1])
    return float(np.mean(vals)) if vals else float("nan")


@torch.no_grad()
def evaluate(model: MarkOne, batches: list[dict], baseline: Baseline, device: torch.device, quantiles: list[float]) -> dict:
    """Per-horizon pinball loss vs baseline, calibration, rank IC and direction hit rate."""
    model.eval()
    qs = np.asarray(quantiles, dtype=np.float32)
    mid = quantiles.index(0.5)
    taus = batches[0]["tau"][0]
    acc = {int(h): {"model": [], "base": [], "pred_mid": [], "y": [], "mask": [], "cover": []} for h in taus}
    for b in batches:
        tb = to_device(b, device)
        pred, _ = model(tb["asset_views"], tb["market_views"], tb["ready"], tb["tau"])
        pred = pred.float().cpu().numpy()                             # [B, K, N, Q]
        base = baseline.predict(b["tau"], pred.shape[2])
        y, mask = b["y"], b["mask"]
        for k, h in enumerate(taus):
            m = mask[:, k]
            if not m.any():
                continue
            yk = y[:, k][m][:, None]
            for name, p in (("model", pred[:, k][m]), ("base", base[:, k][m])):
                d = yk - p
                acc[int(h)][name].append(np.maximum(qs * d, (qs - 1) * d).mean(axis=1))
            acc[int(h)]["cover"].append((yk < pred[:, k][m]).astype(np.float32))
            acc[int(h)]["pred_mid"].append(pred[:, k, :, mid])
            acc[int(h)]["y"].append(y[:, k])
            acc[int(h)]["mask"].append(m)
    model.train()

    report = {}
    for h, a in acc.items():
        if not a["model"]:
            continue
        lm, lb = float(np.concatenate(a["model"]).mean()), float(np.concatenate(a["base"]).mean())
        pm, yy, mm = np.concatenate(a["pred_mid"]), np.concatenate(a["y"]), np.concatenate(a["mask"])
        nz = mm & (np.abs(pm) > 1e-6) & (yy != 0)
        report[h] = {
            "loss_model": lm,
            "loss_base": lb,
            "skill": 1 - lm / lb,
            "coverage": np.concatenate(a["cover"]).mean(axis=0).round(4).tolist(),
            "rank_ic": _xs_spearman(pm, yy, mm),
            "hit_rate": float((np.sign(pm[nz]) == np.sign(yy[nz])).mean()) if nz.any() else float("nan"),
            "n": int(mm.sum()),
        }
    tot_m = np.mean([r["loss_model"] for r in report.values()])
    tot_b = np.mean([r["loss_base"] for r in report.values()])
    return {"horizons": report, "loss_model": float(tot_m), "loss_base": float(tot_b), "skill": float(1 - tot_m / tot_b)}


def format_report(rep: dict, title: str) -> str:
    lines = [f"{title}: loss {rep['loss_model']:.5f} vs baseline {rep['loss_base']:.5f}  skill {100 * rep['skill']:+.2f}%",
             f"  {'horizon':>8} {'skill%':>8} {'rank_ic':>8} {'hit%':>7} {'cov05':>6} {'cov50':>6} {'cov95':>6}"]
    for h, r in sorted(rep["horizons"].items()):
        cov = r["coverage"]
        lines.append(f"  {h:>6}m {100 * r['skill']:>+8.2f} {r['rank_ic']:>+8.4f} {100 * r['hit_rate']:>7.2f} "
                     f"{cov[0]:>6.3f} {cov[len(cov) // 2]:>6.3f} {cov[-1]:>6.3f}")
    return "\n".join(lines)


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.device)
    use_amp = cfg.amp and device.type == "cuda"
    log.info("device %s, amp %s", device, use_amp)

    fs = open_dataset(Path(cfg.dataset_dir))
    splits = make_splits(len(fs.times), cfg.views, cfg.fractions, cfg.gap)
    for s in splits.values():
        log.info("%-5s rows %8d..%8d  (%s -> %s)", s.name, s.start, s.end,
                 np.datetime_as_string(fs.times[s.start], "m"), np.datetime_as_string(fs.times[s.end - 1], "m"))
    norm = Normalizer.fit(fs, splits["train"], seed=cfg.seed)
    quantiles = list(config.QUANTILES)

    train_sampler = WindowSampler(fs, norm, cfg.views, splits["train"], seed=cfg.seed)
    baseline = Baseline.fit(train_sampler, quantiles)
    val_batches = WindowSampler(fs, norm, cfg.views, splits["val"]).fixed_batches(cfg.eval_batches, cfg.batch_size, cfg.report_horizons)

    mcfg = ModelConfig(
        n_asset_features=len(fs.asset_names), n_market_features=len(fs.market_names), n_assets=len(fs.symbols),
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers, n_experts=cfg.n_experts,
        dropout=cfg.dropout, use_asset_embedding=cfg.use_asset_embedding, views=cfg.views,
    )
    model = MarkOne(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Mark I: %.2fM parameters", n_params / 1e6)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    q_t = torch.tensor(quantiles, device=device)

    meta = {
        "train_config": asdict(cfg), "model_config": mcfg.to_dict(), "normalizer": norm.to_dict(),
        "baseline": baseline.to_dict(), "splits": {k: asdict(v) for k, v in splits.items()},
        "symbols": fs.symbols, "asset_names": fs.asset_names, "market_names": fs.market_names,
    }
    save_json(meta, run_dir / "meta.json")

    stream = BatchStream(cfg.dataset_dir, norm, cfg.views, splits["train"], cfg.batch_size, cfg.n_tau, cfg.seed)
    loader = DataLoader(stream, batch_size=None, num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                        persistent_workers=cfg.num_workers > 0, prefetch_factor=4 if cfg.num_workers > 0 else None)

    best, history, t0, running = float("inf"), [], time.time(), []
    for step, batch in enumerate(loader):
        if step >= cfg.steps:
            break
        for g in opt.param_groups:
            g["lr"] = _lr_at(step, cfg)
        b = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            pred, _ = model(b["asset_views"], b["market_views"], b["ready"], b["tau"])
        loss = pinball_loss(pred.float(), b["y"], b["mask"], q_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        running.append(loss.item())

        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            rep = evaluate(model, val_batches, baseline, device, quantiles)
            rate = (step + 1) / (time.time() - t0)
            log.info("step %d  train %.5f  %s  (%.1f steps/s)", step + 1, np.mean(running), format_report(rep, "val"), rate)
            history.append({"step": step + 1, "train_loss": float(np.mean(running)), "val": rep})
            running = []
            if rep["loss_model"] < best:
                best = rep["loss_model"]
                torch.save({"model": model.state_dict(), "step": step + 1}, run_dir / "model.pt")
            save_json({"history": history}, run_dir / "history.json")

    ckpt = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    test_batches = WindowSampler(fs, norm, cfg.views, splits["test"]).fixed_batches(cfg.eval_batches, cfg.batch_size, cfg.report_horizons)
    test_rep = evaluate(model, test_batches, baseline, device, quantiles)
    log.info("best checkpoint (step %d)\n%s", ckpt["step"], format_report(test_rep, "TEST"))
    result = {"best_step": ckpt["step"], "best_val_loss": best, "test": test_rep, "n_params": n_params}
    save_json(result, run_dir / "results.json")
    return result


def load_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "meta.json").read_text())
