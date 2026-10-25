"""Inference: load a trained run and produce return-quantile curves for any horizons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mark1.features.build import FeatureSet
from mark1.model.dataset import Normalizer, assemble_inputs
from mark1.model.network import MarkOne, ModelConfig
from mark1.model.train import load_meta, resolve_device, to_device


class Forecaster:
    def __init__(self, model: MarkOne, norm: Normalizer, meta: dict, device: torch.device):
        self.model, self.norm, self.meta, self.device = model.eval(), norm, meta, device
        self.views = tuple(tuple(v) for v in meta["model_config"]["views"])
        self.quantiles = list(meta["model_config"]["quantiles"])
        self.symbols = meta["symbols"]

    @classmethod
    def load(cls, run_dir: str | Path, device: str = "auto") -> "Forecaster":
        run_dir = Path(run_dir)
        meta = load_meta(run_dir)
        dev = resolve_device(device)
        model = MarkOne(ModelConfig.from_dict(meta["model_config"])).to(dev)
        ckpt = torch.load(run_dir / "model.pt", map_location=dev, weights_only=True)
        model.load_state_dict(ckpt["model"])
        return cls(model, Normalizer.from_dict(meta["normalizer"]), meta, dev)

    @torch.no_grad()
    def predict(self, fs: FeatureSet, t: np.ndarray, taus: list[int] | np.ndarray, batch_size: int = 256) -> dict:
        """Quantiles of future LOG RETURNS for decision rows t and horizons taus.

        Returns {"quantiles": [B, K, N, Q] float32, "ready": [B, N] bool, "taus": [K]}.
        Coins that are not ready get zeros; check `ready` before using them.
        """
        if fs.symbols != self.symbols:
            raise ValueError("Feature set coins differ from the ones the model was trained on")
        t = np.asarray(t, dtype=np.int64)
        taus = np.asarray(taus, dtype=np.int64)
        outs, readies = [], []
        for i in range(0, len(t), batch_size):
            tb = t[i : i + batch_size]
            inp = assemble_inputs(fs, self.norm, self.views, tb)
            tau = np.tile(taus, (len(tb), 1))
            dev = to_device({**inp, "tau": tau}, self.device)
            q, _ = self.model(dev["asset_views"], dev["market_views"], dev["ready"], dev["tau"])
            q = q.float().cpu().numpy()                                   # scaled units
            scale = inp["sigma"][:, None, :, None] * np.sqrt(taus)[None, :, None, None]
            q = q * scale
            q[~np.broadcast_to(inp["ready"][:, None, :, None], q.shape)] = 0.0
            outs.append(q.astype(np.float32))
            readies.append(inp["ready"])
        return {"quantiles": np.concatenate(outs), "ready": np.concatenate(readies), "taus": taus}
