"""Normalised features held as one torch tensor (on the GPU when it fits), with fast window cutting.

The numpy path (`dataset.assemble_inputs`) copies every training window on the CPU. The fused
agent needs inputs at every environment step and every update minibatch, so this store keeps
the whole normalised feature history on the device in float16 and cuts windows with a single
gather. Five years of 20 coins is about 2.7 GB, which fits next to the model on a 12 GB card.
The values are identical to `assemble_inputs` up to float16 rounding.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from mark1.features.build import FeatureSet
from mark1.model.dataset import NORM_CLIP, Normalizer

log = logging.getLogger(__name__)


def pick_store_device(fs: FeatureSet, device: torch.device, requested: str = "auto", max_share: float = 0.45) -> torch.device:
    """Put features on the GPU when they take at most `max_share` of its free memory."""
    if requested != "auto":
        return torch.device(requested)
    if device.type != "cuda":
        return torch.device("cpu")
    need = (fs.asset.size + fs.market.size) * 2
    free, _ = torch.cuda.mem_get_info(device)
    if need <= max_share * free:
        return device
    log.warning("features need %.1f GB but only %.1f GB of GPU memory is free; keeping them in RAM", need / 1e9, free / 1e9)
    return torch.device("cpu")


class TorchFeatureStore:
    def __init__(self, fs: FeatureSet, norm: Normalizer, views, device: torch.device, chunk: int = 200_000):
        self.views = tuple(tuple(v) for v in views)
        self.device = device
        T, N, Fa = fs.asset.shape
        Fm = fs.market.shape[1]
        self.asset = torch.empty((T, N, Fa), dtype=torch.float16, device=device)
        self.market = torch.empty((T, Fm), dtype=torch.float16, device=device)
        for s in range(0, T, chunk):
            e = min(T, s + chunk)
            a = (np.asarray(fs.asset[s:e], dtype=np.float32) - norm.asset_mean) / norm.asset_std
            a = np.clip(a, -NORM_CLIP, NORM_CLIP)
            a[~np.asarray(fs.valid[s:e])] = 0.0
            self.asset[s:e] = torch.from_numpy(a.astype(np.float16)).to(device)
            m = (np.asarray(fs.market[s:e], dtype=np.float32) - norm.market_mean) / norm.market_std
            self.market[s:e] = torch.from_numpy(np.clip(m, -NORM_CLIP, NORM_CLIP).astype(np.float16)).to(device)
        self.ready = torch.from_numpy(np.asarray(fs.ready)).to(device)
        self.sigma = torch.from_numpy(np.asarray(fs.sigma, dtype=np.float32)).to(device)
        self.offsets = [(torch.arange(length - 1, -1, -1) * stride).to(device) for length, stride in self.views]
        log.info("feature store on %s: %.2f GB", device, (self.asset.numel() + self.market.numel()) * 2 / 1e9)

    def inputs(self, t: torch.Tensor, out_device: torch.device | None = None):
        """Model inputs for decision rows t [B] (all rows used are <= t).

        Returns (asset_views [B, N, L, Fa] list, market_views [B, L, Fm] list, ready [B, N], sigma [B, N]),
        float32 on `out_device` (default: the store's device)."""
        t = t.to(self.device, dtype=torch.long)
        dev = out_device or self.device
        av, mv = [], []
        for off in self.offsets:
            rows = t[:, None] - off[None, :]                                     # [B, L]
            av.append(self.asset[rows].permute(0, 2, 1, 3).to(dev, dtype=torch.float32, non_blocking=True))
            mv.append(self.market[rows].to(dev, dtype=torch.float32, non_blocking=True))
        return av, mv, self.ready[t].to(dev), self.sigma[t].to(dev)
