"""The RL allocation policy: forecast curves + portfolio state -> portfolio-weight logits.

Each coin becomes a token built from its forecast curve (quantiles at every horizon, in units
of round-trip cost) and its position state. A portfolio token carries cash, exposure,
drawdown and time left. Self-attention across all tokens lets the policy weigh coins against
each other (one slot of capital, many candidates). Heads:
  * one logit per coin and one for cash, the mean of a diagonal Gaussian over logits
    (the env applies a softmax, so the policy outputs portfolio weights)
  * a value estimate for PPO

Stage 1 reads precomputed forecasts from the observation. Stage 2 ("fused") re-runs the Mark I
network inside the forward pass so policy gradients flow into the forecaster, while an
auxiliary quantile loss on real future returns keeps it an honest forecaster.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from mark1.features.build import FeatureSet
from mark1.features.targets import forward_targets, invalid_cumsum
from mark1.model.dataset import Normalizer, assemble_inputs
from mark1.model.network import MarkOne, pinball_loss
from mark1.rl.distributions import DiagGaussian


@dataclass
class PolicyConfig:
    n_assets: int
    n_taus: int
    n_quantiles: int
    n_coin_features: int
    n_portfolio_features: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0
    init_log_std: float = -0.5

    def to_dict(self) -> dict:
        return asdict(self)


def _mlp(i: int, d: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, d), nn.LayerNorm(d))


class AllocationPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig, fused: "FusedForecaster | None" = None):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.coin_in = _mlp(cfg.n_taus * cfg.n_quantiles + cfg.n_coin_features, d)
        self.port_in = _mlp(cfg.n_portfolio_features, d)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout, batch_first=True, norm_first=True, activation="gelu")
        self.mixer = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.coin_logit = nn.Linear(d, 1)
        self.cash_logit = nn.Linear(d, 1)
        self.value = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.log_std = nn.Parameter(torch.full((cfg.n_assets + 1,), cfg.init_log_std))
        # Start near "mostly cash, small equal bets" so early exploration is not reckless.
        nn.init.zeros_(self.coin_logit.weight)
        nn.init.zeros_(self.coin_logit.bias)
        nn.init.zeros_(self.cash_logit.weight)
        nn.init.constant_(self.cash_logit.bias, 2.0)
        self.fused = fused

    def forecast_features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """[B, K, N, Q] forecast quantiles in round-trip-cost units."""
        if self.fused is not None:
            return self.fused.features(obs["t"][:, 0])
        return obs["forecast"]

    def forward(self, obs: dict[str, torch.Tensor]):
        f = self.forecast_features(obs).float()                 # [B, K, N, Q]
        B, K, N, Q = f.shape
        f = torch.clamp(f, -50, 50).permute(0, 2, 1, 3).reshape(B, N, K * Q)
        coins = self.coin_in(torch.cat([f, obs["coins"].float()], dim=-1))
        port = self.port_in(obs["portfolio"].float())[:, None]
        mask = obs["mask"].bool()
        empty = ~mask.any(dim=1, keepdim=True)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=mask.device), ~mask & ~empty], dim=1)
        h = self.mixer(torch.cat([port, coins], dim=1), src_key_padding_mask=pad)
        port_h, coin_h = h[:, 0], h[:, 1:]
        mean = torch.cat([self.coin_logit(coin_h).squeeze(-1), self.cash_logit(port_h)], dim=-1)   # [B, N+1]
        m = mask.float()[..., None]
        pooled = (coin_h * m).sum(1) / m.sum(1).clamp(min=1.0)
        value = self.value(torch.cat([port_h, pooled], dim=-1)).squeeze(-1)
        # DiagGaussian's mask marks ACTIVE dims: tradable coins plus cash (always active).
        active = torch.cat([mask, torch.ones(B, 1, dtype=torch.bool, device=mask.device)], dim=1)
        return DiagGaussian(mean, self.log_std, mask=active), value


class FusedForecaster(nn.Module):
    """Mark I inside the policy (stage 2): recomputes forecasts for decision rows with gradients."""

    def __init__(self, model: MarkOne, fs: FeatureSet, norm: Normalizer, views, taus: np.ndarray,
                 round_trip: np.ndarray, label_end: int, lr_mult: float = 0.1):
        super().__init__()
        self.model = model
        self.fs, self.norm, self.views = fs, norm, tuple(tuple(v) for v in views)
        self.register_buffer("taus", torch.as_tensor(np.asarray(taus), dtype=torch.long), persistent=False)
        self.register_buffer("cost", torch.as_tensor(np.asarray(round_trip), dtype=torch.float32), persistent=False)
        self.register_buffer("quantiles", torch.tensor(model.cfg.quantiles), persistent=False)
        self.inv_cum = invalid_cumsum(np.asarray(fs.valid))
        self.label_end = label_end
        # The forecaster learns more slowly than the policy: scale its gradients.
        for p in self.model.parameters():
            p.register_hook(lambda g, m=lr_mult: g * m)

    def _inputs(self, t: np.ndarray, device):
        inp = assemble_inputs(self.fs, self.norm, self.views, t)
        conv = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        return [conv(a) for a in inp["asset_views"]], [conv(a) for a in inp["market_views"]], conv(inp["ready"]), conv(inp["sigma"])

    def features(self, t: torch.Tensor) -> torch.Tensor:
        rows = t.detach().cpu().numpy().astype(np.int64)
        dev = self.taus.device
        av, mv, ready, sigma = self._inputs(rows, dev)
        tau = self.taus[None].expand(len(rows), -1)
        q, _ = self.model(av, mv, ready, tau)                                   # [B, K, N, Q] scaled
        scale = sigma[:, None, :, None] * torch.sqrt(self.taus.float())[None, :, None, None]
        logret = q * scale
        logret = logret * ready[:, None, :, None].float()
        return logret / self.cost[None, None, :, None]

    def aux_loss(self, obs: dict[str, torch.Tensor], max_rows: int = 128) -> torch.Tensor:
        """Quantile loss on real future returns for (a sample of) the minibatch rows."""
        rows = obs["t"][:, 0].detach().cpu().numpy().astype(np.int64)
        if len(rows) > max_rows:
            rows = np.random.default_rng().choice(rows, size=max_rows, replace=False)
        dev = self.taus.device
        av, mv, ready, _ = self._inputs(rows, dev)
        taus = self.taus.cpu().numpy()
        tau = np.tile(taus, (len(rows), 1))
        y, mask = forward_targets(np.asarray(self.fs.log_close), np.asarray(self.fs.sigma), self.inv_cum, rows, tau)
        mask &= ((rows[:, None] + tau) < self.label_end)[:, :, None]
        mask &= ready.cpu().numpy()[:, None, :]
        q, _ = self.model(av, mv, ready, torch.from_numpy(tau).to(dev))
        return pinball_loss(q.float(), torch.from_numpy(y).to(dev), torch.from_numpy(mask).to(dev), self.quantiles)
