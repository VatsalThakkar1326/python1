"""The Mark I network.

    per-coin time encoders (1-min, 15-min, 2-hour views)  -> one token per coin
    market time encoders (same views)                      -> one market token
    N x CrossBlock:
        self-attention across [market token + coin tokens] (who matters to whom)
        lead-lag attention: each coin looks at every coin's state 0/5/15/30 min ago
        regime experts: the market token picks a soft mix of K feed-forward experts
    horizon decoder: given any horizon tau (minutes), output quantiles of the
        volatility-scaled future return for every coin

Everything the model sees is at or before the decision time, so causality only matters
for the lag tokens; the time encoders use causal convolutions so those tokens really
describe the past.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from mark1 import config


@dataclass
class ModelConfig:
    n_asset_features: int
    n_market_features: int
    n_assets: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    n_experts: int = 4
    dropout: float = 0.1
    tcn_kernel: int = 3
    tcn_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    lag_minutes: tuple[int, ...] = (0, 5, 15, 30)
    use_asset_embedding: bool = True
    masked_coin_token: bool = False      # learned token for a coin whose inputs are hidden (fused agent's dimension-4 task)
    quantiles: tuple[float, ...] = tuple(config.QUANTILES)
    max_horizon: int = config.MAX_HORIZON
    # Input views: (length, stride in minutes). Must match the dataset.
    views: tuple[tuple[int, int], ...] = ((120, 1), (96, 15), (84, 120))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        d = dict(d)
        for k in ("tcn_dilations", "lag_minutes", "quantiles"):
            d[k] = tuple(d[k])
        d["views"] = tuple(tuple(v) for v in d["views"])
        return cls(**d)


class ChannelNorm(nn.Module):
    """LayerNorm over channels for [B, C, L] tensors."""

    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class CausalConvBlock(nn.Module):
    def __init__(self, d: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(d, 2 * d, kernel, dilation=dilation)
        self.norm = ChannelNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(F.pad(self.norm(x), (self.pad, 0)))
        a, b = h.chunk(2, dim=1)
        return x + self.drop(torch.tanh(a) * torch.sigmoid(b))  # gated activation, WaveNet style


class TemporalEncoder(nn.Module):
    """Dilated causal TCN. [B, L, F] -> hidden sequence [B, L, D]."""

    def __init__(self, n_in: int, d: int, kernel: int, dilations: tuple[int, ...], dropout: float):
        super().__init__()
        self.proj = nn.Conv1d(n_in, d, 1)
        self.blocks = nn.ModuleList(CausalConvBlock(d, kernel, dl, dropout) for dl in dilations)
        self.out_norm = ChannelNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x.transpose(1, 2))
        for blk in self.blocks:
            h = blk(h)
        return self.out_norm(h).transpose(1, 2)


class MultiScaleEncoder(nn.Module):
    """One TCN per view; summarises each view as (last step, mean) and fuses into a token."""

    def __init__(self, n_in: int, cfg: ModelConfig):
        super().__init__()
        self.encoders = nn.ModuleList(
            TemporalEncoder(n_in, cfg.d_model, cfg.tcn_kernel, cfg.tcn_dilations, cfg.dropout) for _ in cfg.views
        )
        self.fuse = nn.Sequential(
            nn.Linear(2 * cfg.d_model * len(cfg.views), cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

    def forward(self, views: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """views[i]: [B, L_i, F]. Returns (token [B, D], finest hidden sequence [B, L_0, D])."""
        parts, fine_seq = [], None
        for i, (enc, v) in enumerate(zip(self.encoders, views)):
            h = enc(v)
            if i == 0:
                fine_seq = h
            parts += [h[:, -1], h.mean(dim=1)]
        return self.fuse(torch.cat(parts, dim=-1)), fine_seq


class RegimeExperts(nn.Module):
    """Soft mixture of K feed-forward experts, weighted by the market token."""

    def __init__(self, d: int, n_experts: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(d, n_experts)
        self.w1 = nn.Parameter(torch.randn(n_experts, d, 4 * d) / math.sqrt(d))
        self.b1 = nn.Parameter(torch.zeros(n_experts, 4 * d))
        self.w2 = nn.Parameter(torch.randn(n_experts, 4 * d, d) / math.sqrt(4 * d))
        self.b2 = nn.Parameter(torch.zeros(n_experts, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, market: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, S, D], market: [B, D]. Returns (output [B, S, D], gate weights [B, K])."""
        g = torch.softmax(self.gate(market), dim=-1)                       # [B, K]
        h = torch.einsum("bsd,kdh->bksh", x, self.w1) + self.b1[None, :, None, :]
        h = self.drop(F.gelu(h))
        y = torch.einsum("bksh,khd->bksd", h, self.w2) + self.b2[None, :, None, :]
        return torch.einsum("bk,bksd->bsd", g, y), g


class CrossBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.norm1 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.lead_lag = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d)
        self.experts = RegimeExperts(d, cfg.n_experts, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, tokens, pad_mask, lag_tokens, lag_mask, need_weights: bool = False):
        """tokens [B, 1+N, D] (market first), pad_mask [B, 1+N] True=ignore,
        lag_tokens [B, N*K, D], lag_mask [B, N*K].
        Returns (tokens, attention weights or None, regime gates, parts) where parts holds what
        this block added to the coin tokens: "cross" (attention over other coins and the market),
        "lead_lag" (lagged coin states) and "experts" (regime-gated feed-forward), each [B, N, D].
        The block is purely residual, so a coin's token is exactly its own-history token plus
        the sum of these parts over all blocks."""
        h = self.norm1(tokens)
        a, attn = self.self_attn(h, h, h, key_padding_mask=pad_mask, need_weights=need_weights)
        a = self.drop(a)
        tokens = tokens + a

        q = self.norm2(tokens[:, 1:])
        kv = self.norm_kv(lag_tokens)
        ll, _ = self.lead_lag(q, kv, kv, key_padding_mask=lag_mask, need_weights=False)
        ll = self.drop(ll)
        tokens = torch.cat([tokens[:, :1], tokens[:, 1:] + ll], dim=1)

        h = self.norm3(tokens)
        e, gates = self.experts(h, h[:, 0])
        e = self.drop(e)
        return tokens + e, attn, gates, {"cross": a[:, 1:], "lead_lag": ll, "experts": e[:, 1:]}


class HorizonDecoder(nn.Module):
    """Maps (coin token, horizon tau) to monotone quantiles of the scaled future return."""

    def __init__(self, cfg: ModelConfig, n_freq: int = 6):
        super().__init__()
        d = cfg.d_model
        self.max_log = math.log(cfg.max_horizon)
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freq)) * math.pi, persistent=False)
        self.tau_mlp = nn.Sequential(nn.Linear(2 * n_freq + 1, d), nn.GELU(), nn.Linear(d, 2 * d))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d), nn.GELU())
        q = list(cfg.quantiles)
        self.mid = q.index(0.5) if 0.5 in q else len(q) // 2
        self.n_q = len(q)
        self.out = nn.Linear(d, self.n_q)

    def tau_features(self, tau: torch.Tensor) -> torch.Tensor:
        u = torch.log(tau.float().clamp(min=1)) / self.max_log      # 0 at 1 minute, 1 at max horizon
        ang = u[..., None] * self.freqs
        return torch.cat([u[..., None], torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(self, coin: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """coin [B, N, D], tau [B, K] minutes -> quantiles [B, K, N, Q]."""
        film = self.tau_mlp(self.tau_features(tau))                  # [B, K, 2D]
        gamma, beta = film.chunk(2, dim=-1)
        h = coin[:, None] * (1 + gamma[:, :, None]) + beta[:, :, None]   # [B, K, N, D]
        raw = self.out(self.head(h))                                 # [B, K, N, Q]
        # Monotone quantiles: a median plus positive steps outward on both sides.
        mid = raw[..., self.mid : self.mid + 1]
        up = torch.cumsum(F.softplus(raw[..., self.mid + 1 :]) + 1e-4, dim=-1)
        down = torch.cumsum(F.softplus(raw[..., : self.mid].flip(-1)) + 1e-4, dim=-1).flip(-1)
        return torch.cat([mid - down, mid, mid + up], dim=-1)


class MarkOne(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.coin_encoder = MultiScaleEncoder(cfg.n_asset_features, cfg)
        self.market_encoder = MultiScaleEncoder(cfg.n_market_features, cfg)
        self.asset_embed = nn.Embedding(cfg.n_assets, d) if cfg.use_asset_embedding else None
        self.lag_embed = nn.Embedding(len(cfg.lag_minutes), d)
        self.lag_proj = nn.Linear(d, d)
        self.blocks = nn.ModuleList(CrossBlock(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.LayerNorm(d)
        self.decoder = HorizonDecoder(cfg)
        self.mask_token = nn.Parameter(torch.zeros(d)) if cfg.masked_coin_token else None

    def encode(self, asset_views, market_views, ready, coin_mask=None, need_weights: bool = False):
        """asset_views[i]: [B, N, L_i, Fa], market_views[i]: [B, L_i, Fm], ready: [B, N] bool.
        coin_mask: optional [B, N] bool; those coins' inputs are hidden (zeroed) and they get a
        learned "hidden" token, so anything the model knows about them must come from other coins.

        Returns coin tokens [B, N, D] and diagnostics, including each pathway separately:
          own       coin tokens from the coin's own history only (before any cross-coin mixing)
          cross     what attention over other coins (and the market token) added
          lead_lag  what lead-lag attention added (coins reacting to other coins' recent states)
          experts   what the regime-gated experts added
          market    final market token [B, D]
        own + cross + lead_lag + experts equals the coin token before the final LayerNorm.
          regime_gates, attention (attention only when need_weights=True)"""
        B, N = ready.shape
        if coin_mask is not None:
            keep = (~coin_mask)[:, :, None, None].to(asset_views[0].dtype)
            asset_views = [v * keep for v in asset_views]
        flat = [v.reshape(B * N, v.shape[2], v.shape[3]) for v in asset_views]
        coin, fine_seq = self.coin_encoder(flat)
        coin = coin.view(B, N, -1)
        if self.asset_embed is not None:
            coin = coin + self.asset_embed.weight[None, :N]
        if coin_mask is not None and self.mask_token is not None:
            coin = torch.where(coin_mask[:, :, None], self.mask_token.to(coin.dtype).expand_as(coin), coin)
        own = coin
        market, _ = self.market_encoder(market_views)

        L0 = fine_seq.shape[1]
        lag_idx = torch.tensor([L0 - 1 - lag for lag in self.cfg.lag_minutes], device=fine_seq.device)
        lag_tokens = self.lag_proj(fine_seq[:, lag_idx])                     # [B*N, K, D]
        lag_tokens = lag_tokens + self.lag_embed.weight[None]
        K = lag_tokens.shape[1]
        lag_tokens = lag_tokens.view(B, N * K, -1)

        not_ready = ~ready
        # A sample with no ready coins would make attention return NaN; unmask it (its loss is masked anyway).
        empty = not_ready.all(dim=1, keepdim=True)
        not_ready = not_ready & ~empty
        pad_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=ready.device), not_ready], dim=1)
        lag_mask = not_ready[:, :, None].expand(B, N, K).reshape(B, N * K)

        tokens = torch.cat([market[:, None], coin], dim=1)
        attns, gates = [], []
        parts = {"cross": 0.0, "lead_lag": 0.0, "experts": 0.0}
        for blk in self.blocks:
            tokens, a, g, p = blk(tokens, pad_mask, lag_tokens, lag_mask, need_weights)
            attns.append(a)
            gates.append(g)
            for k in parts:
                parts[k] = parts[k] + p[k]
        tokens = self.final_norm(tokens)
        return tokens[:, 1:], {"attention": attns, "regime_gates": gates, "own": own, "market": tokens[:, 0], **parts}

    def forward(self, asset_views, market_views, ready, tau):
        coin, diag = self.encode(asset_views, market_views, ready)
        return self.decoder(coin, tau), diag


def pinball_loss(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """pred [B, K, N, Q], y [B, K, N], mask [B, K, N] -> mean quantile loss over valid entries."""
    diff = y[..., None] - pred
    loss = torch.maximum(quantiles * diff, (quantiles - 1) * diff)
    m = mask[..., None].float()
    return (loss * m).sum() / (m.sum() * loss.shape[-1]).clamp(min=1.0)


__all__ = ["ModelConfig", "MarkOne", "pinball_loss"]
