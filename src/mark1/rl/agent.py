"""Mark I Agent: forecaster and trading policy as ONE network, trained in ONE run from scratch.

Inside one forward pass:

  Mark I encoder (multi-scale TCNs + cross blocks)
      |-- own      coin's own-history pathway                     -> dimension 1
      |-- cross    what attention over other coins added           -> dimension 2
      |-- market   market token + regime-gated experts             -> dimension 3
      |-- lead_lag what lagged states of other coins added         -> dimension 4
      |-- horizon decoder: quantile curves at every horizon         -> dimension 5
  policy: per coin, the five dimension groups + position state, mixed across coins with
          attention, plus a portfolio token -> logits for coins + cash, and a value

The encoder's coin token is exactly own + cross + lead_lag + experts (all blocks are residual),
so each group is a genuine separate pathway, not a guess.

What forces learning on every dimension:
  * Dimension dropout. The environment hides whole groups at random (obs["dim_mask"]); a hidden
    group is replaced by a learned "unavailable" vector. The policy cannot lean on one source.
  * One auxiliary task per dimension, trained in the same run alongside PPO:
      1 own history:  predict the coin's own future volatility from the own pathway only
      2 other coins:  rank which coins will do best over the next 1h / 4h (ListNet)
      3 market:       quantiles of the whole market's future move at every horizon
      4 interactions: hide one coin's inputs and recover its recent moves from the other coins
      5 forecast:     quantile loss on real future returns at every horizon
    They run on the rollout's own decision times and on an extra random supervised batch from
    the training period (with dropout and input noise for regularisation).
  * Profit itself: PPO's gradient flows through the policy into every pathway.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mark1.features.build import FeatureSet
from mark1.features.targets import forward_targets, invalid_cumsum
from mark1.model.feature_store import TorchFeatureStore
from mark1.model.network import MarkOne, ModelConfig, pinball_loss
from mark1.rl.distributions import DiagGaussian
from mark1.rl.env import TradingEnv

DIMENSIONS = ("own_history", "other_coins", "market", "interactions", "forecast")
DECISION_TAUS = (5, 15, 30, 60, 120, 240, 480, 720, 1440)


@dataclass
class AuxWeights:
    own_vol: float = 0.3       # dimension 1
    rank: float = 0.5          # dimension 2
    market: float = 0.5        # dimension 3
    hidden_coin: float = 0.3   # dimension 4
    forecast: float = 1.0      # dimension 5


@dataclass
class AgentConfig:
    model: ModelConfig
    decision_taus: tuple[int, ...] = DECISION_TAUS
    rank_horizons: tuple[int, ...] = (60, 240)
    vol_horizons: tuple[int, ...] = (60, 240)
    hidden_targets: tuple[str, ...] = ("ret_15", "ret_60")
    n_coin_state: int = TradingEnv.N_COIN_FEATURES
    n_portfolio: int = TradingEnv.N_PORTFOLIO_FEATURES
    d_policy: int = 64
    n_policy_layers: int = 2
    n_policy_heads: int = 4
    init_log_std: float = -0.5
    aux: AuxWeights = field(default_factory=AuxWeights)
    sup_batch: int = 32         # extra random training rows per update minibatch for the auxiliary tasks
    sup_taus: int = 8           # random horizons per supervised row
    input_noise: float = 0.05   # Gaussian noise on normalised inputs of supervised rows

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AgentConfig":
        d = dict(d)
        d["model"] = ModelConfig.from_dict(d["model"])
        d["aux"] = AuxWeights(**d["aux"])
        for k in ("decision_taus", "rank_horizons", "vol_horizons", "hidden_targets"):
            d[k] = tuple(d[k])
        return cls(**d)


class Labels:
    """Future targets for the auxiliary tasks, computed on the CPU from the feature set."""

    def __init__(self, fs: FeatureSet, label_end: int):
        self.log_close = np.asarray(fs.log_close)
        self.sigma = np.asarray(fs.sigma)
        self.inv_cum = invalid_cumsum(np.asarray(fs.valid))
        r1 = np.zeros_like(self.log_close)
        r1[1:] = np.diff(self.log_close, axis=0)
        r1[~np.asarray(fs.valid)] = 0.0
        self.r2cum = np.zeros((r1.shape[0] + 1, r1.shape[1]))
        self.r2cum[1:] = np.cumsum(r1 ** 2, axis=0)
        self.label_end = label_end

    def returns(self, t: np.ndarray, taus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scaled future returns y [B, K, N] and their mask; labels never reach past label_end."""
        y, mask = forward_targets(self.log_close, self.sigma, self.inv_cum, t, taus)
        mask &= ((t[:, None] + taus) < self.label_end)[:, :, None]
        return y, mask

    def future_vol(self, t: np.ndarray, horizons: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """log(realised volatility over (t, t+h] / sigma_t), shape [B, N, H], and its mask."""
        end = t[:, None] + horizons[None, :]                                  # [B, H]
        ok = end < min(self.label_end, len(self.log_close))
        end_c = np.minimum(end, len(self.log_close) - 1)
        var = (self.r2cum[end_c + 1] - self.r2cum[t + 1][:, None, :]) / horizons[None, :, None]   # [B, H, N]
        sig = self.sigma[t][:, None, :]
        y = np.log(np.sqrt(np.maximum(var, 1e-14)) / np.maximum(sig, 1e-8))
        bad = self.inv_cum[end_c + 1] - self.inv_cum[t + 1][:, None, :]
        mask = ok[:, :, None] & (bad == 0)
        return np.clip(np.where(mask, y, 0.0), -5, 5).transpose(0, 2, 1).astype(np.float32), mask.transpose(0, 2, 1)


def _monotone(raw: torch.Tensor, mid: int) -> torch.Tensor:
    m = raw[..., mid : mid + 1]
    up = torch.cumsum(F.softplus(raw[..., mid + 1 :]) + 1e-4, dim=-1)
    down = torch.cumsum(F.softplus(raw[..., :mid].flip(-1)) + 1e-4, dim=-1).flip(-1)
    return torch.cat([m - down, m, m + up], dim=-1)


def _group(d_in: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out))


class MarkOneAgent(nn.Module):
    def __init__(self, cfg: AgentConfig, round_trip: np.ndarray):
        super().__init__()
        self.cfg = cfg
        mc = cfg.model
        D, P, N = mc.d_model, cfg.d_policy, mc.n_assets
        K, Q = len(cfg.decision_taus), len(mc.quantiles)
        self.core = MarkOne(mc)

        # policy: one projection per dimension group
        self.g_own = _group(D, P)
        self.g_cross = _group(D, P)
        self.g_market = _group(2 * D, P)          # market token + regime experts' contribution
        self.g_lead_lag = _group(D, P)
        self.g_forecast = _group(K * Q, P)
        self.missing = nn.Parameter(torch.zeros(len(DIMENSIONS), P))
        self.g_state = _group(cfg.n_coin_state, P)
        self.coin_fuse = nn.Sequential(nn.Linear(6 * P, P), nn.GELU(), nn.LayerNorm(P))
        self.port_in = nn.Sequential(nn.Linear(cfg.n_portfolio + P, P), nn.GELU(), nn.LayerNorm(P))
        layer = nn.TransformerEncoderLayer(P, cfg.n_policy_heads, 4 * P, 0.0, batch_first=True, norm_first=True, activation="gelu")
        self.mixer = nn.TransformerEncoder(layer, cfg.n_policy_layers, enable_nested_tensor=False)
        self.coin_logit = nn.Linear(P, 1)
        self.cash_logit = nn.Linear(P, 1)
        self.value = nn.Sequential(nn.Linear(2 * P, P), nn.GELU(), nn.Linear(P, 1))
        self.log_std = nn.Parameter(torch.full((N + 1,), cfg.init_log_std))
        nn.init.zeros_(self.coin_logit.weight)
        nn.init.zeros_(self.coin_logit.bias)
        nn.init.zeros_(self.cash_logit.weight)
        nn.init.constant_(self.cash_logit.bias, 2.0)        # start mostly in cash

        # auxiliary heads, one per dimension
        self.h_own_vol = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D), nn.GELU(), nn.Linear(D, len(cfg.vol_horizons)))
        self.h_rank = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, len(cfg.rank_horizons)))
        tau_dim = self.core.decoder.tau_mlp[0].in_features
        self.h_market = nn.Sequential(nn.Linear(D + tau_dim, D), nn.GELU(), nn.Linear(D, Q))
        self.h_hidden = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, len(cfg.hidden_targets)))

        self.register_buffer("taus", torch.tensor(cfg.decision_taus, dtype=torch.long), persistent=False)
        self.register_buffer("cost", torch.as_tensor(np.asarray(round_trip), dtype=torch.float32), persistent=False)
        self.register_buffer("quantiles", torch.tensor(mc.quantiles), persistent=False)
        self.mid = list(mc.quantiles).index(0.5)
        # Data hooks (plain attributes: not parameters, not saved with the weights).
        self.store: TorchFeatureStore | None = None
        self.labels: Labels | None = None
        self.train_rows: np.ndarray | None = None
        self.hidden_idx: list[int] = []
        self.aux_scale = 1.0
        self.last_aux: dict[str, float] = {}
        self._cache: tuple | None = None
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ wiring
    def attach(self, store: TorchFeatureStore, labels: Labels | None, train_rows: np.ndarray | None,
               asset_names: list[str], seed: int = 0) -> "MarkOneAgent":
        self.store, self.labels, self.train_rows = store, labels, train_rows
        self.hidden_idx = [asset_names.index(n) for n in self.cfg.hidden_targets]
        self._rng = np.random.default_rng(seed)
        return self

    @property
    def device(self) -> torch.device:
        return self.log_std.device

    # ------------------------------------------------------------------ encoder
    def encode(self, t: torch.Tensor, coin_mask: torch.Tensor | None = None, noise: float = 0.0, taus: torch.Tensor | None = None) -> dict:
        av, mv, ready, sigma = self.store.inputs(t, out_device=self.device)
        if noise > 0:
            av = [a + noise * torch.randn_like(a) for a in av]
            mv = [m + noise * torch.randn_like(m) for m in mv]
        taus = self.taus[None].expand(len(t), -1) if taus is None else taus
        dev_type = self.device.type
        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16, enabled=dev_type == "cuda"):
            coin, d = self.core.encode(av, mv, ready, coin_mask=coin_mask)
            q = self.core.decoder(coin, taus)
        f = lambda x: x.float()
        return {"coin": f(coin), "own": f(d["own"]), "cross": f(d["cross"]), "lead_lag": f(d["lead_lag"]),
                "experts": f(d["experts"]), "market": f(d["market"]), "q": f(q), "taus": taus,
                "ready": ready, "sigma": sigma}

    # ------------------------------------------------------------------ policy
    def forward(self, obs: dict[str, torch.Tensor]):
        t = obs["t"][:, 0]
        enc = self.encode(t)
        self._cache = (t.data_ptr(), len(t), enc)
        mask = obs["mask"].bool()
        B, N = mask.shape
        dim_mask = obs.get("dim_mask")
        dim_mask = torch.ones(B, len(DIMENSIONS), dtype=torch.bool, device=mask.device) if dim_mask is None else dim_mask.bool()

        fc = enc["q"] * enc["sigma"][:, None, :, None] * torch.sqrt(self.taus.float())[None, :, None, None]
        fc = fc / self.cost[None, None, :, None] * enc["ready"][:, None, :, None].float()
        fc = torch.clamp(fc, -50, 50).permute(0, 2, 1, 3).reshape(B, N, -1)
        market_b = enc["market"][:, None].expand(B, N, -1)
        groups = [
            self.g_own(enc["own"]),
            self.g_cross(enc["cross"]),
            self.g_market(torch.cat([market_b, enc["experts"]], dim=-1)),
            self.g_lead_lag(enc["lead_lag"]),
            self.g_forecast(fc),
        ]
        groups = [torch.where(dim_mask[:, i, None, None], g, self.missing[i].expand_as(g)) for i, g in enumerate(groups)]
        coins = self.coin_fuse(torch.cat(groups + [self.g_state(obs["coins"].float())], dim=-1))
        market_tok = groups[2].mean(dim=1)
        port = self.port_in(torch.cat([obs["portfolio"].float(), market_tok], dim=-1))[:, None]

        empty = ~mask.any(dim=1, keepdim=True)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=mask.device), ~mask & ~empty], dim=1)
        h = self.mixer(torch.cat([port, coins], dim=1), src_key_padding_mask=pad)
        port_h, coin_h = h[:, 0], h[:, 1:]
        mean = torch.cat([self.coin_logit(coin_h).squeeze(-1), self.cash_logit(port_h)], dim=-1)
        m = mask.float()[..., None]
        pooled = (coin_h * m).sum(1) / m.sum(1).clamp(min=1.0)
        value = self.value(torch.cat([port_h, pooled], dim=-1)).squeeze(-1)
        active = torch.cat([mask, torch.ones(B, 1, dtype=torch.bool, device=mask.device)], dim=1)
        return DiagGaussian(mean, self.log_std, mask=active), value

    # ------------------------------------------------------------------ auxiliary tasks
    def _market_quantiles(self, market: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        feats = self.core.decoder.tau_features(taus)                       # [B, K, tau_dim]
        x = torch.cat([market[:, None].expand(-1, taus.shape[1], -1), feats], dim=-1)
        return _monotone(self.h_market(x), self.mid)                       # [B, K, Q]

    def _task_losses(self, rows: np.ndarray, enc: dict) -> dict[str, torch.Tensor]:
        """Dimension 1, 2, 3 and 5 losses for rows with an encoding."""
        dev = self.device
        lab = self.labels
        taus_np = enc["taus"].cpu().numpy()
        y, ymask = lab.returns(rows, taus_np)
        y_t, m_t = torch.from_numpy(y).to(dev), torch.from_numpy(ymask).to(dev)
        out = {"forecast": pinball_loss(enc["q"], y_t, m_t, self.quantiles)}

        # market: average scaled move of the valid coins at each horizon
        cnt = m_t.float().sum(-1)
        ym = (y_t * m_t.float()).sum(-1) / cnt.clamp(min=1)
        mq = self._market_quantiles(enc["market"], enc["taus"])
        out["market"] = pinball_loss(mq[:, :, None, :], ym[:, :, None], (cnt >= 2)[:, :, None], self.quantiles)

        # other coins: ListNet ranking of future returns across coins
        rh = np.asarray(self.cfg.rank_horizons)
        yr, mr = lab.returns(rows, np.tile(rh, (len(rows), 1)))
        yr, mr = torch.from_numpy(yr).to(dev), torch.from_numpy(mr).to(dev)       # [B, H, N]
        score = self.h_rank(enc["coin"]).permute(0, 2, 1)                          # [B, H, N]
        ok = mr.sum(-1) >= 3
        neg = torch.finfo(score.dtype).min / 4
        p_true = torch.softmax(yr.masked_fill(~mr, neg), dim=-1)
        logp = torch.log_softmax(score.masked_fill(~mr, neg), dim=-1)
        rank = -(p_true * logp.masked_fill(~mr, 0.0)).sum(-1)
        out["rank"] = (rank * ok).sum() / ok.sum().clamp(min=1)

        # own history: future volatility from the own pathway alone
        vh = np.asarray(self.cfg.vol_horizons)
        yv, mv = lab.future_vol(rows, vh)
        yv, mv = torch.from_numpy(yv).to(dev), torch.from_numpy(mv).to(dev)
        mv = mv & enc["ready"][:, :, None]
        pv = self.h_own_vol(enc["own"])
        out["own_vol"] = (F.smooth_l1_loss(pv, yv, reduction="none") * mv).sum() / mv.sum().clamp(min=1)
        return out

    def _hidden_coin_loss(self, rows: np.ndarray) -> torch.Tensor:
        """Dimension 4: hide one ready coin per row; recover its recent moves from the others."""
        dev = self.device
        t = torch.as_tensor(rows, device=dev)
        ready = self.store.ready[t.to(self.store.device)].to(dev)
        pick = torch.multinomial(ready.float() + 1e-6, 1).squeeze(1)              # one coin per row
        coin_mask = torch.zeros_like(ready)
        coin_mask[torch.arange(len(rows), device=dev), pick] = True
        enc = self.encode(t, coin_mask=coin_mask)
        target = self.store.asset[t.to(self.store.device)][:, :, self.hidden_idx].to(dev).float()   # [B, N, H]
        sel = torch.arange(len(rows), device=dev)
        pred = self.h_hidden(enc["coin"][sel, pick])
        ok = ready[sel, pick].float()[:, None]
        return (F.smooth_l1_loss(pred, target[sel, pick], reduction="none") * ok).sum() / (ok.sum() * pred.shape[1]).clamp(min=1)

    def aux_total(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """All auxiliary losses for a PPO minibatch (called by PPO right after forward(obs))."""
        w = self.cfg.aux
        t = obs["t"][:, 0]
        if self._cache is not None and self._cache[0] == t.data_ptr() and self._cache[1] == len(t):
            enc = self._cache[2]
        else:
            enc = self.encode(t)
        rows = t.detach().cpu().numpy().astype(np.int64)
        parts = {k: v for k, v in self._task_losses(rows, enc).items()}

        if self.cfg.sup_batch > 0 and self.train_rows is not None:
            srows = self._rng.choice(self.train_rows, size=self.cfg.sup_batch)
            was_training = self.core.training
            self.core.train()                                # dropout on for the supervised rows only
            stau = np.clip(np.floor(np.exp(self._rng.uniform(0, math.log(self.cfg.model.max_horizon + 1),
                                                             size=(len(srows), self.cfg.sup_taus)))), 1, self.cfg.model.max_horizon).astype(np.int64)
            senc = self.encode(torch.as_tensor(srows, device=self.device), noise=self.cfg.input_noise,
                               taus=torch.as_tensor(stau, device=self.device))
            for k, v in self._task_losses(srows, senc).items():
                parts[k] = 0.5 * (parts[k] + v)
            parts["hidden_coin"] = self._hidden_coin_loss(srows)
            self.core.train(was_training)
        total = sum(getattr(w, k) * v for k, v in parts.items())
        self.last_aux = {k: float(v.detach()) for k, v in parts.items()}
        return self.aux_scale * total


class DimensionBlind(nn.Module):
    """Wraps an agent so one dimension is always hidden (for measuring how much it matters)."""

    def __init__(self, agent: MarkOneAgent, dim: int):
        super().__init__()
        self.agent, self.dim = agent, dim

    def forward(self, obs):
        obs = dict(obs)
        dm = obs.get("dim_mask")
        B = obs["t"].shape[0]
        dm = torch.ones(B, len(DIMENSIONS), dtype=torch.bool, device=obs["t"].device) if dm is None else dm.clone().bool()
        dm[:, self.dim] = False
        obs["dim_mask"] = dm
        return self.agent(obs)
