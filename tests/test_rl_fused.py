"""Stage 2: Mark I inside the policy must receive gradients from both the policy loss and the anchor."""

import numpy as np
import torch

from mark1.data.panel import synthetic_panel
from mark1.features.build import WARMUP, build_features
from mark1.model.dataset import Normalizer, make_splits
from mark1.model.network import MarkOne, ModelConfig
from mark1.rl.env import TradingEnv
from mark1.rl.policy import AllocationPolicy, FusedForecaster, PolicyConfig

VIEWS = ((16, 1), (8, 4), (6, 16))
TAUS = np.array([15, 60])


def _setup(lr_mult=0.5):
    fs = build_features(synthetic_panel(n_times=WARMUP + 800, symbols=["AUSDT", "BUSDT", "CUSDT"], seed=11))
    splits = make_splits(len(fs.times), VIEWS, gap=60, first_row=WARMUP)
    norm = Normalizer.fit(fs, splits["train"])
    mcfg = ModelConfig(n_asset_features=fs.asset.shape[2], n_market_features=fs.market.shape[1], n_assets=3,
                       d_model=16, n_heads=2, n_layers=1, n_experts=2, tcn_dilations=(1, 2), lag_minutes=(0, 3),
                       views=VIEWS, max_horizon=240, dropout=0.0)
    torch.manual_seed(0)
    model = MarkOne(mcfg)
    fused = FusedForecaster(model, fs, norm, VIEWS, TAUS, np.full(3, 0.009), label_end=splits["train"].end, lr_mult=lr_mult)
    pcfg = PolicyConfig(n_assets=3, n_taus=len(TAUS), n_quantiles=len(mcfg.quantiles), n_coin_features=TradingEnv.N_COIN_FEATURES,
                        n_portfolio_features=TradingEnv.N_PORTFOLIO_FEATURES, d_model=16, n_heads=2, n_layers=1)
    policy = AllocationPolicy(pcfg, fused)
    t = np.arange(splits["train"].start, splits["train"].start + 6)
    obs = {
        "t": torch.as_tensor(t[:, None]),
        "forecast": torch.zeros(6, len(TAUS), 3, len(mcfg.quantiles)),   # ignored in fused mode
        "coins": torch.zeros(6, 3, TradingEnv.N_COIN_FEATURES),
        "portfolio": torch.zeros(6, TradingEnv.N_PORTFOLIO_FEATURES),
        "mask": torch.ones(6, 3, dtype=torch.bool),
    }
    return policy, fused, model, obs


def test_policy_loss_reaches_the_forecaster():
    policy, fused, model, obs = _setup()
    dist, value = policy(obs)
    loss = -dist.log_prob(dist.sample()).mean() + value.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_anchor_loss_reaches_the_forecaster_and_gradients_are_scaled():
    _, fused_a, model_a, obs = _setup(lr_mult=1.0)
    fused_a.aux_loss(obs).backward()
    ga = torch.cat([p.grad.flatten() for p in model_a.parameters() if p.grad is not None])
    _, fused_b, model_b, obs_b = _setup(lr_mult=0.1)
    fused_b.aux_loss(obs_b).backward()
    gb = torch.cat([p.grad.flatten() for p in model_b.parameters() if p.grad is not None])
    assert ga.abs().sum() > 0
    torch.testing.assert_close(gb, 0.1 * ga, rtol=1e-4, atol=1e-8)
