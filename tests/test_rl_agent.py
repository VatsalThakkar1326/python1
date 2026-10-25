"""The fused Mark I Agent: one network, all dimensions, trained in one run."""

import numpy as np
import pytest
import torch

from mark1.data.panel import synthetic_panel
from mark1.features.build import WARMUP, build_features
from mark1.model.dataset import Normalizer, make_splits
from mark1.model.feature_store import TorchFeatureStore
from mark1.model.network import ModelConfig
from mark1.rl.agent import DIMENSIONS, AgentConfig, DimensionBlind, Labels, MarkOneAgent
from mark1.rl.env import EnvConfig, TradingEnv
from mark1.rl.ppo import PPO, PPOConfig
from mark1.sim import SimConfig

VIEWS = ((16, 1), (8, 4), (6, 16))


@pytest.fixture(autouse=True)
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _setup(dim_dropout=0.3):
    panel = synthetic_panel(n_times=WARMUP + 1500, symbols=["AUSDT", "BUSDT", "CUSDT"], seed=21)
    for a in (panel.volume, panel.quote_volume, panel.taker_buy_volume):
        a *= 1e4
    fs = build_features(panel)
    splits = make_splits(len(fs.times), VIEWS, gap=60, first_row=WARMUP)
    norm = Normalizer.fit(fs, splits["train"])
    store = TorchFeatureStore(fs, norm, VIEWS, torch.device("cpu"))
    labels = Labels(fs, label_end=splits["train"].end)
    mc = ModelConfig(n_asset_features=fs.asset.shape[2], n_market_features=fs.market.shape[1], n_assets=3,
                     d_model=16, n_heads=2, n_layers=1, n_experts=2, tcn_dilations=(1, 2), lag_minutes=(0, 3),
                     views=VIEWS, max_horizon=240, masked_coin_token=True)
    acfg = AgentConfig(model=mc, decision_taus=(15, 60, 240), d_policy=16, n_policy_layers=1, n_policy_heads=2, sup_batch=4, sup_taus=3)
    sim_cfg = SimConfig()
    env_cfg = EnvConfig(decision_every=15, episode_minutes=15 * 8, dim_dropout=dim_dropout)
    train = splits["train"]
    rows = np.arange(train.start, train.end - 240)
    torch.manual_seed(0)
    agent = MarkOneAgent(acfg, np.full(3, 0.009)).attach(store, labels, rows, fs.asset_names)
    make_env = lambda i, full=False: TradingEnv(panel, None, np.asarray(fs.sigma), train.start, train.end, sim_cfg, env_cfg,
                                               seed=i, full_period=full, ready=np.asarray(fs.ready))
    return agent, make_env


def test_env_hides_dimensions_but_never_all_of_them():
    _, make_env = _setup(dim_dropout=0.6)
    env = make_env(0)
    masks = [env.reset(seed=s)["dim_mask"] for s in range(50)]
    assert all(m.shape == (len(DIMENSIONS),) and m.any() for m in masks)
    assert not all(m.all() for m in masks)
    assert make_env(0, full=True).reset()["dim_mask"].all()      # evaluation always sees everything


def test_forward_aux_and_gradients_reach_every_pathway():
    agent, make_env = _setup()
    env = make_env(0)
    obs = [env.reset(seed=s) for s in range(4)]
    batch = {k: torch.as_tensor(np.stack([o[k] for o in obs])) for k in obs[0]}
    dist, value = agent(batch)
    assert dist.mode.shape == (4, 4) and value.shape == (4,)
    loss = -dist.log_prob(dist.sample()).mean() + value.pow(2).mean() + agent.aux_total(batch)
    loss.backward()
    assert set(agent.last_aux) == {"forecast", "market", "rank", "own_vol", "hidden_coin"}
    assert all(np.isfinite(v) for v in agent.last_aux.values())
    for name in ("coin_encoder", "market_encoder", "blocks", "decoder"):
        g = [p.grad for p in getattr(agent.core, name).parameters() if p.grad is not None]
        assert g and any(x.abs().sum() > 0 for x in g), name
    for head in ("h_own_vol", "h_rank", "h_market", "h_hidden", "g_own", "g_cross", "g_market", "g_lead_lag", "g_forecast"):
        g = [p.grad for p in getattr(agent, head).parameters() if p.grad is not None]
        assert g and any(x.abs().sum() > 0 for x in g), head


def test_hidden_dimension_changes_the_policy_input():
    agent, make_env = _setup(dim_dropout=0.0)
    agent.eval()
    # The logit head starts at zero (start in cash), which makes the output input-independent at init.
    torch.nn.init.normal_(agent.coin_logit.weight)
    torch.nn.init.normal_(agent.missing)                # make "unavailable" distinguishable from real inputs
    obs = make_env(0).reset(seed=1)
    batch = {k: torch.as_tensor(v[None]) for k, v in obs.items()}
    with torch.no_grad():
        full = agent(batch)[0].mode
        blind = [DimensionBlind(agent, i)(batch)[0].mode for i in range(len(DIMENSIONS))]
    assert all(not torch.allclose(full, b) for b in blind)


def test_single_run_ppo_with_aux_trains_end_to_end():
    agent, make_env = _setup()
    before = [p.detach().clone() for p in agent.core.parameters()]
    ppo = PPO(agent, lambda i: make_env(i), PPOConfig(n_envs=2, n_steps=16, minibatch_size=16, n_epochs=2,
                                                      reward_scale=100.0, max_grad_norm=1.0, weight_decay=0.01, device="cpu"),
              aux_loss_fn=agent.aux_total)
    hist = ppo.learn(2 * 2 * 16)
    assert len(hist) == 2 and all(np.isfinite(h["aux_loss"]) for h in hist)
    moved = sum(float((p.detach() - b).abs().sum()) for p, b in zip(agent.core.parameters(), before))
    assert moved > 0                                     # the forecaster itself was trained in the same run
