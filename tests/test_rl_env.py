import numpy as np
import pytest
import torch

from mark1.data.panel import synthetic_panel
from mark1.rl.env import EnvConfig, Forecasts, TradingEnv
from mark1.rl.policy import AllocationPolicy, PolicyConfig
from mark1.rl.ppo import PPO, PPOConfig
from mark1.sim import SimConfig

LEVELS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
TAUS = np.array([15, 60, 240])
EVERY = 15


@pytest.fixture(autouse=True)
def _one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def _forecasts(panel, signal_coin=None, drift=0.0):
    rows = np.arange(0, panel.n_times - 1, EVERY)
    K, N, Q = len(TAUS), panel.n_assets, len(LEVELS)
    z = np.array([-1.645, -1.2816, -0.6745, 0.0, 0.6745, 1.2816, 1.645])
    sd = 0.001 * np.sqrt(TAUS)[:, None, None]
    mean = np.zeros((K, N, 1))
    if signal_coin is not None:
        mean[:, signal_coin, 0] = drift * TAUS
    q = np.broadcast_to(mean + sd * z, (K, N, Q))
    return Forecasts(rows, np.repeat(q[None], len(rows), axis=0).astype(np.float32),
                     np.ones((len(rows), N), dtype=bool), TAUS, LEVELS)


def _env(panel, fc, **kw):
    cfg = EnvConfig(decision_every=EVERY, episode_minutes=EVERY * 20, **kw)
    sigma = np.full(panel.close.shape, 0.001, dtype=np.float32)
    return TradingEnv(panel, fc, sigma, 0, panel.n_times, SimConfig(), cfg, seed=0)


def _liquid(panel):
    # Synthetic volumes are tiny; give the coins realistic liquidity so fills are not volume-capped.
    for a in (panel.volume, panel.quote_volume, panel.taker_buy_volume):
        a *= 1e4
    return panel


def test_reset_step_shapes_and_rails():
    panel = _liquid(synthetic_panel(n_times=2000, symbols=["AUSDT", "BUSDT", "CUSDT", "DUSDT"], seed=1))
    env = _env(panel, _forecasts(panel), max_weight=0.3)
    obs = env.reset(seed=3)
    assert obs["forecast"].shape == (3, 4, 7) and obs["coins"].shape == (4, TradingEnv.N_COIN_FEATURES)
    assert obs["portfolio"].shape == (TradingEnv.N_PORTFOLIO_FEATURES,) and obs["mask"].all()
    # All-in on coin 0: the rail caps it at 30%, the rest stays in cash.
    action = np.array([10.0, -10, -10, -10, -10])
    obs, r, done, info = env.step(action)
    assert np.isfinite(r) and info["weights"][0] == pytest.approx(0.3) and info["weights"].sum() <= 1 + 1e-9
    assert env.sim.weights[0] > 0.2
    steps = 1
    while not done:
        obs, r, done, info = env.step(action)
        steps += 1
    assert steps == 20 and info["truncated"]


def test_untradable_coin_gets_no_new_weight():
    panel = synthetic_panel(n_times=2000, symbols=["AUSDT", "BUSDT"], seed=2)
    fc = _forecasts(panel)
    fc.ready[:, 1] = False
    env = _env(panel, fc)
    obs = env.reset(seed=0)
    assert not obs["mask"][1]
    _, _, _, info = env.step(np.array([-5.0, 10.0, -5.0]))
    assert info["weights"][1] == 0.0


def test_stop_loss_forces_exit_and_locks_out():
    panel = _liquid(synthetic_panel(n_times=600, symbols=["AUSDT", "BUSDT"], seed=3))
    panel.close[100:, 0] *= 0.5   # coin A halves after row 100
    panel.open[101:, 0] *= 0.5
    panel.high[100:, 0] *= 0.5
    panel.low[100:, 0] *= 0.5
    env = _env(panel, _forecasts(panel), stop_loss_sigmas=3.0, stop_cooldown=4, max_weight=0.5)
    env.reset(seed=0)
    env.t0 = 0
    env.sim.reset(start=0, end=599)
    env.peak = env.sim.equity
    buy = np.array([10.0, -10.0, -10.0])
    for _ in range(12):
        _, _, _, info = env.step(buy)
        if info["rails"]["stopped"]:
            break
    assert info["rails"]["stopped"] == 1 and info["weights"][0] == 0.0
    assert env.sim.weights[0] < 0.01                  # sold on the next bars
    for _ in range(3):                                # still locked out while the cooldown runs
        _, _, _, info = env.step(buy)
        assert info["weights"][0] == 0.0


def test_full_period_episode_is_deterministic():
    panel = synthetic_panel(n_times=1500, symbols=["AUSDT", "BUSDT"], seed=4)
    sigma = np.full(panel.close.shape, 0.001, dtype=np.float32)
    fc = _forecasts(panel)
    eq = []
    for _ in range(2):
        env = TradingEnv(panel, fc, sigma, 0, panel.n_times, SimConfig(), EnvConfig(decision_every=EVERY), full_period=True)
        env.reset()
        done = False
        while not done:
            _, _, done, info = env.step(np.array([0.5, 0.2, 0.0]))
        eq.append(info["equity"])
    assert eq[0] == eq[1] and env.n_steps == len(env.grid) - 1


def _policy(env):
    cfg = PolicyConfig(n_assets=env.n, n_taus=len(TAUS), n_quantiles=len(LEVELS), n_coin_features=TradingEnv.N_COIN_FEATURES,
                       n_portfolio_features=TradingEnv.N_PORTFOLIO_FEATURES, d_model=32, n_heads=2, n_layers=1)
    return AllocationPolicy(cfg)


def test_policy_forward_and_masking():
    panel = synthetic_panel(n_times=1500, symbols=["AUSDT", "BUSDT", "CUSDT"], seed=5)
    fc = _forecasts(panel)
    fc.ready[:, 2] = False
    env = _env(panel, fc)
    obs = env.reset(seed=0)
    pol = _policy(env)
    batch = {k: torch.as_tensor(np.stack([v, v])) for k, v in obs.items()}
    dist, value = pol(batch)
    assert dist.mode.shape == (2, 4) and value.shape == (2,)
    a = dist.sample()
    lp = dist.log_prob(a)
    a2 = a.clone()
    a2[:, 2] += 100.0                     # changing a masked coin's logit must not change log-prob
    assert torch.allclose(lp, dist.log_prob(a2))


def test_ppo_learns_to_follow_a_true_forecast():
    """Coin A really drifts up and the forecast says so: PPO should shift weight onto A."""
    panel = _liquid(synthetic_panel(n_times=6000, symbols=["AUSDT", "BUSDT", "CUSDT"], seed=6, vol_per_minute=0.0005))
    drift = 0.0004                                         # 4 bps per minute, way above costs over hours
    lc = np.log(panel.close)
    lc[:, 0] += drift * np.arange(panel.n_times)
    panel.close[:] = np.exp(lc)
    panel.open[1:] = panel.close[:-1]
    panel.high[:] = np.maximum(panel.open, panel.close) * 1.0002
    panel.low[:] = np.minimum(panel.open, panel.close) * 0.9998
    fc = _forecasts(panel, signal_coin=0, drift=drift)
    sim_cfg = SimConfig(taker_fee_bps=10, half_spread_bps=1.0, impact_coef=0.0)
    sigma = np.full(panel.close.shape, 0.0005, dtype=np.float32)
    env_cfg = EnvConfig(decision_every=EVERY, episode_minutes=EVERY * 16, max_weight=0.9, stop_loss_sigmas=50)
    make_env = lambda i: TradingEnv(panel, fc, sigma, 0, panel.n_times, sim_cfg, env_cfg, seed=i)
    torch.manual_seed(0)
    pol = _policy(make_env(0))
    ppo = PPO(pol, make_env, PPOConfig(n_envs=4, n_steps=64, minibatch_size=128, n_epochs=4, lr=1e-3,
                                       reward_scale=100.0, seed=0, device="cpu"))
    ppo.learn(25 * 4 * 64)
    env = make_env(99)
    obs = env.reset(seed=99)
    with torch.no_grad():
        dist, _ = pol({k: torch.as_tensor(v[None]) for k, v in obs.items()})
    w, _ = env.target_weights(dist.mode[0].numpy())
    assert w[0] > 0.5, w
