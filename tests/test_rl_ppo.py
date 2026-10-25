"""Tests for the generic PPO core: distribution maths, GAE, rollout bookkeeping, learning on toy envs."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from mark1.rl import PPO, DiagGaussian, PPOConfig, compute_gae, evaluate


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    """Nets here are tiny; multi-threaded kernels only add contention (10-50x slower on a busy
    4-vCPU box), so run them on one thread and put the setting back afterwards."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)

# --------------------------------------------------------------------------- DiagGaussian


def test_diag_gaussian_matches_torch_normal():
    g = torch.Generator().manual_seed(0)
    mean, log_std = torch.randn(6, 4, generator=g), torch.randn(4, generator=g) * 0.3
    a = torch.randn(6, 4, generator=g)
    d = DiagGaussian(mean, log_std)
    ref = torch.distributions.Normal(mean, log_std.exp())
    torch.testing.assert_close(d.log_prob(a), ref.log_prob(a).sum(-1))
    torch.testing.assert_close(d.entropy(), ref.entropy().sum(-1))
    torch.testing.assert_close(d.mode, mean)
    assert d.log_prob(a).shape == (6,) and d.entropy().shape == (6,) and d.sample().shape == (6, 4)

    # per-sample log_std [B, A] works the same way
    ls2 = torch.randn(6, 4, generator=g) * 0.3
    d2 = DiagGaussian(mean, ls2)
    torch.testing.assert_close(d2.log_prob(a), torch.distributions.Normal(mean, ls2.exp()).log_prob(a).sum(-1))


def test_diag_gaussian_sample_statistics():
    torch.manual_seed(0)
    d = DiagGaussian(torch.full((50_000, 2), 1.5), torch.tensor([0.0, -1.0]))
    s = d.sample()
    assert not s.requires_grad
    assert s.mean(0).tolist() == pytest.approx([1.5, 1.5], abs=0.03)
    assert s.std(0).tolist() == pytest.approx([1.0, np.exp(-1.0)], abs=0.03)


def test_diag_gaussian_mask_zeroes_dims():
    g = torch.Generator().manual_seed(1)
    mean = torch.randn(5, 4, generator=g, requires_grad=True)
    log_std = torch.zeros(4, requires_grad=True)
    a = torch.randn(5, 4, generator=g)
    mask = torch.tensor([[1, 1, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1], [1, 0, 0, 0]], dtype=torch.bool)
    ref = torch.distributions.Normal(mean, log_std.exp())
    a_nan = a.clone()
    a_nan[~mask] = float("nan")             # garbage in masked dims must not leak anywhere
    d = DiagGaussian(mean, log_std, mask)
    lp, ent = d.log_prob(a_nan), d.entropy()
    torch.testing.assert_close(lp, (ref.log_prob(a) * mask).sum(-1))
    torch.testing.assert_close(ent, (ref.entropy() * mask).sum(-1))
    assert lp[1] == 0 and ent[1] == 0       # fully masked row contributes nothing

    (lp.sum() + ent.sum()).backward()
    assert torch.isfinite(mean.grad).all() and torch.isfinite(log_std.grad).all()
    assert (mean.grad[~mask] == 0).all()    # no policy gradient for masked dims

    s = d.sample()
    torch.testing.assert_close(s[~mask], mean.detach()[~mask])     # masked dims return the mean
    assert (s[mask] != mean.detach()[mask]).all()
    torch.testing.assert_close(d.mode, mean)
    with pytest.raises(ValueError):
        DiagGaussian(mean, log_std, mask[:, :3])


# --------------------------------------------------------------------------- GAE


def test_gae_hand_computed_with_done_and_truncation():
    # gamma=0.9, lam=0.8. Step 1 is a true termination, step 3 is a truncation with V(final)=5.
    rewards = np.array([[1.0], [2.0], [3.0], [4.0]])
    values = np.array([[0.5], [1.0], [1.5], [2.0]])
    dones = np.array([[0.0], [1.0], [0.0], [1.0]])
    trunc = np.array([[0.0], [0.0], [0.0], [5.0]])
    adv, ret = compute_gae(rewards, values, dones, trunc, last_values=np.array([99.0]), gamma=0.9, lam=0.8)
    # t=3: delta = 4 + 0.9*5 - 2.0 = 6.5, chain cut                     -> A3 = 6.5  (last_values ignored)
    # t=2: delta = 3 + 0.9*2.0 - 1.5 = 3.3; A2 = 3.3 + 0.9*0.8*6.5      -> 7.98
    # t=1: terminal, delta = 2 - 1.0 = 1.0, chain cut                   -> A1 = 1.0  (no leak from step 2)
    # t=0: delta = 1 + 0.9*1.0 - 0.5 = 1.4; A0 = 1.4 + 0.9*0.8*1.0      -> 2.12
    np.testing.assert_allclose(adv[:, 0], [2.12, 1.0, 7.98, 6.5])
    np.testing.assert_allclose(ret[:, 0], [2.62, 2.0, 9.48, 8.5])


def test_gae_bootstraps_from_last_value_and_limits():
    rng = np.random.default_rng(0)
    T, E, gamma = 5, 3, 0.95
    r, v, lv = rng.normal(size=(T, E)), rng.normal(size=(T, E)), rng.normal(size=E)
    zeros = np.zeros((T, E))
    # lam=1 -> returns are the discounted reward sum plus gamma^T * V(last)
    _, ret = compute_gae(r, v, zeros, zeros, lv, gamma, 1.0)
    mc = sum(gamma ** t * r[t] for t in range(T)) + gamma ** T * lv
    np.testing.assert_allclose(ret[0], mc)
    # lam=0 -> one-step TD advantages
    adv, _ = compute_gae(r, v, zeros, zeros, lv, gamma, 0.0)
    nxt = np.vstack([v[1:], lv])
    np.testing.assert_allclose(adv, r + gamma * nxt - v)


# --------------------------------------------------------------------------- rollout bookkeeping


class CountingEnv:
    """obs = step counter, reward 1 per step, episode ends after `length` steps."""

    def __init__(self, length: int, truncate: bool):
        self.length, self.truncate, self.t = length, truncate, 0

    def reset(self, seed=None):
        self.t = 0
        return {"t": np.array([self.t], dtype=np.float32)}

    def step(self, action):
        self.t += 1
        done = self.t >= self.length
        return {"t": np.array([self.t], dtype=np.float32)}, 1.0, done, {"truncated": done and self.truncate}


class CounterPolicy(nn.Module):
    """V(obs) = 0.1 * counter, so V(final obs) differs from V(reset obs)."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(2))

    def forward(self, obs):
        t = obs["t"][:, 0]
        return DiagGaussian(self.dummy.expand(len(t), 2), torch.zeros(2)), 0.1 * t


def test_collect_rollout_handles_dones_resets_and_truncation():
    lengths = {0: (3, True), 1: (4, False)}          # env 0 truncates every 3 steps, env 1 terminates every 4
    cfg = PPOConfig(n_envs=2, n_steps=8, gamma=0.5, gae_lambda=1.0, reward_scale=2.0, device="cpu")
    ppo = PPO(CounterPolicy(), lambda i: CountingEnv(*lengths[i]), cfg)
    buf = ppo.collect_rollout()

    np.testing.assert_array_equal(buf.obs["t"][:, 0, 0], [0, 1, 2, 0, 1, 2, 0, 1])     # reset after each done
    np.testing.assert_array_equal(buf.obs["t"][:, 1, 0], [0, 1, 2, 3, 0, 1, 2, 3])
    np.testing.assert_array_equal(buf.dones[:, 0], [0, 0, 1, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(buf.dones[:, 1], [0, 0, 0, 1, 0, 0, 0, 1])
    # truncation bootstraps from V(final obs, t=3) = 0.3; true terminations store nothing
    np.testing.assert_allclose(buf.trunc_values[:, 0], [0, 0, 0.3, 0, 0, 0.3, 0, 0], atol=1e-6)
    assert not buf.trunc_values[:, 1].any()
    np.testing.assert_allclose(buf.rewards, 2.0)                                      # reward_scale applied
    # env 0, t=5 (truncation): 2 + 0.5*0.3 - V(t=2)=0.2 ; env 1, t=7 (terminal): 2 - V(t=3)=0.3
    assert buf.advantages[5, 0] == pytest.approx(2 + 0.5 * 0.3 - 0.2, abs=1e-6)
    assert buf.advantages[7, 1] == pytest.approx(2 - 0.3, abs=1e-6)
    np.testing.assert_allclose(buf.last_values, [0.2, 0.0], atol=1e-6)                # env 1 was just reset
    adv, ret = compute_gae(buf.rewards, buf.values, buf.dones, buf.trunc_values, buf.last_values, 0.5, 1.0)
    np.testing.assert_allclose(buf.advantages, adv, atol=1e-6)
    np.testing.assert_allclose(buf.returns, ret, atol=1e-6)

    assert sorted(buf.episode_lengths) == [3, 3, 4, 4]
    assert sorted(buf.episode_returns) == [3.0, 3.0, 4.0, 4.0]                        # raw, not scaled
    assert ppo.env_steps == 16


# --------------------------------------------------------------------------- toy environments


class TargetEnv:
    """Contextual bandit: obs x ~ N(0, I), reward = -||a - W x||^2, fixed-length episodes that
    end as time-limit truncations (so the truncation path is exercised in training too)."""

    DIM_X, DIM_A, HORIZON = 3, 2, 8
    W = np.random.default_rng(123).normal(size=(DIM_A, DIM_X)).astype(np.float32) * 0.7

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.t, self.x = 0, np.zeros(self.DIM_X, np.float32)

    def _obs(self):
        self.x = self.rng.normal(size=self.DIM_X).astype(np.float32)
        return {"x": self.x}

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        return self._obs()

    def step(self, action):
        err = np.asarray(action, np.float32) - self.W @ self.x
        self.t += 1
        done = self.t >= self.HORIZON
        return self._obs(), -float(err @ err), done, {"truncated": done}


class ToyPortfolioEnv:
    """Shaped like the real trading env: 4 asset slots (the last one is never tradable) + cash.

    Each episode one tradable asset is "hot" (mean return +2% per step, vs zero-mean noise for the
    others) and is revealed through a noisy one-hot signal. action = logits [5] -> softmax weights
    with untradable slots forced to 0. reward = log(1 + w . r) - cost * |w - w_prev|_1.
    """

    N, HORIZON, COST, MU, SIGMA = 4, 16, 1e-3, 0.02, 0.01
    MASK = np.array([True, True, True, False])

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.hot, self.t, self.w = 0, 0, np.zeros(self.N + 1)

    def _obs(self):
        signal = np.eye(self.N)[self.hot] + 0.2 * self.rng.normal(size=self.N)
        return {"signal": signal.astype(np.float32), "w_prev": self.w.astype(np.float32), "mask": self.MASK.copy()}

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.hot = int(self.rng.integers(0, 3))
        self.t, self.w = 0, np.eye(self.N + 1)[-1]                  # start fully in cash
        return self._obs()

    def weights(self, logits: np.ndarray) -> np.ndarray:
        z = np.where(np.append(self.MASK, True), np.asarray(logits, np.float64), -np.inf)
        e = np.exp(z - z.max())
        return e / e.sum()

    def step(self, action):
        w = self.weights(action)
        r = self.SIGMA * self.rng.normal(size=self.N)
        r[self.hot] += self.MU
        reward = float(np.log1p(w[:-1] @ r) - self.COST * np.abs(w - self.w).sum())
        self.w, self.t = w, self.t + 1
        return self._obs(), reward, self.t >= self.HORIZON, {}


class MLPPolicy(nn.Module):
    """Concatenate `keys` -> tanh MLP -> Gaussian mean + value; state-independent log_std.
    If the obs has a "mask" [B, N] the cash slot is appended and masked dims are ignored."""

    def __init__(self, keys, in_dim, act_dim, hidden=32, init_log_std=-0.5):
        super().__init__()
        self.keys = keys
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.mean_head, self.value_head = nn.Linear(hidden, act_dim), nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.mean_head.weight, 0.01)
        nn.init.zeros_(self.mean_head.bias)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def forward(self, obs):
        h = self.body(torch.cat([obs[k].float() for k in self.keys], dim=-1))
        mask = None
        if "mask" in obs:
            mask = torch.cat([obs["mask"], torch.ones_like(obs["mask"][:, :1])], dim=-1)
        return DiagGaussian(self.mean_head(h), self.log_std, mask), self.value_head(h).squeeze(-1)


def make_target_policy(seed=0):
    torch.manual_seed(seed)
    return MLPPolicy(("x",), TargetEnv.DIM_X, TargetEnv.DIM_A)


def make_portfolio_policy(seed=0):
    torch.manual_seed(seed)
    n = ToyPortfolioEnv.N
    return MLPPolicy(("signal", "w_prev"), n + n + 1, n + 1)


# --------------------------------------------------------------------------- learning sanity


def test_ppo_learns_target_regression():
    cfg = PPOConfig(n_envs=4, n_steps=64, gamma=0.5, gae_lambda=0.9, n_epochs=6, minibatch_size=64,
                    lr=3e-3, lr_schedule="linear", ent_coef=0.0, seed=0, device="cpu")
    ppo = PPO(make_target_policy(0), lambda i: TargetEnv(100 + i), cfg)
    before = evaluate(ppo.policy, TargetEnv(0), n_episodes=40, seed=1000)["return_mean"]
    history = ppo.learn(total_steps=50 * 4 * 64)
    after = evaluate(ppo.policy, TargetEnv(0), n_episodes=40, seed=1000)["return_mean"]
    # Untrained, every step costs ~||Wx||^2 (episode return about -16); a good policy is near 0.
    assert after > 0.05 * before, (before, after)           # both negative: >95% of the error removed
    assert history[-1]["ep_return_mean"] > history[0]["ep_return_mean"]
    assert history[-1]["entropy"] < history[0]["entropy"]    # the policy sharpened


def _hot_weight_and_return(policy, n_episodes, seed, equal_weight=False):
    """(mean weight on the hot asset, mean per-episode return) with deterministic actions."""
    env, weights, returns = ToyPortfolioEnv(0), [], []
    for i in range(n_episodes):
        obs, total, done = env.reset(seed=seed + i), 0.0, False
        while not done:
            with torch.no_grad():
                dist, _ = policy({k: torch.from_numpy(v)[None] for k, v in obs.items()})
            logits = np.zeros(env.N + 1) if equal_weight else dist.mode[0].numpy()
            weights.append(env.weights(logits)[env.hot])
            obs, reward, done, _ = env.step(logits)
            total += reward
        returns.append(total)
    return float(np.mean(weights)), float(np.mean(returns))


def test_ppo_learns_toy_portfolio():
    cfg = PPOConfig(n_envs=8, n_steps=64, gamma=0.9, gae_lambda=0.9, n_epochs=4, minibatch_size=128,
                    lr=2e-3, reward_scale=100.0, ent_coef=0.0, seed=0, device="cpu")
    ppo = PPO(make_portfolio_policy(0), lambda i: ToyPortfolioEnv(200 + i), cfg)
    w0, _ = _hot_weight_and_return(ppo.policy, 40, seed=5000)
    ppo.learn(total_steps=60 * 8 * 64)
    w1, ret_ppo = _hot_weight_and_return(ppo.policy, 40, seed=5000)
    w_eq, ret_eq = _hot_weight_and_return(ppo.policy, 40, seed=5000, equal_weight=True)
    assert w0 == pytest.approx(0.25, abs=0.05)               # untrained policy ~ equal weights over 3 assets + cash
    assert w_eq == pytest.approx(0.25, abs=1e-9)
    assert w1 > 0.6, (w0, w1)                                # clearly concentrated on the signalled asset
    assert ret_ppo > ret_eq + 0.02, (ret_ppo, ret_eq)        # beats equal-weight on the same 40 episodes


class ConstObsEnv:
    """Constant observation, reward 1 per step, 5-step episodes ending as a truncation or a termination."""

    def __init__(self, truncate: bool, length: int = 5):
        self.truncate, self.length, self.t = truncate, length, 0

    def reset(self, seed=None):
        self.t = 0
        return {"o": np.zeros(1, np.float32)}

    def step(self, action):
        self.t += 1
        done = self.t >= self.length
        return {"o": np.zeros(1, np.float32)}, 1.0, done, {"truncated": done and self.truncate}


@pytest.mark.parametrize("truncate, lo, hi", [(True, 9.0, 11.0), (False, 2.3, 3.0)])
def test_value_learning_distinguishes_truncation_from_termination(truncate, lo, hi):
    # gamma=0.9. A time limit must bootstrap, so the value is the infinite-horizon 1/(1-gamma) = 10.
    # A true termination after 5 steps averages the discounted remaining reward over the positions: ~2.63.
    torch.manual_seed(0)
    policy = MLPPolicy(("o",), 1, 1)
    cfg = PPOConfig(n_envs=4, n_steps=40, gamma=0.9, gae_lambda=0.95, n_epochs=4, minibatch_size=80, lr=1e-2,
                    lr_schedule="constant", target_kl=None, seed=0, device="cpu")
    ppo = PPO(policy, lambda i: ConstObsEnv(truncate), cfg)
    ppo.learn(total_steps=100 * 160)
    with torch.no_grad():
        v = policy({"o": torch.zeros(1, 1)})[1].item()
    assert lo < v < hi, v


# --------------------------------------------------------------------------- training loop mechanics


def _small_target_ppo(seed=0, **overrides) -> PPO:
    kw = dict(n_envs=2, n_steps=16, n_epochs=4, minibatch_size=20, lr=1e-2, seed=seed, device="cpu")
    kw.update(overrides)
    return PPO(make_target_policy(seed), lambda i: TargetEnv(10 * seed + i), PPOConfig(**kw))


def test_learn_callback_lr_schedule_and_stats():
    ppo = _small_target_ppo()
    seen = []

    def cb(it, stats):
        seen.append(it)
        return False if it == 4 else None

    hist = ppo.learn(total_steps=10 * 32, callback=cb)
    assert seen == [1, 2, 3, 4] and len(hist) == 4           # callback can stop training
    assert [h["env_steps"] for h in hist] == [32, 64, 96, 128]
    lrs = [h["lr"] for h in hist]
    assert lrs[0] == pytest.approx(1e-2) and lrs[-1] == pytest.approx(1e-2 * 0.7) and lrs == sorted(lrs, reverse=True)
    keys = {"policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "explained_var", "aux_loss",
            "n_epochs_run", "ep_return_mean", "ep_len_mean", "steps_per_sec"}
    assert keys <= set(hist[-1])
    assert all(np.isfinite(h["policy_loss"]) and np.isfinite(h["value_loss"]) for h in hist)
    assert hist[-1]["ep_len_mean"] == TargetEnv.HORIZON       # 32 steps per iteration > 1 episode finished
    # absolute target: asking for the same total again does nothing; a larger one continues
    assert ppo.learn(total_steps=4 * 32) == []
    assert len(ppo.learn(total_steps=5 * 32)) == 1

    ppo_c = _small_target_ppo(lr_schedule="constant")
    assert {h["lr"] for h in ppo_c.learn(total_steps=3 * 32)} == {1e-2}


def test_target_kl_stops_epochs_early():
    stats = _small_target_ppo(target_kl=1e-9).learn(total_steps=32)[0]
    assert stats["n_epochs_run"] == 1
    stats = _small_target_ppo(target_kl=None).learn(total_steps=32)[0]
    assert stats["n_epochs_run"] == 4


def test_config_validation():
    with pytest.raises(ValueError):
        PPOConfig(lr_schedule="cosine")
    with pytest.raises(ValueError):
        PPOConfig(minibatch_size=0)


def test_clip_range_vf_runs():
    stats = _small_target_ppo(clip_range_vf=0.2).learn(total_steps=32)[0]
    assert np.isfinite(stats["value_loss"])


def test_evaluate_is_deterministic_and_reports_stats():
    policy = make_target_policy(0)
    r1 = evaluate(policy, TargetEnv(0), n_episodes=5, seed=7)
    r2 = evaluate(policy, TargetEnv(99), n_episodes=5, seed=7)
    assert r1["returns"] == r2["returns"] and r1["lengths"] == [TargetEnv.HORIZON] * 5
    assert r1["return_mean"] == pytest.approx(np.mean(r1["returns"])) and r1["length_std"] == 0.0
    policy.train()
    evaluate(policy, TargetEnv(0), n_episodes=1)
    assert policy.training                                    # mode restored
    s = evaluate(policy, TargetEnv(0), n_episodes=2, deterministic=False, seed=7)
    assert s["returns"] != r1["returns"][:2]                  # sampling differs from the mean action


# --------------------------------------------------------------------------- save / load


def test_save_load_roundtrip(tmp_path):
    a = _small_target_ppo(seed=0)
    a.learn(total_steps=2 * 32)
    path = tmp_path / "ckpt" / "ppo.pt"
    a.save(path)

    b = _small_target_ppo(seed=1)                             # different init
    obs = {"x": torch.randn(16, TargetEnv.DIM_X, generator=torch.Generator().manual_seed(3))}
    with torch.no_grad():
        (da, va), (db, vb) = a.policy(obs), b.policy(obs)
    assert not torch.equal(da.mode, db.mode)
    b.load(path)
    with torch.no_grad():
        (db, vb) = b.policy(obs)
    assert torch.equal(da.mode, db.mode) and torch.equal(va, vb)      # identical actions and values
    assert (b.iteration, b.env_steps) == (2, 64)
    sa, sb = a.optimizer.state_dict()["state"], b.optimizer.state_dict()["state"]
    assert all(torch.equal(sa[k]["exp_avg"], sb[k]["exp_avg"]) and float(sa[k]["step"]) == float(sb[k]["step"]) for k in sa)
    assert len(b.learn(total_steps=3 * 32)) == 1               # resumes cleanly (envs re-reset)
    assert b.iteration == 3


def test_save_load_restores_extra_params_outside_policy(tmp_path):
    extra_a = nn.Parameter(torch.tensor([1.0, 2.0]))
    a = PPO(make_target_policy(0), lambda i: TargetEnv(i), PPOConfig(n_envs=1, n_steps=8, device="cpu"), extra_params=[extra_a])
    a.save(tmp_path / "a.pt")
    extra_b = nn.Parameter(torch.zeros(2))
    b = PPO(make_target_policy(0), lambda i: TargetEnv(i), PPOConfig(n_envs=1, n_steps=8, device="cpu"), extra_params=[extra_b])
    b.load(tmp_path / "a.pt")
    assert extra_b.tolist() == [1.0, 2.0]


# --------------------------------------------------------------------------- aux loss / extra params


def test_aux_loss_gradient_reaches_extra_params():
    extra = nn.Parameter(torch.tensor([3.0, -2.0]))
    calls = []

    def aux_loss(obs):
        assert isinstance(obs["x"], torch.Tensor) and obs["x"].shape[1:] == (TargetEnv.DIM_X,)
        calls.append(obs["x"].shape[0])
        return (extra ** 2).sum()

    # reward_scale keeps the policy's own gradients small: the grad-norm clip is global, so a huge
    # policy gradient would shrink the aux gradient too (and vice versa).
    cfg = PPOConfig(n_envs=2, n_steps=16, n_epochs=4, minibatch_size=16, lr=0.03, lr_schedule="constant",
                    target_kl=None, reward_scale=0.1, device="cpu")
    ppo = PPO(make_target_policy(0), lambda i: TargetEnv(i), cfg, aux_loss_fn=aux_loss, extra_params=[extra])
    hist = ppo.learn(total_steps=15 * 32)
    assert len(calls) == 15 * 4 * 2 and set(calls) == {16}    # once per minibatch
    assert hist[0]["aux_loss"] > 10.0                          # started at (3, -2): 9 + 4
    assert extra.abs().max() < 1.0                             # moved toward 0
    assert hist[-1]["aux_loss"] < 0.1 * hist[0]["aux_loss"]


def test_params_are_deduplicated_and_frozen_ones_skipped():
    policy = make_target_policy(0)
    policy.value_head.bias.requires_grad_(False)
    n_trainable = sum(p.requires_grad for p in policy.parameters())
    extra = nn.Parameter(torch.zeros(1))
    ppo = PPO(policy, lambda i: TargetEnv(i), PPOConfig(n_envs=1, n_steps=8, device="cpu"),
              extra_params=[*policy.parameters(), extra, extra])      # policy params again + a duplicate
    assert len(ppo.params) == n_trainable + 1
    assert len({id(p) for p in ppo.params}) == len(ppo.params)
    assert not any(p is policy.value_head.bias for p in ppo.params)
    assert len(ppo.optimizer.param_groups[0]["params"]) == n_trainable + 1
