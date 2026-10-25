"""Proximal Policy Optimization (clipped objective) for dict-observation, single-process envs.

Generic RL core: it knows nothing about trading. The pieces it plugs into are duck-typed:

  env     reset(seed=None) -> obs;  step(action) -> (obs, reward, done, info)
          obs is a dict of fixed-shape numpy arrays; info["truncated"] marks a time-limit end.
  policy  nn.Module, policy(obs_batch) -> (dist, value) with a leading batch dim on every tensor.
          dist needs sample(), log_prob(a) -> [B], entropy() -> [B] (see distributions.DiagGaussian).

Episode ends are handled the standard way: a true termination cuts the bootstrap; a truncation
bootstraps from V(final observation) (computed before the env is reset), so a time limit is
never mistaken for the end of the world.
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np
import torch
from torch import nn

log = logging.getLogger(__name__)

Obs = dict[str, np.ndarray]
TorchObs = dict[str, torch.Tensor]

RECENT_EPISODES = 100   # window for the running episode return / length in the logs


class Env(Protocol):
    def reset(self, seed: int | None = None) -> Obs: ...
    def step(self, action: np.ndarray) -> tuple[Obs, float, bool, dict]: ...


@dataclass
class PPOConfig:
    n_envs: int = 8
    n_steps: int = 256              # steps per env per rollout
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: float | None = None   # in scaled-reward units; None = unclipped value loss
    n_epochs: int = 4
    minibatch_size: int = 512
    lr: float = 3e-4
    lr_schedule: str = "linear"     # "linear" (to 0 over learn()) or "constant"
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5      # <= 0 disables clipping
    target_kl: float | None = 0.03  # stop the update early when an epoch's mean approx KL exceeds this
    normalize_advantages: bool = True
    reward_scale: float = 1.0       # rewards are multiplied by this for training (episode stats stay raw)
    seed: int = 0
    device: str = "auto"
    # False (default): policy.eval() during rollouts and updates, so dropout cannot make the
    # old/new log-probs differ for identical weights. True keeps dropout/batch-norm active.
    policy_train_mode: bool = False

    def __post_init__(self) -> None:
        if self.lr_schedule not in ("linear", "constant"):
            raise ValueError(f"lr_schedule must be 'linear' or 'constant', got {self.lr_schedule!r}")
        if min(self.n_envs, self.n_steps, self.n_epochs, self.minibatch_size) < 1:
            raise ValueError("n_envs, n_steps, n_epochs and minibatch_size must all be >= 1")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _clean_obs(obs: Obs) -> Obs:
    """Plain contiguous arrays; float64 is narrowed to float32 (torch policies expect it)."""
    out = {}
    for k, v in obs.items():
        a = np.asarray(v)
        out[k] = a.astype(np.float32) if a.dtype == np.float64 else a
    return out


def _stack_obs(obs_list: list[Obs]) -> Obs:
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}


def _to_torch(obs: Obs, device: torch.device) -> TorchObs:
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in obs.items()}


def compute_gae(
    rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, trunc_values: np.ndarray,
    last_values: np.ndarray, gamma: float, lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimation over [T, E] arrays; returns (advantages, returns).

    dones[t]        1 if the episode ended on step t (terminated or truncated), else 0.
    trunc_values[t] V(final obs) when step t was a truncation, else 0. The step's reward is
                    credited with gamma * trunc_values[t]; the GAE chain is cut either way.
    last_values     V of the observation after the last step (ignored where dones[T-1] is 1).
    """
    rewards, values, dones, trunc_values = (np.asarray(x, dtype=np.float64) for x in (rewards, values, dones, trunc_values))
    next_value = np.asarray(last_values, dtype=np.float64)
    adv = np.zeros_like(rewards)
    last_gae = np.zeros_like(next_value)
    for t in reversed(range(rewards.shape[0])):
        alive = 1.0 - dones[t]
        delta = rewards[t] + gamma * trunc_values[t] + gamma * next_value * alive - values[t]
        last_gae = delta + gamma * lam * alive * last_gae
        adv[t] = last_gae
        next_value = values[t]
    return adv, adv + values


class RolloutBuffer:
    """One rollout. Every array is [n_steps, n_envs, ...]; observations are kept as numpy per key
    and only moved to the device one minibatch at a time."""

    def __init__(self, n_steps: int, n_envs: int, obs_example: Obs, action_shape: tuple[int, ...]):
        self.n_steps, self.n_envs = n_steps, n_envs
        self.obs = {k: np.zeros((n_steps, n_envs, *v.shape), dtype=v.dtype) for k, v in obs_example.items()}
        self.actions = np.zeros((n_steps, n_envs, *action_shape), dtype=np.float32)
        shape = (n_steps, n_envs)
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros(shape, dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)        # already multiplied by reward_scale
        self.dones = np.zeros(shape, dtype=np.float32)
        self.trunc_values = np.zeros(shape, dtype=np.float32)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)
        self.last_values = np.zeros(n_envs, dtype=np.float32)
        self.episode_returns: list[float] = []                  # raw (unscaled) return of episodes that ended here
        self.episode_lengths: list[int] = []

    @property
    def size(self) -> int:
        return self.n_steps * self.n_envs

    def compute_gae(self, last_values: np.ndarray, gamma: float, lam: float) -> None:
        self.last_values[:] = last_values
        adv, ret = compute_gae(self.rewards, self.values, self.dones, self.trunc_values, last_values, gamma, lam)
        self.advantages[:], self.returns[:] = adv, ret

    def flatten(self) -> tuple[Obs, dict[str, np.ndarray]]:
        """(obs, rest) with the [T, E] dims merged into one sample dim (views, no copies)."""
        n = self.size
        obs = {k: v.reshape(n, *v.shape[2:]) for k, v in self.obs.items()}
        rest = {
            "actions": self.actions.reshape(n, *self.actions.shape[2:]),
            "log_probs": self.log_probs.reshape(n), "values": self.values.reshape(n),
            "advantages": self.advantages.reshape(n), "returns": self.returns.reshape(n),
        }
        return obs, rest


class PPO:
    """Clipped-objective PPO with GAE.

    aux_loss_fn(obs_minibatch) -> scalar tensor, added to the loss on every minibatch (e.g. a
    supervised anchor for a forecaster that also receives policy gradients). extra_params are
    optimised too but need not be part of `policy` (they must already be on the policy's device);
    parameters are de-duplicated, so passing a policy submodule's parameters again is harmless.
    Parameters with requires_grad=False are left out of the optimizer.
    """

    def __init__(
        self,
        policy: nn.Module,
        make_env: Callable[[int], Env],
        cfg: PPOConfig,
        aux_loss_fn: Callable[[TorchObs], torch.Tensor] | None = None,
        extra_params: Iterable[nn.Parameter] | None = None,
    ):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self._rng = np.random.default_rng(cfg.seed)     # minibatch order (device independent)

        self.policy = policy.to(self.device)
        self.aux_loss_fn = aux_loss_fn
        policy_ids = {id(p) for p in policy.parameters()}
        extras = list(extra_params) if extra_params is not None else []
        ref = next(self.policy.parameters(), None)      # compare with the real device ("cuda" vs "cuda:0")
        for p in extras:
            if ref is not None and p.device != ref.device:
                raise ValueError(f"extra param on {p.device} but policy is on {ref.device}; move it before building PPO")
        self._extra_only = [p for p in dict.fromkeys(extras) if id(p) not in policy_ids]   # saved by value
        unique = {id(p): p for p in [*policy.parameters(), *extras]}
        self.params = [p for p in unique.values() if p.requires_grad]
        # eps=1e-5 is the usual PPO setting (Adam's default 1e-8 makes tiny-gradient dims jumpy)
        self.optimizer = torch.optim.Adam(self.params, lr=cfg.lr, eps=1e-5)

        self.envs = [make_env(i) for i in range(cfg.n_envs)]
        self.iteration = 0          # completed updates
        self.env_steps = 0
        self._obs: list[Obs] | None = None      # current observation per env (None = not reset yet)
        self._ep_ret = np.zeros(cfg.n_envs)
        self._ep_len = np.zeros(cfg.n_envs, dtype=np.int64)
        self._recent_returns: deque[float] = deque(maxlen=RECENT_EPISODES)
        self._recent_lengths: deque[int] = deque(maxlen=RECENT_EPISODES)

    # ------------------------------------------------------------------ rollout

    def _reset_envs(self) -> None:
        # A resumed run gets different env seeds than the original start.
        base = self.cfg.seed + 10_000 * self.iteration
        self._obs = [_clean_obs(env.reset(seed=base + i)) for i, env in enumerate(self.envs)]
        self._ep_ret[:] = 0.0
        self._ep_len[:] = 0

    def _forward(self, obs: Obs) -> tuple[object, torch.Tensor]:
        """policy on a stacked numpy obs batch -> (dist, value[B] float32)."""
        batch = next(iter(obs.values())).shape[0]
        dist, value = self.policy(_to_torch(obs, self.device))
        value = value.reshape(-1).float()
        if value.shape[0] != batch:
            raise ValueError(f"policy returned {value.shape[0]} values for a batch of {batch}")
        return dist, value

    @torch.no_grad()
    def collect_rollout(self) -> RolloutBuffer:
        cfg = self.cfg
        if self._obs is None:
            self._reset_envs()
        self.policy.train(cfg.policy_train_mode)
        buf: RolloutBuffer | None = None

        for t in range(cfg.n_steps):
            obs = _stack_obs(self._obs)
            dist, value = self._forward(obs)
            action = dist.sample()
            log_prob = dist.log_prob(action).float()
            actions = action.float().cpu().numpy()
            if buf is None:
                buf = RolloutBuffer(cfg.n_steps, cfg.n_envs, self._obs[0], actions.shape[1:])
            for k, v in obs.items():
                buf.obs[k][t] = v
            buf.actions[t] = actions
            buf.log_probs[t] = log_prob.cpu().numpy()
            buf.values[t] = value.cpu().numpy()

            trunc_envs: list[int] = []
            trunc_obs: list[Obs] = []
            for e, env in enumerate(self.envs):
                next_obs, reward, done, info = env.step(actions[e])
                reward = float(reward)
                if not math.isfinite(reward):
                    raise FloatingPointError(f"env {e} returned a non-finite reward ({reward}) at rollout step {t}")
                buf.rewards[t, e] = reward * cfg.reward_scale
                self._ep_ret[e] += reward
                self._ep_len[e] += 1
                next_obs = _clean_obs(next_obs)
                if done:
                    buf.dones[t, e] = 1.0
                    if (info or {}).get("truncated", False):
                        trunc_envs.append(e)
                        trunc_obs.append(next_obs)      # the final obs, not the reset one
                    buf.episode_returns.append(float(self._ep_ret[e]))
                    buf.episode_lengths.append(int(self._ep_len[e]))
                    self._recent_returns.append(float(self._ep_ret[e]))
                    self._recent_lengths.append(int(self._ep_len[e]))
                    self._ep_ret[e], self._ep_len[e] = 0.0, 0
                    next_obs = _clean_obs(env.reset())
                self._obs[e] = next_obs

            if trunc_envs:      # one batched forward for every env that hit a time limit this step
                _, v_final = self._forward(_stack_obs(trunc_obs))
                buf.trunc_values[t, trunc_envs] = v_final.cpu().numpy()

        assert buf is not None
        _, last_values = self._forward(_stack_obs(self._obs))
        buf.compute_gae(last_values.cpu().numpy(), cfg.gamma, cfg.gae_lambda)
        self.env_steps += buf.size
        return buf

    # ------------------------------------------------------------------ update

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        cfg = self.cfg
        self.policy.train(cfg.policy_train_mode)
        obs_all, data = buffer.flatten()
        n = buffer.size

        adv = data["advantages"]
        if cfg.normalize_advantages:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)       # over the whole rollout, not per minibatch
        var_y = float(np.var(data["returns"]))
        explained_var = float("nan") if var_y < 1e-12 else 1.0 - float(np.var(data["returns"] - data["values"])) / var_y

        dev = self.device
        rows: list[torch.Tensor] = []   # per minibatch: pg, vf, entropy, kl, clip_frac, aux, grad_norm
        epochs_run = 0
        for _ in range(cfg.n_epochs):
            order = self._rng.permutation(n)
            epoch_kl: list[torch.Tensor] = []
            for start in range(0, n, cfg.minibatch_size):
                idx = order[start:start + cfg.minibatch_size]
                obs = {k: torch.from_numpy(v[idx]).to(dev) for k, v in obs_all.items()}
                actions = torch.from_numpy(data["actions"][idx]).to(dev)
                old_logp = torch.from_numpy(data["log_probs"][idx]).to(dev)
                old_value = torch.from_numpy(data["values"][idx]).to(dev)
                mb_adv = torch.from_numpy(adv[idx]).to(dev)
                returns = torch.from_numpy(data["returns"][idx]).to(dev)

                dist, value = self.policy(obs)
                value = value.reshape(-1).float()
                new_logp = dist.log_prob(actions).float()
                entropy = dist.entropy().float().mean()
                if new_logp.shape != old_logp.shape or value.shape != returns.shape:
                    raise ValueError(f"policy returned log_prob {tuple(new_logp.shape)} / value {tuple(value.shape)}, "
                                     f"expected [{len(idx)}] each")

                log_ratio = new_logp - old_logp
                ratio = log_ratio.exp()
                pg_loss = -torch.min(mb_adv * ratio, mb_adv * ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range)).mean()
                if cfg.clip_range_vf is None:
                    vf_loss = ((value - returns) ** 2).mean()
                else:
                    v_clipped = old_value + (value - old_value).clamp(-cfg.clip_range_vf, cfg.clip_range_vf)
                    vf_loss = torch.max((value - returns) ** 2, (v_clipped - returns) ** 2).mean()
                loss = pg_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * entropy

                aux = torch.zeros((), device=dev)
                if self.aux_loss_fn is not None:
                    aux = self.aux_loss_fn(obs).float().reshape(())     # reshape enforces a scalar
                    loss = loss + aux

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.params, cfg.max_grad_norm if cfg.max_grad_norm > 0 else float("inf"), error_if_nonfinite=True)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()    # low-variance KL(old || new) estimator
                    clip_frac = ((ratio - 1).abs() > cfg.clip_range).float().mean()
                rows.append(torch.stack([pg_loss.detach(), vf_loss.detach(), entropy.detach(), approx_kl,
                                         clip_frac, aux.detach(), grad_norm.detach().float()]))
                epoch_kl.append(approx_kl)

            epochs_run += 1
            if cfg.target_kl is not None:
                kl = float(torch.stack(epoch_kl).mean())
                if kl > cfg.target_kl:
                    log.info("early stop after epoch %d/%d: approx KL %.4f > target %.4f", epochs_run, cfg.n_epochs, kl, cfg.target_kl)
                    break

        pg, vf, ent, kl, clip, aux, gn = torch.stack(rows).mean(0).tolist()
        self.iteration += 1
        return {"policy_loss": pg, "value_loss": vf, "entropy": ent, "approx_kl": kl, "clip_frac": clip,
                "explained_var": explained_var, "aux_loss": aux, "grad_norm": gn, "n_epochs_run": epochs_run}

    # ------------------------------------------------------------------ training loop

    def _lr_at(self, progress: float) -> float:
        return self.cfg.lr * (1.0 - progress) if self.cfg.lr_schedule == "linear" else self.cfg.lr

    def _episode_stats(self, buf: RolloutBuffer) -> dict[str, float]:
        rets, lens = self._recent_returns, self._recent_lengths
        return {
            "episodes": len(buf.episode_returns),       # finished during this rollout
            "ep_return_mean": float(np.mean(rets)) if rets else float("nan"),     # last RECENT_EPISODES episodes
            "ep_return_std": float(np.std(rets)) if rets else float("nan"),
            "ep_len_mean": float(np.mean(lens)) if lens else float("nan"),
        }

    def learn(
        self, total_steps: int, callback: Callable[[int, dict], bool | None] | None = None, log_every: int = 1,
    ) -> list[dict]:
        """Train until `env_steps` reaches total_steps (an absolute target, so after load() it
        resumes the lr schedule where it stopped). callback(iteration, stats) runs every
        iteration; returning False stops training. Returns one stats dict per iteration."""
        per_iter = self.cfg.n_envs * self.cfg.n_steps
        n_iters = max(1, math.ceil(total_steps / per_iter))
        history: list[dict] = []
        log.info("PPO: device %s, %d params, %d envs x %d steps = %d steps/iteration, %d iterations",
                 self.device, sum(p.numel() for p in self.params), self.cfg.n_envs, self.cfg.n_steps, per_iter, n_iters)
        while self.iteration < n_iters:
            lr = self._lr_at(self.iteration / n_iters)
            for g in self.optimizer.param_groups:
                g["lr"] = lr
            t0 = time.perf_counter()
            buf = self.collect_rollout()
            t1 = time.perf_counter()
            upd = self.update(buf)
            t2 = time.perf_counter()
            stats = {"iteration": self.iteration, "env_steps": self.env_steps, "lr": lr, **self._episode_stats(buf), **upd,
                     "steps_per_sec": per_iter / (t2 - t0), "rollout_sec": t1 - t0, "update_sec": t2 - t1}
            history.append(stats)
            if self.iteration % max(1, log_every) == 0:
                log.info("iter %d  steps %d  ep_ret %+.4f  ep_len %.0f  pg %+.4f  vf %.4f  ent %.3f  kl %.4f  clip %.3f  "
                         "ev %.3f  aux %.4f  epochs %d  %.0f steps/s",
                         stats["iteration"], stats["env_steps"], stats["ep_return_mean"], stats["ep_len_mean"],
                         stats["policy_loss"], stats["value_loss"], stats["entropy"], stats["approx_kl"], stats["clip_frac"],
                         stats["explained_var"], stats["aux_loss"], stats["n_epochs_run"], stats["steps_per_sec"])
            if callback is not None and callback(self.iteration, stats) is False:
                log.info("stopped by callback at iteration %d", self.iteration)
                break
        return history

    # ------------------------------------------------------------------ checkpoints

    def save(self, path: str | Path) -> None:
        """Policy + optimizer + counters + config. extra_params that are not part of the policy
        are stored by value, in the order given (buffers of modules outside `policy` are not)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "extra_params": [p.detach().cpu() for p in self._extra_only],
            "optimizer": self.optimizer.state_dict(),
            "iteration": self.iteration, "env_steps": self.env_steps, "cfg": asdict(self.cfg),
        }, path)

    def load(self, path: str | Path) -> None:
        """Restore a checkpoint. The current cfg is kept (so lr etc. can be changed on resume);
        envs are reset on the next rollout."""
        state = torch.load(Path(path), map_location=self.device, weights_only=True)
        saved_extra = state["extra_params"]
        if len(saved_extra) != len(self._extra_only):
            raise ValueError(f"checkpoint has {len(saved_extra)} extra params, this PPO has {len(self._extra_only)}")
        self.policy.load_state_dict(state["policy"])
        with torch.no_grad():
            for p, v in zip(self._extra_only, saved_extra):
                p.copy_(v)
        self.optimizer.load_state_dict(state["optimizer"])
        self.iteration, self.env_steps = int(state["iteration"]), int(state["env_steps"])
        self._obs = None
        self._recent_returns.clear()
        self._recent_lengths.clear()
        cur = asdict(self.cfg)
        changed = {k: (state["cfg"].get(k), v) for k, v in cur.items() if state["cfg"].get(k) != v}
        if changed:
            log.warning("loaded checkpoint was trained with a different config (saved, current): %s", changed)
        log.info("loaded %s (iteration %d, %d env steps)", path, self.iteration, self.env_steps)


@torch.no_grad()
def evaluate(
    policy: nn.Module, env: Env, n_episodes: int = 10, deterministic: bool = True,
    seed: int | None = None, device: torch.device | str | None = None, max_steps: int | None = None,
) -> dict:
    """Run complete episodes and report raw (unscaled) returns and lengths.

    deterministic=True acts with dist.mode, otherwise samples. With a seed, episode i is reset
    with seed + i, so two policies can be compared on identical episodes. max_steps is an
    optional safety cap for envs without a time limit. The policy's train/eval mode is restored.
    """
    if device is None:
        device = next(policy.parameters(), torch.empty(0)).device
    was_training = policy.training
    policy.eval()
    returns, lengths = [], []
    try:
        for i in range(n_episodes):
            obs = _clean_obs(env.reset(seed=None if seed is None else seed + i))
            total, steps, done = 0.0, 0, False
            while not done and (max_steps is None or steps < max_steps):
                dist, _ = policy(_to_torch(_stack_obs([obs]), torch.device(device)))
                action = dist.mode if deterministic else dist.sample()
                obs, reward, done, _ = env.step(action[0].float().cpu().numpy())
                obs = _clean_obs(obs)
                total += float(reward)
                steps += 1
            returns.append(total)
            lengths.append(steps)
    finally:
        policy.train(was_training)
    return {
        "return_mean": float(np.mean(returns)), "return_std": float(np.std(returns)),
        "length_mean": float(np.mean(lengths)), "length_std": float(np.std(lengths)),
        "returns": returns, "lengths": lengths,
    }
