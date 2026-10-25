"""Trading environment for the RL decision layer, built on our simulator.

Every `decision_every` minutes the agent sees:
  * forecast: Mark I's return quantiles for every coin at every horizon, divided by the
    round-trip cost, so 1.0 means "this move exactly pays for a round trip"
  * coins:    per-coin state (current weight, P&L since entry and time held, both
    cost-relative, volatility level, tradable flag)
  * portfolio: cash share, gross exposure, drawdown from peak, time left in the episode
  * t:        the decision row, so the stage-2 policy can re-run Mark I with gradients

It answers with N+1 logits (coins + cash). The environment turns them into long-only
weights with a softmax, then applies the safety rails the agent cannot override:
  * per-coin cap and gross cap
  * stop-loss: a coin that falls more than `stop_loss_sigmas` forecast-sigmas below its entry
    is forced to zero and locked out for `stop_cooldown` decisions
  * drawdown brake: exposure scaled down while equity is far below its peak
  * minimum trade size, so tiny weight wiggles do not burn fees

Reward = log equity growth over the step (after every fee, spread and impact cost the
simulator charges) minus `dd_penalty` times any increase in drawdown.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mark1.data.panel import Panel
from mark1.decide.engine import moments_from_quantiles
from mark1.sim import SimConfig, Simulator


@dataclass
class EnvConfig:
    decision_every: int = 15          # minutes between decisions
    episode_minutes: int = 3 * 1440   # episode length (truncated, not terminal)
    max_weight: float = 0.25          # rail: per coin
    max_gross: float = 1.0            # rail: long-only, no leverage
    min_trade_weight: float = 0.01    # rail: skip rebalances smaller than this share of equity
    stop_loss_sigmas: float = 3.0     # rail: forced exit beyond this many forecast-sigmas of loss
    stop_cooldown: int = 8            # decisions a stopped coin stays locked out
    drawdown_brake: float = 0.15      # rail: start cutting exposure beyond this drawdown
    drawdown_floor: float = 0.25      # rail: exposure multiplier at twice the brake level
    dd_penalty: float = 1.0           # reward: penalty per unit increase in drawdown
    impact_estimate_bps: float = 2.0  # used only to express forecasts relative to cost
    dim_dropout: float = 0.0          # chance each step that a whole input dimension is hidden from the agent
    n_dims: int = 5                   # own history, other coins, market, interactions, forecast


class Forecasts:
    """Precomputed forecasts on a regular decision grid: rows[k] = start + k * every."""

    def __init__(self, rows: np.ndarray, quantiles: np.ndarray, ready: np.ndarray, taus: np.ndarray, levels: list[float]):
        self.rows, self.quantiles, self.ready = rows, quantiles, ready     # [R], [R, K, N, Q], [R, N]
        self.taus, self.levels = np.asarray(taus), list(levels)
        self.every = int(rows[1] - rows[0]) if len(rows) > 1 else 1

    def index(self, t: int) -> int:
        k = (t - int(self.rows[0])) // self.every
        if k < 0 or k >= len(self.rows) or self.rows[k] != t:
            raise KeyError(f"no forecast for row {t}")
        return k


def round_trip_cost(sim_cfg: SimConfig, n_assets: int, impact_bps: float) -> np.ndarray:
    hs = np.broadcast_to(np.asarray(sim_cfg.half_spread_bps, dtype=float), (n_assets,))
    return 2 * (sim_cfg.taker_fee_bps + hs + impact_bps) * 1e-4


class TradingEnv:
    """Gym-style env: reset(seed) -> obs, step(action) -> (obs, reward, done, info)."""

    N_COIN_FEATURES = 6
    N_PORTFOLIO_FEATURES = 4

    def __init__(self, panel: Panel, forecasts: Forecasts | None, sigma: np.ndarray, start: int, end: int,
                 sim_cfg: SimConfig | None = None, cfg: EnvConfig | None = None, seed: int = 0,
                 full_period: bool = False, ready: np.ndarray | None = None):
        """start/end bound the split; episodes are drawn inside it. `full_period=True` makes one
        deterministic episode over the whole range (for evaluation).

        With precomputed `forecasts` the observation carries them (stage 1/2 policies). Without
        them (the fused agent computes its own) pass `ready` [T, N], the feature-readiness mask."""
        self.panel, self.fc, self.sigma = panel, forecasts, sigma
        self.sim_cfg, self.cfg = sim_cfg or SimConfig(), cfg or EnvConfig()
        if forecasts is None and ready is None:
            raise ValueError("pass either precomputed forecasts or a ready mask")
        if forecasts is not None and forecasts.every != self.cfg.decision_every:
            raise ValueError("forecast grid and decision_every differ")
        self.ready_mask = ready
        self.sim = Simulator(panel, self.sim_cfg)
        self.rng = np.random.default_rng(seed)
        self.full_period = full_period
        self.n = panel.n_assets
        self.cost = round_trip_cost(self.sim_cfg, self.n, self.cfg.impact_estimate_bps)
        # decision rows usable as episode starts: on the decision grid and inside [start, end)
        rows = forecasts.rows if forecasts is not None else np.arange(start, end, self.cfg.decision_every)
        self.grid = rows[(rows >= start) & (rows < end - self.cfg.decision_every - 1)]
        if len(self.grid) < 2:
            raise ValueError("split too short for this decision grid")
        self.split_end = end
        self._mu_sd_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------ episode control
    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        steps = max(1, self.cfg.episode_minutes // self.cfg.decision_every)
        if self.full_period:
            k0, self.n_steps = 0, len(self.grid) - 1
        else:
            k0 = int(self.rng.integers(0, max(1, len(self.grid) - steps)))
            self.n_steps = min(steps, len(self.grid) - 1 - k0)
        self.t0 = int(self.grid[k0])
        self.k = 0
        end = min(self.t0 + self.n_steps * self.cfg.decision_every + 2, self.split_end)
        self.sim.reset(start=self.t0, end=end)
        self.peak = self.sim.equity
        self.entry_price = np.full(self.n, np.nan)
        self.entry_risk = np.full(self.n, np.nan)
        self.entry_t = np.full(self.n, -1)
        self.lockout = np.zeros(self.n, dtype=int)
        return self._obs()

    @property
    def t(self) -> int:
        return self.sim.t

    def step(self, action: np.ndarray):
        eq0 = self.sim.equity
        dd0 = 1 - eq0 / self.peak
        weights, rails = self.target_weights(np.asarray(action, dtype=np.float64))
        prev = self.sim.weights
        if np.abs(weights - prev).max() >= self.cfg.min_trade_weight:
            self.sim.rebalance(weights)
        self._track_entries(weights, prev)
        for _ in range(self.cfg.decision_every):
            if self.sim.done:
                break
            self.sim.step()
        eq1 = self.sim.equity
        self.peak = max(self.peak, eq1)
        dd1 = 1 - eq1 / self.peak
        reward = float(np.log(max(eq1, 1e-9) / eq0) - self.cfg.dd_penalty * max(0.0, dd1 - dd0))
        self.k += 1
        self.lockout = np.maximum(self.lockout - 1, 0)
        done = self.k >= self.n_steps or self.sim.done
        info = {"equity": eq1, "drawdown": dd1, "weights": weights, "rails": rails,
                "truncated": done}   # episodes end on a time limit, never on a true terminal state
        return self._obs(), reward, done, info

    # ------------------------------------------------------------------ actions and rails
    def target_weights(self, logits: np.ndarray) -> tuple[np.ndarray, dict]:
        """Softmax over tradable coins + cash, then the safety rails."""
        ready = self._ready(self.t)
        z = np.where(np.append(ready, True), logits, -np.inf)
        z = z - z.max()
        w = np.exp(z)
        w = w / w.sum()
        coins = w[:-1].copy()
        rails = {"capped": 0, "stopped": 0, "brake": 1.0}

        over = coins > self.cfg.max_weight
        rails["capped"] = int(over.sum())
        coins = np.minimum(coins, self.cfg.max_weight)

        price = self.panel.close[self.t]
        held = self.sim.weights > 1e-6
        loss = np.where(held & np.isfinite(self.entry_price), np.log(price / self.entry_price), 0.0)
        stop = held & (loss < -self.cfg.stop_loss_sigmas * np.nan_to_num(self.entry_risk, nan=np.inf))
        self.lockout[stop] = self.cfg.stop_cooldown
        rails["stopped"] = int(stop.sum())
        coins[self.lockout > 0] = 0.0

        gross = coins.sum()
        if gross > self.cfg.max_gross:
            coins *= self.cfg.max_gross / gross
        dd = 1 - self.sim.equity / self.peak
        if dd > self.cfg.drawdown_brake:
            frac = min(1.0, (dd - self.cfg.drawdown_brake) / self.cfg.drawdown_brake)
            mult = 1.0 - frac * (1.0 - self.cfg.drawdown_floor)
            coins *= mult
            rails["brake"] = mult
        # Coins we cannot trade right now keep their current weight.
        coins = np.where(ready, coins, self.sim.weights)
        return coins, rails

    def _ready(self, t: int) -> np.ndarray:
        if self.fc is not None:
            return self.fc.ready[self._safe_index(t)] & self.panel.valid[t]
        return np.asarray(self.ready_mask[t]) & self.panel.valid[t]

    def _track_entries(self, w: np.ndarray, prev: np.ndarray) -> None:
        price = self.panel.close[self.t]
        new = (w > 1e-6) & (prev <= 1e-6)
        gone = w <= 1e-6
        # risk of a new position over ~4 hours: the forecast's spread if we have one, else realised volatility
        if self.fc is not None:
            _, sd = self._moments(self._safe_index(self.t))
            risk = sd[int(np.argmin(np.abs(self.fc.taus - 240)))]
        else:
            risk = self.sigma[self.t] * np.sqrt(240.0)
        self.entry_price[new], self.entry_risk[new], self.entry_t[new] = price[new], risk[new], self.t
        self.entry_price[gone], self.entry_risk[gone], self.entry_t[gone] = np.nan, np.nan, -1

    def _moments(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        if k not in self._mu_sd_cache:
            if len(self._mu_sd_cache) > 4096:
                self._mu_sd_cache.clear()
            self._mu_sd_cache[k] = moments_from_quantiles(self.fc.quantiles[k], self.fc.levels)
        return self._mu_sd_cache[k]

    # ------------------------------------------------------------------ observations
    def _obs(self) -> dict[str, np.ndarray]:
        t = self.t
        ready = self._ready(t)

        w = self.sim.weights
        price = self.panel.close[t]
        held = np.isfinite(self.entry_price)
        pnl = np.where(held, np.log(price / np.where(held, self.entry_price, 1.0)), 0.0) / self.cost
        age = np.where(self.entry_t >= 0, (t - self.entry_t) / 60.0, 0.0)
        coins = np.stack([
            w,
            np.clip(pnl, -20, 20),
            np.log1p(age),
            np.log(np.maximum(self.sigma[t], 1e-6)) + 7.5,          # roughly centred volatility level
            (self.lockout > 0).astype(float),
            ready.astype(float),
        ], axis=1).astype(np.float32)
        eq = self.sim.equity
        portfolio = np.array([
            max(0.0, 1 - w.sum()),
            w.sum(),
            1 - eq / self.peak,
            1 - self.k / max(1, self.n_steps),
        ], dtype=np.float32)
        obs = {"t": np.array([t], dtype=np.int64), "coins": coins, "portfolio": portfolio,
               "mask": ready.astype(bool), "dim_mask": self._dim_mask()}
        if self.fc is not None:
            q = self.fc.quantiles[self._safe_index(t)]               # [K, N, Q] log returns
            obs["forecast"] = (q / self.cost[None, :, None]).astype(np.float32)
        return obs

    def _dim_mask(self) -> np.ndarray:
        """Which input dimensions the agent may use this step (all of them in evaluation)."""
        p = self.cfg.dim_dropout
        if p <= 0 or self.full_period:
            return np.ones(self.cfg.n_dims, dtype=bool)
        keep = self.rng.random(self.cfg.n_dims) >= p
        if not keep.any():
            keep[self.rng.integers(self.cfg.n_dims)] = True
        return keep

    def _safe_index(self, t: int) -> int:
        # The simulator can stop a few bars past the grid at the very end of a split; use the
        # latest forecast at or before t.
        k = (t - int(self.fc.rows[0])) // self.fc.every
        return int(np.clip(k, 0, len(self.fc.rows) - 1))
