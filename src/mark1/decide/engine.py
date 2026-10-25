"""Decision engine: forecast curves -> target portfolio weights.

Input at each decision time: for every coin, quantiles of the future log return at a grid of
horizons (the model's forecast curve). Output: long-only target weights.

The logic, per coin, with c = one-way trading cost (fee + half spread + impact estimate):

  * Expected move and risk at each horizon come from the quantiles
    (mean via Pearson-Tukey weights, spread via the 5%-95% range).
  * ENTER only if, at some horizon, the expected move beats a round trip:  mu(tau) - 2c > margin.
  * HOLD while some horizon still has mu(tau) > 0. The exit fee is paid whether we sell now
    or later, so it does not count against holding.
  * DIP EXIT: sell a held coin if a nearer horizon predicts a drop bigger than selling and
    buying back:  mu(tau_short) < -(2c + margin). This is the "down 0.5% in an hour,
    up 1.3% tomorrow" case: worth acting on only when the dip beats the round-trip cost.
  * SIZE with fractional Kelly on the horizon with the best risk-adjusted edge:
    w = kelly_fraction * edge / sigma^2, capped per coin, gross exposure capped at max_gross.
  * Risk overlays: position stop-loss in units of forecast risk, and a portfolio drawdown
    brake that scales exposure down while equity is far below its peak.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DecisionConfig:
    kelly_fraction: float = 0.25        # quarter Kelly: forecasts are noisy, full Kelly is reckless
    max_weight: float = 0.20            # per coin, fraction of equity
    max_gross: float = 1.00             # long-only spot, no leverage
    entry_margin: float = 0.0005        # extra log-return an entry must clear beyond round-trip cost (5 bps)
    dip_margin: float = 0.0010          # extra drop a dip exit must clear beyond round-trip cost (10 bps)
    dip_max_horizon: int = 240          # dips are only acted on within this many minutes
    min_trade_weight: float = 0.01      # ignore rebalances smaller than this (fraction of equity)
    stop_loss_sigmas: float = 3.0       # exit if a position loses more than this many forecast-sigmas
    drawdown_brake: float = 0.15        # start scaling exposure down beyond this drawdown from peak
    drawdown_floor: float = 0.25        # exposure multiplier when drawdown reaches 2 x drawdown_brake
    min_horizon: int = 5                # ignore horizons shorter than this for entries (too noisy for costs)


@dataclass
class DecisionInfo:
    expected: np.ndarray                # [K, N] expected log return per horizon
    risk: np.ndarray                    # [K, N] std of log return per horizon
    best_tau: np.ndarray                # [N] horizon with the best risk-adjusted edge
    action: list[str] = field(default_factory=list)   # per coin: enter / hold / exit / dip_exit / stop / flat


def moments_from_quantiles(q: np.ndarray, levels: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Approximate mean and std from quantiles [..., Q] (needs the 5%, 50% and 95% levels)."""
    i05, i50, i95 = levels.index(0.05), levels.index(0.5), levels.index(0.95)
    mean = 0.185 * q[..., i05] + 0.630 * q[..., i50] + 0.185 * q[..., i95]   # Pearson-Tukey
    std = np.maximum((q[..., i95] - q[..., i05]) / 3.29, 1e-6)
    return mean, std


class DecisionEngine:
    def __init__(self, cfg: DecisionConfig | None = None):
        self.cfg = cfg or DecisionConfig()
        self.peak_equity: float | None = None
        self.entry_price: np.ndarray | None = None
        self.entry_risk: np.ndarray | None = None

    def reset(self) -> None:
        self.peak_equity, self.entry_price, self.entry_risk = None, None, None

    def decide(
        self,
        quantiles: np.ndarray,        # [K, N, Q] future log-return quantiles
        taus: np.ndarray,             # [K] minutes
        levels: list[float],          # quantile levels, e.g. [0.05, ..., 0.95]
        current_weights: np.ndarray,  # [N]
        cost_oneway: np.ndarray,      # [N] one-way cost as a log return (e.g. 0.0045 for 45 bps)
        ready: np.ndarray,            # [N] bool
        prices: np.ndarray,           # [N] latest close
        equity: float,
    ) -> tuple[np.ndarray, DecisionInfo]:
        c = self.cfg
        K, N = quantiles.shape[:2]
        if self.entry_price is None:
            self.entry_price = np.full(N, np.nan)
            self.entry_risk = np.full(N, np.nan)
        self.peak_equity = equity if self.peak_equity is None else max(self.peak_equity, equity)

        mu, sd = moments_from_quantiles(quantiles, levels)            # [K, N]
        taus = np.asarray(taus)
        usable = (taus >= c.min_horizon)[:, None]
        round_trip = 2 * cost_oneway[None, :]

        edge_enter = np.where(usable, mu - round_trip, -np.inf)       # [K, N]
        score = edge_enter / sd
        best_k = np.argmax(score, axis=0)
        cols = np.arange(N)
        best_edge = edge_enter[best_k, cols]
        best_sd = sd[best_k, cols]

        short = (taus <= c.dip_max_horizon)[:, None]
        worst_short = np.where(short, mu, np.inf).min(axis=0)          # most negative near-term move
        any_positive = (np.where(usable, mu, -np.inf) > 0).any(axis=0)

        held = current_weights > c.min_trade_weight / 2
        target = np.zeros(N)
        actions = []
        for i in range(N):
            if not ready[i]:
                target[i] = current_weights[i] if held[i] else 0.0    # cannot trade, keep as is
                actions.append("hold" if held[i] else "flat")
                continue
            kelly_w = c.kelly_fraction * max(best_edge[i], 0.0) / best_sd[i] ** 2
            if held[i]:
                stop = self._stopped(i, prices[i])
                if stop:
                    actions.append("stop")
                elif worst_short[i] < -(2 * cost_oneway[i] + c.dip_margin):
                    actions.append("dip_exit")
                elif any_positive[i]:
                    # keep at least the current size unless the edge says to grow it
                    target[i] = min(c.max_weight, max(current_weights[i], kelly_w) if best_edge[i] > 0 else current_weights[i])
                    actions.append("hold")
                else:
                    actions.append("exit")
            elif best_edge[i] > c.entry_margin and worst_short[i] > -(2 * cost_oneway[i] + c.dip_margin):
                target[i] = min(c.max_weight, kelly_w)
                actions.append("enter" if target[i] >= c.min_trade_weight else "flat")
                if target[i] < c.min_trade_weight:
                    target[i] = 0.0
            else:
                actions.append("flat")

        gross = target.sum()
        if gross > c.max_gross:
            target *= c.max_gross / gross
        target *= self._drawdown_multiplier(equity)
        # Small changes are not worth the fees.
        small = np.abs(target - current_weights) < c.min_trade_weight
        target = np.where(small & (target > 0), current_weights, target)

        for i in range(N):
            if target[i] > 0 and not held[i]:
                self.entry_price[i], self.entry_risk[i] = prices[i], best_sd[i]
            elif target[i] == 0:
                self.entry_price[i], self.entry_risk[i] = np.nan, np.nan

        return target, DecisionInfo(expected=mu, risk=sd, best_tau=taus[best_k], action=actions)

    def _stopped(self, i: int, price: float) -> bool:
        if np.isnan(self.entry_price[i]):
            return False
        loss = np.log(price / self.entry_price[i])
        return loss < -self.cfg.stop_loss_sigmas * self.entry_risk[i]

    def _drawdown_multiplier(self, equity: float) -> float:
        c = self.cfg
        dd = 1 - equity / self.peak_equity
        if dd <= c.drawdown_brake:
            return 1.0
        # Linear from 1 at the brake level down to the floor at twice the brake level.
        frac = min(1.0, (dd - c.drawdown_brake) / c.drawdown_brake)
        return 1.0 - frac * (1.0 - c.drawdown_floor)
