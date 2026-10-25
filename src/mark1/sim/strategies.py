"""Tiny reference strategies, used for benchmarks, tests and speed checks."""

from __future__ import annotations

import numpy as np

from mark1.sim.orders import Decision


class EqualWeightBuyAndHold:
    """Buy an equal-weight basket of the assets valid at the first call, then do nothing.

    Stateless: it buys only while the account is flat with no open orders, so it is safe to
    reuse across runs. Pair it with a large `decision_every` to call it once.
    """

    def __init__(self, gross: float = 1.0):
        self.gross = gross

    def __call__(self, t, market, portfolio):
        if portfolio.n_open_orders or np.any(portfolio.positions != 0.0):
            return None
        valid = market.valid[-1]
        if not valid.any():
            return None
        return Decision(target_weights=valid / valid.sum() * self.gross)


class FixedWeights:
    """Re-target the same weights at every decision step (drift is rebalanced away)."""

    def __init__(self, weights: np.ndarray):
        self.weights = np.asarray(weights, dtype=np.float64)

    def __call__(self, t, market, portfolio):
        return self.weights
