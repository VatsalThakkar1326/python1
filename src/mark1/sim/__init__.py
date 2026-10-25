"""Mark I market simulator: a minute-bar exchange model, run loop and metrics.

Quick start::

    from mark1.data.panel import synthetic_panel
    from mark1.sim import SimConfig, run_backtest, Decision

    def strategy(t, market, portfolio):
        return Decision(target_weights=[0.5, 0.0, 0.0])   # or just return the array

    result = run_backtest(synthetic_panel(), strategy, SimConfig(), decision_every=60)
    print(result.summary())

See `mark1.sim.engine` for the execution model and its limits.
"""

from mark1.sim.costs import SimConfig, cost_inputs, sqrt_impact_bps
from mark1.sim.engine import REBALANCE_TAG, Simulator, StepInfo
from mark1.sim.metrics import compute_metrics, format_metrics
from mark1.sim.orders import (Decision, Fill, Order, OrderStatus, OrderType, Side, TimeInForce)
from mark1.sim.portfolio import Portfolio, PortfolioView
from mark1.sim.runner import (BacktestResult, MarketView, Strategy, buy_and_hold_benchmark,
                              cash_benchmark, run_backtest)
from mark1.sim.strategies import EqualWeightBuyAndHold, FixedWeights

__all__ = [
    "BacktestResult", "Decision", "EqualWeightBuyAndHold", "Fill", "FixedWeights", "MarketView",
    "Order", "OrderStatus", "OrderType", "Portfolio", "PortfolioView", "REBALANCE_TAG", "Side",
    "SimConfig", "Simulator", "StepInfo", "Strategy", "TimeInForce", "buy_and_hold_benchmark",
    "cash_benchmark", "compute_metrics", "cost_inputs", "format_metrics", "run_backtest",
    "sqrt_impact_bps",
]
