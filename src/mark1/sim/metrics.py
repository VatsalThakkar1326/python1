"""Performance metrics from a per-minute equity curve (crypto trades 24/7, so no trading calendar)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from mark1.config import MINUTES_PER_YEAR

_ZERO_VOL = 1e-12  # per-period std below this is treated as "no volatility" (float noise)


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of consecutive True values."""
    if not mask.any():
        return 0
    edges = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
    return int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max())


def drawdown_series(equity: np.ndarray) -> np.ndarray:
    """Equity relative to its running peak minus one (<= 0)."""
    eq = np.asarray(equity, dtype=np.float64)
    return eq / np.maximum.accumulate(eq) - 1.0


def compute_metrics(
    equity: Sequence[float] | np.ndarray | pd.Series,
    *,
    exposure: np.ndarray | pd.Series | None = None,
    fees_paid: float = 0.0,
    slippage_cost: float = 0.0,
    traded_notional: float = 0.0,
    n_trades: int = 0,
    round_trip_pnls: Sequence[float] | None = None,
    periods_per_year: float = MINUTES_PER_YEAR,
    risk_free: float = 0.0,
) -> dict[str, float]:
    """Summary statistics of an equity curve sampled once per period (default: per minute).

    Conventions: returns are simple per-period returns; Sharpe and Sortino are annualised with
    sqrt(periods_per_year) on excess-over-`risk_free` (annual rate, default 0); max drawdown is
    negative (-0.25 = 25% below the peak); Calmar = CAGR / |max drawdown|; turnover_annual is
    traded notional (buys + sells) per year as a multiple of average equity; durations count
    bars (minutes) with equity below its running peak. Ratios with a zero denominator are nan.
    """
    eq = np.asarray(equity, dtype=np.float64)
    n = len(eq)
    if n < 2:
        raise ValueError("need at least two equity points")
    years = (n - 1) / periods_per_year
    e0, e1 = float(eq[0]), float(eq[-1])
    total_return = e1 / e0 - 1.0
    if e1 <= 0:
        cagr = -1.0
    elif years <= 0:
        cagr = float("nan")
    else:  # exp/log form so very short windows overflow to inf instead of raising
        log_growth = math.log(e1 / e0) / years
        cagr = math.expm1(log_growth) if log_growth < 700.0 else float("inf")

    with np.errstate(divide="ignore", invalid="ignore"):
        r = eq[1:] / eq[:-1] - 1.0
    r = np.where(np.isfinite(r), r, 0.0)
    rf = risk_free / periods_per_year
    mean_ex = float(r.mean()) - rf
    std = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    ann = math.sqrt(periods_per_year)
    vol_ok = std > _ZERO_VOL
    downside = float(np.sqrt(np.mean(np.minimum(r - rf, 0.0) ** 2)))

    dd = drawdown_series(eq)
    max_dd = float(dd.min())
    dd_bars = _longest_run(dd < 0)

    pnls = np.asarray(round_trip_pnls if round_trip_pnls is not None else [], dtype=np.float64)
    exp = None if exposure is None else np.asarray(exposure, dtype=np.float64)
    avg_equity = float(eq.mean())
    return {
        "n_bars": n - 1,
        "years": years,
        "initial_equity": e0,
        "final_equity": e1,
        "total_return": total_return,
        "cagr": cagr,
        "ann_vol": std * ann,
        "sharpe": mean_ex / std * ann if vol_ok else float("nan"),
        "sortino": mean_ex / downside * ann if downside > _ZERO_VOL else float("nan"),
        "max_drawdown": max_dd,
        "max_dd_duration_bars": dd_bars,
        "max_dd_duration_days": dd_bars / (periods_per_year / 365.0),
        "calmar": cagr / abs(max_dd) if max_dd < 0 else float("nan"),
        "turnover_annual": traded_notional / avg_equity / years if years > 0 else float("nan"),
        "traded_notional": float(traded_notional),
        "fees_paid": float(fees_paid),
        "slippage_cost": float(slippage_cost),
        "total_costs": float(fees_paid + slippage_cost),
        "n_trades": int(n_trades),
        "n_round_trips": int(len(pnls)),
        "win_rate": float((pnls > 0).mean()) if len(pnls) else float("nan"),
        "avg_exposure": float(np.nanmean(exp)) if exp is not None and len(exp) else float("nan"),
    }


_PCT = {"total_return", "cagr", "ann_vol", "max_drawdown", "win_rate", "avg_exposure"}
_MONEY = {"initial_equity", "final_equity", "traded_notional", "fees_paid", "slippage_cost",
          "total_costs"}
_INT = {"n_bars", "n_trades", "n_round_trips", "max_dd_duration_bars"}
_LABELS = {
    "n_bars": "Bars", "years": "Years", "initial_equity": "Initial equity", "final_equity": "Final equity",
    "total_return": "Total return", "cagr": "CAGR", "ann_vol": "Annualised vol", "sharpe": "Sharpe",
    "sortino": "Sortino", "max_drawdown": "Max drawdown", "max_dd_duration_bars": "Max DD duration (bars)",
    "max_dd_duration_days": "Max DD duration (days)", "calmar": "Calmar",
    "turnover_annual": "Turnover (x equity / yr)", "traded_notional": "Traded notional",
    "fees_paid": "Fees paid", "slippage_cost": "Slippage + impact", "total_costs": "Total costs",
    "n_trades": "Fills", "n_round_trips": "Round trips", "win_rate": "Win rate (round trips)",
    "avg_exposure": "Avg exposure",
}


def format_metrics(metrics: dict[str, float], title: str | None = None) -> str:
    """Aligned plain-text table of a metrics dict."""
    lines = [title] if title else []
    width = max(len(_LABELS.get(k, k)) for k in metrics)
    for key, v in metrics.items():
        if isinstance(v, float) and math.isnan(v):
            s = "n/a"
        elif key in _PCT:
            s = f"{v * 100:,.2f}%"
        elif key in _MONEY:
            s = f"{v:,.2f}"
        elif key in _INT:
            s = f"{int(v):,}"
        else:
            s = f"{v:,.3f}"
        lines.append(f"{_LABELS.get(key, key):<{width}}  {s:>16}")
    return "\n".join(lines)
