"""Shared helpers for the simulator tests (no tests in here)."""

from __future__ import annotations

import numpy as np

from mark1.data.panel import Panel
from mark1.sim import SimConfig

# No costs at all: fills happen exactly at the reference price.
FREE = SimConfig(taker_fee_bps=0, maker_fee_bps=0, half_spread_bps=0, impact_coef=0, min_notional=0,
                 participation_cap=1.0)


def make_panel(o, h=None, l=None, c=None, v=1e6, valid=None, symbols=None) -> Panel:
    """Panel from explicit per-bar arrays ([T] or [T, N]); h/l/c default to the open."""
    o = np.asarray(o, dtype=np.float64)
    if o.ndim == 1:
        o = o[:, None]
    T, N = o.shape

    def arr(x, default):
        if x is None:
            return default.copy()
        x = np.asarray(x, dtype=np.float64)
        return np.broadcast_to(x[:, None] if x.ndim == 1 else x, (T, N)).copy()

    h, l, c = arr(h, o), arr(l, o), arr(c, o)
    vol = np.broadcast_to(np.asarray(v, dtype=np.float64).reshape(-1, 1) if np.ndim(v) == 1 else v,
                          (T, N)).copy()
    ok = np.ones((T, N), dtype=bool) if valid is None else np.broadcast_to(
        np.asarray(valid, dtype=bool).reshape(T, -1), (T, N)).copy()
    return Panel(
        times=np.datetime64("2024-01-01T00:00:00", "ns") + np.arange(T).astype("timedelta64[m]"),
        symbols=symbols or [f"S{i}USDT" for i in range(N)],
        open=o, high=h, low=l, close=c, volume=vol, quote_volume=vol * c,
        trades=np.round(vol / 10), taker_buy_volume=vol / 2, valid=ok,
    )


def flat_panel(T=10, price=100.0, **kw) -> Panel:
    """Constant-price panel."""
    return make_panel(np.full(T, price), **kw)


def bounce_panel(T=400, spread_bps=10.0, vol_bps=1.0, seed=0, price=100.0, volume=1e5) -> Panel:
    """One asset whose closes bounce between bid and ask around a slow random-walk mid.

    The bar range is exactly the quoted spread, so both spread estimators should recover
    `spread_bps` (full spread) when the mid barely moves.
    """
    rng = np.random.default_rng(seed)
    mid = price * np.exp(np.cumsum(rng.normal(0, vol_bps * 1e-4, T)))
    half = spread_bps * 1e-4 / 2
    u = rng.choice([-1.0, 1.0], size=T)
    open_ = np.concatenate([[mid[0]], mid[:-1]])
    return make_panel(open_, h=mid * (1 + half), l=mid * (1 - half), c=mid * (1 + half * u), v=volume)
