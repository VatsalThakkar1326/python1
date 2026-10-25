"""Minute-bar exchange simulator: steppable core with a realistic execution model.

Timing (no look-ahead)
----------------------
Row t is the bar opening at times[t]. After `reset(start)` the simulator sits at t = start: the
decision point at the CLOSE of bar `start`. `submit()` orders there; they become active at bar
t + 1 + latency_bars (default: the next bar). `step()` processes the next bar b = t + 1 (fills for
active orders), moves to t = b and marks the portfolio to the bar's close. A strategy deciding
at t may only look at rows <= t. Equity point k of the curve is the mark at the close of bar
start + k (point 0 is the initial cash).

What is modelled
----------------
* Market orders fill at the bar OPEN, worse by the half-spread and a square-root market impact
  (`SimConfig` in `costs.py`), and pay the taker fee. Sigma and the spread estimate use bars
  strictly before the fill bar; the fill bar's volume is the liquidity traded against.
* Participation cap: per asset and bar at most `participation_cap` * bar volume fills in total
  (all orders together). The unfilled rest of a market order carries over to later bars; IOC
  orders get exactly one bar (the first valid one after activation) and the remainder is cancelled.
* LIMIT / TAKE_PROFIT: filled only if the bar trades strictly THROUGH the price (buy: low <
  limit, sell: high > limit). Touching is not enough, a crude proxy for queue position. A
  resting order fills at its limit price (even if the bar opens through it), with the maker
  fee, no spread or impact, still subject to the participation cap. A limit that is already
  through the market on the bar it becomes active is marketable: it fills at the open like a
  market order (spread, impact, taker fee).
* STOP (stop-market): triggers when the bar trades at or through the stop (sell: low <= stop,
  buy: high >= stop), then behaves like a market order at min(stop, open) (sell) / max(stop,
  open) (buy), with spread, impact and the taker fee. A gap through the stop fills at the open.
  Any remainder left by the participation cap continues as a market order on later bars.
* Intrabar ordering is unknowable from OHLC, so ambiguity is resolved against the trader. Per
  bar: first everything that executes at the open (market orders, gapped stops and gapped
  limits; their order is certain), then orders triggered later in the bar. Within each group
  SELLS go before BUYS (cash is released first, and a sell never relies on a buy that filled in
  the same group), and on one side stops go before market orders before limits / take-profits.
  So when a stop-loss and a take-profit on one position both trade inside a bar, the stop
  executes and the take-profit finds no position (unless linked by `oco`, in which case it is
  cancelled). A take-profit that gaps through at the open is certain to fill first, though.
* Invalid bars (`valid` False): no fills, orders wait (expiry still counts bars).
* Cash: no leverage by default; buys that cost more than the cash available are scaled down and
  the rest is cancelled. Long-only by default: sells are clamped to the position held. Orders
  below `min_notional` are rejected on submission; `qty_step` floors quantities to a lot size.
* Latency delays both new orders and cancel requests by `latency_bars` bars.
* Mark-to-market uses bar closes; an asset without a valid bar keeps its last valid close.

What is NOT modelled
--------------------
* Queue position beyond trade-through, hidden or iceberg liquidity, order book depth, tick size.
* Permanent impact or impact decay: impact is temporary and per bar, and our fills never change
  the bars that follow. Cross-asset impact is ignored.
* Which of high / low came first inside a bar, and the volume actually available after a stop
  triggers (the whole bar's volume is available).
* Fee tiers that depend on traded volume, fees paid in base currency, withdrawals, exchange
  outages beyond the `valid` mask.
* Funding / borrow costs and margin liquidation. They do not exist for spot long-only; with
  `long_only=False` or `allow_leverage=True` positions are simply allowed with no financing cost.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from mark1.data.panel import Panel
from mark1.sim.costs import MarketCosts, SimConfig
from mark1.sim.orders import (Fill, Order, OrderStatus, OrderType, Side, TimeInForce)
from mark1.sim.portfolio import Portfolio, PortfolioView, _readonly

REBALANCE_TAG = "rebalance"

_MARKET, _STOP = OrderType.MARKET, OrderType.STOP
_OPEN, _FILLED, _CANCELLED, _EXPIRED, _REJECTED = (
    OrderStatus.OPEN, OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED,
    OrderStatus.REJECTED)
_IOC = TimeInForce.IOC


@dataclass(slots=True)
class StepInfo:
    """Result of `reset()` / `step()`: state at the close of bar `t`."""

    t: int
    time: np.datetime64
    equity: float
    cash: float
    fills: list[Fill]
    done: bool


class _Log:
    """Append-only float64 table that grows geometrically (compact for millions of rows)."""

    def __init__(self, ncols: int, capacity: int = 1024):
        self._a = np.empty((capacity, ncols))
        self.n = 0

    def append(self, row: tuple) -> None:
        if self.n == len(self._a):
            self._a = np.concatenate([self._a, np.empty_like(self._a)])
        self._a[self.n] = row
        self.n += 1

    def array(self) -> np.ndarray:
        return self._a[: self.n]


FILL_COLUMNS = ("t", "asset", "side", "qty", "price", "fee", "slippage", "ref_price", "spread_bps",
                "impact_bps", "order_id", "order_type", "maker", "tag", "realized_pnl")
ORDER_COLUMNS = ("order_id", "asset", "side", "order_type", "qty", "price", "tif", "expire_bar", "tag",
                 "oco", "created_t", "active_bar", "end_t", "status", "filled_qty", "avg_fill_price",
                 "reason")


class Simulator:
    """Steppable exchange simulator over a `Panel`. See the module docstring for the model.

    Typical loop::

        sim = Simulator(panel, SimConfig())
        sim.reset(start=1_000, end=5_000)
        while not sim.done:
            # decide using rows <= sim.t only
            sim.submit(Order.market(asset=0, side=Side.BUY, qty=0.1))
            info = sim.step()          # processes bar sim.t + 1
    """

    def __init__(self, panel: Panel, config: SimConfig | None = None):
        self.panel = panel
        self.cfg = config or SimConfig()
        n = panel.n_assets
        self._hs_floor = self.cfg.half_spread_array(n)
        self._qstep = self.cfg.qty_step_array(n)
        self._costs = MarketCosts(panel, self.cfg, self._hs_floor)
        self._portfolio_view = PortfolioView(self)
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self, start: int = 0, end: int | None = None) -> StepInfo:
        """Start a fresh episode on rows [start, end). The first decision point is t = start."""
        T = self.panel.n_times
        end = T if end is None else int(end)
        start = int(start)
        if not (0 <= start < end <= T) or end - start < 2:
            raise ValueError(f"need 0 <= start < end <= {T} with at least 2 bars, got [{start}, {end})")
        n = self.panel.n_assets
        self._start, self._end, self._t = start, end, start
        self.pf = Portfolio(n, self.cfg.initial_cash)
        row, ok = self.panel.close[start], self.panel.valid[start]
        self.pf.mark(np.where(row > 0, row, 0.0), ok)
        self._orders: dict[int, Order] = {}
        self._next_id = 1
        self._tags: list[str | None] = [None]
        self._tag_codes: dict[str | None, int] = {None: 0}
        self._reasons: list[str] = [""]
        self._reason_codes: dict[str, int] = {"": 0}
        self._flog, self._olog = _Log(len(FILL_COLUMNS)), _Log(len(ORDER_COLUMNS))
        self._equity = np.empty(end - start)
        self._exposure = np.empty(end - start)
        self._equity[0], self._exposure[0] = self.pf.equity(), 0.0
        self._costs.prepare()
        self._vol = self._cap_left = self._sig = self._hs = None
        return StepInfo(start, self.panel.times[start], float(self._equity[0]), self.pf.cash, [], False)

    # ------------------------------------------------------------------ read-only state
    @property
    def t(self) -> int:
        """Current decision point: the last completed bar."""
        return self._t

    @property
    def start(self) -> int:
        return self._start

    @property
    def end(self) -> int:
        return self._end

    @property
    def time(self) -> np.datetime64:
        return self.panel.times[self._t]

    @property
    def done(self) -> bool:
        return self._t + 1 >= self._end

    @property
    def cash(self) -> float:
        return self.pf.cash

    @property
    def equity(self) -> float:
        return self.pf.equity()

    @property
    def positions(self) -> np.ndarray:
        return _readonly(self.pf.pos)

    @property
    def prices(self) -> np.ndarray:
        return _readonly(self.pf.last_price)

    @property
    def weights(self) -> np.ndarray:
        return self.pf.weights()

    @property
    def portfolio(self) -> PortfolioView:
        return self._portfolio_view

    @property
    def open_orders(self) -> list[Order]:
        """Copies of the live orders."""
        return [copy.copy(o) for o in self._orders.values()]

    @property
    def n_open_orders(self) -> int:
        return len(self._orders)

    def equity_curve(self) -> np.ndarray:
        """Equity at the close of bars start..t (a view; copy before mutating the simulation)."""
        return self._equity[: self._t - self._start + 1]

    def exposure_curve(self) -> np.ndarray:
        """Gross exposure (|position value| / equity) at the close of bars start..t."""
        return self._exposure[: self._t - self._start + 1]

    # ------------------------------------------------------------------ orders
    def submit(self, orders: Order | Iterable[Order]) -> list[int]:
        """Submit order requests at the current decision point; returns one id per request.

        Requests that break a rule (below `min_notional`, lot size rounds to zero, already expired)
        are rejected: they get an id, are logged with status REJECTED and never trade.
        """
        if isinstance(orders, Order):
            orders = [orders]
        ids = []
        cfg = self.cfg
        n = self.panel.n_assets
        for spec in orders:
            o = spec.copy()
            o.id, o.created_t = self._next_id, self._t
            o.filled, o.value, o.cancel_at, o.triggered, o.status = 0.0, 0.0, -1, False, OrderStatus.NEW
            self._next_id += 1
            ids.append(o.id)
            if not 0 <= o.asset < n:
                raise ValueError(f"asset index {o.asset} out of range for {n} assets")
            o.active_bar = self._t + 1 + cfg.latency_bars
            if self._qstep is not None and self._qstep[o.asset] > 0:
                step = self._qstep[o.asset]
                o.qty = math.floor(o.qty / step + 1e-9) * step
                if o.qty <= 0:
                    self._reject(o, "lot_size")
                    continue
            ref = o.price if o.price is not None else float(self.pf.last_price[o.asset])
            if ref > 0 and o.qty * ref < cfg.min_notional:
                self._reject(o, "min_notional")
                continue
            if o.expire_bar is not None and o.expire_bar < o.active_bar:
                self._reject(o, "expired")
                continue
            o.status = _OPEN
            self._orders[o.id] = o
        return ids

    def cancel(self, order_id: int) -> bool:
        """Request cancellation. Effective now with zero latency, else `latency_bars` bars later
        (the order can still fill meanwhile). Returns False for unknown / finished orders."""
        o = self._orders.get(order_id)
        if o is None or o.cancel_at >= 0:
            return False
        if self.cfg.latency_bars == 0:
            self._close(o, _CANCELLED, "user", self._t)
        else:
            o.cancel_at = self._t + 1 + self.cfg.latency_bars
        return True

    def cancel_all(self, *, tag: str | None = None, asset: int | None = None) -> int:
        """Cancel every live order (optionally only a tag and / or an asset); returns the count."""
        n = 0
        for o in tuple(self._orders.values()):
            if (tag is None or o.tag == tag) and (asset is None or o.asset == asset):
                n += self.cancel(o.id)
        return n

    def _reject(self, o: Order, reason: str) -> None:
        o.status = _REJECTED
        self._log_order(o, self._t, reason)

    def _close(self, o: Order, status: OrderStatus, reason: str, t: int) -> None:
        self._orders.pop(o.id, None)
        o.status = status
        self._log_order(o, t, reason)

    def _code(self, table: list, codes: dict, key) -> int:
        c = codes.get(key)
        if c is None:
            c = codes[key] = len(table)
            table.append(key)
        return c

    def _order_row(self, o: Order, end_t: int, reason: str) -> tuple:
        return (o.id, o.asset, int(o.side), int(o.type), o.qty,
                np.nan if o.price is None else o.price, int(o.tif),
                np.nan if o.expire_bar is None else o.expire_bar,
                self._code(self._tags, self._tag_codes, o.tag),
                -1 if o.oco is None else o.oco, o.created_t, o.active_bar, end_t, int(o.status),
                o.filled, o.avg_fill_price, self._code(self._reasons, self._reason_codes, reason))

    def _log_order(self, o: Order, end_t: int, reason: str) -> None:
        if self.cfg.log_orders:
            self._olog.append(self._order_row(o, end_t, reason))

    # ------------------------------------------------------------------ target weights
    def target_weight_orders(self, weights: np.ndarray, rebalance_threshold: float = 0.0,
                             tag: str = REBALANCE_TAG) -> list[Order]:
        """Market orders that move current holdings to `weights` (fraction of equity per asset).

        Sizing uses the last close. Trades smaller than max(rebalance_threshold * equity,
        min_notional) are skipped. Without leverage the buys are scaled down proportionally if
        fees and spread would otherwise make the rebalance cost more than the available cash. With
        latency > 0, orders from earlier rebalances that will still fill before the new ones become
        active count as holdings. Sells come first in the returned list.
        """
        cfg, pf = self.cfg, self.pf
        n = self.panel.n_assets
        w = np.asarray(weights, dtype=np.float64)
        total = float(w.sum()) if w.shape == (n,) else math.nan
        if not math.isfinite(total):
            raise ValueError(f"target weights must be {n} finite numbers, got shape {w.shape}")
        if cfg.long_only:
            lowest = float(w.min())
            if lowest < -1e-9:
                raise ValueError("negative target weight while long_only=True")
            if lowest < 0.0:
                w = np.maximum(w, 0.0)
            gross = total
        else:
            gross = float(np.abs(w).sum())
        if not cfg.allow_leverage and gross > 1.0 + 1e-6:
            raise ValueError(f"target weights sum to {gross:.6f} > 1 without leverage")

        eq = pf.equity()
        price = pf.last_price
        pos = pf.pos
        if cfg.latency_bars > 0:
            pos = pos.copy()
            for o in self._orders.values():
                in_window = o.active_bar <= self._t + cfg.latency_bars and (
                    o.cancel_at < 0 or o.cancel_at > max(o.active_bar, self._t + 1))
                if o.tag == tag and o.type == _MARKET and in_window:
                    pos[o.asset] += o.sign * o.remaining
        target_qty = np.divide(w * eq, price, out=np.zeros(n), where=price > 0.0)
        delta = target_qty - pos
        thr = max(rebalance_threshold * eq, cfg.min_notional, 1e-12)  # unpriced assets have notional 0
        idx = np.flatnonzero(np.abs(delta) * price >= thr)
        if idx.size == 0:
            return []
        qty, px, ids = delta[idx].tolist(), price[idx].tolist(), idx.tolist()
        if cfg.long_only and not cfg.allow_leverage:
            need, have = 0.0, pf.cash
            for q, p, h in zip(qty, px, self._hs_floor[idx].tolist()):
                unit = (cfg.taker_fee_bps + h) * 1e-4
                if q > 0:
                    need += q * p * (1.0 + unit)
                else:
                    have -= q * p * (1.0 - unit)
            if need > have:
                scale = max(have, 0.0) / need
                qty = [q * scale if q > 0 else q for q in qty]
        sells = [Order(i, Side.SELL, -q, tag=tag) for i, q, p in zip(ids, qty, px) if -q * p >= thr]
        buys = [Order(i, Side.BUY, q, tag=tag) for i, q, p in zip(ids, qty, px) if q * p >= thr]
        return sells + buys

    def rebalance(self, weights: np.ndarray, rebalance_threshold: float = 0.0) -> list[int]:
        """Cancel still-pending rebalance orders, then submit market orders towards `weights`."""
        orders = self.target_weight_orders(weights, rebalance_threshold)
        self.cancel_all(tag=REBALANCE_TAG)
        return self.submit(orders)

    # ------------------------------------------------------------------ stepping
    def step(self) -> StepInfo:
        """Advance one bar: process fills for active orders, then mark to the bar's close."""
        if self.done:
            raise RuntimeError("simulation finished; call reset()")
        b = self._t + 1
        fills: list[Fill] = []
        if self._orders:
            self._process_bar(b, fills)
        self._t = b
        pf, P = self.pf, self.panel
        pf.mark(P.close[b], P.valid[b])
        eq = pf.cash + float(pf.pos @ pf.last_price)
        k = b - self._start
        self._equity[k] = eq
        if eq > 0.0:
            gross = (eq - pf.cash) if self.cfg.long_only else pf.gross_value()
            self._exposure[k] = gross / eq
        else:
            self._exposure[k] = 0.0
        return StepInfo(b, P.times[b], eq, pf.cash, fills, b + 1 >= self._end)

    def _process_bar(self, b: int, fills: list[Fill]) -> None:
        orders = self._orders
        eligible = []
        for o in tuple(orders.values()):
            if 0 <= o.cancel_at <= b:
                self._close(o, _CANCELLED, "user", b)
            elif o.expire_bar is not None and b > o.expire_bar:
                self._close(o, _EXPIRED, "expired", b)
            elif o.active_bar <= b:
                eligible.append(o)
        if not eligible:
            return
        P = self.panel
        op, hi, lo = P.open[b].tolist(), P.high[b].tolist(), P.low[b].tolist()
        vol, valid = P.volume[b].tolist(), P.valid[b].tolist()
        sig, hs = self._costs.row(b)
        self._vol, self._sig, self._hs = vol, sig.tolist(), hs.tolist()
        cap = self.cfg.participation_cap
        self._cap_left = [cap * v if v > 0.0 else 0.0 for v in vol]  # nan / negative volume: no liquidity

        # Classify: (phase, is_buy, type rank, id, order, base price, reference price, maker).
        # phase 0 = executes at the open, 1 = triggered later in the bar.
        cands = []
        for o in eligible:
            i = o.asset
            if not valid[i]:
                continue                                  # no trading: the order waits
            s, typ, px, o_ = o.sign, o.type, o.price, op[i]
            if typ == _MARKET or o.triggered:
                cands.append((0, s > 0, 1, o.id, o, o_, o_, False))
                continue
            if typ == _STOP:
                if s < 0:
                    hit = 0 if o_ <= px else (1 if lo[i] <= px else -1)
                else:
                    hit = 0 if o_ >= px else (1 if hi[i] >= px else -1)
                if hit < 0:
                    continue
                o.triggered = True                        # the rest continues as a market order
                cands.append((hit, s > 0, 0, o.id, o, o_ if hit == 0 else px, px, False))
                continue
            if s > 0:                                     # LIMIT / TAKE_PROFIT buy
                hit = 0 if o_ < px else (1 if lo[i] < px else -1)
            else:
                hit = 0 if o_ > px else (1 if hi[i] > px else -1)
            if hit < 0:
                if o.tif == _IOC:
                    self._close(o, _CANCELLED, "ioc_no_fill", b)
                continue
            if hit == 0 and o.active_bar == b:
                # Arrives already through the market: it crosses the spread like a market order.
                cands.append((0, s > 0, 1, o.id, o, o_, o_, False))
            else:
                # Resting in the book: the market moved through it, so it fills at its own price.
                cands.append((hit, s > 0, 2, o.id, o, px, px, True))

        cands.sort(key=lambda c: c[:4])
        for _, _, _, _, o, base, ref, maker in cands:
            if o.status != _OPEN:                         # cancelled by an OCO sibling this bar
                continue
            self._execute(o, b, base, ref, maker, fills)
            if o.tif == _IOC and o.status == _OPEN:
                self._close(o, _CANCELLED, "ioc_remainder", b)

    def _taker_price(self, base: float, s: int, q: float, i: int) -> tuple[float, float, float]:
        """Fill price after half-spread and square-root impact: (price, spread_bps, impact_bps)."""
        cfg = self.cfg
        sp = self._hs[i]
        imp = cfg.impact_coef * self._sig[i] * math.sqrt(q / self._vol[i])
        if imp > cfg.impact_cap_bps:
            imp = cfg.impact_cap_bps
        return base * (1.0 + s * (sp + imp) * 1e-4), sp, imp

    def _execute(self, o: Order, b: int, base: float, ref: float, maker: bool,
                 fills: list[Fill]) -> None:
        """Try to fill `o` on bar b at `base` (before costs). Applies caps, funds, costs, accounting."""
        cfg, pf = self.cfg, self.pf
        i, s = o.asset, o.sign
        q = o.qty - o.filled
        pos = float(pf.pos[i])
        if cfg.long_only and s < 0:
            if pos <= 0.0:
                self._close(o, _CANCELLED, "no_position", b)
                return
            q = min(q, pos)
        q = min(q, self._cap_left[i])
        step = self._qstep[i] if self._qstep is not None else 0.0
        if step > 0.0:
            q = math.floor(q / step + 1e-9) * step
        if q <= 0.0:
            return                                        # no liquidity left this bar: wait
        fee_rate = (cfg.maker_fee_bps if maker else cfg.taker_fee_bps) * 1e-4
        sp = imp = 0.0
        price = base
        if not maker:
            price, sp, imp = self._taker_price(base, s, q, i)

        funds_limited = False
        if not cfg.allow_leverage and (s > 0 or not cfg.long_only):
            free = pf.cash if cfg.long_only else pf.equity() - pf.gross_value()
            closing = max(0.0, -s * pos)                  # part of the trade that reduces |position|
            qmax = (free + 2.0 * closing * price) / (price * (1.0 + fee_rate)) * (1.0 - 1e-12)
            qmax = max(qmax, closing)
            if q > qmax:
                funds_limited = True
                q = qmax
                if step > 0.0:
                    q = math.floor(q / step + 1e-9) * step
                if q > 0.0 and not maker:
                    price, sp, imp = self._taker_price(base, s, q, i)
                if q <= 0.0 or q * price < cfg.min_notional:
                    self._close(o, _CANCELLED, "insufficient_funds", b)
                    return

        fee = fee_rate * q * price
        realized = pf.apply_fill(i, s * q, price, fee, b)
        pf.slippage_paid += s * (price - ref) * q
        self._cap_left[i] -= q
        o.filled += q
        o.value += q * price
        fill = Fill(b, i, o.side, q, price, fee, s * (price - ref) * q, ref, sp, imp, o.id, o.type,
                    maker, o.tag, realized)
        fills.append(fill)
        if cfg.log_fills:
            self._flog.append((b, i, s, q, price, fee, fill.slippage, ref, sp, imp, o.id, int(o.type),
                               maker, self._code(self._tags, self._tag_codes, o.tag), realized))
        if o.oco is not None:
            for x in tuple(self._orders.values()):
                if x is not o and x.oco == o.oco and x.asset == i:
                    self._close(x, _CANCELLED, "oco", b)
        if o.qty - o.filled <= 1e-12 * o.qty:
            self._close(o, _FILLED, "filled", b)
        elif funds_limited:
            self._close(o, _CANCELLED, "insufficient_funds", b)
        elif cfg.long_only and s < 0 and pf.pos[i] <= 0.0:
            self._close(o, _CANCELLED, "no_position", b)

    # ------------------------------------------------------------------ results
    def stats(self) -> dict[str, float]:
        """Running totals: fees, slippage (spread + impact + gaps), traded notional, counts, PnL."""
        pf = self.pf
        return {"fees_paid": pf.fees_paid, "slippage_cost": pf.slippage_paid,
                "traded_notional": pf.traded_notional, "n_fills": pf.n_fills,
                "n_orders": self._next_id - 1, "realized_pnl": pf.realized_pnl,
                "unrealized_pnl": pf.unrealized_pnl()}

    def equity_series(self) -> pd.Series:
        idx = pd.DatetimeIndex(self.panel.times[self._start: self._t + 1], name="time")
        return pd.Series(self.equity_curve().copy(), index=idx, name="equity")

    def exposure_series(self) -> pd.Series:
        idx = pd.DatetimeIndex(self.panel.times[self._start: self._t + 1], name="time")
        return pd.Series(self.exposure_curve().copy(), index=idx, name="exposure")

    def fills_frame(self) -> pd.DataFrame:
        """All fills so far. `time` is the open time of the bar the fill happened in."""
        a = self._flog.array()
        symbols = np.asarray(self.panel.symbols, dtype=object)
        t, asset = a[:, 0].astype(np.int64), a[:, 1].astype(np.int64)
        tags = np.asarray(self._tags, dtype=object)
        df = pd.DataFrame({
            "time": pd.DatetimeIndex(self.panel.times[t]),
            "t": t,
            "asset": asset,
            "symbol": symbols[asset],
            "side": np.where(a[:, 2] > 0, "buy", "sell").astype(object),
            "qty": a[:, 3], "price": a[:, 4], "fee": a[:, 5], "slippage": a[:, 6],
            "ref_price": a[:, 7], "spread_bps": a[:, 8], "impact_bps": a[:, 9],
            "order_id": a[:, 10].astype(np.int64),
            "order_type": np.asarray([m.name for m in OrderType], dtype=object)[a[:, 11].astype(int)],
            "liquidity": np.where(a[:, 12] > 0, "maker", "taker").astype(object),
            "tag": tags[a[:, 13].astype(int)],
            "realized_pnl": a[:, 14],
        })
        df["notional"] = df["qty"] * df["price"]
        return df

    def orders_frame(self) -> pd.DataFrame:
        """All finished orders plus the ones still open (status OPEN, reason 'open')."""
        rows = self._olog.array()
        live = [self._order_row(o, -1, "open") for o in self._orders.values()]
        a = np.vstack([rows, np.asarray(live).reshape(-1, len(ORDER_COLUMNS))]) if live else rows
        symbols = np.asarray(self.panel.symbols, dtype=object)
        tags = np.asarray(self._tags, dtype=object)
        reasons = np.asarray(self._reasons, dtype=object)
        asset = a[:, 1].astype(np.int64)
        times = self.panel.times
        end_t = a[:, 12].astype(np.int64)
        df = pd.DataFrame({
            "order_id": a[:, 0].astype(np.int64),
            "asset": asset,
            "symbol": symbols[asset],
            "side": np.where(a[:, 2] > 0, "buy", "sell").astype(object),
            "order_type": np.asarray([m.name for m in OrderType], dtype=object)[a[:, 3].astype(int)],
            "qty": a[:, 4], "price": a[:, 5],
            "tif": np.asarray([m.name for m in TimeInForce], dtype=object)[a[:, 6].astype(int)],
            "expire_bar": a[:, 7], "tag": tags[a[:, 8].astype(int)],
            "oco": a[:, 9].astype(np.int64),
            "created_t": a[:, 10].astype(np.int64),
            "created_time": pd.DatetimeIndex(times[a[:, 10].astype(np.int64)]),
            "active_bar": a[:, 11].astype(np.int64),
            "end_t": end_t,
            "end_time": pd.DatetimeIndex(np.where(end_t >= 0, times[np.maximum(end_t, 0)],
                                                  np.datetime64("NaT", "ns"))),
            "status": np.asarray([m.name for m in OrderStatus], dtype=object)[a[:, 13].astype(int)],
            "filled_qty": a[:, 14], "avg_fill_price": a[:, 15],
            "reason": reasons[a[:, 16].astype(int)],
        })
        return df.sort_values("order_id", kind="stable").reset_index(drop=True)

    def round_trips_frame(self) -> pd.DataFrame:
        """Closed round trips (flat -> position -> flat) with PnL net of fees."""
        rt = self.pf.round_trips
        cols = ["asset", "open_t", "close_t", "pnl"]
        df = pd.DataFrame(rt, columns=cols) if rt else pd.DataFrame(
            {"asset": pd.Series(dtype=np.int64), "open_t": pd.Series(dtype=np.int64),
             "close_t": pd.Series(dtype=np.int64), "pnl": pd.Series(dtype=float)})
        df.insert(1, "symbol", np.asarray(self.panel.symbols, dtype=object)[df["asset"].to_numpy(int)])
        df["open_time"] = pd.DatetimeIndex(self.panel.times[df["open_t"].to_numpy(np.int64)])
        df["close_time"] = pd.DatetimeIndex(self.panel.times[df["close_t"].to_numpy(np.int64)])
        return df
