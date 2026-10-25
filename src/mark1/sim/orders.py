"""Order, fill and decision types used by the simulator and by strategies."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence

import numpy as np


class Side(IntEnum):
    BUY = 1
    SELL = -1


class OrderType(IntEnum):
    MARKET = 0       # fills at the open of the first eligible bar, plus costs
    LIMIT = 1        # maker order, fills only when the bar trades strictly through the price
    STOP = 2         # stop-market: triggers on touch, then fills like a market order
    TAKE_PROFIT = 3  # limit-like exit at a better price; same fill rule as LIMIT


class TimeInForce(IntEnum):
    GTC = 0  # good till cancelled (or until `expire_bar`)
    IOC = 1  # one attempt (for stops: the bar that triggers them); the unfilled rest is cancelled


class OrderStatus(IntEnum):
    NEW = 0
    OPEN = 1       # live in the book (possibly partially filled)
    FILLED = 2
    CANCELLED = 3
    EXPIRED = 4
    REJECTED = 5


@dataclass(slots=True)
class Order:
    """An order request. Quantities are in base units; `asset` is the column index in the panel.

    `price` is the limit price (LIMIT / TAKE_PROFIT) or the trigger price (STOP) and is ignored
    for MARKET. `expire_bar` is an absolute row index: the order may fill on bars <= expire_bar
    and is expired afterwards. `oco` links orders on the same asset: when one of them fills,
    the others are cancelled (use it for stop-loss / take-profit brackets).

    The simulator copies the request on submit and assigns `id`, so a request can be reused.
    The engine-owned fields below the divider are filled in by the simulator only.
    """

    asset: int
    side: Side
    qty: float
    type: OrderType = OrderType.MARKET
    price: float | None = None
    tif: TimeInForce = TimeInForce.GTC
    expire_bar: int | None = None
    tag: str | None = None
    oco: int | None = None
    # ---- engine-owned state -------------------------------------------------
    id: int = field(default=-1, init=False)
    created_t: int = field(default=-1, init=False)    # step at which the order was submitted
    active_bar: int = field(default=-1, init=False)   # first bar on which it may fill
    cancel_at: int = field(default=-1, init=False)    # bar from which a requested cancel is effective
    filled: float = field(default=0.0, init=False)
    value: float = field(default=0.0, init=False)     # sum of fill notional (for the average price)
    status: OrderStatus = field(default=OrderStatus.NEW, init=False)
    triggered: bool = field(default=False, init=False)  # STOP already triggered
    sign: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if self.side.__class__ is not Side:   # skip the (slow) enum lookup for ready-made members
            self.side = Side(self.side)
        if self.type.__class__ is not OrderType:
            self.type = OrderType(self.type)
        if self.tif.__class__ is not TimeInForce:
            self.tif = TimeInForce(self.tif)
        self.sign = int(self.side)
        self.asset = int(self.asset)
        self.qty = float(self.qty)
        if not (math.isfinite(self.qty) and self.qty > 0.0):
            raise ValueError(f"order qty must be a positive finite number, got {self.qty!r}")
        if self.type != OrderType.MARKET:
            if self.price is None or not (math.isfinite(self.price) and self.price > 0.0):
                raise ValueError(f"{self.type.name} order needs a positive price, got {self.price!r}")
            self.price = float(self.price)

    def copy(self) -> "Order":
        """A fresh request with the same parameters and no engine state."""
        return Order(self.asset, self.side, self.qty, self.type, self.price, self.tif, self.expire_bar,
                     self.tag, self.oco)

    @property
    def remaining(self) -> float:
        return self.qty - self.filled

    @property
    def avg_fill_price(self) -> float:
        return self.value / self.filled if self.filled > 0 else float("nan")

    # ---- convenience constructors ------------------------------------------
    @classmethod
    def market(cls, asset: int, side: Side, qty: float, *, tif: TimeInForce = TimeInForce.GTC,
               tag: str | None = None, expire_bar: int | None = None) -> "Order":
        return cls(asset, side, qty, OrderType.MARKET, None, tif, expire_bar, tag)

    @classmethod
    def limit(cls, asset: int, side: Side, qty: float, price: float, *,
              tif: TimeInForce = TimeInForce.GTC, tag: str | None = None,
              expire_bar: int | None = None, oco: int | None = None) -> "Order":
        return cls(asset, side, qty, OrderType.LIMIT, price, tif, expire_bar, tag, oco)

    @classmethod
    def stop(cls, asset: int, side: Side, qty: float, stop_price: float, *,
             tif: TimeInForce = TimeInForce.GTC, tag: str | None = None,
             expire_bar: int | None = None, oco: int | None = None) -> "Order":
        return cls(asset, side, qty, OrderType.STOP, stop_price, tif, expire_bar, tag, oco)

    @classmethod
    def take_profit(cls, asset: int, side: Side, qty: float, price: float, *,
                    tif: TimeInForce = TimeInForce.GTC, tag: str | None = None,
                    expire_bar: int | None = None, oco: int | None = None) -> "Order":
        return cls(asset, side, qty, OrderType.TAKE_PROFIT, price, tif, expire_bar, tag, oco)


@dataclass(slots=True)
class Fill:
    """One execution (a single order can produce several fills over several bars).

    `t` is the panel row of the bar in which the fill happened. `slippage` is the signed cost in
    quote currency versus `ref_price` (bar open for market orders, the limit / stop price for the
    others): positive means worse than the reference, negative is price improvement.
    """

    t: int
    asset: int
    side: Side
    qty: float
    price: float
    fee: float
    slippage: float
    ref_price: float
    spread_bps: float
    impact_bps: float
    order_id: int
    order_type: OrderType
    maker: bool
    tag: str | None
    realized_pnl: float  # gross of this fill's fee, on the part that closed an existing position

    @property
    def notional(self) -> float:
        return self.qty * self.price


@dataclass
class Decision:
    """What a strategy returns at a decision step.

    target_weights: fraction of equity per asset, shape [N]. Converted by the run loop into
        market orders for the difference versus current holdings (see `Simulator.rebalance`).
    orders: explicit order requests, submitted after the rebalance.
    cancel: ids of open orders to cancel (before anything else happens).
    """

    target_weights: np.ndarray | None = None
    orders: Sequence[Order] = ()
    cancel: Sequence[int] = ()
