"""Cash and position accounting, plus the read-only view handed to strategies."""

from __future__ import annotations

import numpy as np


class Portfolio:
    """Cash (quote currency), signed base-unit positions and running cost totals.

    Position accounting uses average cost: adding to a position blends the entry price, reducing
    realises PnL at (fill price - average entry), flipping through zero (only possible when
    shorting is enabled) resets the entry price. Identity that always holds:

        equity = initial_cash + realized_pnl + unrealized_pnl - fees_paid

    where realized/unrealized PnL are gross of fees (fill prices already include spread and
    impact). A "round trip" is one stretch from a flat position back to flat, with PnL net of fees.
    """

    def __init__(self, n_assets: int, cash: float):
        self.n_assets = n_assets
        self.initial_cash = float(cash)
        self.cash = float(cash)
        self.pos = np.zeros(n_assets)
        self.avg_entry = np.zeros(n_assets)
        self.realized = np.zeros(n_assets)     # cumulative gross realised PnL per asset
        self.last_price = np.zeros(n_assets)   # last valid close; 0 = never priced
        self.fees_paid = 0.0
        self.slippage_paid = 0.0               # spread + impact + gap cost versus reference prices
        self.traded_notional = 0.0
        self.n_fills = 0
        self.round_trips: list[tuple[int, int, int, float]] = []  # (asset, open_t, close_t, net pnl)
        self._trip_open = [-1] * n_assets
        self._trip_pnl = [0.0] * n_assets

    # ---- marking --------------------------------------------------------------
    def mark(self, close_row: np.ndarray, valid_row: np.ndarray) -> None:
        """Mark to the bar's close; assets without a valid bar keep their last valid close."""
        np.copyto(self.last_price, close_row, where=valid_row)

    def equity(self) -> float:
        return self.cash + float(self.pos @ self.last_price)

    def gross_value(self) -> float:
        return float(np.abs(self.pos) @ self.last_price)

    def unrealized_pnl(self) -> float:
        return float(np.where(self.pos != 0.0, self.pos * (self.last_price - self.avg_entry), 0.0).sum())

    @property
    def realized_pnl(self) -> float:
        return float(self.realized.sum())

    def weights(self) -> np.ndarray:
        eq = self.equity()
        return self.pos * self.last_price / eq if eq > 0 else np.zeros(self.n_assets)

    # ---- trading --------------------------------------------------------------
    def apply_fill(self, i: int, qty: float, price: float, fee: float, t: int) -> float:
        """Apply a signed fill (qty > 0 buys). Returns the gross realised PnL it produced."""
        p0 = float(self.pos[i])
        a0 = float(self.avg_entry[i])
        new = p0 + qty
        realized = 0.0
        if p0 == 0.0 or (p0 > 0.0) == (qty > 0.0):          # opening or adding
            avg = (abs(p0) * a0 + abs(qty) * price) / abs(new)
        else:                                               # reducing, closing or flipping
            closed = min(abs(qty), abs(p0))
            realized = closed * (price - a0) * (1.0 if p0 > 0.0 else -1.0)
            avg = price if abs(qty) > abs(p0) else a0
        if abs(new) <= 1e-12 * max(abs(p0), abs(qty)):      # float dust from chained partial fills
            new = 0.0
        if new == 0.0:
            avg = 0.0
        self.cash -= qty * price + fee
        self.pos[i] = new
        self.avg_entry[i] = avg
        self.realized[i] += realized
        self.fees_paid += fee
        self.traded_notional += abs(qty) * price
        self.n_fills += 1
        self._track_round_trip(i, p0, new, realized - fee, t)
        return realized

    def _track_round_trip(self, i: int, before: float, after: float, pnl: float, t: int) -> None:
        if before == 0.0:
            self._trip_open[i], self._trip_pnl[i] = t, 0.0
        self._trip_pnl[i] += pnl
        flipped = before != 0.0 and after != 0.0 and (before > 0.0) != (after > 0.0)
        if after == 0.0 or flipped:
            self.round_trips.append((i, self._trip_open[i], t, self._trip_pnl[i]))
            self._trip_open[i], self._trip_pnl[i] = t, 0.0


def _readonly(a: np.ndarray) -> np.ndarray:
    v = a.view()
    v.flags.writeable = False
    return v


class PortfolioView:
    """Read-only window onto the simulator's account, passed to strategies.

    Arrays are live read-only views: copy them if you want to keep a snapshot.
    """

    def __init__(self, sim) -> None:
        self._sim = sim

    @property
    def t(self) -> int:
        return self._sim.t

    @property
    def cash(self) -> float:
        return self._sim.pf.cash

    @property
    def equity(self) -> float:
        return self._sim.pf.equity()

    @property
    def positions(self) -> np.ndarray:
        """Signed base-unit holdings per asset."""
        return _readonly(self._sim.pf.pos)

    @property
    def prices(self) -> np.ndarray:
        """Last valid close per asset (0 if the asset has never had a valid bar)."""
        return _readonly(self._sim.pf.last_price)

    @property
    def avg_entry(self) -> np.ndarray:
        return _readonly(self._sim.pf.avg_entry)

    @property
    def weights(self) -> np.ndarray:
        """Position value as a fraction of equity (cash weight = 1 - sum)."""
        return self._sim.pf.weights()

    @property
    def gross_exposure(self) -> float:
        eq = self._sim.pf.equity()
        return self._sim.pf.gross_value() / eq if eq > 0 else 0.0

    @property
    def realized_pnl(self) -> float:
        return self._sim.pf.realized_pnl

    @property
    def unrealized_pnl(self) -> float:
        return self._sim.pf.unrealized_pnl()

    @property
    def fees_paid(self) -> float:
        return self._sim.pf.fees_paid

    @property
    def n_open_orders(self) -> int:
        return self._sim.n_open_orders

    @property
    def open_orders(self) -> list:
        """Copies of the live orders (changing them has no effect on the simulator)."""
        return self._sim.open_orders
