"""Scheduled macro-event calendar: FOMC policy decisions and US CPI releases (2021-2026).

The table lives in ``events.csv`` next to this module (columns ``event, datetime_utc, scheduled,
source_url, notes``). ``datetime_utc`` is the exact release instant in UTC, converted from US
Eastern time with ``zoneinfo`` ("America/New_York"), so daylight saving time is handled by the
tz database and not by hand-computed offsets:

* ``FOMC_DECISION``: the policy statement, 2:00 pm ET on the last day of a scheduled two-day meeting
  (18:00Z in EDT, 19:00Z in EST). One documented exception: the 2025-08-22 notation vote
  (10:00 am EDT, 14:00Z), flagged ``TIME_EXCEPTION`` in ``notes`` and ``scheduled=false``.
* ``CPI_RELEASE``: BLS Consumer Price Index news release, 8:30 am ET (12:30Z in EDT, 13:30Z in EST).
  Delayed releases (2025 and 2026 lapses in appropriations) use the ACTUAL release time and say so
  in ``notes``; October 2025 CPI was never published, so it has no row.

Sources (also in each row's ``source_url``/``notes``): Federal Reserve ``fomccalendars.htm``, the
per-meeting statement pages (which print "For release at 2:00 p.m. EST/EDT") and the Fed's
"tentative meeting schedule" announcements (a second, independent document per year); BLS archived
CPI release pages (index https://www.bls.gov/bls/news-release/cpi.htm), BLS yearly schedule pages
(https://www.bls.gov/schedule/YYYY/home.htm), the BLS "revised release dates" page for the
2025/2026 lapses, and https://www.bls.gov/schedule/news_release/cpi.htm for upcoming releases.

Causality
---------
Scheduled dates are public long before they happen (the Fed publishes meeting dates 1-2 years ahead,
BLS publishes the yearly release calendar in advance). Using "hours until the next scheduled event" as
a feature is therefore NOT look-ahead. Two things must still be respected, and are:

1. Unscheduled actions (``scheduled == False``: emergency meetings, notation votes) were not known in
   advance. They never contribute to ``hours_to_next`` and they never raise the window flag before they
   happen; they count for ``hours_since_last`` and the post-event half of the window like any event.
2. "Hours since last" only looks at events at or before the bar time.

Caveat: for the three releases BLS moved at short notice (2025-10-24, 2025-12-18, 2026-02-13) the table
holds the actual date, flagged ``scheduled=true``. A live system would have used the original date
(2025-10-15, 2025-12-10, 2026-02-11) until the change was announced, so ``hours_to_next`` is very
slightly optimistic for the few days between the original calendar and the announcement.

Time convention: for a time ``t`` equal to an event instant, ``hours_to_next == 0`` and
``hours_since_last == 0`` and the window flag is 1. The table ends at 2026-12-10; after the last event the
"next" distance is the cap. Earlier than the first row the "since" distance is the cap (the previous
events, Dec 2020, are more than 14 days before 2021-01-01, so the default cap is exact there).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["KINDS", "WINDOW_HOURS", "event_features", "load_events"]

KINDS = ("FOMC_DECISION", "CPI_RELEASE")
WINDOW_HOURS = 2.0

_EVENTS_CSV = Path(__file__).with_name("events.csv")
_NS_PER_HOUR = 3_600_000_000_000


def load_events(kinds: list[str] | None = None) -> pd.DataFrame:
    """Read ``events.csv`` and return it sorted by time.

    Returns a DataFrame with columns ``event`` (str), ``time`` (``datetime64[ns]``, UTC, tz-naive) and
    ``scheduled`` (bool: True when the date was on the published calendar in advance, False for
    emergency/unscheduled actions). ``kinds`` optionally restricts to some of ``KINDS``; an unknown name
    raises ``ValueError`` so a typo cannot silently yield an empty table.
    """
    raw = pd.read_csv(_EVENTS_CSV, dtype=str, keep_default_na=False)
    flag = raw["scheduled"].str.strip().str.lower()
    bad = ~flag.isin(["true", "false"])
    if bad.any():
        raise ValueError(f"events.csv: 'scheduled' must be true/false, got {sorted(set(raw.loc[bad, 'scheduled']))}")
    df = pd.DataFrame({
        "event": raw["event"].str.strip(),
        "time": pd.to_datetime(raw["datetime_utc"], format="%Y-%m-%dT%H:%M:%SZ").astype("datetime64[ns]"),
        "scheduled": (flag == "true").to_numpy(),
    })
    if kinds is not None:
        unknown = sorted(set(kinds) - set(KINDS))
        if unknown:
            raise ValueError(f"unknown event kinds {unknown}; valid kinds: {list(KINDS)}")
        df = df[df["event"].isin(list(kinds))]
    return df.sort_values("time", kind="stable").reset_index(drop=True)


def _to_ns(values) -> np.ndarray:
    """Any datetime-like (numpy, pandas, tz-aware or naive) -> int64 nanoseconds since the UTC epoch."""
    if isinstance(values, (pd.Series, pd.DatetimeIndex)):
        values = pd.DatetimeIndex(values)
        if values.tz is not None:
            values = values.tz_convert("UTC").tz_localize(None)
        values = values.to_numpy()
    arr = np.asarray(values).astype("datetime64[ns]")
    if np.isnat(arr).any():
        raise ValueError("NaT is not allowed in event times or query times")
    return arr.astype(np.int64)


def event_features(
    times: np.ndarray,
    events_df: pd.DataFrame,
    kinds=KINDS,
    cap_hours: float = 24 * 14,
) -> dict[str, np.ndarray]:
    """Event-timing features for each query time, vectorised with ``np.searchsorted``.

    ``times`` is a 1-D datetime64[ns] array (UTC, tz-naive; need not be sorted). ``events_df`` is the
    output of :func:`load_events` (columns ``event``, ``time`` and optionally ``scheduled``; a missing
    ``scheduled`` column means every row was scheduled). For each kind the result has three float32
    arrays, each the same length as ``times``:

    ``{kind}_hours_to_next``
        Hours until the next SCHEDULED event at or after ``t``, clipped to ``[0, cap_hours]``; equals
        ``cap_hours`` when there is none in the table. Scheduled dates are public in advance, so this
        is not look-ahead. Unscheduled/emergency rows are ignored here: they cannot be anticipated,
        so they must not shorten the countdown before they happened.
    ``{kind}_hours_since_last``
        Hours since the latest event (scheduled or not) at or before ``t``, clipped to
        ``[0, cap_hours]``; ``cap_hours`` when there is none yet. Only past events are used.
    ``{kind}_window``
        1.0 when ``t`` lies within [-2 h, +2 h] of an event (inclusive), else 0.0. The -2 h half
        uses scheduled events only; the +2 h half uses all events, so an unscheduled action never
        lights the window before it happens.

    Distances are computed in exact integer nanoseconds (window membership is decided before any
    float conversion or capping, so the flag does not depend on ``cap_hours``). A kind with no rows
    yields ``cap_hours`` for both distances and a zero window.
    """
    t = _to_ns(times)
    if t.ndim != 1:
        raise ValueError("times must be a 1-D array")
    cap_ns = int(round(cap_hours * _NS_PER_HOUR))
    win_ns = int(round(WINDOW_HOURS * _NS_PER_HOUR))
    has_flag = "scheduled" in events_df.columns
    out: dict[str, np.ndarray] = {}
    for kind in kinds:
        sub = events_df[events_df["event"] == kind]
        all_ns = np.sort(_to_ns(sub["time"]))
        sched_ns = np.sort(_to_ns(sub.loc[sub["scheduled"].astype(bool), "time"])) if has_flag else all_ns

        to_next = np.full(t.shape, cap_ns, dtype=np.int64)
        in_window = np.zeros(t.shape, dtype=bool)
        if sched_ns.size:
            i = np.searchsorted(sched_ns, t, side="left")                # first scheduled event >= t
            ok = i < sched_ns.size
            gap = sched_ns[np.minimum(i, sched_ns.size - 1)] - t
            to_next = np.where(ok, gap, cap_ns)
            in_window |= ok & (gap <= win_ns)

        since = np.full(t.shape, cap_ns, dtype=np.int64)
        if all_ns.size:
            j = np.searchsorted(all_ns, t, side="right") - 1             # last event <= t
            ok = j >= 0
            gap = t - all_ns[np.maximum(j, 0)]
            since = np.where(ok, gap, cap_ns)
            in_window |= ok & (gap <= win_ns)

        out[f"{kind}_hours_to_next"] = (np.clip(to_next, 0, cap_ns) / _NS_PER_HOUR).astype(np.float32)
        out[f"{kind}_hours_since_last"] = (np.clip(since, 0, cap_ns) / _NS_PER_HOUR).astype(np.float32)
        out[f"{kind}_window"] = in_window.astype(np.float32)
    return out
