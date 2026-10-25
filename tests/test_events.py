"""Offline tests for the scheduled macro-event calendar (mark1.data.events + events.csv)."""

import datetime as dt
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from mark1.data.events import KINDS, event_features, load_events

NY = ZoneInfo("America/New_York")
UTC = dt.timezone.utc
AS_OF = dt.date(2026, 10, 25)            # date the calendar was compiled: later rows must say "scheduled"
H = np.timedelta64(1, "h")


def _ts(s):
    return np.datetime64(s, "ns")


def _raw():
    from mark1.data import events as mod

    return pd.read_csv(mod._EVENTS_CSV, dtype=str, keep_default_na=False)


def _local(ts: pd.Timestamp) -> dt.datetime:
    return ts.to_pydatetime().replace(tzinfo=UTC).astimezone(NY)


def _nth_sunday(year: int, month: int, n: int) -> dt.date:
    first = dt.date(year, month, 1)
    first_sunday = first + dt.timedelta(days=(6 - first.weekday()) % 7)
    return first_sunday + dt.timedelta(weeks=n - 1)


def _is_dst(d: dt.date) -> bool:
    """US rule: DST from the 2nd Sunday of March to the 1st Sunday of November (events are after 2 am local)."""
    return _nth_sunday(d.year, 3, 2) <= d < _nth_sunday(d.year, 11, 1)


@pytest.fixture(scope="module")
def ev():
    return load_events()


# ---------------------------------------------------------------- loading and structure -------------
def test_csv_loads_with_expected_schema(ev):
    raw = _raw()
    assert list(raw.columns) == ["event", "datetime_utc", "scheduled", "source_url", "notes"]
    assert list(ev.columns) == ["event", "time", "scheduled"]
    assert ev["time"].dtype == np.dtype("datetime64[ns]")            # UTC, tz-naive
    assert ev["scheduled"].dtype == np.dtype(bool)
    assert set(ev["event"]) == set(KINDS)
    assert ev["time"].is_monotonic_increasing
    assert raw["datetime_utc"].str.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:00Z").all()
    assert not raw.duplicated(["event", "datetime_utc"]).any()


def test_kind_filter_and_validation():
    only = load_events(["CPI_RELEASE"])
    assert set(only["event"]) == {"CPI_RELEASE"} and only["time"].is_monotonic_increasing
    assert len(only) + len(load_events(["FOMC_DECISION"])) == len(load_events())
    with pytest.raises(ValueError):
        load_events(["FOMC"])


def test_times_strictly_increasing_per_kind(ev):
    for kind, g in ev.groupby("event"):
        assert g["time"].is_monotonic_increasing and g["time"].is_unique, kind


def test_every_row_has_official_source_and_notes():
    raw = _raw()
    assert (raw["notes"].str.len() > 0).all()
    assert raw["source_url"].str.match(r"https://www\.(federalreserve|bls)\.gov/").all()
    fomc = raw[raw["event"] == "FOMC_DECISION"]
    cpi = raw[raw["event"] == "CPI_RELEASE"]
    assert fomc["source_url"].str.contains("federalreserve.gov").all()
    assert cpi["source_url"].str.contains("bls.gov").all()


def test_coverage_and_future_rows_marked_scheduled(ev):
    assert ev["time"].min() >= pd.Timestamp("2021-01-01") and ev["time"].max() < pd.Timestamp("2027-01-01")
    raw = _raw()
    when = pd.to_datetime(raw["datetime_utc"], format="%Y-%m-%dT%H:%M:%SZ")
    future = when.dt.date > AS_OF
    assert future.sum() == 4                                           # FOMC 2026-10-28, 2026-12-09; CPI 2026-11-10, 2026-12-10
    assert raw.loc[future, "notes"].str.startswith("scheduled").all()
    assert not raw.loc[~future, "notes"].str.startswith("scheduled").any()


# ---------------------------------------------------------------- release times and DST -------------
def test_fomc_times_are_1800_or_1900_utc_unless_noted(ev):
    raw = _raw()
    for r, t in zip(raw.itertuples(), pd.to_datetime(raw["datetime_utc"], format="%Y-%m-%dT%H:%M:%SZ")):
        if r.event != "FOMC_DECISION":
            continue
        if "TIME_EXCEPTION" in r.notes:
            continue
        assert (t.hour, t.minute) in {(18, 0), (19, 0)}, (r.datetime_utc, r.notes)


def test_cpi_times_are_1230_or_1330_utc_unless_noted():
    raw = _raw()
    for r, t in zip(raw.itertuples(), pd.to_datetime(raw["datetime_utc"], format="%Y-%m-%dT%H:%M:%SZ")):
        if r.event != "CPI_RELEASE" or "TIME_EXCEPTION" in r.notes:
            continue
        assert (t.hour, t.minute) in {(12, 30), (13, 30)}, (r.datetime_utc, r.notes)


def test_only_known_time_exception_is_the_2025_notation_vote():
    raw = _raw()
    exc = raw[raw["notes"].str.contains("TIME_EXCEPTION")]
    assert list(exc["datetime_utc"]) == ["2025-08-22T14:00:00Z"]       # 10:00 am EDT
    row = exc.iloc[0]
    assert row["event"] == "FOMC_DECISION" and row["scheduled"] == "false" and "NOTATION VOTE" in row["notes"]


def test_local_wall_clock_matches_release_convention(ev):
    """Convert back with zoneinfo: FOMC must be 14:00 ET, CPI 08:30 ET, on weekdays (notation vote: 10:00 ET)."""
    for r in ev.itertuples():
        loc = _local(r.time)
        want = (10, 0) if r.time == pd.Timestamp("2025-08-22 14:00") else ((14, 0) if r.event == "FOMC_DECISION" else (8, 30))
        assert (loc.hour, loc.minute) == want, (r.event, r.time)
        assert loc.weekday() < 5


def test_utc_offset_follows_us_dst_rule_for_every_row(ev):
    """Independent of zoneinfo: UTC hour = local hour + 4 in DST (2nd Sun Mar - 1st Sun Nov), +5 otherwise."""
    for r in ev.itertuples():
        loc = _local(r.time)
        offset_h = 4 if _is_dst(loc.date()) else 5
        assert r.time.hour == loc.hour + offset_h, (r.event, r.time)


@pytest.mark.parametrize(
    "event, stamp",
    [
        # 2021 DST started Sun Mar 14: CPI before (EST) and after (EDT); FOMC on both sides
        ("CPI_RELEASE", "2021-03-10T13:30:00Z"),
        ("FOMC_DECISION", "2021-03-17T18:00:00Z"),
        ("CPI_RELEASE", "2021-04-13T12:30:00Z"),
        ("FOMC_DECISION", "2021-01-27T19:00:00Z"),
        # 2021 DST ended Sun Nov 7: FOMC Nov 3 still EDT, CPI Nov 10 already EST
        ("FOMC_DECISION", "2021-11-03T18:00:00Z"),
        ("CPI_RELEASE", "2021-11-10T13:30:00Z"),
        ("FOMC_DECISION", "2021-12-15T19:00:00Z"),
        # 2022 DST started Mar 13: CPI Mar 10 (EST) then FOMC Mar 16 (EDT)
        ("CPI_RELEASE", "2022-03-10T13:30:00Z"),
        ("FOMC_DECISION", "2022-03-16T18:00:00Z"),
        # 2023 DST ended Nov 5: FOMC Nov 1 EDT, CPI Nov 14 EST
        ("FOMC_DECISION", "2023-11-01T18:00:00Z"),
        ("CPI_RELEASE", "2023-11-14T13:30:00Z"),
        # 2024 DST ended Nov 3: FOMC Nov 7 is EST
        ("FOMC_DECISION", "2024-11-07T19:00:00Z"),
        ("CPI_RELEASE", "2024-11-13T13:30:00Z"),
        # 2026 DST starts Mar 8 and ends Nov 1: CPI Mar 11 EDT, FOMC Oct 28 EDT, FOMC Dec 9 EST, CPI Nov 10 EST
        ("CPI_RELEASE", "2026-03-11T12:30:00Z"),
        ("FOMC_DECISION", "2026-10-28T18:00:00Z"),
        ("CPI_RELEASE", "2026-11-10T13:30:00Z"),
        ("FOMC_DECISION", "2026-12-09T19:00:00Z"),
    ],
)
def test_dst_spot_checks_present(event, stamp):
    raw = _raw()
    assert ((raw["event"] == event) & (raw["datetime_utc"] == stamp)).sum() == 1


def test_dst_rule_helper_matches_known_switch_dates():
    assert _nth_sunday(2021, 3, 2) == dt.date(2021, 3, 14) and _nth_sunday(2021, 11, 1) == dt.date(2021, 11, 7)
    assert _nth_sunday(2024, 3, 2) == dt.date(2024, 3, 10) and _nth_sunday(2024, 11, 1) == dt.date(2024, 11, 3)
    assert _nth_sunday(2026, 3, 2) == dt.date(2026, 3, 8) and _nth_sunday(2026, 11, 1) == dt.date(2026, 11, 1)
    # the instant itself: 8:30 am ET the day before vs the day after the March switch
    assert dt.datetime(2021, 3, 13, 8, 30, tzinfo=NY).astimezone(UTC).hour == 13
    assert dt.datetime(2021, 3, 15, 8, 30, tzinfo=NY).astimezone(UTC).hour == 12
    assert dt.datetime(2021, 11, 6, 14, 0, tzinfo=NY).astimezone(UTC).hour == 18
    assert dt.datetime(2021, 11, 8, 14, 0, tzinfo=NY).astimezone(UTC).hour == 19


# ---------------------------------------------------------------- counts per year -------------------
def test_fomc_scheduled_decisions_are_eight_per_year(ev):
    f = ev[(ev["event"] == "FOMC_DECISION") & ev["scheduled"]]
    counts = f.groupby(f["time"].dt.year).size().to_dict()
    assert counts == {y: 8 for y in range(2021, 2027)}


def test_fomc_scheduled_decisions_fall_on_wednesdays_except_post_election_2024(ev):
    f = ev[(ev["event"] == "FOMC_DECISION") & ev["scheduled"]]
    for t in f["time"]:
        wd = _local(t).weekday()
        assert wd == 2 or (t == pd.Timestamp("2024-11-07 19:00") and wd == 3), t


def test_unscheduled_fomc_rows_are_only_the_2025_notation_vote(ev):
    u = ev[~ev["scheduled"]]
    assert list(u["time"]) == [pd.Timestamp("2025-08-22 14:00")] and list(u["event"]) == ["FOMC_DECISION"]
    # no emergency (intermeeting) policy action occurred 2021-2026
    raw = _raw()
    assert raw.loc[raw["scheduled"] == "false", "notes"].str.contains("NOTATION VOTE").all()


def test_cpi_releases_per_year_only_2025_differs(ev):
    c = ev[ev["event"] == "CPI_RELEASE"]
    counts = c.groupby(c["time"].dt.year).size().to_dict()
    assert counts == {2021: 12, 2022: 12, 2023: 12, 2024: 12, 2025: 11, 2026: 12}
    raw = _raw()
    note = raw.loc[raw["datetime_utc"] == "2025-12-18T13:30:00Z", "notes"].iloc[0]
    assert "October 2025 CPI" in note and "CANCELED" in note and "11 releases" in note


def test_cpi_reference_months_are_consecutive_except_october_2025():
    raw = _raw()
    c = raw[raw["event"] == "CPI_RELEASE"].sort_values("datetime_utc")
    refs = []
    for n in c["notes"]:
        m = re.search(r"CPI for ([A-Z][a-z]{2}) (\d{4})", n)
        refs.append(dt.datetime.strptime(f"{m.group(1)} {m.group(2)}", "%b %Y"))
    months = [r.year * 12 + r.month for r in refs]
    expected = [m for m in range(2020 * 12 + 12, 2026 * 12 + 11 + 1) if m != 2025 * 12 + 10]
    assert months == expected


def test_shutdown_delays_use_actual_datetimes_and_say_so():
    raw = _raw().set_index("datetime_utc")
    sep = raw.loc["2025-10-24T12:30:00Z", "notes"]
    nov = raw.loc["2025-12-18T13:30:00Z", "notes"]
    jan = raw.loc["2026-02-13T13:30:00Z", "notes"]
    assert "DELAYED" in sep and "2025-10-15" in sep and "Sep 2025" in sep
    assert "DELAYED" in nov and "2025-12-10" in nov and "Nov 2025" in nov
    assert "DELAYED" in jan and "2026-02-11" in jan and "Jan 2026" in jan
    # the originally scheduled dates must NOT be in the table
    for original in ("2025-10-15T12:30:00Z", "2025-11-13T13:30:00Z", "2025-12-10T13:30:00Z", "2026-02-11T13:30:00Z"):
        assert original not in raw.index
    assert "Oct 2025" not in " ".join(raw["notes"].str.extract(r"(CPI for \w{3} \d{4})")[0].dropna())


# ---------------------------------------------------------------- event_features: hand-made example ---
@pytest.fixture()
def toy():
    return pd.DataFrame(
        {
            "event": ["FOMC_DECISION", "FOMC_DECISION", "FOMC_DECISION", "CPI_RELEASE", "CPI_RELEASE"],
            "time": pd.to_datetime(
                ["2024-01-10 19:00", "2024-01-15 12:00", "2024-01-20 19:00", "2024-01-12 13:30", "2024-01-25 13:30"]
            ).astype("datetime64[ns]"),
            "scheduled": [True, False, True, True, True],        # Jan 15 12:00 is an emergency action
        }
    )


TOY_TIMES = np.array(
    [
        "2024-01-10T00:00", "2024-01-10T17:00", "2024-01-10T19:00", "2024-01-10T21:00", "2024-01-10T21:01",
        "2024-01-15T11:00", "2024-01-15T12:00", "2024-01-15T13:30", "2024-01-20T19:00", "2024-02-20T00:00",
    ],
    dtype="datetime64[ns]",
)


def test_event_features_hand_made_example(toy):
    out = event_features(TOY_TIMES, toy)
    assert set(out) == {f"{k}_{s}" for k in KINDS for s in ("hours_to_next", "hours_since_last", "window")}
    for v in out.values():
        assert v.dtype == np.float32 and v.shape == TOY_TIMES.shape
    one_min = 1 / 60
    np.testing.assert_allclose(
        out["FOMC_DECISION_hours_to_next"],
        [19, 2, 0, 238, 238 - one_min, 128, 127, 125.5, 0, 336], rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        out["FOMC_DECISION_hours_since_last"],
        [336, 336, 0, 2, 2 + one_min, 112, 0, 1.5, 0, 336], rtol=0, atol=1e-4)
    np.testing.assert_array_equal(out["FOMC_DECISION_window"], [0, 1, 1, 1, 0, 0, 1, 1, 1, 0])
    np.testing.assert_allclose(
        out["CPI_RELEASE_hours_to_next"],
        [61.5, 44.5, 42.5, 40.5, 40.5 - one_min, 242.5, 241.5, 240, 114.5, 336], rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        out["CPI_RELEASE_hours_since_last"],
        [336, 336, 336, 336, 336, 69.5, 70.5, 72, 197.5, 336], rtol=0, atol=1e-4)
    np.testing.assert_array_equal(out["CPI_RELEASE_window"], np.zeros(10))


def test_window_is_inclusive_plus_minus_two_hours(toy):
    cpi = pd.Timestamp("2024-01-12 13:30")
    mins = [-121, -120, -1, 0, 1, 119, 120, 121]
    t = np.array([(cpi + pd.Timedelta(minutes=m)).to_datetime64() for m in mins], dtype="datetime64[ns]")
    out = event_features(t, toy, kinds=("CPI_RELEASE",))
    np.testing.assert_array_equal(out["CPI_RELEASE_window"], [0, 1, 1, 1, 1, 1, 1, 0])
    np.testing.assert_allclose(out["CPI_RELEASE_hours_to_next"][:4], [121 / 60, 2, 1 / 60, 0], atol=1e-5)
    np.testing.assert_allclose(out["CPI_RELEASE_hours_since_last"][3:7], [0, 1 / 60, 119 / 60, 2], atol=1e-5)


def test_cap_clipping_and_missing_neighbours(toy):
    t = np.array(["2024-01-01T00:00", "2025-01-01T00:00"], dtype="datetime64[ns]")
    out = event_features(t, toy, cap_hours=100)
    assert out["FOMC_DECISION_hours_since_last"][0] == 100 and out["FOMC_DECISION_hours_to_next"][1] == 100
    assert out["FOMC_DECISION_hours_to_next"][0] == 100              # 232 h away -> clipped
    assert out["CPI_RELEASE_hours_since_last"][1] == 100            # 2024-01-25 is far in the past
    empty = event_features(t, toy.iloc[0:0])
    assert (empty["FOMC_DECISION_hours_to_next"] == 336).all() and (empty["CPI_RELEASE_window"] == 0).all()


def test_unsorted_and_pandas_inputs_give_identical_results(toy):
    shuffled = TOY_TIMES[::-1]
    a = event_features(TOY_TIMES, toy)
    b = event_features(shuffled, toy)
    c = event_features(pd.DatetimeIndex(TOY_TIMES), toy)
    d = event_features(pd.DatetimeIndex(TOY_TIMES).tz_localize("UTC"), toy)
    for k in a:
        np.testing.assert_array_equal(a[k][::-1], b[k])
        np.testing.assert_array_equal(a[k], c[k])
        np.testing.assert_array_equal(a[k], d[k])
    with pytest.raises(ValueError):
        event_features(np.array(["NaT"], dtype="datetime64[ns]"), toy)


def test_missing_scheduled_column_treats_all_rows_as_scheduled(toy):
    out_all = event_features(TOY_TIMES, toy.assign(scheduled=True))
    out_missing = event_features(TOY_TIMES, toy.drop(columns="scheduled"))
    for k in out_all:
        np.testing.assert_array_equal(out_all[k], out_missing[k])


# ---------------------------------------------------------------- unscheduled events are causal --------
def test_unscheduled_event_never_counts_before_it_occurs_toy(toy):
    # the emergency action at 2024-01-15 12:00 must not shorten the countdown or open the window beforehand
    before = np.array(["2024-01-15T06:00", "2024-01-15T10:00", "2024-01-15T11:59"], dtype="datetime64[ns]")
    out = event_features(before, toy)
    to_next = out["FOMC_DECISION_hours_to_next"]
    expected = (np.datetime64("2024-01-20T19:00", "ns") - before) / H            # next SCHEDULED event
    np.testing.assert_allclose(to_next, expected, atol=1e-4)
    assert (out["FOMC_DECISION_window"] == 0).all()
    # ... and after it occurred it is the "last" event and opens the +2 h window
    after = np.array(["2024-01-15T12:00", "2024-01-15T13:00", "2024-01-15T14:00", "2024-01-15T14:01"], dtype="datetime64[ns]")
    out = event_features(after, toy)
    np.testing.assert_allclose(out["FOMC_DECISION_hours_since_last"], [0, 1, 2, 2 + 1 / 60], atol=1e-4)
    np.testing.assert_array_equal(out["FOMC_DECISION_window"], [1, 1, 1, 0])


def test_unscheduled_rows_in_real_table_never_enter_hours_to_next(ev):
    u = ev[~ev["scheduled"]]
    assert len(u) >= 1
    sched = ev[ev["scheduled"]]
    for row in u.itertuples():
        t0 = row.time.to_datetime64()
        before = np.array([t0 - np.timedelta64(m, "m") for m in (1, 30, 60, 119, 121, 600)], dtype="datetime64[ns]")
        out = event_features(before, ev, kinds=(row.event,))
        nxt = sched[(sched["event"] == row.event) & (sched["time"] > row.time)]["time"].min()
        want = np.minimum((nxt.to_datetime64() - before) / H, 336.0)
        np.testing.assert_allclose(out[f"{row.event}_hours_to_next"], want, atol=1e-3)
        assert (out[f"{row.event}_hours_to_next"] > 1.9).all()
        assert (out[f"{row.event}_window"] == 0).all()                  # no scheduled event nearby either
        after = np.array([t0, t0 + np.timedelta64(1, "h"), t0 + np.timedelta64(121, "m")], dtype="datetime64[ns]")
        out = event_features(after, ev, kinds=(row.event,))
        np.testing.assert_allclose(out[f"{row.event}_hours_since_last"], [0, 1, 121 / 60], atol=1e-4)
        np.testing.assert_array_equal(out[f"{row.event}_window"], [1, 1, 0])


def test_every_scheduled_event_is_anticipated_in_real_table(ev):
    s = ev[ev["scheduled"]]
    for kind in KINDS:
        times = s[s["event"] == kind]["time"].to_numpy()
        out = event_features(times - np.timedelta64(1, "h"), ev, kinds=(kind,))
        np.testing.assert_allclose(out[f"{kind}_hours_to_next"], 1.0, atol=1e-5)
        assert (out[f"{kind}_window"] == 1).all()
        out0 = event_features(times, ev, kinds=(kind,))
        assert (out0[f"{kind}_hours_to_next"] == 0).all() and (out0[f"{kind}_hours_since_last"] == 0).all()


def test_event_features_matches_brute_force_on_real_table(ev):
    rng = np.random.default_rng(0)
    lo, hi = np.datetime64("2020-12-31", "ns").astype(np.int64), np.datetime64("2027-01-02", "ns").astype(np.int64)
    ns = rng.integers(lo, hi, 4000)
    # force coverage of exact event instants and +/- 2 h boundaries
    e = ev["time"].to_numpy().astype(np.int64)
    ns = np.concatenate([ns, e, e - 2 * 3600 * 10**9, e + 2 * 3600 * 10**9, e - 1, e + 1])
    t = ns.astype("datetime64[ns]")
    out = event_features(t, ev)
    cap = 24 * 14 * 3600 * 10**9
    win = 2 * 3600 * 10**9
    for kind in KINDS:
        sub = ev[ev["event"] == kind]
        allv = sub["time"].to_numpy().astype(np.int64)
        sch = sub.loc[sub["scheduled"], "time"].to_numpy().astype(np.int64)
        d_next = sch[None, :] - ns[:, None]
        d_next = np.where(d_next >= 0, d_next, np.iinfo(np.int64).max).min(axis=1)
        d_prev = ns[:, None] - allv[None, :]
        d_prev = np.where(d_prev >= 0, d_prev, np.iinfo(np.int64).max).min(axis=1)
        to_next = np.minimum(d_next, cap) / 3.6e12
        since = np.minimum(d_prev, cap) / 3.6e12
        w = ((d_next <= win) | (d_prev <= win)).astype(np.float32)
        np.testing.assert_allclose(out[f"{kind}_hours_to_next"], to_next.astype(np.float32), rtol=0, atol=1e-6)
        np.testing.assert_allclose(out[f"{kind}_hours_since_last"], since.astype(np.float32), rtol=0, atol=1e-6)
        np.testing.assert_array_equal(out[f"{kind}_window"], w)
        assert out[f"{kind}_hours_to_next"].min() >= 0 and out[f"{kind}_hours_to_next"].max() <= 336
        assert out[f"{kind}_hours_since_last"].min() >= 0 and out[f"{kind}_hours_since_last"].max() <= 336
