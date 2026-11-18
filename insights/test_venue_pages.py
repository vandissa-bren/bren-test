"""Tests for venue_pages.py: the pictures on a venue's Insights page."""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import privacy  # noqa: E402
import venue_pages as VP  # noqa: E402
from test_figures import MON, at, bookings, sess  # noqa: E402

END = MON + timedelta(days=55)          # 8 weeks of Mondays from MON
KM = {(1, 2): 3.0, (1, 3): 20.0, (2, 3): 18.0}


def km(a, b):
    a, b = int(a), int(b)
    return 0.0 if a == b else KM.get((min(a, b), max(a, b)), float("nan"))


def _week(n, **kw):
    return [sess(d=MON + timedelta(days=7 * w), **{k: (v(w) if callable(v) else v) for k, v in kw.items()}) for w in range(n)]


def _frame():
    rows = []
    # venue 1: Mon 7pm Open Play (full), Wed 10am Social (half), both weekly for 8 weeks
    rows += _week(8, fid=1, title="Open Play", h=19, booked=16, cap=16)
    rows += _week(8, fid=1, title="Morning Social", h=10, booked=8, cap=16, fgroup="Social & open play", lvl="all levels")
    # venue 2 (3 km): Mon 7pm intermediate too, 75% full
    rows += _week(8, fid=2, title="Club Night", h=19, booked=12, cap=16)
    # venue 3 (20 km): advanced social on Thursdays
    rows += [dict(r, starts_at=r["starts_at"] + pd.Timedelta(days=3)) for r in
             _week(8, fid=3, title="Advanced Social", h=19, booked=14, cap=16, lvl="advanced")]
    f = pd.DataFrame(rows)
    f["price_hr"] = [{1: 16.0, 2: 12.0, 3: 14.0}[v] for v in f["fid"]]
    f["hour"] = f["starts_at"].dt.tz_convert("Australia/Melbourne").dt.hour
    return f


def _bookings(f):
    """Players 100-111 play Open Play only; 200-206 play both venue-1 sessions;
    300-307 play Open Play and travel to venue 3's advanced social (8 of them)."""
    keys = {(r.fid, r.title, r.starts_at): r.session_key for r in f.itertuples()}
    o = []
    for w in range(8):
        d = MON + timedelta(days=7 * w)
        op, ms = keys[(1, "Open Play", at(d, 19))], keys[(1, "Morning Social", at(d, 10))]
        adv = keys[(3, "Advanced Social", at(d, 19) + pd.Timedelta(days=3))]
        for p in range(100, 112):
            o.append((p, 1, op, at(d, 19)))
        for p in range(200, 207):
            o.append((p, 1, op, at(d, 19)))
            o.append((p, 1, ms, at(d, 10)))
        for p in range(300, 308):
            o.append((p, 1, op, at(d, 19)))
            o.append((p, 3, adv, at(d, 19) + pd.Timedelta(days=3)))
    return bookings(o)


def build():
    f = _frame()
    return VP.build(f, _bookings(f), None, END, km, [1, 2, 3])


def test_trend_has_eight_weeks_against_the_city():
    v = build()["venues"]["1"]
    assert len(v["trend"]["here"]) == 8 and v["trend"]["here"][-1] == 75.0
    assert v["trend"]["city"][-1] == round(100 * (1 + .5 + .75 + 14 / 16) / 4, 1)


def test_anchors_count_players_who_come_for_nothing_else():
    a = {r["title"]: r for r in build()["venues"]["1"]["anchors"]}
    op = a["Open Play"]
    assert op["players"] == 27 and op["fill"] == 100.0
    assert op["only"] == round(100 * 20 / 27, 1)            # 12 + 8 play only Open Play there
    ms = a["Morning Social"]
    assert ms["players"] == 7 and ms["only"] == 0.0


def test_supply_flags_its_own_kinds_within_10_km():
    s = build()["venues"]["1"]["supply"]
    kinds = {(r["kind"], r["part"]): r for r in s}
    assert kinds[("Intermediate social & open play", "evenings")]["mine"] is True
    assert kinds[("Intermediate social & open play", "evenings")]["perWeek"] == 2.0     # its own and venue 2's
    assert not any(r["kind"].startswith("Advanced") for r in s)                        # venue 3 is 20 km away


def test_travel_shows_what_regulars_drive_for():
    t = build()["venues"]["1"]["travel"]
    assert t == [{"kind": "Advanced social & open play", "v": 3, "players": 8, "km": 20.0, "runsIt": False}]


def test_prices_against_nearby_sessions():
    p = build()["venues"]["1"]["prices"]
    assert p["median"] == 12.0 and p["others"] == [12.0]
    assert {m["title"]: m["price"] for m in p["mine"]} == {"Open Play": 16.0, "Morning Social": 16.0}


def test_share_compares_two_28_day_periods():
    s = build()["venues"]["1"]["share"]
    assert s["ready"]
    row = next(r for r in s["rows"] if r["kind"] == "Intermediate social & open play")
    assert row["before"] == 100.0 and row["now"] == 100.0     # venue 2's sessions have no rosters


def test_small_groups_are_held_back():
    f = _frame()
    o = _bookings(f)
    o = o[~o["pbp_user_id"].between(203, 206)]                 # 3 play both sessions now
    v = VP.build(f, o, None, END, km, [1, 2, 3])["venues"]["1"]
    a = {r["title"]: r for r in v["anchors"]}
    assert a["Morning Social"]["players"] is None
    assert a["Open Play"]["only"] is None                      # 20 of 23: the other 3 are a small group


def test_newcomers_back_within_28_days():
    f = _frame()
    o = _bookings(f)
    # 12 first-timers at Open Play on 12 Oct; 6 come back a week later
    k12 = f[(f["fid"] == 1) & (f["title"] == "Open Play") & (f["starts_at"] == at(MON + timedelta(days=14), 19))]["session_key"].iloc[0]
    k19 = f[(f["fid"] == 1) & (f["title"] == "Open Play") & (f["starts_at"] == at(MON + timedelta(days=21), 19))]["session_key"].iloc[0]
    extra = [(900 + i, 1, k12, at(MON + timedelta(days=14), 19)) for i in range(12)]
    extra += [(900 + i, 1, k19, at(MON + timedelta(days=21), 19)) for i in range(6)]
    o = pd.concat([o[o["starts_at"] >= at(MON + timedelta(days=14), 0)], bookings(extra)])
    nc = VP.build(f, o, None, END, km, [1, 2, 3])["venues"]["1"]["newcomers"]
    row = next(r for r in nc["rows"] if r["title"] == "Open Play")
    assert row["n"] >= 12 and row["back"] is not None


def test_privacy_paths_cover_the_pages():
    d = {"figures": {"pages": build()}}
    d["figures"]["pages"]["venues"]["1"]["anchors"][0]["players"] = 3
    privacy.enforce(d)
    assert d["figures"]["pages"]["venues"]["1"]["anchors"][0]["players"] is None
