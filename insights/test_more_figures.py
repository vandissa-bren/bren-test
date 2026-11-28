"""Price changes, social groups, the player path, session health (more_figures.py)."""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import more_figures as MF  # noqa: E402
import privacy  # noqa: E402
from test_figures import MON, at, bookings, sess  # noqa: E402

END = MON + timedelta(days=76)


def weekly(fid, title, h, n, price, booked, cap=16):
    rows = []
    for w in range(n):
        r = sess(d=MON + timedelta(days=7 * w), fid=fid, title=title, h=h, booked=booked(w) if callable(booked) else booked, cap=cap)
        r["price_n"] = price(w) if callable(price) else price
        rows.append(r)
    return rows


def frame():
    rows = weekly(1, "Open Play", 19, 11, lambda w: 20.0 if w < 5 else 22.0, lambda w: 16 if w < 5 else 12)
    rows += weekly(1, "Morning Social", 10, 11, 15.0, 8)
    rows += weekly(2, "Club Night", 19, 11, 18.0, 14)
    f = pd.DataFrame(rows)
    f["fill"] = f["booked"] / f["capacity"]
    return f


def okv_for(f):
    keys = {(r.fid, r.title, r.starts_at): r.session_key for r in f.itertuples()}
    o = []
    for w in range(11):
        d = MON + timedelta(days=7 * w)
        op, ms = keys[(1, "Open Play", at(d, 19))], keys[(1, "Morning Social", at(d, 10))]
        cn = keys[(2, "Club Night", at(d, 19))]
        # a group of 6 who book together: Open Play every third week, Club Night the week after
        for p in range(10, 16):
            if w % 3 == 0:
                o.append((p, 1, op, at(d, 19)))
            elif w % 3 == 1:
                o.append((p, 2, cn, at(d, 19)))
        # regulars who come every week (no evidence of a group: they never miss)
        for p in range(100, 120):
            o.append((p, 1, op, at(d, 19)))
        for p in range(300, 310):
            o.append((p, 2, cn, at(d, 19)))
        # a scatter of others at random weeks
        for p in range(200, 230):
            if w % 2 == p % 2:
                o.append((p, 1, ms, at(d, 10)))
    # first-timers on 12 Oct (week 2): 10 at Morning Social, 5 come back
    d = MON + timedelta(days=14)
    for p in range(500, 510):
        o.append((p, 1, keys[(1, "Morning Social", at(d, 10))], at(d, 10)))
    for p in range(500, 505):
        for w in (3, 4, 5, 6):
            dd = MON + timedelta(days=7 * w)
            o.append((p, 1, keys[(1, "Morning Social", at(dd, 10))], at(dd, 10)))
    return bookings(o)


def build():
    f = frame()
    return MF.build(f, okv_for(f), None, END, [1, 2])


def test_price_change_found_with_fill_either_side():
    p = build()["prices"]
    ch = p["venues"]["1"]["sessions"]
    assert len(ch) == 1 and ch[0]["from"] == 20 and ch[0]["to"] == 22
    assert ch[0]["fillBefore"] == 100.0 and ch[0]["fillAfter"] == 75.0
    assert p["city"]["changes"] == 1 and p["city"]["up"] == 1


def test_hire_rate_changes():
    f = frame()
    R = lambda c, p, a, b: {"facility_id": 1, "court_id": c, "shift": "peak", "price": p, "member_price": None,
                            "first_seen": a, "last_seen": b}
    rates = pd.DataFrame([
        R("c1", 40, "2026-10-01T00:00:00Z", "2026-10-31T00:00:00Z"), R("c1", 44, "2026-11-01T00:00:00Z", "2026-11-26T00:00:00Z"),
        R("c2", 40, "2026-10-01T00:00:00Z", "2026-10-31T00:00:00Z"), R("c2", 44, "2026-11-01T00:00:00Z", "2026-11-26T00:00:00Z"),
        # a one-day flicker on c3 isn't a change
        R("c3", 40, "2026-10-01T00:00:00Z", "2026-10-20T00:00:00Z"), R("c3", 30, "2026-10-21T00:00:00Z", "2026-10-21T12:00:00Z"),
        R("c3", 40, "2026-10-22T00:00:00Z", "2026-11-26T00:00:00Z")])
    h = MF.build(f, okv_for(f), None, END, [1, 2], rates)["prices"]["venues"]["1"]["hire"]
    assert h == [{"shift": "peak", "date": "2026-11-01", "from": 40.0, "to": 44.0, "courts": 2}]


def test_a_group_that_moves_between_venues():
    g = build()["groups"]
    assert g["city"]["groups"] >= 1
    sizes = [x["size"] for v in g["venues"].values() for x in v["list"]]
    assert 6 in sizes
    assert g["city"]["acrossVenues"] >= 1


def test_player_path():
    p = build()["paths"]["venues"]["1"]
    assert p["firsts"] == 10 and p["once"] == 50.0 and p["cameBack"] == 50.0
    assert p["regular"] == 50.0                   # the 5 who came back played 4+ times
    assert p["regularsN"] == 5 and p["lapsed"] == 100.0      # none played there in the last 4 weeks


def test_health_scores_and_grades():
    h = build()["health"]["venues"]["1"]
    by = {r["title"]: r for r in h}
    op = by["Open Play"]
    assert op["trend"] < 0 and op["grade"] in "ABCDE"
    assert by["Open Play"]["score"] > by["Morning Social"]["score"]


def test_privacy_holds_small_counts():
    d = {"figures": build()}
    d["figures"]["paths"]["venues"]["1"]["firsts"] = 3
    privacy.enforce(d)
    assert d["figures"]["paths"]["venues"]["1"]["firsts"] is None


def test_path_window_is_4_weeks_while_the_data_is_young():
    p = build()["paths"]
    assert p["days"] == 28 and p["regularSessions"] == 3


def test_health_grades_by_fifths_of_the_city():
    h = build()["health"]
    assert sum(h["city"]["grades"].values()) == h["city"]["sessions"]


def test_groups_need_a_third_player_in_common():
    from more_figures import core_groups
    # two trios joined by one pair link stay two groups; a chain is no group
    assert sorted(map(sorted, core_groups([(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 6), (6, 4)]))) == [[1, 2, 3], [4, 5, 6]]
    assert core_groups([(1, 2), (2, 3), (3, 4)]) == []
