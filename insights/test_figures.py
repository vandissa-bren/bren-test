"""Tests for figures.py: the seven figures and sessions to cut or move."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figures as FG  # noqa: E402
import privacy  # noqa: E402

MEL = ZoneInfo("Australia/Melbourne")
MON = date(2026, 9, 28)


def at(d: date, h=19, m=0):
    return pd.Timestamp(datetime(d.year, d.month, d.day, h, m, tzinfo=MEL)).tz_convert("UTC")


def sess(fid=1, title="Intermediate Open Play", d=MON, h=19, cap=16, booked=10, so_h=None, ahead_h=240,
         wl=0, wfa_h=None, key=None, fgroup="Social & open play", lvl="intermediate", slug=None):
    st = at(d, h)
    full = booked >= cap
    return {"session_key": key or f"k-{fid}-{d}-{h}-{title}", "fid": fid, "starts_at": st, "title": title,
            "label": title, "program_slug": slug, "session_type": "open play", "lvl": lvl, "fgroup": fgroup,
            "capacity": cap, "booked": booked, "fill": booked / cap, "sold_out": full,
            "so_h": so_h, "first_obs": st - pd.Timedelta(hours=ahead_h),
            "first_full_at": (st - pd.Timedelta(hours=so_h)) if so_h is not None else (st - pd.Timedelta(hours=ahead_h) if full else pd.NaT),
            "waitlist_max": wl,
            "waitlist_first_at": (st - pd.Timedelta(hours=wfa_h)) if wfa_h is not None else pd.NaT}


def weekly(n, **kw):
    return [sess(d=MON + timedelta(days=7 * w), **{k: (v(w) if callable(v) else v) for k, v in kw.items()})
            for w in range(n)]


def F(rows):
    return pd.DataFrame(rows)


def bookings(rows):
    """rows: (player, fid, session_key, starts_at[, first_seen])"""
    out = []
    for r in rows:
        p, v, k, st = r[:4]
        out.append({"pbp_user_id": p, "fid": v, "session_key": k, "starts_at": st,
                    "first_seen": r[4] if len(r) > 4 else st - pd.Timedelta(hours=100)})
    return pd.DataFrame(out)


# ── 1. another court ──
def test_need_court_flags_three_of_four_early():
    f = F(weekly(5, so_h=lambda w: 60 if w != 2 else None, booked=lambda w: 16 if w != 2 else 10))
    out = FG.need_court(f)
    assert len(out["city"]) == 1
    r = out["city"][0]
    assert r["runs"] == [True, False, True, True] and r["hoursEarly"] == 60.0 and r["waitlist"] is False
    assert r["label"] == "Mon 7pm Intermediate Open Play"


def test_need_court_already_full_counts_only_when_seen_48h_out():
    # full when first seen (so_h None), first seen 30 h ahead: not watched from 48 h, not early
    f = F(weekly(4, booked=16, so_h=None, ahead_h=30))
    assert FG.need_court(f)["city"] == []
    f = F(weekly(4, booked=16, so_h=None, ahead_h=72))
    assert len(FG.need_court(f)["city"]) == 1


def test_need_court_waitlist_must_appear_a_day_ahead_when_known():
    late = F(weekly(4, wl=3, wfa_h=5))
    assert FG.need_court(late)["city"] == []
    early = F(weekly(4, wl=3, wfa_h=30, booked=15))
    assert FG.need_court(early)["city"][0]["waitlist"] is True
    one = F(weekly(4, wl=1, wfa_h=30, booked=15))           # a waitlist of 1 isn't pressure
    assert FG.need_court(one)["city"] == []


def test_need_court_needs_four_runs():
    assert FG.need_court(F(weekly(3, so_h=60, booked=16)))["city"] == []


# ── 2. regulars who've gone quiet ──
def _quiet_frame(n_quiet_else=6, n_quiet_gone=6, n_active=3):
    rows, p = [], 1000
    end = MON + timedelta(days=7 * 8)
    for group, n in (("else", n_quiet_else), ("gone", n_quiet_gone), ("active", n_active)):
        for _ in range(n):
            p += 1
            weeks = range(0, 5) if group != "active" else range(0, 9)
            for w in weeks:
                d = MON + timedelta(days=7 * w)
                rows.append((p, 1, f"s1-{w}", at(d)))
            if group == "else":
                rows.append((p, 2, "s2-7", at(MON + timedelta(days=7 * 7))))
    return bookings(rows), end


def test_quiet_regulars_split():
    o, end = _quiet_frame()
    out = FG.quiet_regulars(o, end, tracked={1, 2})
    v = out["venues"]["1"]
    assert v["regulars"] == 15 and v["quiet"] == 12 and v["elsewhere"] == 6 and v["notSeen"] == 6
    assert v["to"] == [[2, 6]]
    assert out["provisional"] is True           # 23 Nov is 63 days after 21 Sep: under 10 weeks


def test_quiet_split_hidden_when_a_part_is_small():
    o, end = _quiet_frame(n_quiet_else=3, n_quiet_gone=7)
    v = FG.quiet_regulars(o, end, tracked={1, 2})["venues"]["1"]
    assert v["quiet"] == 10 and v["elsewhere"] is None and v["notSeen"] is None and v["to"] == []


def test_quiet_needs_four_weeks_and_fourteen_days():
    rows = [(1, 1, f"s{w}", at(MON + timedelta(days=7 * w))) for w in range(3)]   # only 3 weeks
    out = FG.quiet_regulars(bookings(rows), MON + timedelta(days=60), tracked={1})
    assert "1" not in out["venues"] or out["venues"]["1"]["regulars"] in (0, None)
    # weekly regular, last seen 10 days ago: due after 14 days, not quiet yet
    rows = [(p, 1, f"s{w}", at(MON + timedelta(days=7 * w))) for p in range(5) for w in range(5)]
    end = MON + timedelta(days=7 * 4 + 10)
    assert FG.quiet_regulars(bookings(rows), end, tracked={1})["venues"]["1"]["quiet"] == 0


def test_quiet_provisional_until_ten_weeks():
    o, _ = _quiet_frame()
    assert FG.quiet_regulars(o, FG.DATA_START + timedelta(days=69), {1})["provisional"] is True
    assert FG.quiet_regulars(o, FG.DATA_START + timedelta(days=70), {1})["provisional"] is False


# ── 3. new players coming back ──
def test_retention_cohorts_and_windows():
    rows = []
    c0 = date(2026, 10, 12)
    for p in range(25):                            # first seen week of 12 Oct; 10 back within a week
        rows.append((p, 1, "a", at(c0)))
        if p < 10:
            rows.append((p, 1, "b", at(c0 + timedelta(days=5))))
    for p in range(100, 125):                      # first seen before the cohorts start: left out
        rows.append((p, 1, "z", at(date(2026, 10, 5))))
    end = c0 + timedelta(days=6 + 14)
    out = FG.retention(bookings(rows), None, end)
    c = out["city"][0]
    assert c["from"] == "2026-10-12" and c["n"] == 25 and c["w1"] == 40.0 and c["w2"] == 40.0
    assert c["w4"] is None and c["w8"] is None     # their windows haven't passed
    assert len(out["city"]) == 1


def test_retention_small_weeks_join_their_month():
    rows = []
    for i, wk in enumerate((date(2026, 10, 12), date(2026, 10, 19))):
        for p in range(12):
            rows.append((1000 * (i + 1) + p, 1, f"k{i}", at(wk)))
    out = FG.retention(bookings(rows), None, date(2026, 12, 31))
    assert [c["by"] for c in out["city"]] == ["month"] and out["city"][0]["n"] == 24
    rows = [(p, 1, "k", at(date(2026, 10, 12))) for p in range(12)]   # 12 alone: dropped
    assert FG.retention(bookings(rows), None, date(2026, 12, 31))["city"] == []


def test_retention_rates_pass_privacy():
    d = {"figures": {"retention": {"city": [{"n": 25, "w1": 12.0, "w2": 40.0, "w4": 96.0, "w8": None}], "venues": {}}}}
    privacy.enforce(d)
    c = d["figures"]["retention"]["city"][0]
    assert c["w1"] is None            # 3 of 25
    assert c["w2"] == 40.0            # 10 of 25
    assert c["w4"] is None            # 24 of 25: 1 didn't


# ── 4. same faces ──
def test_same_faces_median_of_last_four():
    f = F(weekly(5))
    keys = list(f["session_key"])
    rows = []
    for i, k in enumerate(keys):
        st = f.loc[i, "starts_at"]
        core = list(range(8))                      # the same 8 every week
        extra = [100 + 10 * i + j for j in range(2)]
        for p in core + extra:
            rows.append((p, 1, k, st))
    out = FG.same_faces(f, bookings(rows))
    r = out["venues"]["1"][0]
    assert r["pct"] == 80.0 and r["players"] == 10 and r["runs"] == 4


def test_same_faces_needs_eight_players_and_skips_events():
    f = F(weekly(5))
    rows = [(p, 1, k, st) for k, st in zip(f["session_key"], f["starts_at"]) for p in range(6)]
    assert FG.same_faces(f, bookings(rows))["venues"] == {}
    rows = [(p, 1, k, st) for k, st in zip(f["session_key"], f["starts_at"]) for p in range(60)]
    assert FG.same_faces(f, bookings(rows))["venues"] == {}


# ── 5. where to run a new session ──
def _km(a, b):
    return {frozenset((1, 2)): 2.0, frozenset((1, 3)): 20.0}.get(frozenset((a, b)), 99.0)


def _where_world(own_half_empty=False):
    rows_f, rows_o = [], []
    # venue 2: Wed 7pm intermediate open play, sells out early, every week
    for w in range(6):
        d = MON + timedelta(days=2 + 7 * w)
        s = sess(fid=2, d=d, h=19, booked=16, so_h=60, key=f"v2-{w}")
        rows_f.append(s)
        for p in range(8):                         # venue 1's players who came twice or more
            rows_o.append((p, 2, s["session_key"], s["starts_at"]))
    # venue 1: Mon 10am beginner social, its own players
    for w in range(6):
        d = MON + timedelta(days=7 * w)
        s = sess(fid=1, d=d, h=10, title="Beginner Social", lvl="beginner", booked=10, key=f"v1-{w}")
        rows_f.append(s)
        for p in range(8):
            rows_o.append((p, 1, s["session_key"], s["starts_at"]))
    if own_half_empty:                             # venue 1 runs a half-empty session Wed 6:30pm
        for w in range(6):
            d = MON + timedelta(days=2 + 7 * w)
            rows_f.append(sess(fid=1, d=d, h=18, title="Round Robin", lvl="no level stated",
                               fgroup="Round robin", booked=4, key=f"v1rr-{w}"))
    return F(rows_f), bookings(rows_o)


def test_where_candidates_and_finish():
    f, o = _where_world()
    cand = FG.where_candidates(f, o, _km, lambda v: f"Venue {v}", [1, 2])
    rows = [r for r in cand["1"] if r["dow"] == 2 and r["h"] == 19]
    assert rows and rows[0]["players"] == 8 and rows[0]["nearby"] == 1
    d = {"_where": cand, "courtFree": {"cols": ["v", "dow", "h", "weeks", "free2", "freeMed", "courts", "hire", "prog", "hireH"],
                                       "rows": [[1, 2, 19, 6, 5, 3, 6, 0.2, 0.0, 10], [1, 2, 18, 6, 5, 3, 6, 0.2, 0.0, 10],
                                                [1, 2, 20, 6, 1, 1, 6, 0.8, 0.0, 10]]}}
    FG.where_finish(d)
    assert "_where" not in d
    got = d["figures"]["where"]["venues"]["1"]
    top = got[0]
    assert (top["dow"], top["h"], top["confidence"]) == (2, 19, "High")
    assert top["text"].startswith("8 of your players who come twice or more book intermediate social & open play elsewhere on Wednesday evenings around 7pm")
    assert "Venue 2 (2.0 km) runs intermediate social & open play on Wednesdays at 7pm" in top["text"]
    assert "sold out at least 2 days early in 6 of its last 6 weeks" in top["text"]
    assert "You have 2 or more courts free on Wednesdays at 7pm in 5 of the last 6 weeks (usually 3)." in top["text"]
    # 6pm rests on the same format the same evening: one suggestion, not two
    assert not any(r["dow"] == 2 and r["h"] == 18 for r in got)
    assert "more sold out" not in top["text"] and " and 1 more" not in top["text"]
    assert all(r["h"] != 20 for r in got)          # 20:00 has courts free in only 1 of 6 weeks


def test_where_blocked_by_own_half_empty_session_and_no_courts():
    f, o = _where_world(own_half_empty=True)
    cand = FG.where_candidates(f, o, _km, str, [1, 2])
    assert not any(r["dow"] == 2 and r["h"] in (18, 19) for r in cand["1"])
    d = {"_where": {"1": [{"dow": 2, "h": 19, "fmt": "x", "lvl": "y", "kind": "x", "players": 8, "nearby": 0, "nearbyBest": None}]},
         "courtFree": {"cols": [], "rows": []}}
    FG.where_finish(d)
    assert d["figures"]["where"]["venues"] == {} and "Court records" in d["figures"]["where"]["reason"]


def test_where_ignores_small_player_counts():
    f, o = _where_world()
    o = o[o["pbp_user_id"] < 4]                    # only 4 of venue 1's players go to venue 2
    cand = FG.where_candidates(f, o, _km, str, [1, 2])
    r = [x for x in cand.get("1", []) if x["dow"] == 2 and x["h"] == 19]
    assert r and r[0]["players"] is None and r[0]["nearby"] == 1   # still suggested on nearby demand alone


# ── 6. freed places refilled ──
def test_refilled_minimums():
    rows_f, rows_o = [], []
    for i in range(12):
        s = sess(d=MON + timedelta(days=i), booked=16, so_h=50, wl=2, key=f"w{i}")
        rows_f.append(s)
        seen = s["starts_at"] - pd.Timedelta(hours=10 if i < 7 else 80)   # 7 refilled after selling out
        rows_o.append((1, 1, s["session_key"], s["starts_at"], seen))
    out = FG.refilled(F(rows_f), bookings(rows_o))
    assert out["city"] == {"sessions": 12, "refilled": 7, "pct": 58.3}
    out = FG.refilled(F(rows_f[:9]), bookings(rows_o))
    assert out["city"]["pct"] is None              # under 10 waitlisted sessions


# ── 7. leaning on a few players ──
def test_leaning_top_fifth():
    rows = []
    end = MON + timedelta(days=27)
    for p in range(50):
        n = 5 if p < 10 else 1
        for j in range(n):
            rows.append((p, 1, f"k{p}-{j}", at(MON + timedelta(days=j))))
    out = FG.leaning(bookings(rows), end)
    v = out["venues"]["1"]
    assert v == {"players": 50, "top": 10, "bookings": 90, "pct": 55.6} and out["cityMedian"] == 55.6
    assert FG.leaning(bookings(rows[:45]), end)["venues"] == {}   # under 50 players


# ── 8. cut or move ──
def test_cut_or_move_with_better_nearby():
    f = F(weekly(4, booked=4) + weekly(4, fid=2, booked=13, key=None, title="Open Play Intermediate"))
    out = FG.cut_or_move(f, _km, lambda v: f"Venue {v}")
    r = out["venues"]["1"][0]
    assert r["fills"] == [25.0, 25.0, 25.0, 25.0]
    assert r["better"]["v"] == 2 and r["better"]["fill"] == 81.2
    assert "Venue 2 (2.0 km) runs intermediate social & open play at 7pm on Mondays and fills 81%." in r["text"]
    assert "2" not in out["venues"]


# ── privacy of the quiet split ──
def test_quiet_partitions():
    d = {"figures": {"quiet": {"city": {"regulars": 20, "quiet": 17}, "venues": {
        "1": {"regulars": 30, "quiet": 12, "elsewhere": 8, "notSeen": None, "to": [[2, 6]]}}}}}
    privacy.enforce(d)
    q = d["figures"]["quiet"]
    assert q["city"]["quiet"] is None              # 20 - 17 = 3 regulars still playing
    v = q["venues"]["1"]
    assert v["elsewhere"] is None                  # 12 - 8 would give "not seen" = 4
    assert v["to"] == []                           # with "elsewhere" hidden, where they went goes too
    d = {"figures": {"quiet": {"venues": {"1": {"regulars": 30, "quiet": 20, "elsewhere": 8, "notSeen": 12,
                                                "to": [[2, 6], [3, 8]]}}}}}
    privacy.enforce(d)
    v = d["figures"]["quiet"]["venues"]["1"]
    assert v["elsewhere"] == 8 and v["to"][0][1] is None and v["to"][1][1] == 8   # 8 - 6 = 2 gives one away


# ── the fixture build end to end ──
def test_fixture_build_has_every_figure():
    import fixture_build
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        fixture_build.write(t / "csv")
        out = t / "out.json"
        subprocess.run([sys.executable, str(HERE / "build_insights.py"), "--csv", str(t / "csv"), "--out", str(out)],
                       check=True, capture_output=True)
        fg = json.loads(out.read_text())["figures"]
    assert fg["needCourt"]["city"]
    assert any(v.get("quiet") for v in fg["quiet"]["venues"].values())
    assert any(c["w1"] is not None for c in fg["retention"]["city"])
    assert any(fg["sameFaces"]["venues"].values())
    assert fg["refilled"]["city"]["sessions"] >= 10
    assert fg["leaning"]["venues"] and fg["leaning"]["cityMedian"] is not None
    assert fg["cutOrMove"]["city"]
    assert any(fg["where"]["venues"].values())
    assert "_where" not in json.dumps(fg)
    # no player count of 1-4 anywhere in the figures
    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from walk(v, f"{path}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                yield from walk(v, f"{path}[{i}]")
        else:
            yield path, o
    player_keys = ("regulars", "quiet", "elsewhere", "notSeen", ".n", "players", ".top")
    for path, v in walk(fg):
        if any(path.endswith(k) for k in player_keys):
            assert not privacy.small(v), path


# ── review fixes (9 Nov) ──────────────────────────────────────────────────────
def test_privacy_quiet_hidden_total_not_recoverable_from_split():
    # critic #1: regulars 20, quiet 17 -> quiet hidden (3 still playing); the
    # split 10 + 7 would give it back, so the split goes too
    d = {"figures": {"quiet": {"venues": {"1": {"regulars": 20, "quiet": 17, "elsewhere": 10, "notSeen": 7,
                                                "to": [[2, 6]]}}}}}
    privacy.enforce(d)
    v = d["figures"]["quiet"]["venues"]["1"]
    assert v["quiet"] is None and v["elsewhere"] is None and v["notSeen"] is None and v["to"] == []
    tot = v["regulars"] - sum(x for x in (v["quiet"], v["elsewhere"], v["notSeen"]) if x is not None)
    assert tot == 20                                     # nothing to subtract


def test_privacy_city_quiet_minus_visible_venues():
    # critic #2: city 15 quiet, the only venue shown has 12 -> 3 at hidden venues
    d = {"figures": {"quiet": {"city": {"regulars": 60, "quiet": 15}, "venues": {
        "1": {"regulars": 40, "quiet": 12, "elsewhere": 6, "notSeen": 6, "to": []},
        "2": {"regulars": 8, "quiet": None, "elsewhere": None, "notSeen": None, "to": []}}}}}
    privacy.enforce(d)
    q = d["figures"]["quiet"]
    assert q["city"]["quiet"] is None                   # the city figure goes, not the venue's
    assert q["venues"]["1"]["quiet"] == 12
    for total, parts in privacy.partitions(d):
        rem = privacy.remainder(total, parts)
        assert rem is None or not privacy.small(rem)


def test_privacy_exact_breakdown_only_when_it_adds_up():
    d = {"newret": {"city": {"players": 30, "new": 3, "returning": 27}}}
    privacy.enforce(d)
    c = d["newret"]["city"]
    assert c["new"] is None and c["returning"] is None  # 30 - 27 would give the 3 back
    d = {"scorecard": {"city": {"newHere": 40, "series": {"newHere": [12, 3, 0]}}}}   # 6 weeks of a longer window
    privacy.enforce(d)
    assert d["scorecard"]["city"]["series"]["newHere"] == [12, None, 0]


def test_need_court_full_at_any_reading_and_hours_from_sold_out_weeks():
    # full 60 h ahead, then a late cancellation: not sold out at the start, still early
    rows = weekly(4, so_h=60, booked=15)
    for r in rows:
        r["first_full_at"] = r["starts_at"] - pd.Timedelta(hours=60)
        r["sold_out"] = False
    assert len(FG.need_court(F(rows))["city"]) == 1
    # 3 sold-out weeks at 72 h and a waitlist-only week: hoursEarly from the sold-out weeks
    rows = weekly(4, so_h=lambda w: 72 if w < 3 else None, booked=lambda w: 16 if w < 3 else 15,
                  wl=lambda w: 0 if w < 3 else 4, wfa_h=lambda w: None if w < 3 else 30)
    r = FG.need_court(F(rows))["city"][0]
    assert r["hoursEarly"] == 72.0 and r["runs"] == [True, True, True, True]


def test_need_court_waitlist_only_seen_after_start_does_not_count():
    rows = weekly(4, wl=4, wfa_h=None)                  # column present, no reading before the start
    assert FG.need_court(F(rows))["city"] == []


def test_need_court_and_cut_skip_stale_sessions():
    rows = weekly(4, so_h=60, booked=16)
    end = MON + timedelta(days=7 * 3 + 30)               # last run 30 days before the end
    assert FG.need_court(F(rows), end)["city"] == []
    assert FG.cut_or_move(F(weekly(4, booked=4)), _km, str, end)["city"] == []


def test_cut_needs_latest_week_under_40_and_skips_new_series_start():
    f = F(weekly(4, booked=lambda w: 4 if w < 3 else 12))   # 25, 25, 25, 75
    assert FG.cut_or_move(f, _km, str)["city"] == []
    late = [sess(d=MON + timedelta(days=21 + 7 * w), booked=4) for w in range(5)]   # began 19 Oct
    assert FG.cut_or_move(F(late), _km, str)["city"] == []   # its 5th run is all that's left after the start-up
    late = [sess(d=MON + timedelta(days=21 + 7 * w), booked=4) for w in range(8)]
    assert len(FG.cut_or_move(F(late), _km, str)["city"]) == 1


def test_cut_better_session_must_be_weekly():
    one_off = [sess(fid=2, d=MON + timedelta(days=21), booked=13, title="Special Night", key="one")]
    f = F(weekly(4, booked=4) + one_off)
    r = FG.cut_or_move(f, _km, lambda v: f"Venue {v}")["venues"]["1"][0]
    assert r["better"] is None


def test_same_faces_skips_comparison_across_a_missing_week():
    f = F(weekly(6))
    f = f[f.index != 2].reset_index(drop=True)          # week 3 didn't run
    rows = [(p, 1, k, st) for k, st in zip(f["session_key"], f["starts_at"]) for p in range(10)]
    # 5 runs, 4 comparisons, one across the gap: only 3 count, so not enough
    assert FG.same_faces(f, bookings(rows))["venues"] == {}


def test_where_same_format_any_level_blocks_and_recent_only():
    f, o = _where_world()
    extra = [sess(fid=1, d=MON + timedelta(days=2 + 7 * w), h=19, title="Open Play", lvl="all levels",
                  booked=14, key=f"v1op-{w}") for w in range(4, 6)]    # recent all-levels open play at 7pm
    f2 = pd.concat([f, F(extra)], ignore_index=True)
    cand = FG.where_candidates(f2, o, _km, str, [1, 2])
    assert not any(r["dow"] == 2 and r["h"] == 19 and r["fmt"] == "Social & open play" for r in cand["1"])
    old = [sess(fid=1, d=MON + timedelta(days=2), h=19, title="Open Play", lvl="all levels",
                booked=14, key="v1op-old")]                          # one run, 5 weeks before the end
    f3 = pd.concat([f, F(old)], ignore_index=True)
    cand = FG.where_candidates(f3, o, _km, str, [1, 2])
    assert any(r["dow"] == 2 and r["h"] == 19 for r in cand["1"])


def test_where_nearby_counts_venues_once_and_courts_capped():
    f, o = _where_world()
    cand = FG.where_candidates(f, o, _km, lambda v: f"Venue {v}", [1, 2])
    r = [x for x in cand["1"] if x["dow"] == 2 and x["h"] == 19][0]
    assert r["nearby"] == 1                              # venue 2's 6pm, 7pm, 8pm neighbours are one venue
    cols = ["v", "dow", "h", "weeks", "free2", "freeMed", "courts", "hire", "prog", "hireH"]
    d = {"_where": cand, "venues": [{"id": 1, "courts": 4}],
         "courtFree": {"cols": cols, "rows": [[1, 2, 19, 6, 5, 5.6, 4, 0.2, 0.0, 10]]}}
    FG.where_finish(d)
    top = d["figures"]["where"]["venues"]["1"][0]
    assert top["courts"]["freeMed"] == 4 and "(usually 4)" in top["text"]
    d = {"_where": cand, "venues": [{"id": 1, "courts": 4}],
         "courtFree": {"cols": cols, "rows": [[1, 2, 19, 6, 5, 7.2, 6, 0.2, 0.0, 10]]}}   # log says 6 courts
    FG.where_finish(d)
    top = d["figures"]["where"]["venues"]["1"][0]          # the log is PlayByPoint's own count: trust it over the registry
    assert top["courts"]["courts"] == 6 and top["courts"]["freeMed"] == 6
    d = {"_where": cand, "venues": [{"id": 1, "courts": 4}],
         "courtFree": {"cols": cols, "rows": [[1, 2, 19, 6, 5, 5.6, None, 0.2, 0.0, 10]]}}  # no count in the log
    FG.where_finish(d)
    assert d["figures"]["where"]["venues"]["1"][0]["courts"]["freeMed"] == 4   # falls back to the registry


def test_refilled_only_before_start_and_when_someone_left():
    rows_f, rows_o, allr = [], [], []
    for i in range(12):
        s = sess(d=MON + timedelta(days=i), booked=16, so_h=50, wl=2, key=f"w{i}", cap=16)
        rows_f.append(s)
        when = s["starts_at"] + pd.Timedelta(hours=1) if i < 3 else s["starts_at"] - pd.Timedelta(hours=10)
        rows_o.append((1, 1, s["session_key"], s["starts_at"], when))
        n_ever = 17 if i < 10 else 16                     # 2 sessions: places added, nobody left
        allr += [{"session_key": s["session_key"], "pbp_user_id": p} for p in range(n_ever)]
    out = FG.refilled(F(rows_f), bookings(rows_o), pd.DataFrame(allr))
    assert out["city"]["refilled"] == 7                 # 12 - 3 after the start - 2 with no one leaving


def test_carry_over_names_the_window_that_built():
    sys.path.insert(0, str(HERE))
    import build_insights as BI
    figs, dg = None, None
    d28 = {"figures": {"needCourt": {}}, "digest": {"from": "2026-11-02", "city": []}}
    figs, dg = BI.carry_over(d28, 28, 28, figs, dg)       # the 90-day window was skipped
    assert d28["figures"]["fromWindow"] == 28 and d28["figures"]["fromCovered"] == 28
    d7 = {"figures": {"needCourt": {"x": 1}}, "digest": {}}
    figs, dg = BI.carry_over(d7, 7, 7, figs, dg)
    assert d7["figures"]["fromWindow"] == 28 and d7["figures"]["needCourt"] == {}
    assert d7["digest"]["fromWindow"] == 28
    d90 = {"figures": {"needCourt": {}}, "digest": {"from": "x"}}
    f2, _ = BI.carry_over(d90, 90, 49, None, None)
    assert d90["figures"]["fromCovered"] == 49


# ── 9 Nov fixes after the first real build ──
def test_need_court_sold_out_early_but_started_half_empty_is_not_listed():
    # a term programme: full a week ahead every week, but only 9 of 16 there at the start
    rows = weekly(4, so_h=170, booked=9)
    for r in rows:
        r["sold_out"] = False
    assert FG.need_court(F(rows))["city"] == []


def test_need_court_coached_sessions_listed_apart():
    rows = weekly(4, so_h=60, booked=16, title="Youth Pickleball Growth Program", fgroup="Unknown")
    out = FG.need_court(F(rows))
    assert out["city"] == [] and out["venues"] == {}
    assert len(out["needCoach"]) == 1 and out["needCoach"][0]["coached"] is True
    rows = weekly(4, so_h=60, booked=16, title="Fundamentals", fgroup="Coaching & clinics")
    assert len(FG.need_court(F(rows))["needCoach"]) == 1
    assert FG.need_court(F(weekly(4, so_h=60, booked=16)))["city"][0]["coached"] is False


def test_where_skips_unknown_formats_and_thin_nearby_evidence():
    f, o = _where_world()
    f = f.copy()
    f["fgroup"] = "Unknown"
    cand = FG.where_candidates(f, o, _km, lambda v: f"Venue {v}", [1, 2])
    assert all(r["kind"] != "unknown" and r["fmt"] != "Unknown" for rows in cand.values() for r in rows)
    # a nearby session seen only twice is no evidence
    f2, o2 = _where_world()
    keep = f2[f2["fid"] == 2].sort_values("starts_at").tail(2).index
    f2 = f2[(f2["fid"] != 2) | f2.index.isin(keep)]
    cand = FG.where_candidates(f2, o2.iloc[0:0], _km, lambda v: f"Venue {v}", [1, 2])
    assert all(r["nearby"] == 0 for rows in cand.values() for r in rows)


def test_where_same_nearby_session_once():
    near = {"v": 2, "km": 1.0, "runs": 4, "earlyRuns": 4, "early": 1.0, "fill": 1.0, "name": "Venue 2",
            "dow": 4, "h": 16, "kind": "social & open play", "label": "Venue 2 Fri 4pm", "key": "2|4|16"}
    rows = [{"dow": 4, "h": h, "fmt": fm, "lvl": "all levels", "kind": "x", "players": None, "nearby": 1,
             "nearbyBest": near} for h, fm in ((15, "Social & open play"), (16, "Social & open play"), (16, "Round robin"),
                                              (17, "Round robin"))]
    cols = ["v", "dow", "h", "weeks", "free2", "freeMed", "courts"]
    d = {"_where": {"1": rows}, "courtFree": {"cols": cols, "rows": [[1, 4, h, 5, 5, 3, 6] for h in (15, 16, 17)]}}
    FG.where_finish(d)
    got = d["figures"]["where"]["venues"]["1"]
    assert len(got) == 1 and "on Fridays at 4pm" in got[0]["text"]
