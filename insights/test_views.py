"""
The restructured page's views (views.py): venues that share players and how
that compares with chance, sessions that share regulars, "what changed",
"came back", hire takings and the scorecard's change.

    python -m pytest -q test_views.py
"""
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import views as V  # noqa: E402

UTC = timezone.utc


def bookings(rows):
    """rows: (player, fid, 'YYYY-MM-DD')"""
    return pd.DataFrame([{"pbp_user_id": p, "fid": v,
                          "starts_at": pd.Timestamp(d + " 09:00", tz="Australia/Melbourne").tz_convert(UTC)}
                         for p, v, d in rows])


# ── shares ──
def test_shares_share_and_chance():
    rows = []
    # venue 1: 10 players with 2 sessions each; 6 of them also once at venue 2
    for p in range(10):
        rows += [(p, 1, "2026-10-01"), (p, 1, "2026-10-08")]
    for p in range(6):
        rows.append((p, 2, "2026-10-02"))
    # venue 2 also has 14 players of its own (once each) -> 20 seen at venue 2
    for p in range(100, 114):
        rows.append((p, 2, "2026-10-03"))
    # venue 3: 20 keen players (2 sessions each); 5 of them also seen at venue 2
    for p in range(200, 220):
        rows += [(p, 3, "2026-10-01"), (p, 3, "2026-10-08")]
    for p in range(200, 205):
        rows.append((p, 2, "2026-10-04"))
    out = V.shares(bookings(rows), [1, 2, 3])
    r = [x for x in out["rows"] if x[0] == 1 and x[1] == 2]
    assert r, out
    a, b, n, share, chance = r[0]
    assert n == 6 and share == 60.0
    # chance: of the other players with 2+ sessions (venue 3's 20; the one-off
    # players at venue 2 don't count), 5 were seen at venue 2 -> 0.25.
    # Expected 10 x 0.25 = 2.5 -> 6 / 2.5 = 2.4
    assert chance == 2.4
    assert out["base"] == {"1": 10, "3": 20}
    # venue 2 has no one with 2+ sessions there, so no 2 -> 1 row
    assert not [x for x in out["rows"] if x[0] == 2]


def test_shares_chance_not_inflated_by_keen_players():
    """A's regulars at B at the same rate as other keen players read 1.0x,
    however much more they play than the one-off crowd."""
    rows = []
    for p in range(20):                       # A: 20 keen, 10 also at B
        rows += [(p, 1, "2026-10-01"), (p, 1, "2026-10-08")]
    for p in range(10):
        rows.append((p, 2, "2026-10-02"))
    for p in range(100, 120):                 # C: 20 keen, 10 also at B
        rows += [(p, 3, "2026-10-01"), (p, 3, "2026-10-08")]
    for p in range(100, 110):
        rows.append((p, 2, "2026-10-02"))
    for p in range(300, 400):                 # 100 one-off players at B
        rows.append((p, 2, "2026-10-05"))
    out = V.shares(bookings(rows), [1, 2, 3])
    r = [x for x in out["rows"] if x[0] == 1 and x[1] == 2][0]
    assert r[4] == 1.0


def test_shares_hide_pairs_under_five():
    rows = []
    for p in range(10):
        rows += [(p, 1, "2026-10-01"), (p, 1, "2026-10-08")]
    for p in range(4):
        rows.append((p, 2, "2026-10-02"))
    out = V.shares(bookings(rows), [1, 2])
    assert out["rows"] == []


# ── sessions that share regulars ──
def series(j, v, title, dow=1, hm="19:00"):
    return {"j": j, "v": v, "t": title, "dow": dow, "hm": hm}


def test_shared_regulars_thresholds():
    S = [series(0, 1, "Open Play"), series(1, 1, "Ladder", 3, "18:30"), series(2, 2, "Social", 5, "10:00")]
    # 0 and 1 share 6 regulars out of 10 and 40 (60% of the smaller)
    sp = {0: Counter({p: 2 for p in range(10)}),
          1: Counter({p: 3 for p in list(range(6)) + list(range(50, 84))}),
          # 2 shares 5 regulars with 1, but the smaller of the two has 40: 12.5%, out
          2: Counter({p: 2 for p in list(range(50, 55)) + list(range(200, 245))})}
    out = V.shared_regulars(S, sp, lambda v: f"Venue {v}")
    assert [(p["a"], p["b"], p["shared"]) for p in out["city"]] == [(0, 1, 6)]
    p = out["city"][0]
    assert p["pct"] == 60.0
    assert p["labelA"] == "Venue 1 · Tue 7pm · Open Play" and p["labelB"] == "Venue 1 · Thu 6:30pm · Ladder"
    assert list(out["venues"]) == ["1"]


def test_one_visit_players_are_not_regulars():
    S = [series(0, 1, "A"), series(1, 1, "B", 2)]
    sp = {0: Counter({p: 1 for p in range(10)}), 1: Counter({p: 1 for p in range(10)})}
    assert V.shared_regulars(S, sp, str)["city"] == []


# ── what changed ──
def sessions_frame(end=date(2026, 11, 8), base_fill=0.5, last_fill=0.5, per_day=3, venues=(1, 2), early_last=0.0):
    rows = []
    d0 = end - timedelta(days=40)
    for i in range(41):
        d = d0 + timedelta(days=i)
        last = d > end - timedelta(days=7)
        for v in venues:
            for k in range(per_day):
                rows.append({"fid": v, "fgroup": "Social & open play", "lvl": "intermediate",
                             "starts_at": pd.Timestamp(f"{d} 18:00", tz="Australia/Melbourne").tz_convert(UTC),
                             "fill": last_fill if last else base_fill,
                             "so_h": (60.0 if (last and k < per_day * early_last) else np.nan)})
    return pd.DataFrame(rows)


def test_digest_reports_big_moves_only():
    f = sessions_frame(base_fill=0.5, last_fill=0.72)
    okv = bookings([(p, 1, "2026-10-20") for p in range(30)])
    d = V.digest(f, okv, lambda v: f"Venue {v}")
    assert d["from"] == "2026-11-02" and d["baseFrom"] == "2026-10-05"
    fills = [i for i in d["items"] if i["metric"] == "fill" and "scope" not in i]
    assert fills and fills[0]["dir"] == "up" and fills[0]["now"] == 72.0 and fills[0]["before"] == 50.0
    assert fills[0]["text"].startswith("Average fill rose to 72%")
    assert len(d["items"]) <= V.DIGEST_MAX


def test_digest_ignores_small_moves():
    f = sessions_frame(base_fill=0.5, last_fill=0.55)
    d = V.digest(f, bookings([(1, 1, "2026-10-20")]), str)
    assert d["items"] == [] and d["reason"]


def test_digest_needs_five_weeks():
    f = sessions_frame()
    f = f[f["starts_at"] >= pd.Timestamp("2026-10-20", tz=UTC)]
    d = V.digest(f, bookings([(1, 1, "2026-10-25")]), str)
    assert d["items"] == [] and "5 weeks" in d["reason"] and "from" not in d


def test_digest_never_names_a_count_under_five():
    # players: 4 this week vs 2 a week before -> +100%, but both under 5: no sentence
    f = sessions_frame()
    rows = [(p, 1, "2026-11-05") for p in range(4)] + [(p, 1, "2026-10-10") for p in range(2)]
    d = V.digest(f, bookings(rows), str)
    assert not [i for i in d["items"] if i["metric"] == "players"]


def _weekly_players_rows(per_week, last_week, v=1):
    """per_week different players in each of the 4 baseline weeks, last_week in the last 7 days."""
    rows, pid = [], 0
    for w, d in enumerate(["2026-10-07", "2026-10-14", "2026-10-21", "2026-10-28"]):
        for _ in range(per_week):
            rows.append((pid, v, d)); pid += 1
    for _ in range(last_week):
        rows.append((pid, v, "2026-11-05")); pid += 1
    return rows


def test_digest_needs_a_big_enough_baseline():
    # 64 players this week against 8 a week: a small baseline, no sentence
    f = sessions_frame()
    d = V.digest(f, bookings(_weekly_players_rows(8, 64)), str)
    assert not [i for i in d["items"] if i["metric"] == "players"]


def test_digest_says_counts_for_a_big_rise():
    f = sessions_frame()
    d = V.digest(f, bookings(_weekly_players_rows(25, 60)), str)
    p = [i for i in d["items"] if i["metric"] == "players"]
    assert p and "%" not in p[0]["text"]
    assert p[0]["text"] == "60 players booked in the last 7 days, against about 25 a week over the 4 weeks before."
    # a smaller rise keeps the percentage
    d = V.digest(f, bookings(_weekly_players_rows(25, 40)), str)
    p = [i for i in d["items"] if i["metric"] == "players"]
    assert p and "up 60%" in p[0]["text"]


def test_digest_needs_every_baseline_week():
    # venue 3 only started in the last baseline week: no comparison for it,
    # however big its move looks
    f = sessions_frame(base_fill=0.5, last_fill=0.5)
    late = sessions_frame(base_fill=0.2, last_fill=0.9, venues=(3,))
    late = late[late["starts_at"] >= pd.Timestamp("2026-10-27", tz=UTC)]
    f = pd.concat([f, late], ignore_index=True)
    d = V.digest(f, bookings([(1, 1, "2026-10-20")]), lambda v: f"Venue {v}")
    assert not [i for i in d["items"] if i.get("scope") == {"v": 3}]
    assert "3" not in d["venues"]


def test_digest_capitalises_level_sentences():
    f = sessions_frame(base_fill=0.5, last_fill=0.5)
    f2 = sessions_frame(base_fill=0.3, last_fill=0.8, venues=(1,))
    f2["lvl"] = "advanced"
    f2["fgroup"] = "Coaching & clinics"
    d = V.digest(pd.concat([f, f2], ignore_index=True), bookings([(1, 1, "2026-10-20")]), str)
    lv = [i for i in d["items"] if i.get("scope", {}).get("lvl")]
    assert lv and lv[0]["text"].startswith("Advanced sessions")


# ── sold out early ──
def test_sold_early_leaves_out_sessions_full_when_listed():
    g = pd.DataFrame({"sold_out": [True, True, True, False],
                      "so_h": [60.0, 10.0, np.nan, np.nan]})   # the 3rd was full when first seen
    assert abs(V.early_rate(g) - 100 / 3) < 1e-9          # 1 of 3; not 1 of 4


# ── came back ──
def test_came_back_within_fourteen_days():
    rows = []
    for p in range(10):                     # first visit 1 Oct; 7 back on 10 Oct, 3 back on 30 Oct (too late)
        rows.append((p, 1, "2026-10-01"))
        rows.append((p, 1, "2026-10-10" if p < 7 else "2026-10-30"))
    rows.append((99, 1, "2026-11-01"))      # first visit too late to judge: not in the base
    rate, n = V.came_back(bookings(rows), 1, date(2026, 11, 8))
    assert n == 10 and rate == 70.0


# ── hire takings ──
def test_hire_takings_weighted_by_hireable_court_hours():
    cols = ["v", "dow", "h", "weeks", "free2", "freeMed", "courts", "hire", "prog", "hireH"]
    d = {"courtFree": {"cols": cols,
                       "rows": [[1, 1, 18, 4, 0, 0, 4, 1.0, 0.0, 16.0], [1, 1, 10, 4, 4, 4, 4, 0.0, 0.0, 16.0],
                                [2, 1, 18, 2, 0, 0, 2, None, 0.0, 4.0]]},
         "venues": [{"id": 1, "hireRatePerCourtHour": 40.0}, {"id": 2, "hireRatePerCourtHour": 30.0}],
         "city": {}}
    V.hire_takings(d)
    assert d["venues"][0]["hireTakingsPerCourtHour"] == 20.0     # half the court-hours hired x $40
    assert d["venues"][0]["hireableCourtHours"] == 32
    assert d["venues"][1]["hireTakingsPerCourtHour"] is None     # no hire share
    assert d["city"]["hireTakingsPerCourtHour"] == 20.0
    # an hour mostly given to the venue's own sessions (1 hireable court-hour,
    # all hired) barely moves the figure; weighting by courts would double it
    d["courtFree"]["rows"] = [[1, 1, 18, 4, 0, 0, 4, 1.0, 0.9, 1.0], [1, 1, 10, 4, 4, 4, 4, 0.0, 0.0, 16.0]]
    V.hire_takings(d)
    assert d["venues"][0]["hireTakingsPerCourtHour"] == round(40 * 1 / 17, 1)


# ── scorecard change ──
def test_scorecard_change_uses_the_dates_asked():
    """The period before ran sessions only on its last 7 days of 28: sessions a
    week still divides by 4 weeks, not 1."""
    cols = ["v", "booked", "cap", "price"]
    d = {"scorecard": {"city": {"perWeek": 7.0, "fill": 50.0, "takings": 100}, "venues": {}},
         "prev": {"cols": cols, "sessions": [[1, 5, 10, 10.0]] * 28,
                  "from": "2026-10-25", "to": "2026-10-31", "asked": ["2026-10-04", "2026-10-31"]}}
    V.scorecard_changes(d)
    assert d["scorecard"]["city"]["change"]["perWeek"] == 0.0


def test_scorecard_change_against_prev():
    d = {"scorecard": {"city": {"perWeek": 30.0, "fill": 70.0, "takings": 1200},
                       "venues": {"1": {"perWeek": 10.0, "fill": 50.0, "takings": None}}},
         "prev": {"from": "2026-10-01", "to": "2026-10-14",
                  "cols": ["v", "date", "dow", "hour", "fmt", "lvl", "cap", "booked", "so", "price", "mprice"],
                  "sessions": [[1, "2026-10-01", 2, 18, "x", "y", 10, 6, 0, 20.0, None]] * 40}}
    V.scorecard_changes(d)
    c = d["scorecard"]["city"]["change"]
    # prev: 40 sessions over 14 days = 20 a week, 60% full, $4,800 taken
    assert c["perWeek"] == 10.0 and c["fill"] == 10.0 and c["takings"] == -75.0
    v = d["scorecard"]["venues"]["1"]["change"]
    assert v["perWeek"] == -10.0 and v["takings"] is None and v["soldEarly"] is None
