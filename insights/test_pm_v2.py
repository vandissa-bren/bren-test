"""Tests for pm_v2 on made-up players whose TRUE level is known.

600 players at 6 venues, 8 weeks. Each venue runs four level-banded weekly
sessions and one open to all. Players book the banded sessions nearest their
true level. 45% show a DUPR close to their true level; 30 "strivers" carry a
DUPR 0.7 below how they really play (they play up, their DUPR hasn't caught
up) and their DUPR rises after week 5. A few coaches are booked into every
learning session at their venue.
"""
import numpy as np
import pandas as pd
import pytest

import pm_v2 as V2

MEL = "Australia/Melbourne"
BANDS = [("Beginner Social", 2.0, 2.5, 2.35), ("Low Intermediate", 2.5, 3.0, 2.8),
         ("Intermediate Social", 3.0, 3.5, 3.25), ("Advanced Social", 3.5, 4.5, 3.8)]


def world(seed=3, players=600, weeks=8):
    rng = np.random.default_rng(seed)
    venues = [101, 102, 103, 104, 105, 106]
    start = pd.Timestamp("2026-09-21 18:00", tz=MEL)
    sessions, rows = [], []
    for v in venues:
        for b, (title, lo, hi, _) in enumerate(BANDS):
            for w in range(weeks):
                k = f"s{v}-{b}-{w}"
                sessions.append(dict(session_key=k, fid=v, starts_at=start + pd.Timedelta(days=7 * w + b, hours=v % 3),
                                     title=title, label=title.lower(), family="social", band_lo=lo, band_hi=hi,
                                     program_slug=f"p{v}-{b}", session_type="social", level_class=None, coaches=None))
        for w in range(weeks):     # open to all
            sessions.append(dict(session_key=f"s{v}-o-{w}", fid=v, starts_at=start + pd.Timedelta(days=7 * w + 5),
                                 title="Open Play All Levels", label="all levels", family="social", band_lo=np.nan,
                                 band_hi=np.nan, program_slug=f"p{v}-o", session_type="open play", level_class=None,
                                 coaches=None))
        for w in range(weeks):     # a second learning session, another day
            sessions.append(dict(session_key=f"s{v}-L2-{w}", fid=v, starts_at=start + pd.Timedelta(days=7 * w + 4, hours=2),
                                 title="Skills Clinic", label="beginner clinic", family="learning", band_lo=2.0,
                                 band_hi=2.5, program_slug=f"p{v}-L2", session_type="clinic", level_class=None,
                                 coaches=None))
        for w in range(weeks):     # a learning session with a coach booked in
            sessions.append(dict(session_key=f"s{v}-L-{w}", fid=v, starts_at=start + pd.Timedelta(days=7 * w + 6),
                                 title="Learn to Play", label="learn to play", family="learning", band_lo=2.0,
                                 band_hi=2.5, program_slug=f"p{v}-L", session_type="clinic", level_class=None,
                                 coaches=None))
    s = pd.DataFrame(sessions)
    s["starts_at"] = s["starts_at"].dt.tz_convert("UTC")
    true = np.clip(rng.normal(3.15, 0.5, players), 2.1, 4.8)
    pids = np.arange(1000, 1000 + players)
    has_d = rng.random(players) < 0.45
    strivers = rng.choice(np.nonzero(has_d & (true > 3.2))[0], 30, replace=False)
    dval = np.round(true + rng.normal(0, 0.1, players), 3)
    dval[strivers] = np.round(true[strivers] - 0.7, 3)
    home = rng.choice(venues, players)
    risen = set(strivers.tolist())

    def rating(i, w):       # what the roster shows: strivers' DUPR rises by 0.3 from week 6
        if not has_d[i]:
            return None
        return round(dval[i] + 0.3, 3) if i in risen and w >= 6 else dval[i]
    for i, p in enumerate(pids):
        near = np.argsort([abs(t[3] - (true[i] + rng.normal(0, 0.12))) for t in BANDS])[:2]
        keen = rng.uniform(0.4, 0.95)
        for w in range(8):
            for b in near[: 1 + (rng.random() < 0.4)]:
                if rng.random() < keen:
                    rows.append((p, f"s{home[i]}-{b}-{w}", rating(i, w)))
            if rng.random() < 0.15:
                rows.append((p, f"s{home[i]}-o-{w}", rating(i, w)))
            if true[i] < 2.5 and rng.random() < 0.5:
                rows.append((p, f"s{home[i]}-L-{w}", rating(i, w)))
    for v in venues:                  # a strong coach per venue in every learning session
        for w in range(8):
            rows.append((9000 + v, f"s{v}-L-{w}", None))
            rows.append((9100 + v, f"s{v}-L-{w}", None))
        for w in range(8):            # ...and in the second learning session
            rows.append((9000 + v, f"s{v}-L2-{w}", None))
            rows.append((9100 + v, f"s{v}-L2-{w}", None))
        for w in range(8):            # ...and the same coaches in the beginner session
            rows.append((9000 + v, f"s{v}-0-{w}", None))
            rows.append((9100 + v, f"s{v}-0-{w}", None))
    m = pd.DataFrame(rows, columns=["pbp_user_id", "session_key", "rating_at_time"]).drop_duplicates(
        ["pbp_user_id", "session_key"])
    m = m.merge(s[["session_key", "fid", "starts_at"]], on="session_key")
    m["last_seen"] = m["starts_at"]
    m["cancelled"] = False
    # strivers' DUPR rises in week 6 (rating log), everyone else's stays
    log = []
    for i in strivers:
        log.append(dict(pbp_user_id=pids[i], rating=round(dval[i] + 0.3, 3), prev_rating=dval[i],
                        seen_at=start + pd.Timedelta(weeks=6, days=2), prev_seen_at=start, session_key=None, source="live"))
    ratelog = pd.DataFrame(log)
    ratelog["seen_at"] = pd.to_datetime(ratelog["seen_at"]).dt.tz_convert("UTC")
    ratelog["prev_seen_at"] = pd.to_datetime(ratelog["prev_seen_at"]).dt.tz_convert("UTC")
    truth = pd.DataFrame({"true": true, "has_d": has_d, "dupr": dval}, index=pids)
    truth["striver"] = False
    truth.iloc[strivers, truth.columns.get_loc("striver")] = True
    return m, s, ratelog, truth


@pytest.fixture(scope="module")
def fitted():
    m, s, ratelog, truth = world()
    summary, levels = V2.fit(m, s, ratelog, "2026-09-21", "2026-11-15")
    lv = pd.DataFrame(levels).set_index("id")
    return summary, lv, truth


def test_players_without_dupr_land_near_their_true_level(fitted):
    _, lv, truth = fitted
    j = lv.join(truth, how="inner")
    x = j[~j["has_d"] & (j["n"] >= 4)]
    err = (x["level"] - x["true"]).abs()
    assert len(x) > 100
    assert err.median() < 0.3, err.describe()


def test_strivers_are_pulled_up_from_their_dupr(fitted):
    _, lv, truth = fitted
    j = lv.join(truth, how="inner")
    st = j[j["striver"]]
    # their DUPR (latest) is 0.4 under true after the rise; the PM level should sit well above the old DUPR
    # latest DUPR = true - 0.4 (after the rise); the PM level should sit above it, towards true
    assert (st["level"] - (st["true"] - 0.4)).median() > 0.05
    ok = j[j["has_d"] & ~j["striver"]]
    assert (ok["level"] - ok["dupr"]).abs().median() < 0.15


def test_rooms_rank_by_level_and_coaches_are_found(fitted):
    summary, _, _ = fitted
    r = pd.DataFrame(summary["rooms"])
    by = r.groupby("title")["level"].median()
    assert by["Beginner Social"] < by["Low Intermediate"] < by["Intermediate Social"] < by["Advanced Social"]
    assert summary["coaches"] is None or summary["coaches"] >= 5     # 12 coaches, shown only as 5+
    assert all(x["players"] >= 5 for x in summary["rooms"])


def test_forward_test_sees_strivers_rise(fitted):
    summary, _, _ = fitted
    f = summary["tests"]["forward"]
    assert f is not None and f["above"]["move"] is not None and f["above"]["move"] > 0.05
    assert f["hitPct"] is None or f["hitPct"] >= 70


def test_no_player_count_of_one_to_four_leaves(fitted):
    summary, _, _ = fitted
    bad = []

    def walk(o, path):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif isinstance(o, int) and not isinstance(o, bool) and 1 <= o <= 4:
            if not any(t in path for t in ("rounds", "dow", ".v", "runs", "sessions", "weeks", "settings", "seconds")):
                bad.append(path)
    walk(summary, "pm")
    assert not bad, bad


def test_protect_hides_what_subtraction_would_give_away():
    assert V2._protect(20, {"a": 10, "b": 7}) == {"a": 10, "b": None}           # 20-17 = 3
    # c (3) was already held at source; alone it would be 20 - 12 - 5, so b goes too
    out = V2._protect(20, {"a": 12, "b": 5, "c": None}, exact=True)
    assert out == {"a": 12, "b": None, "c": None}
    assert V2._protect(30, {"a": 10, "b": 20}, exact=True) == {"a": 10, "b": 20}
    assert V2._protect(None, {"a": 9}) == {"a": None}


def test_a_venue_with_no_duprs_keeps_its_spread():
    """Strip every DUPR at one venue: its players must still spread out by what
    they book (labels, anchored by other venues' DUPRs), not all collapse to
    one middle or drift off (the PickleBoom case, 14 Nov)."""
    m, s, rl, truth = world()
    fid = m["session_key"].map(s.set_index("session_key")["fid"])
    home = m.assign(f=fid).groupby("pbp_user_id")["f"].agg(lambda x: x.mode().iat[0])
    blind = set(home[home == 101].index) & set(truth.index)
    m.loc[m["pbp_user_id"].isin(blind), "rating_at_time"] = None
    rl = rl[~rl["pbp_user_id"].isin(blind)]
    _, lv = V2.fit(m, s, rl, "2026-09-21", "2026-11-15")
    L = pd.DataFrame(lv).set_index("id")["level"]
    ids = sorted(blind & set(L.index))
    a, b = L[ids], truth.loc[ids, "true"]
    assert abs((a - b).mean()) < 0.1, (a - b).mean()          # no drift
    assert np.polyfit(b, a, 1)[0] > 0.65                        # spread kept
    assert (a - b).abs().mean() < 0.18
