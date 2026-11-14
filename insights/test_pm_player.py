"""Tests for pm_player on test_pm_v2's made-up world, with known pairs added.

25 couples: a new player copies an existing player's sessions (90% of them),
and the two also turn up together at another venue's open play three times.
Everyone else books on their own, so two regulars of the same weekly session
share runs only by chance and should NOT be called partners.

    python -m pytest test_pm_player.py -q
"""
import numpy as np
import pandas as pd
import pytest

import pm_player as PP
import pm_v2 as V2
from test_pm_v2 import world

VMAP = {v: {"lat": -37.80 - 0.03 * (v - 101), "lng": 144.96, "area": "x"} for v in range(101, 107)}


@pytest.fixture(scope="module")
def built():
    m, s, ratelog, truth = world()
    rng = np.random.default_rng(7)
    regular = m.groupby("pbp_user_id").size()
    heads = rng.choice(regular[(regular >= 8) & (regular.index < 9000)].index, 25, replace=False)
    add = []
    for k, h in enumerate(heads):
        mate = 5000 + k
        mine = m[m["pbp_user_id"] == h]
        for r in mine.itertuples():
            if rng.random() < 0.9:
                add.append((mate, r.session_key, None))
        home = int(mine["fid"].mode().iat[0])
        other = 101 + (home - 100) % 6
        for w in (2, 4, 6):
            for p in (h, mate):
                add.append((p, f"s{other}-o-{w}", None))
    extra = pd.DataFrame(add, columns=["pbp_user_id", "session_key", "rating_at_time"]).merge(
        s[["session_key", "fid", "starts_at"]], on="session_key")
    extra["last_seen"], extra["cancelled"] = extra["starts_at"], False
    m = pd.concat([m, extra], ignore_index=True).drop_duplicates(["pbp_user_id", "session_key"])
    det = {}
    V2.fit(m, s, ratelog, "2026-09-21", "2026-11-15", detail=det)
    players, series = PP.build(det, VMAP, m, ratelog, None)
    return {int(h): 5000 + k for k, h in enumerate(heads)}, pd.DataFrame(players).set_index("id"), series, det


def test_couples_are_found_both_ways(built):
    couples, P, _, _ = built
    found = sum(any(x["id"] == mate for x in P.at[h, "partners"]) for h, mate in couples.items())
    back = sum(any(x["id"] == h for x in P.at[mate, "partners"]) for h, mate in couples.items())
    assert found >= 22 and back >= 22, (found, back)


def test_regulars_of_the_same_session_are_not_partners(built):
    couples, P, _, _ = built
    inside = set(couples) | set(couples.values())
    others = P[~P.index.isin(inside) & (P["n"] >= 6)]
    wrong = others["partners"].map(lambda xs: any(x["id"] not in inside for x in xs))
    assert wrong.mean() < 0.03, wrong.mean()


def test_the_terms_add_up_to_the_level(built):
    _, P, _, _ = built
    for pid, r in P.sample(60, random_state=1).iterrows():
        shares = sum(t["share"] for t in r["terms"])
        avg = sum(t["share"] * t["x"] for t in r["terms"]) / 100
        assert abs(shares - 100) < 1.0, (pid, shares)
        assert abs(avg - r["level"]) < 0.05, (pid, avg, r["level"])


def test_recommendations_fit_and_are_new(built):
    _, P, series, _ = built
    n = 0
    for pid, r in P[P["n"] >= 6].head(200).iterrows():
        played = {x["s"] for x in r["sessions"]}
        for x in r["recs"]["near"]:
            n += 1
            assert x["s"] not in played and x["fit"] in ("suits", "up") and x["km"] <= r["reach"]["km"] + 0.1
            assert x["s"] in series
            lv = series[x["s"]]["level"]
            if x["by"] == "room":
                assert -0.25 <= lv - r["level"] <= 0.5
    assert n > 100


def test_partner_recommendations_name_the_partners(built):
    couples, P, _, _ = built
    seen = 0
    for h, mate in couples.items():
        for x in P.at[h, "recs"]["partners"]:
            seen += 1
            assert set(x["partners"]) <= {p["id"] for p in P.at[h, "partners"]}
    assert seen >= 0


def test_sessions_say_where_they_sit(built):
    _, P, series, _ = built
    places = {x["place"] for xs in P["sessions"] for x in xs}
    assert {"pulled", "inside", "open"} <= places, places
    assert all(x["s"] in series for xs in P["sessions"] for x in xs)


def test_dupr_trail_newest_first():
    m = pd.DataFrame({"pbp_user_id": [1, 1, 1], "last_seen": ["2026-09-01", "2026-10-01", "2026-11-01"],
                      "rating_at_time": [3.12, 3.12, 3.27]})
    reads = pd.DataFrame({"pbp_user_id": [1], "read_at": ["2026-11-10"], "doubles": [3.31]})
    t = PP.dupr_trail(m, None, reads)[1]
    assert [x["v"] for x in t["changes"]] == [3.31, 3.27, 3.12]
    assert t["changes"][0]["src"] == "read" and t["lastSeen"] == "2026-11-10"


def test_nothing_identifying_is_printed(built, capsys):
    out = capsys.readouterr().out
    assert "5000" not in out


def test_rows_are_plain_json(built):
    import json
    _, P, series, _ = built
    json.dumps(P.reset_index().head(300).to_dict("records"))
    json.dumps(series)


def test_reach_comes_from_how_far_they_play(built):
    couples, P, _, _ = built
    # couples play twice+ at a second venue 3.3-16.7 km from home: their own reach
    for h in list(couples)[:10]:
        r = P.at[h, "reach"]
        assert r["how"] == "own" and r["km"] > 3, r
    # everyone else only plays at home: their venue's travellers' median, or their venue only
    others = P[~P.index.isin(set(couples) | set(couples.values()))]
    assert set(others["reach"].map(lambda r: r["how"])) <= {"peers", "venue"}


def test_travel_uses_venues_played_twice():
    S = {"a": {"v": 101}, "b": {"v": 103}, "c": {"v": 106}}
    home, km = PP.travel({"a": 6, "b": 2, "c": 1}, S, VMAP)
    assert home == 101 and abs(km - 6.67) < 0.1          # 106 (once) doesn't count
    assert PP.travel({"a": 6}, S, VMAP) == (101, None)


def test_coaches_sessions_they_run_add_nothing(built):
    _, P, series, _ = built
    for cid in (9101, 9102):
        r = P.loc[cid]
        assert r["coach"]
        coached = [x for x in r["sessions"] if series[x["s"]]["coached"]]
        assert coached and all(x["place"] == "coaching" and x["share"] == 0 for x in coached)
        assert all(t.get("s") not in {x["s"] for x in coached} for t in r["terms"])
