"""
The 5-or-more rule (privacy.py): no published build carries a player count of
1-4, or a percentage that rests on 1-4 players.

    python -m pytest -q test_privacy.py
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import privacy as P  # noqa: E402

# keys that name players wherever they appear (a second net under COUNTS)
PLAYERISH = {"players", "regulars", "size", "netPlayers", "homeHere", "new", "returning",
             "verified", "travellers", "giantN", "inGroups", "venuePlayers", "shared", "newHere"}


def test_counts_under_five_are_held_back():
    d = {"net": {"series": [{"players": 4, "regulars": 3}],
                 "shared": {"city": [{"shared": 4}, {"shared": 6}], "venues": {"597": [{"shared": 2}]}}},
         "shares": {"rows": [[597, 755, 4, 20.0, 1.2], [597, 1557, 9, 45.0, 2.0]], "base": {"597": 20, "755": 3}},
         "scorecard": {"city": {"newHere": 40, "series": {"newHere": [12, 3, 0]}},
                       "venues": {"597": {"newHere": 4, "series": {"newHere": [None, 2, 6]}}}},
         "prev": {"venuePlayers": {"597": 3, "755": 40}}}
    P.enforce(d)
    assert d["net"]["series"][0]["players"] is None and d["net"]["series"][0]["regulars"] is None
    assert [x["shared"] for x in d["net"]["shared"]["city"]] == [None, 6]
    assert d["net"]["shared"]["venues"]["597"][0]["shared"] is None
    assert d["shares"]["rows"][0][2] is None and d["shares"]["rows"][1][2] == 9
    assert d["shares"]["base"] == {"597": 20, "755": None}
    assert d["scorecard"]["city"]["series"]["newHere"] == [12, None, 0]
    assert d["scorecard"]["venues"]["597"]["newHere"] is None
    assert d["scorecard"]["venues"]["597"]["series"]["newHere"] == [None, None, 6]
    assert d["prev"]["venuePlayers"] == {"597": None, "755": 40}


def test_rates_resting_on_few_players_are_held_back():
    d = {"venues": [{"players": 40, "loyal": 5.0, "fromAfar": 50.0, "fromAfarN": 6,
                     "cameBack": 30.0, "cameBackN": 10},
                    {"players": 3, "loyal": 100.0}],
         "ratings": {"progression": [30, 0.1, 10.0, 50.0]}}
    P.enforce(d)
    v = d["venues"][0]
    assert v["loyal"] is None               # 2 of 40
    assert v["fromAfar"] is None            # 3 of 6
    assert v["cameBack"] is None            # 3 of 10
    assert "fromAfarN" not in v and "cameBackN" not in v
    assert d["venues"][1]["players"] is None and d["venues"][1]["loyal"] is None
    assert d["ratings"]["progression"] == [30, 0.1, None, 50.0]   # 3 of 30 went up


def test_rate_with_few_players_left_over_is_held():
    d = {"venues": [{"id": 1, "players": 20, "cameBack": 95.0, "cameBackN": 20}]}
    P.enforce(d)
    assert d["venues"][0]["cameBack"] is None        # 19 of 20: says 1 didn't


def test_subtraction_never_leaves_one_to_four():
    d = {"scorecard": {"city": {"newHere": 40, "series": {"newHere": [18, 14, 7, 0]}}, "venues": {}},
         "venues": [{"id": 597, "players": 7}], "shares": {"base": {"597": 5}, "rows": [[597, 755, 5, 100.0, 2.0]]},
         "newret": {"city": {"players": 50, "new": 47, "returning": 3}}}
    P.enforce(d)
    wk = d["scorecard"]["city"]["series"]["newHere"]
    assert P.remainder((d["scorecard"]["city"], "newHere"),
                       [(wk, i) for i in range(len(wk))]) in (0,) or P.remainder(
        (d["scorecard"]["city"], "newHere"), [(wk, i) for i in range(len(wk))]) >= 5
    assert wk == [18, 14, None, 0]                    # 40 - 18 - 14 = 8, not 1
    assert d["shares"]["base"]["597"] is None         # 7 - 5 = 2 one-session players
    assert d["shares"]["rows"][0][3] is None and d["shares"]["rows"][0][4] is None
    assert d["newret"]["city"]["returning"] is None and d["newret"]["city"]["new"] is None


def test_pm_regulars_rate_of_four_players_is_held():
    d = {"pm": {"regulars": [8, 50.0]}}
    P.enforce(d)
    assert d["pm"]["regulars"] == [8, None]


def _walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, o


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    import fixture_build
    t = tmp_path_factory.mktemp("fx")
    fixture_build.write(t / "csv")
    out = t / "out.json"
    subprocess.run([sys.executable, str(HERE / "build_insights.py"), "--csv", str(t / "csv"), "--out", str(out)],
                   check=True, capture_output=True, cwd=HERE)
    return json.loads(out.read_text())


def test_fixture_has_small_groups_to_hide(built):
    # the fixture's tiny venue (3 players) must reach the build, with its counts hidden
    v = [x for x in built["venues"] if x["id"] == 1883]
    assert v and v[0]["players"] is None


def test_no_listed_count_is_under_five(built):
    bad = []
    for path in P.COUNTS:
        for parent, k in P._parents(built, P._parse(path)):
            if P.small(parent[k]):
                bad.append((path, k, parent[k]))
    assert not bad


def test_no_player_named_field_is_under_five(built):
    bad = [(p, v) for p, v in _walk(built)
           if p.rsplit(".", 1)[-1].split("[")[0] in PLAYERISH and P.small(v)]
    assert not bad


def test_no_total_minus_parts_is_one_to_four(built):
    bad = []
    for total, parts in P.partitions(built):
        rem = P.remainder(total, parts)
        if rem is not None and P.small(rem):
            bad.append((total[1], rem))
    assert not bad


def test_fixture_publishes_venues_that_share_players(built):
    rows = built["shares"]["rows"]
    assert rows, "the fixture should give shares rows end to end"
    for a, b, n, share, chance in rows:
        assert n >= 5 and a != b
        if share is not None:
            assert 0 < share <= 100
        if chance is not None:
            assert chance > 0


def test_internal_bases_are_not_published(built):
    for v in built["venues"]:
        assert "fromAfarN" not in v and "cameBackN" not in v


def test_player_level_network_is_not_published(built):
    assert set(built["net"]) == {"city", "series", "shared"}
    assert set(built["net"]["city"]) <= {"events", "eventCap", "rosterSessions", "sessions", "bookings"}
    for x in built["net"]["series"]:
        assert "cm" not in x and "also" not in x
    assert "links" not in built
    for v in built["venues"]:
        for k in ("netPlayers", "partners", "reach2", "oneTimers", "regularPairs", "groups", "groupMix",
                  "repeat", "avgRating", "pph"):
            assert k not in v


def test_new_keys_carry_no_small_counts(built):
    assert all(not P.small(n) for n in built["shares"]["base"].values())
    assert all(r[2] >= 5 for r in built["shares"]["rows"])
    for item in built["digest"]["items"] + [i for lst in built["digest"]["venues"].values() for i in lst]:
        if item["metric"] in ("players", "sessions"):
            assert item["now"] >= 5 and item["before"] >= 5
    assert built["glossary"]["players"]["definition"]
