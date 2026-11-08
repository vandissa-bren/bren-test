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
             "verified", "travellers", "giantN", "inGroups", "venuePlayers"}


def test_counts_under_five_are_held_back():
    d = {"net": {"groups": [{"size": 12, "lvl": {"advanced": 2, "beginner": 0, "unrated": 9}}],
                 "series": [{"players": 4, "regulars": 3, "also": [[1, 40, 2], [2, 10, 6]]}],
                 "city": {"degHist": [["0", 0], ["1–5", 3], ["81+", 1]]}},
         "prev": {"venuePlayers": {"597": 3, "755": 40}}}
    P.enforce(d)
    assert d["net"]["groups"][0]["lvl"] == {"advanced": None, "beginner": 0, "unrated": 9}
    assert d["net"]["series"][0]["players"] is None and d["net"]["series"][0]["regulars"] is None
    assert d["net"]["series"][0]["also"] == [[1, 40, None], [2, 10, 6]]
    assert d["net"]["city"]["degHist"] == [["0", 0], ["1–5", None], ["81+", None]]
    assert d["prev"]["venuePlayers"] == {"597": None, "755": 40}


def test_rates_resting_on_few_players_are_held_back():
    d = {"venues": [{"players": 40, "repeat": 95.0, "loyal": 5.0, "fromAfar": 50.0, "fromAfarN": 6,
                     "cameBack": 30.0, "cameBackN": 10, "oneTimers": 0.0, "netPlayers": 40},
                    {"players": 3, "repeat": 100.0, "loyal": 100.0}],
         "ratings": {"progression": [30, 0.1, 10.0, 50.0]}}
    P.enforce(d)
    v = d["venues"][0]
    assert v["repeat"] == 95.0              # 38 players
    assert v["loyal"] is None               # 2 of 40
    assert v["fromAfar"] is None            # 3 of 6
    assert v["cameBack"] is None            # 3 of 10
    assert v["oneTimers"] == 0.0            # none
    assert "fromAfarN" not in v and "cameBackN" not in v
    assert d["venues"][1]["players"] is None and d["venues"][1]["repeat"] is None
    assert d["ratings"]["progression"] == [30, 0.1, None, 50.0]   # 3 of 30 went up


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


def test_internal_bases_are_not_published(built):
    for v in built["venues"]:
        assert "fromAfarN" not in v and "cameBackN" not in v
