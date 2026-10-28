"""Tests for league_rosters.py (host leagues phase 2). No network: PlayByPoint
and Supabase are both faked.

    python -m pytest test_league_rosters.py -q      (or: python test_league_rosters.py)
"""
from __future__ import annotations

import asyncio
import json

import httpx

import league_rosters as lr

lr.PAUSE = 0


class FakeAPI:
    """PlayByPoint: night lists by lesson id, ratings by user id; a value of
    Exception makes that call fail."""
    def __init__(self, nights, ratings):
        self.nights, self.ratings, self.calls = nights, ratings, []

    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False

    async def _get_json(self, path, params=None):
        self.calls.append(path)
        if path.endswith("lesson_players"):
            v = self.nights.get(params["lesson_id"])
            if v is Exception:
                raise RuntimeError("boom")
            return {"users": v}
        uid = int(path.split("/")[3])
        v = self.ratings.get(uid)
        if v is Exception:
            raise RuntimeError("boom")
        return {"ratings": v, "rpr_linked": True}


def run(api, todo, due, fail_rpc=(), ttodo=()):
    """Run the job against a fake Supabase; returns the RPC bodies sent."""
    sent = []

    def handler(request: httpx.Request):
        name = request.url.path.rsplit("/", 1)[-1]
        body = json.loads(request.content or b"{}")
        sent.append((name, body))
        if name in fail_rpc:
            return httpx.Response(500, text="nope")
        if name == "league_rosters_to_read":
            return httpx.Response(200, json=todo)
        if name == "league_roster_ingest":
            return httpx.Response(200, json={"applied": 3, "held": 0, "failed": 0, "added": 2, "missing": 0})
        if name == "tournament_rosters_to_read":
            return httpx.Response(200, json=list(ttodo))
        if name == "tournament_roster_ingest":
            return httpx.Response(200, json={"applied": True, "held": 0, "failed": 0, "added": 4, "missing": 0})
        if name == "pbp_people_due_ratings":
            return httpx.Response(200, json=[{"pbp_user_id": u} for u in due])
        if name == "pbp_ratings_ingest":
            return httpx.Response(200, json={"read": len(body["p_rows"]), "new_readings": 1})
        return httpx.Response(404)

    real = httpx.AsyncClient
    lr.httpx.AsyncClient = lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    lr.SUPABASE_KEY = "test"
    try:
        asyncio.run(lr.read_league_rosters({"c": "1"}, 7, lambda c, s: api, "venue"))
    finally:
        lr.httpx.AsyncClient = real
    return sent


TODO = [{"division_id": "d1", "program_slug": "lg", "batch_key": "B",
         "nights": [{"night_no": 1, "lesson_id": 11, "night_date": "2026-10-20"},
                    {"night_no": 2, "lesson_id": 12, "night_date": "2026-10-27"},
                    {"night_no": 3, "lesson_id": 13, "night_date": "2026-11-03"}]}]
U = lambda i, **k: {"id": i, "name": f" Player {i} ", "name_initials": "P", "avatar": None, **k}


def test_reads_three_nights_and_sends_them():
    api = FakeAPI({11: [U(1), U(2)], 12: [U(1)], 13: [U(1), U(3, is_child=True)]}, {})
    sent = run(api, TODO, [])
    ing = [b for n, b in sent if n == "league_roster_ingest"]
    assert len(ing) == 1 and ing[0]["p_division"] == "d1"
    reads = ing[0]["p_reads"]
    assert [r["night_no"] for r in reads] == [1, 2, 3]
    assert [u["id"] for u in reads[0]["users"]] == [1, 2]
    assert reads[0]["users"][0]["name"] == "Player 1"          # trimmed
    assert reads[0]["users"][0]["avatar"] == ""                # None -> ""
    assert reads[2]["users"][1]["is_child"] is True


def test_failed_night_is_sent_as_not_read():
    api = FakeAPI({11: [U(1)], 12: Exception, 13: None}, {})
    reads = [b for n, b in run(api, TODO, []) if n == "league_roster_ingest"][0]["p_reads"]
    assert reads[1]["users"] is None                            # call raised
    assert reads[2]["users"] is None                            # no list in the reply


def test_rows_without_an_id_are_dropped():
    api = FakeAPI({11: [U(1), {"name": "Guest"}, {"id": "x"}], 12: [], 13: []}, {})
    reads = [b for n, b in run(api, TODO, []) if n == "league_roster_ingest"][0]["p_reads"]
    assert [u["id"] for u in reads[0]["users"]] == [1]
    assert reads[1]["users"] == []                              # an empty list is still a read


def test_ratings_read_for_due_players_and_failures_sent_as_null():
    dupr = {"provider": "dupr", "double": "3.41", "external_user_id": "AB12CD"}
    api = FakeAPI({11: [], 12: [], 13: []}, {1: [dupr], 2: Exception, 3: [dupr, "junk"]})
    sent = run(api, TODO, [1, 2, 3])
    rows = [b for n, b in sent if n == "pbp_ratings_ingest"][0]["p_rows"]
    assert rows[0] == {"pbp_user_id": 1, "ratings": [dupr]}
    assert rows[1] == {"pbp_user_id": 2, "ratings": None}
    assert rows[2]["ratings"] == [dupr]                         # non-objects dropped


def test_ratings_saved_in_chunks():
    lr_chunk = lr.RATINGS_CHUNK
    lr.RATINGS_CHUNK = 2
    try:
        api = FakeAPI({}, {i: [] for i in range(5)})
        sent = run(api, [], list(range(5)))
    finally:
        lr.RATINGS_CHUNK = lr_chunk
    sizes = [len(b["p_rows"]) for n, b in sent if n == "pbp_ratings_ingest"]
    assert sizes == [2, 2, 1]


def test_a_failed_ingest_does_not_stop_ratings():
    api = FakeAPI({11: [U(1)], 12: [U(1)], 13: [U(1)]}, {1: []})
    sent = run(api, TODO, [1], fail_rpc=("league_roster_ingest",))
    assert any(n == "pbp_ratings_ingest" for n, _ in sent)


TTODO = [{"division_id": "t1", "program_slug": "tour-mens",
          "lessons": [{"lesson_id": 21, "date": "2026-12-12"}]},
         {"division_id": "t2", "program_slug": "tour-empty", "lessons": []}]


def test_tournament_divisions_read_and_sent():
    api = FakeAPI({21: [U(5), U(6)]}, {})
    sent = run(api, [], [], ttodo=TTODO)
    ing = [b for n, b in sent if n == "tournament_roster_ingest"]
    assert len(ing) == 1 and ing[0]["p_division"] == "t1"      # a division with no session is skipped
    assert ing[0]["p_reads"] == [{"lesson_id": 21, "users": [
        {"id": 5, "name": "Player 5", "name_initials": "P", "avatar": "", "is_child": False},
        {"id": 6, "name": "Player 6", "name_initials": "P", "avatar": "", "is_child": False}]}]


def test_tournament_failed_read_sent_as_not_read():
    api = FakeAPI({21: Exception}, {})
    ing = [b for n, b in run(api, [], [], ttodo=TTODO) if n == "tournament_roster_ingest"]
    assert ing[0]["p_reads"][0]["users"] is None


def test_tournament_trouble_does_not_stop_ratings():
    api = FakeAPI({21: [U(5)]}, {9: []})
    for fail in ("tournament_rosters_to_read", "tournament_roster_ingest"):
        sent = run(api, [], [9], fail_rpc=(fail,), ttodo=TTODO)
        assert any(n == "pbp_ratings_ingest" for n, _ in sent), fail


def test_supabase_down_never_raises():
    api = FakeAPI({}, {})
    sent = run(api, TODO, [], fail_rpc=("league_rosters_to_read",))
    assert [n for n, _ in sent] == ["league_rosters_to_read"]   # stopped quietly
    assert api.calls == []


def test_ratings_stop_after_failures_in_a_row():
    api = FakeAPI({}, {i: Exception for i in range(20)})
    sent = run(api, [], list(range(20)))
    rows = [b for n, b in sent if n == "pbp_ratings_ingest"][0]["p_rows"]
    assert len(rows) == lr.GIVE_UP_AFTER                        # not all 20
    assert all(r["ratings"] is None for r in rows)


def test_a_failed_ratings_save_does_not_stop_the_rest():
    lr_chunk = lr.RATINGS_CHUNK
    lr.RATINGS_CHUNK = 1
    try:
        api = FakeAPI({}, {1: [], 2: []})
        sent = run(api, [], [1, 2], fail_rpc=("pbp_ratings_ingest",))
    finally:
        lr.RATINGS_CHUNK = lr_chunk
    assert [n for n, _ in sent].count("pbp_ratings_ingest") == 2


def test_due_ids_accepts_both_shapes():
    assert lr._due_ids([{"pbp_user_id": 3}, 4, {"x": 1}, "5", None]) == [3, 4, 5]


def test_is_child_only_when_true():
    assert lr._player({"id": 1, "is_child": "false"})["is_child"] is False
    assert lr._player({"id": 1, "is_child": True})["is_child"] is True


def test_no_key_skips():
    lr.SUPABASE_KEY = ""
    asyncio.run(lr.read_league_rosters({}, 0, lambda c, s: FakeAPI({}, {}), ""))


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as e:
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
