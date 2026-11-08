"""Tests for dupr_sample.py. No network: PlayByPoint's HTTP session and
Supabase are both faked; the real PlayByPointAPI._get_json (with its retries)
and league_rosters.read_ratings run on top.

    python -m pytest test_dupr_sample.py -q
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import httpx
import pytest

import dupr_sample as ds
from extract_thejar import PlayByPointAPI


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    real = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda *a, **k: real(0))


# ── round robin ──────────────────────────────────────────────────────────────

def T(uid, venue):
    return {"pbp_user_id": uid, "venue": venue}


def test_round_robin_one_per_venue_in_turn():
    raw = [T(1, "a"), T(2, "a"), T(3, "a"), T(4, "b"), T(5, "c"), T(6, "c")]
    assert [t["pbp_user_id"] for t in ds.round_robin(raw, 10)] == [1, 4, 5, 2, 6, 3]


def test_round_robin_limit_keeps_every_venue():
    raw = [T(i, "a") for i in range(1, 11)] + [T(20, "b"), T(30, "c")]
    out = ds.round_robin(raw, 3)
    assert [t["venue"] for t in out] == ["a", "b", "c"]
    assert [t["pbp_user_id"] for t in out] == [1, 20, 30]


def test_round_robin_drops_duplicates_and_junk():
    raw = [T(1, "a"), T("1", "b"), {"pbp_user_id": None, "venue": "a"}, "x", T(2, None)]
    out = ds.round_robin(raw, 10)
    assert out == [T(1, "a"), T(2, "unknown")]


def test_round_robin_already_ordered_is_unchanged():
    raw = [T(1, "a"), T(4, "b"), T(2, "a"), T(5, "b"), T(3, "a")]
    assert ds.round_robin(raw, 10) == raw


# ── reading ──────────────────────────────────────────────────────────────────

DUPR = [{"provider": "dupr", "double": "3.812", "single": None, "external_user_id": "X1",
         "double_verified": "true"},
        {"provider": "ntrp_default", "double": "3.0"}]
NO_DUPR = [{"provider": "ntrp_default", "double": "3.0"}]


class Resp:
    def __init__(self, status, body=None):
        self.status_code, self._body, self.headers = status, body, {}
        self.text = json.dumps(body) if status < 400 else "null"

    def json(self):
        return self._body


class FakeSession:
    """PlayByPoint's HTTP session: user id -> ratings list or HTTP status."""
    def __init__(self, replies):
        self.replies, self.calls = replies, []

    async def get(self, path, params=None, headers=None):
        uid = int(path.split("/")[3])
        self.calls.append(uid)
        v = self.replies[uid]
        return Resp(v) if isinstance(v, int) else Resp(200, {"ratings": v})

    async def aclose(self):
        pass


def run(replies, targets):
    sent = []

    def handler(request: httpx.Request):
        name = request.url.path.rsplit("/", 1)[-1]
        body = json.loads(request.content or b"{}")
        sent.append((name, body))
        if name == "dupr_sample_register":
            return httpx.Response(200, json=len(body["p_ids"]))
        if name == "pbp_ratings_ingest":
            return httpx.Response(200, json={"read": len(body["p_rows"]), "new_readings": 2})
        return httpx.Response(404)

    ds.lr.SUPABASE_KEY = "test"
    session = FakeSession(replies)

    async def go():
        api = PlayByPointAPI(cookies={"c": "1"}, rate_limit=0)
        await api.aclose()
        api._client = session
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await ds.run_sample(client, api, targets, sleep=0)

    return asyncio.run(go()), session, sent


def test_all_read_counts_and_saves():
    targets = [T(1, "a"), T(2, "b"), T(3, "a")]
    res, session, sent = run({1: DUPR, 2: NO_DUPR, 3: []}, targets)
    t = res["totals"]
    assert res["stopped"] is None
    assert (t["read"], t["dupr"], t["doubles"], t["id"], t["doubles_verified"]) == (3, 1, 1, 1, 1)
    assert res["per_venue"]["a"]["read"] == 2 and res["per_venue"]["a"]["dupr"] == 1
    assert [n for n, _ in sent] == ["dupr_sample_register", "pbp_ratings_ingest"]
    assert sent[0][1] == {"p_ids": [1, 2, 3]}
    assert [r["pbp_user_id"] for r in sent[1][1]["p_rows"]] == [1, 2, 3]


@pytest.mark.parametrize("status", [403, 429, 401])
def test_stops_at_first_refusal_without_retrying(status):
    targets = [T(1, "a"), T(2, "b"), T(3, "a"), T(4, "b")]
    res, session, sent = run({1: DUPR, 2: status, 3: DUPR, 4: DUPR}, targets)
    assert str(status) in res["stopped"]
    # one request for player 2, none after: PlayByPointAPI's retries never reach the network
    assert session.calls == [1, 2]
    # what was read before the stop is still saved; the refused player is not
    assert sent[0][1] == {"p_ids": [1]}
    assert [r["pbp_user_id"] for r in sent[1][1]["p_rows"]] == [1]


def test_stops_after_three_failures_in_a_row():
    targets = [T(i, "a") for i in range(1, 7)]
    res, session, sent = run({1: DUPR, 2: 500, 3: 500, 4: 500, 5: DUPR, 6: DUPR}, targets)
    assert res["stopped"] == "3 failed reads in a row"
    assert 5 not in session.calls and 6 not in session.calls
    assert res["totals"]["failed"] == 3 and res["totals"]["read"] == 1


def test_two_failures_then_success_carries_on():
    targets = [T(i, "a") for i in range(1, 5)]
    res, session, sent = run({1: 500, 2: 500, 3: DUPR, 4: 500}, targets)
    assert res["stopped"] is None
    assert res["totals"]["read"] == 1 and res["totals"]["failed"] == 3


def test_nightly_window():
    at = lambda h, m: datetime(2026, 11, 8, h, m, tzinfo=timezone.utc)
    assert ds.in_nightly_window(at(15, 0)) and ds.in_nightly_window(at(14, 45))
    assert not ds.in_nightly_window(at(14, 44)) and not ds.in_nightly_window(at(17, 30))


# ── review fixes ──

def test_stops_cleanly_when_the_nightly_window_starts():
    times = iter([datetime(2026, 11, 8, 14, 44, tzinfo=timezone.utc),
                  datetime(2026, 11, 8, 14, 44, 30, tzinfo=timezone.utc),
                  datetime(2026, 11, 8, 14, 45, tzinfo=timezone.utc)])
    targets = [T(i, "a") for i in range(1, 5)]
    ds.lr.SUPABASE_KEY = "test"
    session = FakeSession({i: DUPR for i in range(1, 5)})
    sent = []

    def handler(request):
        name = request.url.path.rsplit("/", 1)[-1]
        body = json.loads(request.content or b"{}")
        sent.append((name, body))
        return httpx.Response(200, json=({"new_readings": 1} if name == "pbp_ratings_ingest" else 1))

    async def go():
        api = PlayByPointAPI(cookies={"c": "1"}, rate_limit=0)
        await api.aclose()
        api._client = session
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await ds.run_sample(client, api, targets, sleep=0, clock=lambda: next(times), force=False)

    res = asyncio.run(go())
    assert "nightly" in res["stopped"]
    assert session.calls == [1, 2]
    assert [r["pbp_user_id"] for r in sent[-1][1]["p_rows"]] == [1, 2]   # what was read is saved


def test_save_errors_never_print_ids(capsys):
    def handler(request):
        name = request.url.path.rsplit("/", 1)[-1]
        if name == "pbp_ratings_ingest":
            return httpx.Response(409, json={"message": "Key (pbp_user_id)=(424242) is not present"})
        return httpx.Response(200, json=1)

    ds.lr.SUPABASE_KEY = "test"

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await ds._save(client, [{"pbp_user_id": 424242, "ratings": DUPR}])

    assert asyncio.run(go()) == 0
    out = capsys.readouterr().out
    assert "424242" not in out and "HTTP 409" in out
