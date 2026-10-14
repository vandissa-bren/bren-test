"""
test_league_nights.py -- pbp_league_nights rows (20261110100000, section 5).

Every league or tournament night on a programme's PlayByPoint page goes to
pbp_league_nights, past and future, one row per lesson: batch_id kept (null
when PlayByPoint gives none), deleted lessons skipped, times as HH:MM:SS from
hour_start / hour_end (seconds after midnight), nights more than 400 days away
dropped. Non-competitive programmes write nothing. A failed write never
raises.

Run: python3 test_league_nights.py   (or pytest test_league_nights.py)
"""
import asyncio
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import push_to_supabase as p

TODAY = date(2026, 10, 14)
T = TODAY.isoformat()
def d(n): return (TODAY + timedelta(days=n)).isoformat()


def lesson(i, n=7, **kw):
    base = {"id": i, "lesson_date": d(n), "hour_start": 68400, "hour_end": 73800,
            "formatted_hour_start": "7:00 PM", "batch_id": "0WVhLXqM", "capacity": 16,
            "player_count": 12, "status": "active", "created_at": "2026-09-01T10:30:00.000+10:00",
            "deleted_at": None, "clinic_id": 99}
    base.update(kw)
    return base


def test_row_shape_and_time_parsing():
    rows = p._league_night_rows("mon-league", 1234, [lesson(5001)], T)
    assert rows == [{
        "lesson_id": 5001, "program_slug": "mon-league", "venue_id": "pbp-1234",
        "night_date": d(7), "start_time": "19:00:00", "end_time": "20:30:00",
        "batch_id": "0WVhLXqM", "capacity": 16, "player_count": 12, "status": "active",
        "pbp_created_at": "2026-09-01T10:30:00.000+10:00"}]


def test_time_edges():
    r = p._league_night_rows("s", 1, [lesson(1, hour_start=0, hour_end=None),
                                      lesson(2, hour_start="32400", hour_end=34200 + 30),
                                      lesson(3, hour_start=None, hour_end=-5)], T)
    by = {x["lesson_id"]: x for x in r}
    assert (by[1]["start_time"], by[1]["end_time"]) == ("00:00:00", None)
    assert (by[2]["start_time"], by[2]["end_time"]) == ("09:00:00", "09:30:30")
    assert (by[3]["start_time"], by[3]["end_time"]) == (None, None)


def test_batch_present_and_null():
    r = p._league_night_rows("s", 1, [lesson(1), lesson(2, batch_id=None), lesson(3, batch_id=""),
                                      lesson(4, batch_id=77)], T)
    assert [x["batch_id"] for x in r] == ["0WVhLXqM", None, None, "77"]


def test_deleted_and_unplaceable_skipped():
    r = p._league_night_rows("s", 1, [
        lesson(1), lesson(2, deleted_at="2026-10-01T00:00:00Z"), lesson(None, id=None),
        lesson(4, lesson_date=None), lesson(5, lesson_date="not-a-date"), "junk", None], T)
    assert [x["lesson_id"] for x in r] == [1]


def test_past_and_future_kept_within_400_days():
    r = p._league_night_rows("s", 1, [
        lesson(1, n=-400), lesson(2, n=-401), lesson(3, n=-30), lesson(4, n=400), lesson(5, n=401)], T)
    assert sorted(x["lesson_id"] for x in r) == [1, 3, 4]


def test_bad_created_at_and_counts_become_null():
    r = p._league_night_rows("s", 1, [lesson(1, created_at="yesterday-ish", capacity=None,
                                             player_count="x", status=None)], T)[0]
    assert r["pbp_created_at"] is None and r["capacity"] is None
    assert r["player_count"] is None and r["status"] is None


# ── only competitive programmes are collected ──────────────────────────────

_FAIL_SLUGS = set()


class _FakeAPI:
    def __init__(self, **kw): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def program_detail_html(self, slug):
        return "" if slug in _FAIL_SLUGS else "<html>" + slug + "</html>"
    async def _get_json(self, *a, **k): return {}


def _scrape(stubs, competitive_slugs=()):
    saved = (p.PlayByPointAPI, p.fetch_clinic_stubs, p._extract_react_props_from_html, p.COMPETITIVE_SLUGS)
    async def fake_stubs(api, fid): return stubs, 1, False
    p.PlayByPointAPI = _FakeAPI
    p.fetch_clinic_stubs = fake_stubs
    p._extract_react_props_from_html = lambda html: {"sessions": [lesson(11), lesson(12, batch_id=None, n=30)]}
    p.COMPETITIVE_SLUGS = set(competitive_slugs)
    try:
        return asyncio.run(p.scrape_pbp_venue({}, 0, 1234, "Venue", "venue", [p.mel_today()]))
    finally:
        p.PlayByPointAPI, p.fetch_clinic_stubs, p._extract_react_props_from_html, p.COMPETITIVE_SLUGS = saved


def test_seen_ids_collected_only_for_competitive_pages_read_in_full():
    stubs = [
        {"id": 1, "url": "https://x/programs/monday-league", "name": "Monday Night League"},
        {"id": 2, "url": "https://x/programs/open-play", "name": "Open Play", "category": "Open Play"},
        {"id": 3, "url": "https://x/programs/tuesday-league", "name": "Tuesday League"},
    ]
    _FAIL_SLUGS.add("tuesday-league")          # challenged / unreadable page
    try:
        res = _scrape(stubs)
    finally:
        _FAIL_SLUGS.clear()
    assert res["_failed_slugs"] == ["tuesday-league"]
    assert res["_nights_seen"] == [("monday-league", [11, 12])]


def test_non_competitive_programme_skipped():
    stubs = [
        {"id": 1, "url": "https://x/programs/monday-league", "name": "Monday Night League"},
        {"id": 2, "url": "https://x/programs/open-play", "name": "Open Play", "category": "Open Play"},
        {"id": 3, "url": "https://x/programs/mixers", "name": "Monday Night Mixers"},
    ]
    res = _scrape(stubs, competitive_slugs={"mixers"})
    slugs = {r["program_slug"] for r in res["_nights"]}
    assert slugs == {"monday-league", "mixers"}
    assert len(res["_nights"]) == 4
    assert all(r["venue_id"] == "pbp-1234" for r in res["_nights"])
    assert {r["batch_id"] for r in res["_nights"]} == {"0WVhLXqM", None}


# ── the writer ─────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, code): self.status_code, self.text = code, "boom"


def _fake_client(codes, calls, raise_on=None):
    class C:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json=None, headers=None):
            calls.append((url, json, headers))
            if raise_on is not None and len(calls) - 1 == raise_on:
                raise RuntimeError("network down")
            return _Resp(codes[min(len(calls) - 1, len(codes) - 1)])
    return C


def _run_record(rows, codes=(201,), raise_on=None):
    calls = []
    real = p.httpx.AsyncClient
    p.httpx.AsyncClient = _fake_client(codes, calls, raise_on)
    try:
        asyncio.run(p.record_league_nights(rows))
    finally:
        p.httpx.AsyncClient = real
    return calls


def _rows(n, start=1):
    return [{**p._league_night_rows("s", 1, [lesson(i)], T)[0]} for i in range(start, start + n)]


def test_empty_makes_no_call():
    assert _run_record([]) == []


def test_upsert_target_headers_and_read_at():
    calls = _run_record(_rows(3))
    assert len(calls) == 1
    url, body, headers = calls[0]
    assert url.endswith("/rest/v1/pbp_league_nights?on_conflict=lesson_id")
    assert "merge-duplicates" in headers["Prefer"]
    assert len(body) == 3 and all(r["read_at"].endswith("Z") and "T" in r["read_at"] for r in body)
    assert all("first_seen" not in r for r in body)


def test_chunks_of_500_and_dedupe():
    calls = _run_record(_rows(1200) + _rows(10))   # 10 duplicates of lesson ids 1-10
    assert [len(c[1]) for c in calls] == [500, 500, 200]


def test_failure_never_raises_and_later_chunks_still_sent():
    calls = _run_record(_rows(1100), codes=(500, 201, 201))
    assert len(calls) == 3
    calls = _run_record(_rows(1100), raise_on=0)
    assert len(calls) == 3


# ── pruning moved / deleted nights ────────────────────────────────────────

class _PResp:
    def __init__(self, code=204, data=None): self.status_code, self.text, self._d = code, "boom", data or []
    def json(self): return self._d


def _prune(seen, stored=None, codes=(204,), today=T):
    """Run prune_league_nights against a fake client; returns the calls made."""
    calls = []
    def client_cls():
        class C:
            def __init__(self, **kw): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
            async def delete(self, url, params=None, headers=None):
                calls.append(("DELETE", url, dict(params), headers))
                return _PResp(codes[min(len(calls) - 1, len(codes) - 1)])
            async def get(self, url, params=None, headers=None):
                calls.append(("GET", url, dict(params), headers))
                rows = sorted(stored or [])[params["offset"]:params["offset"] + params["limit"]]
                return _PResp(200, [{"lesson_id": i} for i in rows])
        return C
    real = p.httpx.AsyncClient
    p.httpx.AsyncClient = client_cls()
    try:
        asyncio.run(p.prune_league_nights(seen, today))
    finally:
        p.httpx.AsyncClient = real
    return calls


def test_prune_deletes_stale_future_nights_for_a_read_slug():
    calls = _prune([("monday-league", [11, 12])])
    assert len(calls) == 1
    verb, url, params, headers = calls[0]
    assert verb == "DELETE" and url.endswith("/rest/v1/pbp_league_nights")
    assert params == {"program_slug": "eq.monday-league", "night_date": f"gte.{T}",
                      "lesson_id": "not.in.(11,12)"}
    assert "apikey" in headers and headers["Authorization"].startswith("Bearer ")


def test_prune_nothing_for_a_slug_whose_page_failed():
    # a failed page never reaches the seen list; with no pairs there is no call at all
    assert _prune([]) == []
    _FAIL_SLUGS.add("tuesday-league")
    try:
        failed = _scrape([{"id": 3, "url": "https://x/programs/tuesday-league", "name": "Tuesday League"}])
    finally:
        _FAIL_SLUGS.clear()
    assert failed["_nights_seen"] == [] and _prune(failed["_nights_seen"]) == []


def test_prune_deletes_all_future_rows_when_page_lists_no_future_nights():
    calls = _prune([("monday-league", [])])
    assert len(calls) == 1
    assert calls[0][2] == {"program_slug": "eq.monday-league", "night_date": f"gte.{T}"}   # no lesson_id filter


def test_prune_keeps_past_rows_by_bounding_on_today():
    assert _prune([("s", [1])])[0][2]["night_date"] == f"gte.{T}"


def test_prune_one_delete_per_slug_and_ids_merged():
    calls = _prune([("a", [2, 1]), ("b", [9]), ("a", [3])])
    assert [c[2]["program_slug"] for c in calls] == ["eq.a", "eq.b"]
    assert calls[0][2]["lesson_id"] == "not.in.(1,2,3)"


def test_prune_large_id_lists_read_stored_ids_and_delete_only_stale_in_chunks():
    seen_ids = list(range(1, 401))                       # 400 > 300
    stale = list(range(1000, 1650))                      # 650 stale -> 3 chunks of <=300
    calls = _prune([("big", seen_ids)], stored=set(seen_ids[:100]) | set(stale))
    gets = [c for c in calls if c[0] == "GET"]
    dels = [c for c in calls if c[0] == "DELETE"]
    assert gets and all("lesson_id" not in c[2] or c[2]["lesson_id"] for c in dels)
    deleted = []
    for c in dels:
        assert c[2]["program_slug"] == "eq.big" and c[2]["lesson_id"].startswith("in.(")
        ids = [int(x) for x in c[2]["lesson_id"][4:-1].split(",")]
        assert len(ids) <= 300
        deleted += ids
    assert sorted(deleted) == stale                       # nothing kept is touched
    assert len(dels) == 3


def test_prune_failure_never_raises_and_other_slugs_still_pruned():
    calls = _prune([("a", [1]), ("b", [2])], codes=(500, 204))
    assert len(calls) == 2


def test_prune_slug_is_url_encoded_by_the_client_not_interpolated():
    calls = _prune([("odd slug&x=1", [1])])
    assert calls[0][1].endswith("/pbp_league_nights") and "odd slug" not in calls[0][1]
    assert calls[0][2]["program_slug"] == "eq.odd slug&x=1"      # httpx encodes params


def test_record_league_nights_reports_failure_so_run_skips_prune():
    calls = []
    real = p.httpx.AsyncClient
    p.httpx.AsyncClient = _fake_client((500,), calls)
    try:
        assert asyncio.run(p.record_league_nights(_rows(2))) is False
    finally:
        p.httpx.AsyncClient = real
    p.httpx.AsyncClient = _fake_client((201,), [])
    try:
        assert asyncio.run(p.record_league_nights(_rows(2))) is True
    finally:
        p.httpx.AsyncClient = real


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"  PASS  {name}")
            except Exception as e:
                failed += 1; print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print("ALL PASS" if not failed else f"{failed} FAILURES")
    sys.exit(1 if failed else 0)
