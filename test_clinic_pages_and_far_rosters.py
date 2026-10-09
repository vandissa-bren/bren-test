"""
test_clinic_pages_and_far_rosters.py -- 9 Oct.

1. The programme list is read page by page. It used to ask for one page of 50
   and never the next, so a venue with more than 50 programmes silently lost
   the rest. Eastern Indoor was reading 43.
2. Far-out competitive sessions (tournament divisions, league nights beyond
   the 14-day window) get their roster saved in data.far_rosters, for
   Discover's opened division. Everything else beyond the window is unchanged:
   no roster read, fill log only.
3. A roster that couldn't be read this run keeps the stored one.

No network: a fake PlayByPoint client stands in. Run: python3 test_clinic_pages_and_far_rosters.py
"""
import asyncio
import sys
from datetime import date, timedelta

import push_to_supabase as pts

failures = []
def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + ("" if cond else f"  {detail}"))
    if not cond:
        failures.append(label)


class PagedAPI:
    """Serves /api/public/clinics in pages. ignore_page=True repeats page 1."""
    def __init__(self, n, ignore_page=False, fail_page=None):
        self.all = [{"id": i, "url": f"https://x/programs/p{i}", "name": f"P{i}"} for i in range(n)]
        self.ignore_page, self.fail_page, self.calls = ignore_page, fail_page, []
    async def _get_json(self, path, params=None):
        page = 1 if self.ignore_page else params.get("page", 1)
        self.calls.append(page)
        if page == self.fail_page:
            raise RuntimeError("HTTP 403")
        size = params["per_page"]
        return {"clinics": self.all[(page - 1) * size: page * size]}

def run(coro):
    return asyncio.run(coro)

print("1 · programme list paging")
api = PagedAPI(43)
stubs, pages, more = run(pts.fetch_clinic_stubs(api, 1009))
check("43 programmes: one page, all read", len(stubs) == 43 and pages == 1 and not more, (len(stubs), pages, more))

api = PagedAPI(50)
stubs, pages, more = run(pts.fetch_clinic_stubs(api, 1009))
check("exactly 50: asks once more, finds nothing, stops", len(stubs) == 50 and pages == 2 and not more, (len(stubs), pages, more))

api = PagedAPI(123)
stubs, pages, more = run(pts.fetch_clinic_stubs(api, 1009))
check("123 programmes: three pages, all read, none twice",
      len(stubs) == 123 and pages == 3 and len({s["id"] for s in stubs}) == 123, (len(stubs), pages))

api = PagedAPI(80, ignore_page=True)
stubs, pages, more = run(pts.fetch_clinic_stubs(api, 1009))
check("PlayByPoint ignoring 'page': no duplicates, never fewer than before, flagged",
      len(stubs) == 50 and more, (len(stubs), more))

api = PagedAPI(80, fail_page=2)
try:
    run(pts.fetch_clinic_stubs(api, 1009)); raised = False
except RuntimeError:
    raised = True
check("a later page failing raises (the venue is kept as stored, not saved half-read)", raised)

print("2 · which programmes are competitive")
check("tournament category", pts.is_competitive({"category": "Tournament", "name": "Spring Open"}))
check("league by name", pts.is_competitive({"category": "Event", "name": "Monday Social League Beginner"}))
check("numbered event by name", pts.is_competitive({"category": "Event", "name": "Open Singles Slam IV"}))
check("open play is not", not pts.is_competitive({"category": "Open Play", "name": "Tuesday Open Play"}))
check("clinic is not", not pts.is_competitive({"category": "Clinic", "name": "Beginner clinic"}))

print("3 · far rosters inside scrape_pbp_venue")
TODAY = date.today()
d = lambda n: (TODAY + timedelta(days=n)).isoformat()
LESSONS = {
    "spring-open": [  # competitive, far: roster read when someone's entered
        {"id": 501, "lesson_date": d(30), "capacity": 16, "player_count": 11, "hour_start": 28800},
        {"id": 502, "lesson_date": d(31), "capacity": 16, "player_count": 0, "hour_start": 28800},   # nobody yet: no read
        {"id": 503, "lesson_date": d(200), "capacity": 16, "player_count": 4, "hour_start": 28800},  # past FAR_ROSTER_DAYS
        {"id": 504, "lesson_date": d(40), "capacity": 16, "player_count": 5, "hour_start": 28800},   # read fails
    ],
    "open-play": [  # not competitive: no roster read beyond the window
        {"id": 601, "lesson_date": d(30), "capacity": 12, "player_count": 6, "hour_start": 28800},
    ],
}
STUBS = [{"id": 1, "url": "https://x/programs/spring-open", "name": "Spring Open", "category": "Tournament"},
         {"id": 2, "url": "https://x/programs/open-play", "name": "Open Play", "category": "Open Play"}]
roster_calls = []

class FakePBP:
    def __init__(self, **kw): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def _get_json(self, path, params=None):
        if path == "/api/public/clinics":
            return {"clinics": STUBS if params.get("page", 1) == 1 else []}
        if path == "/api/public/clinics/lesson_players":
            roster_calls.append(params["lesson_id"])
            if params["lesson_id"] == 504:
                raise RuntimeError("HTTP 500")
            return {"users": [{"id": 9, "name": "Sam Kerr", "name_initials": "SK", "rating": "3.41", "avatar": ""}]}
        raise AssertionError(path)
    async def program_detail_html(self, slug):
        return slug

orig_api, orig_props = pts.PlayByPointAPI, pts._extract_react_props_from_html
pts.PlayByPointAPI = FakePBP
pts._extract_react_props_from_html = lambda html: {"sessions": LESSONS[html], "name": html}
try:
    dates = [TODAY + timedelta(days=i) for i in range(pts.DAYS_AHEAD)]
    res = run(pts.scrape_pbp_venue({}, 1, 999, "Test venue", "test", dates))
finally:
    pts.PlayByPointAPI, pts._extract_react_props_from_html = orig_api, orig_props

fr = res.get("far_rosters", {})
check("competitive, far, entrants: roster saved, name shortened", "501" in fr and fr["501"]["roster"][0]["name"] == "Sam K.", fr.get("501"))
check("saved with its programme and date", fr.get("501", {}).get("program_slug") == "spring-open" and fr.get("501", {}).get("date") == d(30))
check("nobody entered: not read", 502 not in roster_calls and "502" not in fr)
check("beyond FAR_ROSTER_DAYS: not read", 503 not in roster_calls and "503" not in fr)
check("read failed: listed for carry-over, nothing saved", "504" in res.get("_roster_failed", []) and "504" not in fr)
check("not competitive: no roster read", 601 not in roster_calls)
check("fill log unchanged: all far sessions still logged", {o["session_key"] for o in res["_far"]} >= {"pbp-501", "pbp-502", "pbp-503", "pbp-504", "pbp-601"},
      [o["session_key"] for o in res["_far"]])

check("short_name", (pts.short_name("Sam Kerr"), pts.short_name("Ana"), pts.short_name("  "), pts.short_name("Jo-Ann  de  Silva")) == ("Sam K.", "Ana", None, "Jo-Ann S."))

print("4 · carry-over in supabase_upsert")
class FakeResp:
    def __init__(self, code, body=None): self.status_code, self._b, self.text = code, body, ""
    def json(self): return self._b
posted = {}
EXISTING = {"sessions": [], "far_rosters": {
    "504": {"program_slug": "spring-open", "date": d(40), "roster": [{"name": "Old Read"}]},   # read failed this run: keep
    "700": {"program_slug": "gone", "date": d(10), "roster": []},                               # programme read fine, no longer listed: drop
    "701": {"program_slug": "unreadable", "date": d(20), "roster": [{"name": "Kept"}]},        # programme page failed: keep
    "702": {"program_slug": "unreadable", "date": d(-1), "roster": []},                         # date passed: drop
}}
class FakeClient:
    def __init__(self, **kw): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def get(self, url, params=None, headers=None):
        return FakeResp(200, [{"id": "pbp-999", "data": EXISTING}])
    async def post(self, url, json=None, headers=None):
        posted["records"] = json; return FakeResp(201)
orig_client = pts.httpx.AsyncClient
pts.httpx.AsyncClient = FakeClient
try:
    rec = {"id": "pbp-999", "data": {"sessions": [], "far_rosters": {"501": fr["501"]}},
           "_keep_slugs": ["unreadable"], "_keep_rosters": ["504"]}
    run(pts.supabase_upsert([rec]))
finally:
    pts.httpx.AsyncClient = orig_client
out = posted["records"][0]["data"]["far_rosters"]
check("new read kept", "501" in out)
check("failed read: stored roster kept", out.get("504", {}).get("roster") == [{"name": "Old Read"}])
check("unreadable programme: stored roster kept", "701" in out)
check("no longer listed: dropped", "700" not in out)
check("date passed: dropped", "702" not in out)
check("private keys not written", "_keep_rosters" not in posted["records"][0] and "_keep_slugs" not in posted["records"][0])

print(f"\n{'ALL PASS' if not failures else f'{len(failures)} FAILED'}")
sys.exit(1 if failures else 0)
