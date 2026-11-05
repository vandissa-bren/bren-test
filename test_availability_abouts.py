"""
test_availability_abouts.py -- /api/pbp/availability sends no descriptions;
/api/pbp/abouts gives them by key (6 Nov 2026).

Production: the availability list was 188 KB compressed, three quarters of it
descriptions repeated for every night of the fortnight.

Run: python3 test_availability_abouts.py
"""
import asyncio, gzip, json, sys
import api_server as a

failures = []
def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + ("" if cond else f"  {detail}"))
    if not cond: failures.append(label)

HTML = "<p>Social night. <b>All levels</b>.</p>" + "x" * 3000
rows = [{
    "id": 101, "name": "Test Hall", "slug": "testhall", "by_date": {},
    "sessions": [
        {"title": "Social", "date": f"2026-11-{d:02d}", "start": "18:00", "end": "20:00",
         "lesson_id": 900 + d, "description_html": HTML, "description": "Social night. All levels."}
        for d in range(6, 20)
    ] + [{"title": "No words", "date": "2026-11-07", "start": "09:00", "end": "10:00", "lesson_id": 1}],
}]

async def fake_read(platform, attempts=3):
    return json.loads(json.dumps(rows))
a._read_from_supabase = fake_read
a._abouts.clear()

async def main():
    print("\n-- the list --")
    r = await a.pbp_availability(date=None, from_time="00:00", to_time="23:30",
                                 venue_ids=None, pm_user_id=None, authorization=None)
    sessions = r["venues"][0]["sessions"]
    check("15 sessions", len(sessions) == 15, len(sessions))
    check("no description on any session",
          not any("description" in s or "description_html" in s for s in sessions))
    keys = {s.get("about") for s in sessions if s.get("about")}
    check("the 14 nights share one about key", len(keys) == 1, keys)
    check("a session with no description has no key", "about" not in sessions[-1])
    check("the source rows are untouched", "description_html" in rows[0]["sessions"][0])
    key = keys.pop()

    print("\n-- the warm cache builds the same list --")
    a._availability_cache.clear()
    from datetime import date as D
    await a._warm_single_date(D.today(), json.loads(json.dumps(rows)))
    warm = a._cache_get(f"availability:{D.today().isoformat()}:00:00:23:30:all")
    check("warm list has no descriptions",
          warm and not any("description_html" in s for v in warm["venues"] for s in v["sessions"]))

    print("\n-- the descriptions --")
    one = await a.pbp_abouts(keys=key)
    check("read by key", one.get(key, {}).get("description_html") == HTML)
    bad = await a.pbp_abouts(keys="../etc,ZZZ")
    check("keys that aren't keys are ignored", bad == {}, bad)

    print("\n-- after a restart --")
    a._abouts.clear(); a._abouts_reread_at = 0.0
    again = await a.pbp_abouts(keys=key)
    check("read again from the cache table", key in again)
    a._abouts.clear()
    check("but not twice in a minute", await a.pbp_abouts(keys=key) == {})

    print("\n-- size --")
    before = len(gzip.compress(json.dumps(rows).encode()))
    after = len(gzip.compress(json.dumps(r).encode()))
    check("smaller than before", after < before, (before, after))

asyncio.run(main())
print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}"); sys.exit(1)
print("all checks passed")
