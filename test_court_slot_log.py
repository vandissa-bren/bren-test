"""
test_court_slot_log.py -- the court-hire log rows written by fetch_court_blocks.

What has to hold:
  * every slot PlayByPoint lists is logged, TAKEN ones included (they never
    reach court_slots, which holds free courts only)
  * free courts are listed by id, sorted, and only courts in the inventory
  * n_courts is the inventory count (settles "8 or 10 courts")
  * slots that have already started today are left out
  * a day with a read error is never logged (a court we failed to read
    must not look booked)
  * weekend shifts carry the _weekend tag, as the price cache does

Run: python3 test_court_slot_log.py
"""
import asyncio, os, sys
from datetime import date, datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

os.environ.setdefault("SUPABASE_URL", "x")
os.environ.setdefault("SUPABASE_KEY", "x")
os.environ.setdefault("PBP_COOKIES_JSON", "{}")

import fetch_court_blocks as fcb
import court_surfaces, court_inventory

failures = []
def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + ("" if cond else f"  {detail}"))
    if not cond: failures.append(label)

MEL = ZoneInfo("Australia/Melbourne")
H = lambda h, m=0: h * 3600 + m * 60

print("\n-- court_slot_rows on its own --")
day = date(2030, 1, 2)                       # a Wednesday
universe = {H(18): "primetime", H(18, 30): "primetime", H(19): "primetime"}
court_slots = {"11|Court 1": [H(18), H(18, 30)], "12|Court 2": [H(18)],
               "99|Not a court": [H(18), H(19)]}
obs = datetime(2030, 1, 1, 9, tzinfo=timezone.utc)
rows = fcb.court_slot_rows(1557, day, universe, court_slots, {"11", "12", "13"}, obs,
                           now_local=datetime(2030, 1, 1, 12, tzinfo=MEL))
by = {r["slot_start"]: r for r in rows}
check("one row per listed slot", sorted(by) == ["18:00", "18:30", "19:00"], sorted(by))
check("free courts by id, sorted", by["18:00"]["free_court_ids"] == ["11", "12"], by["18:00"])
check("a fully taken slot is still logged, with nothing free",
      by["19:00"]["free_court_ids"] == [], by["19:00"])
check("courts outside the inventory are left out",
      all("99" not in r["free_court_ids"] for r in rows))
check("n_courts is the inventory count", all(r["n_courts"] == 3 for r in rows))
check("shift carried", by["18:30"]["shift"] == "primetime")
check("observed_at is the read time", by["18:00"]["observed_at"] == obs.isoformat())

print("\n-- today: slots already started are not logged --")
today = date(2030, 1, 2)
rows = fcb.court_slot_rows(1557, today, universe, court_slots, {"11", "12"}, obs,
                           now_local=datetime(2030, 1, 2, 18, 15, tzinfo=MEL))
check("18:00 (started) dropped, later slots kept",
      [r["slot_start"] for r in rows] == ["18:30", "19:00"], [r["slot_start"] for r in rows])
rows = fcb.court_slot_rows(1557, date(2030, 1, 1), universe, court_slots, {"11"}, obs,
                           now_local=datetime(2030, 1, 2, 9, tzinfo=MEL))
check("a past day logs nothing", rows == [], rows)


print("\n-- through fetch_court_blocks_for_venue --")
class FakeAPI:
    def __init__(self, fail_slot=None):
        self.fail_slot = fail_slot
    async def court_types(self, fid, kind=None):
        return [{"surface": "standard_courts"}]
    async def courts(self, fid):
        return []
    async def available_hours(self, fid, d, surface=None):
        return {"available_hours": [
            {"seconds_from_midnight": H(18), "available": True, "shift": "primetime"},
            {"seconds_from_midnight": H(18, 30), "available": True, "shift": "primetime"},
            {"seconds_from_midnight": H(19), "available": False, "shift": "primetime"},
        ]}
    async def available_courts(self, fid, d, start, end, surface=None):
        if self.fail_slot == start:
            raise RuntimeError("timeout")
        return {H(18): [{"id": 11, "name": "Court 1"}, {"id": 12, "name": "Court 2"}],
                H(18, 30): [{"id": 11, "name": "Court 1"}]}.get(start, [])

court_surfaces.resolve_surfaces = lambda fid, ct: SimpleNamespace(
    court=["standard_courts"], unknown=[], non_court=[], alternate=[], diagnostic=lambda: "")
court_inventory.build_inventory = lambda fid, ct, courts: SimpleNamespace(
    courts=[SimpleNamespace(id=11), SimpleNamespace(id=12), SimpleNamespace(id=13)])
async def no_prices(api, blocks, d, uid, prices, fetched_at=None):
    return prices, fetched_at or {}
fcb.fetch_missing_prices = no_prices
async def no_sleep(*a, **k): return None
fcb.asyncio.sleep = no_sleep

saturday = date(2030, 1, 5)
log, errors = [], []
blocks, _, _ = asyncio.run(fcb.fetch_court_blocks_for_venue(
    FakeAPI(), 1557, saturday, 1, {}, {}, errors=errors, slot_log=log))
by = {r["slot_start"]: r for r in log}
check("no errors on a clean read", errors == [], errors)
check("all three listed slots logged, including the taken 19:00",
      sorted(by) == ["18:00", "18:30", "19:00"], sorted(by))
check("19:00 logged with nothing free", by.get("19:00", {}).get("free_court_ids") == [], by.get("19:00"))
check("18:00 has courts 11 and 12 free", by["18:00"]["free_court_ids"] == ["11", "12"])
check("weekend shift tagged", by["18:00"]["shift"] == "primetime_weekend", by["18:00"]["shift"])
check("n_courts = 3 from the inventory", by["18:00"]["n_courts"] == 3)
check("blocks still produced as before", len(blocks) == 1 and blocks[0]["court_id"] == "11", blocks)
check("slot_log is optional (old call shape still works)",
      asyncio.run(fcb.fetch_court_blocks_for_venue(FakeAPI(), 1557, saturday, 1, {}, {}))[0] == blocks)

print("\n-- a read error marks the day, so main() drops its rows --")
log, errors = [], []
asyncio.run(fcb.fetch_court_blocks_for_venue(
    FakeAPI(fail_slot=H(18, 30)), 1557, saturday, 1, {}, {}, errors=errors, slot_log=log))
check("the error is reported", len(errors) == 1, errors)
src = open("fetch_court_blocks.py").read()
check("main keeps a day's rows only when it read without errors",
      "if not date_errors:\n                        court_log_rows.extend(day_slot_rows)" in src)
check("main sends the log through record_court_slots",
      "rpc/record_court_slots" in src)
check("a failed log write fails the run (joins save_failed)",
      'save_failed.append(("court_slot_states"' in src)

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all checks passed")
