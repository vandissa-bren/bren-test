"""
geo_probe.py -- can GitHub's runners read the geo-restricted venues?

SportsWell (885), Raya (1770) and Pickle4Real (1783) have their court hire
fetched by fetch_sportswell.py on a Sydney server, because in August they came
back empty from anywhere but an Australian address. Fixed servers get blocked
by PlayByPoint after a couple of weeks of steady use (the API server in
September, the Sydney server around 30 Sep). GitHub runs each job from a
different address, which is why fetch_court_blocks.py there has not been
blocked. If GitHub can read these three venues now, the job can move there.

For each venue, for tomorrow: the venue's court list, the day's hourly slots,
and the free courts in the first open slot. A control venue the GitHub jobs
already read (The Jar | South Melbourne, 597) is read the same way, so a
failure there means the cookies, not the venue.

Prints counts and status only, never a cookie or a court name. Saves nothing.

    python geo_probe.py            (needs PBP_COOKIES_JSON)

Exit code: 0 when all three geo venues returned slots and courts, 1 otherwise.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from extract_thejar import PlayByPointAPI

GEO = [(885, "sportswellpickleballpalace", "SportsWell"),
       (1770, "rayapickleballclub", "Raya"),
       (1783, "PICKLE4REAL", "Pickle4Real")]
CONTROL = (597, "nplpickleball", "The Jar South Melbourne (control)")


def hhmm(sec: int) -> str:
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}"


async def probe(cookies: dict, fid: int, slug: str, label: str, day) -> tuple[bool, bool]:
    """(signed in, read the day). Signed in = the court-types call answered."""
    print(f"\n{label} ({fid})")
    ok = True
    signed_in = True
    async with PlayByPointAPI(cookies=cookies, club_slug=slug) as api:
        # 1. the venue's court types (signed-in endpoint)
        try:
            ct = await api.court_types(fid, kind=None)
            print(f"  court types        {len(ct or [])} returned")
        except Exception as e:
            print(f"  court types        REFUSED  {type(e).__name__}: {str(e)[:100]}")
            ok = signed_in = False
        # 2. tomorrow's slots, as fetch_sportswell asks for them
        slots = []
        try:
            h = await api.available_hours(fid, day, surface="pickleball")
            slots = [s for s in ((h or {}).get("available_hours") or []) if isinstance(s, dict)]
            free = [s for s in slots if s.get("available")]
            print(f"  slots tomorrow     {len(slots)} listed, {len(free)} with a free court")
        except Exception as e:
            print(f"  slots tomorrow     REFUSED  {type(e).__name__}: {str(e)[:100]}")
            return signed_in, False
        if not slots:
            print("  slots tomorrow     NONE returned")
            return signed_in, False
        # 3. free courts in the first open slot
        first = next((s for s in slots if s.get("available")), None)
        if first is None:
            print("  free courts        no open slot tomorrow (fully booked?) -- slots still prove access")
            return signed_in, ok
        sec = int(first["seconds_from_midnight"])
        try:
            courts = await api.available_courts(fid, day, sec, sec + 1800, surface="pickleball")
            print(f"  free courts at {hhmm(sec)} {len(courts or [])}")
            if not courts:
                ok = False
        except Exception as e:
            print(f"  free courts        REFUSED  {type(e).__name__}: {str(e)[:100]}")
            ok = False
    return signed_in, ok


async def main() -> int:
    raw = os.environ.get("PBP_COOKIES_JSON", "")
    try:
        cookies = (json.loads(raw) or {}).get("cookies") or {}
    except Exception:
        cookies = {}
    if not cookies:
        print("PBP_COOKIES_JSON is missing or not valid JSON with a 'cookies' object.")
        return 1
    day = datetime.now(ZoneInfo("Australia/Melbourne")).date() + timedelta(days=1)
    print(f"Reading {day.isoformat()} (tomorrow, Melbourne) from this runner.")

    # The control only has to prove the cookies are signed in: its courts may sit
    # on a surface other than "pickleball", so its slots aren't judged.
    control_ok, _ = await probe(cookies, *CONTROL, day)
    results = {}
    for fid, slug, label in GEO:
        results[label] = (await probe(cookies, fid, slug, label, day))[1]
        await asyncio.sleep(1)

    print("\nVERDICT")
    if not control_ok:
        print("  The control venue failed too: the cookies or this runner are the problem, not the")
        print("  geo restriction. Run 'PBP Probe' to see which, then run this again.")
        return 1
    if all(results.values()):
        print("  All three geo venues answered from GitHub. Their court hire can move to GitHub,")
        print("  like the other venues, and the Sydney server can go.")
        return 0
    if not any(results.values()):
        print("  None of the three geo venues answered, while the control did: the geo restriction")
        print("  still applies to GitHub. Keep a Sydney server (new address, slower schedule).")
        return 1
    print("  Mixed: " + ", ".join(f"{k} {'ok' if v else 'FAILED'}" for k, v in results.items()))
    print("  The ones that failed still need the Sydney server.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
