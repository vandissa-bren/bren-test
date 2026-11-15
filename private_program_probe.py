"""
private_program_probe.py -- 16 Nov 2026, read only

Can the scraper's account read a programme that PlayByPoint doesn't list
publicly (a private weekly session), given its page address? For each slug:
does the page open, does it carry its sessions, and do the public roster reads
answer for them. Prints counts and dates only: no names, ids or ratings.

    PROBE_SLUGS=monday-funday-session python private_program_probe.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import date

from extract_thejar import PlayByPointAPI, _extract_react_props_from_html

SLUGS = [s.strip() for s in os.environ.get("PROBE_SLUGS", "").split(",") if s.strip()]
ROSTERS = 4          # roster reads per programme: a few past, one upcoming


async def probe(api, slug: str) -> None:
    html = await api.program_detail_html(slug)
    if not html:
        print(f"{slug}: page did not open")
        return
    props = _extract_react_props_from_html(html)
    if not props:
        print(f"{slug}: page opened ({len(html) // 1024} KB) but carries no session data "
              f"({'sign-in page' if 'sign_in' in html or 'Log in' in html else 'unknown layout'})")
        return
    lessons = [x for x in (props.get("sessions") or props.get("clinic_lessons") or []) if isinstance(x, dict)]
    today = date.today().isoformat()
    past = sorted((x for x in lessons if (x.get("lesson_date") or "") < today), key=lambda x: x["lesson_date"])
    ahead = sorted((x for x in lessons if (x.get("lesson_date") or "") >= today), key=lambda x: x["lesson_date"])
    keys = sorted(k for k in props.keys())[:40]
    print(f"{slug}: page opened; {len(lessons)} sessions on it ({len(past)} past, {len(ahead)} ahead)"
          + (f"; dates {lessons and min(x.get('lesson_date') or '' for x in lessons)} to "
             f"{max(x.get('lesson_date') or '' for x in lessons)}" if lessons else ""))
    print(f"  page fields: {', '.join(keys)}")
    for x in past[-(ROSTERS - 1):] + ahead[:1]:
        try:
            rd = await api._get_json("/api/public/clinics/lesson_players",
                                     params={"lesson_id": x.get("id"), "rating_provider": "dupr"})
            users = (rd or {}).get("users")
            print(f"  roster {x.get('lesson_date')}: "
                  + (f"{len(users)} players (capacity {x.get('capacity')}, player_count {x.get('player_count')})"
                     if isinstance(users, list) else "no roster in the answer"))
        except Exception as e:
            print(f"  roster {x.get('lesson_date')}: failed ({type(e).__name__})")
        await asyncio.sleep(1.0)


async def main() -> int:
    if not SLUGS:
        print("Set PROBE_SLUGS")
        return 1
    from push_to_supabase import PBP_SLUG_MAP, PROXY_URL, _load_cookies
    cookies, user_id = _load_cookies()
    if not cookies:
        print("No PBP_COOKIES_JSON")
        return 1
    async with PlayByPointAPI(cookies=cookies, club_slug=next(iter(PBP_SLUG_MAP.values()), ""), proxy=PROXY_URL) as api:
        api._user_id = user_id
        for slug in SLUGS[:5]:
            await probe(api, slug)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
