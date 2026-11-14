"""
dupr_monthly.py -- monthly DUPR read at the venues that hide DUPR (14 Nov 2026)

Runs on the 2nd of each month (.github/workflows/dupr_monthly.yml), or by hand.

13 venues' public rosters hide DUPR, so the PM level (insights/pm_v2.py) has
little to anchor their regulars to. Each month this reads PlayByPoint's
signed-in ratings for the regulars there:

1. dupr_monthly_targets(p_limit, p_min_sessions) (migration 20270215100000)
   picks players with 4+ sessions in the last 60 days, seen only at those
   venues, no DUPR on any roster, not read in the last 25 days; round-robin
   by home venue.
2. Each is read and saved exactly as the one-off sample did (dupr_sample.py:
   the nightly job's read_ratings and pbp_ratings_ingest, the same sign-in,
   about one read a second, stopping at the first 401/403/429 or after 3 failed
   reads in a row, and outside 14:45-17:30 UTC).
3. The nightly insights build reads the DUPR rows of pbp_rating_readings
   (build_insights "pbpreads") into the PM level's DUPR history.

Prints aggregate counts only: never names, PlayByPoint ids or DUPR ids.

    DUPR_MONTHLY_LIMIT=1000 python dupr_monthly.py
    DUPR_MONTHLY_DRY_RUN=1 python dupr_monthly.py     (pick and count only)
"""
from __future__ import annotations

import asyncio
import os
import sys
from collections import Counter
from datetime import datetime, timezone

import httpx

import dupr_sample as S
import league_rosters as lr

console = lr.console

LIMIT = int(os.environ.get("DUPR_MONTHLY_LIMIT") or 1000)
MAX_LIMIT = 2000
MIN_SESSIONS = int(os.environ.get("DUPR_MONTHLY_MIN_SESSIONS") or 4)
DRY_RUN = os.environ.get("DUPR_MONTHLY_DRY_RUN", "").lower() in ("1", "true", "yes")


async def main() -> int:
    if not lr.SUPABASE_KEY:
        console.print("[red]No SUPABASE_KEY[/red]")
        return 1
    if not DRY_RUN and not S.FORCE and S.in_nightly_window(datetime.now(timezone.utc)):
        console.print("[red]The nightly scrape runs 15:00 UTC; run this outside 14:45-17:30 UTC.[/red]")
        return 1
    limit = max(1, min(LIMIT, MAX_LIMIT))
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            raw = await lr._rpc(client, "dupr_monthly_targets", {"p_limit": limit, "p_min_sessions": MIN_SESSIONS})
        except Exception as e:
            console.print(f"[red]Could not pick targets: {S._why(e)}[/red]")
            return 1
        targets = S.round_robin(raw, limit)
        by_venue = Counter(t["venue"] for t in targets)
        console.print(f"Targets: {len(targets)} across {len(by_venue)} venues: "
                      + ", ".join(f"{v} {n}" for v, n in sorted(by_venue.items())))
        if DRY_RUN or not targets:
            return 0

        from push_to_supabase import PBP_SLUG_MAP, PROXY_URL, _load_cookies  # same sign-in as the nightly job
        from extract_thejar import PlayByPointAPI
        cookies, user_id = _load_cookies()
        if not cookies:
            console.print("[red]No PBP_COOKIES_JSON[/red]")
            return 1
        async with PlayByPointAPI(cookies=cookies, club_slug=next(iter(PBP_SLUG_MAP.values()), ""),
                                  proxy=PROXY_URL) as api:
            api._user_id = user_id
            res = await S.run_sample(client, api, targets)
        S.report(res, len(targets), title="DUPR monthly read")
        return 2 if res["stopped"] else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
