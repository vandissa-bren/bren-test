#!/usr/bin/env python3
"""
reread_played_rosters.py  --  8 Oct 2026

Re-reads the rosters of sessions played 1, 3, 7 and 14 days ago and records
each player's DUPR as it is now (record_rating_checks). With the rating on the
day (roster_sightings.rating_at_time) this measures how ratings move after a
session, for everyone who played, including players who don't book again
soon. New DUPRs also go into player_rating_log (source 'reread').

The past_roster_probe (8 Oct) showed PlayByPoint still serves past rosters in
full. Uses the same roster call as the roster job. Writes ONLY the rating
checks and the rating log: never roster_sightings, so each session's
before-rating and cancellation timing stay as they were.

Sends player ids and ratings only (no names). Prints counts only.

Env: SUPABASE_URL, SUPABASE_KEY (service), PBP_COOKIES_JSON,
     REREAD_DAYS (default "1,3,7,14").
"""
import asyncio
import collections
import json
import re
import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv

load_dotenv()

from extract_thejar import PlayByPointAPI  # noqa: E402

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://stwohmddmdwttasbyblt.supabase.co").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY", "")
PBP_BASE = os.environ.get("PBP_BASE_OVERRIDE") or "https://app.playbypoint.com"
DAYS = [int(x) for x in os.environ.get("REREAD_DAYS", "1,3,7,14").split(",") if x.strip()]
BATCH = 150   # sessions per RPC call


def headers() -> dict:
    # apikey only (F73): new-style keys are rejected on Bearer.
    return {"apikey": SUPABASE_KEY, "Content-Type": "application/json"}


def load_cookies():
    try:
        d = json.loads(os.environ.get("PBP_COOKIES_JSON", "")) or {}
    except Exception:
        d = {}
    return d.get("cookies") or {}, d.get("user_id") or 0


async def played_sessions(client: httpx.AsyncClient, dates: dict) -> list[dict]:
    """Sessions we recorded on each target date: [{session_key, lesson_id, session_date, days_after}]."""
    out, seen = [], set()
    start, page = 0, 1000
    while True:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/roster_sightings",
            params={"select": "session_key,lesson_id,session_date",
                    "session_date": f"in.({','.join(dates)})", "lesson_id": "not.is.null"},
            headers={**headers(), "Range-Unit": "items", "Range": f"{start}-{start + page - 1}"},
        )
        if r.status_code not in (200, 206):
            print(f"Reading roster_sightings failed: HTTP {r.status_code}")
            sys.exit(1)
        rows = r.json()
        for x in rows:
            if x["session_key"] in seen:
                continue
            seen.add(x["session_key"])
            out.append({"session_key": x["session_key"], "lesson_id": x["lesson_id"],
                        "session_date": x["session_date"], "days_after": dates[x["session_date"]]})
        if len(rows) < page:
            return out
        start += page


async def main():
    if not SUPABASE_KEY:
        print("ERROR: no SUPABASE_KEY / SUPABASE_SERVICE_KEY in the environment")
        sys.exit(2)
    cookies, user_id = load_cookies()
    if not cookies:
        print("ERROR: PBP_COOKIES_JSON is missing or has no cookies")
        sys.exit(2)
    kw = {"app_base_url": PBP_BASE} if os.environ.get("PBP_BASE_OVERRIDE") else {}
    today = datetime.now(ZoneInfo("Australia/Melbourne")).date()
    dates = {(today - timedelta(days=k)).isoformat(): k for k in DAYS}

    async with httpx.AsyncClient(timeout=30.0) as client:
        targets = await played_sessions(client, dates)
        print(f"Re-read: {len(targets)} played sessions, {len(DAYS)} offsets ({', '.join(map(str, DAYS))} days)")
        run_at = datetime.now(timezone.utc).isoformat()
        checks, ok, empty, failed, players = [], 0, 0, 0, 0
        per_offset = {k: [0, 0] for k in DAYS}          # offset -> [sessions read, players]
        async with PlayByPointAPI(cookies=cookies, club_slug="thejar", **kw) as api:
            api._user_id = user_id

            async def read(lesson_id):
                return await api._get_json("/api/public/clinics/lesson_players",
                                           params={"lesson_id": lesson_id, "rating_provider": "dupr"})

            # The client already retries each read three times, so a read that
            # still fails is refused for good (the first runs lost the same 11 of
            # 154 twice). Count why, as HTTP status or error type only.
            results, why = [], collections.Counter()
            for t in targets:
                try:
                    results.append((t, await read(t["lesson_id"])))
                except Exception as e:
                    failed += 1
                    m = re.search(r"\b(\d{3}) ", str(e)[:40])
                    why[f"HTTP {m.group(1)}" if m else type(e).__name__] += 1
                await asyncio.sleep(0.3)
            for t, rd in results:
                users = [{"id": u.get("id"), "rating": u.get("rating")}
                         for u in (rd or {}).get("users", []) if u.get("id") is not None]
                if users:
                    ok += 1
                    players += len(users)
                    per_offset[t["days_after"]][0] += 1
                    per_offset[t["days_after"]][1] += len(users)
                    checks.append({**{k: t[k] for k in ("session_key", "session_date", "days_after")},
                                   "observed_at": run_at, "players": users})
                else:
                    empty += 1

        written = {"checks": 0, "changes": 0}
        for i in range(0, len(checks), BATCH):
            r = await client.post(f"{SUPABASE_URL}/rest/v1/rpc/record_rating_checks",
                                  headers=headers(), json={"p_checks": checks[i:i + BATCH]})
            if r.status_code != 200:
                print(f"  RPC FAILED: HTTP {r.status_code} {r.text[:200]}")
                sys.exit(1)
            res = r.json() or {}
            written["checks"] += int(res.get("checks", 0))
            written["changes"] += int(res.get("changes", 0))

    print("=" * 56)
    for k in DAYS:
        print(f"  {k:>2} days after: {per_offset[k][0]} sessions read, {per_offset[k][1]} players")
    if failed:
        print(f"  failed reads by reason: {', '.join(f'{k}: {n}' for k, n in why.most_common())}")
    print(f"REREAD SUMMARY  sessions={len(targets)}  read_ok={ok}  empty={empty}  failed={failed}  "
          f"players={players}  dupr_checks={written['checks']}  new_duprs={written['changes']}")
    if targets and ok == 0:
        print("READ NOTHING FROM PLAYBYPOINT — session may be expired or blocked")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
