"""
past_roster_probe.py -- can PlayByPoint still give us a session's roster after it has been played?

The PM level plan (8 Oct) tracks DUPR changes after DUPR-rated sessions. The
roster job only reads sessions from today on, so a player who plays a rated
session and doesn't book again soon has no "after" reading. If the roster
endpoint still answers for past sessions, re-reading each rated session's
roster 3 and 10 days later gives an after-reading for everyone who played.

This takes sessions we recorded 1 to 14 days ago (from roster_sightings),
re-reads a sample of their rosters with the same call the roster job uses, and
reports, by how many days ago the session was:

  · how many reads worked, failed or came back empty
  · how many of the players we recorded are still on the roster
  · how many carry a DUPR-style rating, and how many of those DUPRs have
    changed since our last reading

Prints counts only: never a name, an id, a rating, an email or a cookie.
Writes nothing.

    python past_roster_probe.py     (needs SUPABASE_URL, SUPABASE_KEY, PBP_COOKIES_JSON)
"""
from __future__ import annotations

import asyncio
import collections
import json
import os
import random
import sys
from datetime import date, timedelta

import httpx

from extract_thejar import PlayByPointAPI

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY", "")
PER_BUCKET = int(os.environ.get("PROBE_PER_BUCKET", "8"))
BUCKETS = [("1 day ago", 1, 1), ("2 days ago", 2, 2), ("3 days ago", 3, 3), ("4-7 days ago", 4, 7), ("8-14 days ago", 8, 14)]


def dupr(v):
    """The build's rule: two or three decimals that aren't a half step, 2 to 8."""
    try:
        x = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    if not 2 <= x <= 8 or round(x * 1000) % 500 == 0:
        return None
    return round(x, 3)


async def recorded(client: httpx.AsyncClient) -> dict:
    """lesson_id -> {"date": d, "ratings": {player: rating text}} for sessions 1-14 days ago."""
    today = date.today()
    lo, hi = (today - timedelta(days=14)).isoformat(), (today - timedelta(days=1)).isoformat()
    out: dict = {}
    start, page = 0, 1000
    while True:
        r = await client.get(
            f"{SUPABASE_URL}/rest/v1/roster_sightings",
            params={"select": "lesson_id,session_date,pbp_user_id,rating_at_time",
                    "session_date": f"gte.{lo}", "and": f"(session_date.lte.{hi},lesson_id.not.is.null)"},
            headers={"apikey": SUPABASE_KEY, "Range-Unit": "items", "Range": f"{start}-{start + page - 1}"},
        )
        if r.status_code not in (200, 206):
            print(f"Reading roster_sightings failed: HTTP {r.status_code}")
            sys.exit(1)
        rows = r.json()
        for x in rows:
            s = out.setdefault(x["lesson_id"], {"date": x["session_date"], "ratings": {}})
            s["ratings"][x["pbp_user_id"]] = x.get("rating_at_time")
        if len(rows) < page:
            return out
        start += page


async def main() -> int:
    if not (SUPABASE_URL and SUPABASE_KEY):
        print("SUPABASE_URL / SUPABASE_KEY missing.")
        return 1
    try:
        data = json.loads(os.environ.get("PBP_COOKIES_JSON", "")) or {}
    except Exception:
        data = {}
    cookies, user_id = data.get("cookies") or {}, data.get("user_id") or 0
    if not cookies:
        print("PBP_COOKIES_JSON is missing or has no cookies.")
        return 1

    async with httpx.AsyncClient(timeout=30.0) as client:
        sessions = await recorded(client)
    print(f"Sessions recorded 1-14 days ago: {len(sessions)}")
    today = date.today()

    # Sample per bucket, sessions with 2+ DUPR players first (the ones that matter).
    rng = random.Random(8)
    picks: dict = {}
    for label, a, b in BUCKETS:
        pool = [(lid, s) for lid, s in sessions.items()
                if a <= (today - date.fromisoformat(s["date"])).days <= b]
        rng.shuffle(pool)
        pool.sort(key=lambda t: -sum(1 for v in t[1]["ratings"].values() if dupr(v) is not None))
        picks[label] = pool[:PER_BUCKET]

    fields = collections.Counter()
    async with PlayByPointAPI(cookies=cookies, club_slug="thejar") as api:
        api._user_id = user_id
        print()
        print(f"{'Played':<15}{'read':>6}{'failed':>8}{'empty':>7}{'recorded':>10}{'still on':>10}"
              f"{'DUPRs':>7}{'DUPR changed':>14}")
        for label, _, _ in BUCKETS:
            ok = failed = empty = rec = still = duprs = changed = 0
            for lid, s in picks[label]:
                rec += len(s["ratings"])
                try:
                    rd = await api._get_json("/api/public/clinics/lesson_players",
                                             params={"lesson_id": lid, "rating_provider": "dupr"})
                except Exception:
                    failed += 1
                    await asyncio.sleep(0.3)
                    continue
                users = (rd or {}).get("users") or []
                fields.update((rd or {}).keys() if isinstance(rd, dict) else ["<not an object>"])
                if not users:
                    empty += 1
                else:
                    ok += 1
                for u in users:
                    pid = u.get("id")
                    if pid in s["ratings"]:
                        still += 1
                        now, then = dupr(u.get("rating")), dupr(s["ratings"][pid])
                        if now is not None:
                            duprs += 1
                            if then is not None and now != then:
                                changed += 1
                await asyncio.sleep(0.3)
            print(f"{label:<15}{ok:>6}{failed:>8}{empty:>7}{rec:>10}{still:>10}{duprs:>7}{changed:>14}")
    print()
    print("Top-level fields in the roster responses (how many responses had each):")
    for k, n in fields.most_common():
        print(f"  {k}: {n}")
    print()
    print("Read: 'still on' close to 'recorded' means past rosters are served in full;")
    print("'empty' or 'failed' for older buckets shows how far back they go. 'DUPR changed'")
    print("counts players whose DUPR has moved since our last reading of that session.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
