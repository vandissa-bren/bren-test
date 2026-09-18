#!/usr/bin/env python3
"""
capture_roster_sightings.py  —  M1, 19 Sep 2026

Append-only capture of WHO is on each PlayByPoint session roster, several times
a day, so the model can see joins and — the signal that matters — cancellations.
The nightly scrape only keeps one overwritten snapshot; this keeps history.

Per run:
  1. read the session list + lesson_ids from availability_cache (already there;
     no new PBP catalogue calls),
  2. for each session with a lesson_id, call /api/public/clinics/lesson_players
     (the same endpoint the scrape and the count-refresh already use),
  3. call the Supabase function record_roster_sightings once, with every
     session's roster, which writes only the deltas (joins + departures).

Runs on GitHub Actions (ubuntu runners reach PlayByPoint; repo is public so
minutes are free), using the PBP_COOKIES_JSON + SUPABASE_* secrets — the same
ones the nightly scrape uses. NOT on .117 (Cloudflare-blocked).

The Supabase key goes on the `apikey` header only: new-style sb_ keys are
rejected on Authorization: Bearer (the F73 lesson). Legacy eyJ keys still work
on apikey too, so apikey-only is correct for both.
"""
import asyncio, json, os, sys, httpx
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

from extract_thejar import PlayByPointAPI

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://stwohmddmdwttasbyblt.supabase.co").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_KEY", "")
PBP_BASE = os.environ.get("PBP_BASE_OVERRIDE") or "https://app.playbypoint.com"
# Only capture rosters for sessions from today out to this many days ahead:
# rosters for long-past sessions never change, and far-future ones are usually
# empty. The band that actually moves is the near future.
DAYS_AHEAD = int(os.environ.get("ROSTER_DAYS_AHEAD", "21"))


def supabase_headers() -> dict:
    # apikey only (F73): new-style keys are rejected on Bearer.
    return {"apikey": SUPABASE_KEY, "Content-Type": "application/json"}


def load_cookies():
    raw = os.environ.get("PBP_COOKIES_JSON", "")
    if not raw:
        return {}, 0
    try:
        d = json.loads(raw)
        return d.get("cookies", {}) or {}, d.get("user_id", 0) or 0
    except Exception:
        return {}, 0


async def read_catalogue(client) -> list[dict]:
    """The stored PBP catalogue: one row per venue, each with its sessions."""
    r = await client.get(
        f"{SUPABASE_URL}/rest/v1/availability_cache",
        params={"select": "data", "platform": "eq.playbypoint"},
        headers=supabase_headers(),
    )
    if r.status_code != 200:
        print(f"  CATALOGUE READ FAILED: HTTP {r.status_code} {r.text[:200]}")
        return []
    return [row["data"] for row in r.json() if row.get("data")]


def collect_target_sessions(catalogue: list[dict]) -> list[dict]:
    """Every session with a lesson_id, in the near-future band, deduped by lesson_id."""
    today = datetime.now(timezone.utc).date()
    seen, out = set(), []
    for venue in catalogue:
        for s in venue.get("sessions", []) or []:
            lid = s.get("lesson_id")
            if not lid or lid in seen:
                continue
            date_str = s.get("date")
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else None
            except Exception:
                d = None
            if d is not None and not (today <= d <= today.fromordinal(today.toordinal() + DAYS_AHEAD)):
                continue
            seen.add(lid)
            out.append({"lesson_id": lid, "date": date_str})
    return out


async def fetch_roster(api, lesson_id) -> list[dict] | None:
    """The current roster for one session. None on failure (so the caller can
    tell 'empty' from 'could not read' and NOT record a spurious mass-departure)."""
    try:
        rd = await api._get_json(
            "/api/public/clinics/lesson_players",
            params={"lesson_id": lesson_id, "rating_provider": "dupr"},
        )
    except Exception as e:
        print(f"  lesson {lesson_id}: fetch failed ({type(e).__name__})")
        return None
    return [
        {"id": u.get("id"), "name": u.get("name"),
         "initials": u.get("name_initials"), "rating": u.get("rating")}
        for u in (rd or {}).get("users", [])
        if u.get("id") is not None
    ]


async def main():
    if not SUPABASE_KEY:
        print("ERROR: no SUPABASE_KEY / SUPABASE_SERVICE_KEY in the environment")
        sys.exit(2)
    cookies, user_id = load_cookies()
    kw = {"app_base_url": PBP_BASE} if os.environ.get("PBP_BASE_OVERRIDE") else {}

    async with httpx.AsyncClient(timeout=30.0) as client:
        catalogue = await read_catalogue(client)
        if not catalogue:
            print("No catalogue rows; nothing to capture.")
            sys.exit(1)
        targets = collect_target_sessions(catalogue)
        print(f"Roster capture: {len(targets)} sessions in the next {DAYS_AHEAD} days")

        run_at = datetime.now(timezone.utc).isoformat()
        observations, failures, empty, withppl = [], 0, 0, 0
        # One shared client session for all lesson_players calls.
        async with PlayByPointAPI(cookies=cookies, club_slug="thejar", **kw) as api:
            api._user_id = user_id
            for t in targets:
                players = await fetch_roster(api, t["lesson_id"])
                if players is None:
                    failures += 1
                    continue                      # could-not-read: send nothing for it
                if players:
                    withppl += 1
                else:
                    empty += 1
                observations.append({
                    "session_key": f"pbp-{t['lesson_id']}",
                    "lesson_id": t["lesson_id"],
                    "session_date": t.get("date"),
                    "observed_at": run_at,
                    "players": players,           # [] here = genuinely empty (read OK)
                })
                await asyncio.sleep(0.3)

        # Send everything we read (including genuine empties -> departures) in
        # one RPC call. Sessions we could NOT read are omitted, so a fetch
        # failure never fabricates a mass departure.
        written = 0
        if observations:
            r = await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/record_roster_sightings",
                headers=supabase_headers(),
                json={"p_observations": observations},
            )
            if r.status_code == 200:
                try:
                    written = int(r.json())
                except Exception:
                    written = -1
            else:
                print(f"  RPC FAILED: HTTP {r.status_code} {r.text[:200]}")
                sys.exit(1)

        print("=" * 56)
        print(f"CAPTURE SUMMARY  sessions={len(targets)}  read_ok={withppl + empty}  "
              f"with_players={withppl}  empty={empty}  read_failed={failures}  "
              f"rows_written={written}")
        # Loud only if we read NOTHING at all (session dead / blocked).
        if targets and (withppl + empty) == 0:
            print("READ NOTHING FROM PLAYBYPOINT — session may be expired or blocked")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
