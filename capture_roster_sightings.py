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
DAYS_START = int(os.environ.get("ROSTER_DAYS_START", "0"))
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
        params={"select": "id,data", "platform": "eq.playbypoint"},
        headers=supabase_headers(),
    )
    if r.status_code != 200:
        print(f"  CATALOGUE READ FAILED: HTTP {r.status_code} {r.text[:200]}")
        return []
    return [(row["id"], row["data"]) for row in r.json() if row.get("data")]


def collect_target_sessions(catalogue: list[dict]) -> list[dict]:
    """Every session with a lesson_id, in the near-future band, deduped by lesson_id."""
    today = datetime.now(timezone.utc).date()
    seen, out = set(), []
    for row_id, venue in catalogue:
        for s in venue.get("sessions", []) or []:
            lid = s.get("lesson_id")
            if not lid or lid in seen:
                continue
            date_str = s.get("date")
            try:
                d = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else None
            except Exception:
                d = None
            lo = today.fromordinal(today.toordinal() + DAYS_START)
            hi = today.fromordinal(today.toordinal() + DAYS_AHEAD - 1)
            if d is not None and not (lo <= d <= hi):
                continue
            seen.add(lid)
            out.append({"row_id": row_id, "lesson_id": lid, "date": date_str,
                        "capacity": s.get("capacity"),
                        "start": s.get("start"),
                        "session_type": s.get("type") or s.get("category"),
                        "price": s.get("price"),
                        "status": s.get("status"),
                        "title": s.get("title"),
                        "program_slug": s.get("program_slug"),
                        "skill_level": s.get("skill_level")})
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
        print(f"Roster capture: {len(targets)} sessions, day window {DAYS_START}..{DAYS_AHEAD-1}")

        run_at = datetime.now(timezone.utc).isoformat()
        observations, spot_updates, inv_observations, failures, empty, withppl = [], [], [], 0, 0, 0
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
                    # the session's facts, stored with its roster so the history
                    # survives the session leaving the catalogue (F114)
                    "venue_id": t["row_id"],
                    "start_time": t.get("start"),
                    "session_type": t.get("session_type"),
                    "title": t.get("title"),
                    "program_slug": t.get("program_slug"),
                    "skill_level": t.get("skill_level"),
                    "price": t.get("price"),
                    "capacity": t.get("capacity"),
                })
                # Count write-back: spots_left = capacity - roster size, when we
                # know the capacity. This is the (previously dead) count refresh,
                # served by the same fetch. Only sessions we READ contribute; a
                # failed fetch never touches spots.
                cap = t.get("capacity")
                if cap is not None:
                    left = max(0, int(cap) - len(players))
                    spot_updates.append({
                        "id": t["row_id"],
                        "lesson_id": t["lesson_id"],
                        "spots_left": left,
                        "status": "Full" if left == 0 else "Available",
                        "observed_at": run_at,
                    })
                    # inventory_snapshots observation — the durable demand-curve
                    # history, now with venue + type so it can later be sliced by
                    # venue / type / day / price band (analytics groundwork).
                    inv_observations.append({
                        "session_key": f"pbp-{t['lesson_id']}",
                        "spots_left": left,
                        "capacity": int(cap),
                        "session_date": t.get("date"),
                        "start_time": t.get("start"),
                        "price": t.get("price"),
                        "status": "Full" if left == 0 else "Available",
                        "venue_id": t["row_id"],
                        "session_type": t.get("session_type"),
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

        # Count write-back through the session-level merge (M2), so only the
        # named sessions' spots change -- never a whole-array overwrite (F30).
        spots_written = 0
        if spot_updates and not os.environ.get("SKIP_SPOTS"):
            r = await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/merge_session_spots",
                headers=supabase_headers(),
                json={"p_updates": spot_updates},
            )
            if r.status_code == 200:
                try:
                    spots_written = int(r.json())
                except Exception:
                    spots_written = -1
            else:
                print(f"  SPOTS RPC FAILED: HTTP {r.status_code} {r.text[:200]}")

        # inventory_snapshots: the durable demand-curve history (server-side,
        # complete every run — not browsing-dependent like the client write).
        inv_written = 0
        if inv_observations and not os.environ.get("SKIP_INVENTORY"):
            r = await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/record_inventory_snapshot",
                headers=supabase_headers(),
                json={"p_observations": inv_observations, "p_source": "pbp"},
            )
            if r.status_code == 200:
                try:
                    inv_written = int(r.json())
                except Exception:
                    inv_written = -1
            else:
                print(f"  INVENTORY RPC FAILED: HTTP {r.status_code} {r.text[:160]}")

        # Roster history upkeep (F114): fill any session facts still missing, and
        # roll sightings older than 12 months into per-player counts. Idempotent,
        # and does nothing when nothing is due. A failure here never fails the run.
        upkeep = None
        if not os.environ.get("SKIP_UPKEEP"):
            r = await client.post(
                f"{SUPABASE_URL}/rest/v1/rpc/maintain_roster_history",
                headers=supabase_headers(),
                json={},
            )
            if r.status_code == 200:
                upkeep = r.json()
            else:
                print(f"  UPKEEP RPC FAILED: HTTP {r.status_code} {r.text[:160]}")

        print("=" * 56)
        print(f"CAPTURE SUMMARY  sessions={len(targets)}  read_ok={withppl + empty}  "
              f"with_players={withppl}  empty={empty}  read_failed={failures}  "
              f"rows_written={written}  spots_updated={spots_written}  inv_snapshots={inv_written}  "
              f"upkeep={upkeep}")
        # Loud only if we read NOTHING at all (session dead / blocked).
        if targets and (withppl + empty) == 0:
            print("READ NOTHING FROM PLAYBYPOINT — session may be expired or blocked")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
