"""
league_rosters.py -- host leagues phase 2: season rosters and ratings (14 Oct 2026)

Run at the end of push_to_supabase.py's nightly run (read_league_rosters).

1. Asks Supabase which claimed season divisions still need their roster read
   (league_rosters_to_read: nights 1 to 3 of each live division, until two
   days after its third night).
2. Reads PlayByPoint's player list for each of those nights and hands the lot
   to league_roster_ingest, one division at a time. A night that can't be read
   is sent as "not read" (users null), so what we had for it is kept; the
   database also holds back a list that comes back empty or sharply shorter.
3. Asks which roster players are due a ratings read (pbp_people_due_ratings:
   never read, or not in the last 20 hours) and reads each one's ratings from
   PlayByPoint (signed in), then saves them with pbp_ratings_ingest. Only
   changes are stored.

Never fails the nightly run: every problem is printed and skipped.
"""
from __future__ import annotations

import asyncio
import os

import httpx
from rich.console import Console

console = Console()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://stwohmddmdwttasbyblt.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
RATINGS_PER_RUN = 400        # most players read for ratings in one night
RATINGS_CHUNK = 100          # players per save
PAUSE = 0.3                  # between PlayByPoint calls, on top of the client's own pacing
GIVE_UP_AFTER = 5            # ratings reads failing in a row: signed out, stop for tonight


def _headers() -> dict:
    return {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"}


async def _rpc(client: httpx.AsyncClient, name: str, body: dict):
    resp = await client.post(f"{SUPABASE_URL}/rest/v1/rpc/{name}", headers=_headers(), json=body)
    if resp.status_code != 200:
        raise RuntimeError(f"{name}: HTTP {resp.status_code} {resp.text[:200]}")
    return resp.json()


def _player(u: dict) -> dict | None:
    """The fields we keep from one row of PlayByPoint's player list."""
    if not isinstance(u, dict) or u.get("id") is None:
        return None
    try:
        uid = int(u["id"])
    except (TypeError, ValueError):
        return None
    return {"id": uid, "name": (u.get("name") or "").strip(), "name_initials": u.get("name_initials") or "",
            "avatar": u.get("avatar") or "", "is_child": u.get("is_child") is True}


async def read_night(api, lesson_id) -> list[dict] | None:
    """One night's players, or None when the read failed."""
    try:
        rd = await api._get_json("/api/public/clinics/lesson_players",
                                 params={"lesson_id": lesson_id, "rating_provider": "dupr"})
    except Exception:
        return None
    users = (rd or {}).get("users") if isinstance(rd, dict) else None
    if not isinstance(users, list):
        return None
    return [p for p in (_player(u) for u in users) if p]


async def read_ratings(api, pbp_user_id: int) -> list[dict] | None:
    """One player's ratings entries (one per provider), or None when the read failed."""
    try:
        rd = await api._get_json(f"/api/users/{int(pbp_user_id)}/ratings")
    except Exception:
        return None
    entries = (rd or {}).get("ratings") if isinstance(rd, dict) else None
    return [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else None


def _due_ids(due) -> list[int]:
    """pbp_people_due_ratings as PostgREST returns it: [{"pbp_user_id": n}]
    (or a bare [n] list, just in case)."""
    out = []
    for r in due or []:
        v = r.get("pbp_user_id") if isinstance(r, dict) else r
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            pass
    return out


async def _save_ratings(client, rows) -> int:
    """One chunk; a failed save is printed and the run carries on."""
    try:
        return (await _rpc(client, "pbp_ratings_ingest", {"p_rows": rows})).get("new_readings", 0)
    except Exception as e:
        console.print(f"  [yellow]League ratings: {len(rows)} players not saved: {e}[/yellow]")
        return 0


async def read_league_rosters(cookies: dict, user_id: int, api_factory, slug: str) -> None:
    """The whole job. `api_factory(cookies, slug)` returns an async context
    manager giving a signed-in PlayByPoint client (PlayByPointAPI)."""
    if not SUPABASE_KEY:
        console.print("  [yellow]League rosters: no SUPABASE_KEY, skipped[/yellow]")
        return
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            todo = await _rpc(client, "league_rosters_to_read", {})
            due_first = None
            async with api_factory(cookies, slug) as api:
                api._user_id = user_id

                # 1-2. rosters
                done = held = failed = added = missing = 0
                for div in todo or []:
                    reads = []
                    for n in div.get("nights") or []:
                        users = await read_night(api, n.get("lesson_id"))
                        reads.append({"night_no": n.get("night_no"), "lesson_id": n.get("lesson_id"),
                                      "night_date": n.get("night_date"), "users": users})
                        await asyncio.sleep(PAUSE)
                    try:
                        res = await _rpc(client, "league_roster_ingest",
                                         {"p_division": div["division_id"], "p_reads": reads})
                        done += 1
                        held += res.get("held", 0)
                        failed += res.get("failed", 0)
                        added += res.get("added", 0)
                        missing += res.get("missing", 0)
                    except Exception as e:
                        console.print(f"  [yellow]League roster {div.get('program_slug')} not saved: {e}[/yellow]")
                console.print(f"  League rosters: {done} of {len(todo or [])} divisions read · {added} added"
                              f" · {missing} now missing" + (f" · {held} nights held back" if held else "")
                              + (f" · {failed} nights unreadable" if failed else ""))

                # 3. ratings
                due = await _rpc(client, "pbp_people_due_ratings", {"p_limit": RATINGS_PER_RUN})
                ids = _due_ids(due)
                due_first = len(ids)
                rows, ok, saved, in_a_row, gave_up = [], 0, 0, 0, False
                for uid in ids:
                    entries = await read_ratings(api, uid)
                    ok += entries is not None
                    in_a_row = 0 if entries is not None else in_a_row + 1
                    rows.append({"pbp_user_id": uid, "ratings": entries})
                    await asyncio.sleep(PAUSE)
                    if len(rows) >= RATINGS_CHUNK:
                        saved += await _save_ratings(client, rows)
                        rows = []
                    if in_a_row >= GIVE_UP_AFTER:
                        gave_up = True
                        break
                if rows:
                    saved += await _save_ratings(client, rows)
            if due_first is not None:
                console.print(f"  League ratings: {ok} of {due_first} players read · {saved} changed"
                              + (f" · stopped after {GIVE_UP_AFTER} failed reads in a row (signed out?)"
                                 if gave_up else ""))
    except Exception as e:
        console.print(f"  [yellow]League rosters stopped: {type(e).__name__}: {e}[/yellow]")
