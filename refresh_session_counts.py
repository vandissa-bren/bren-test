"""
refresh_session_counts.py
Refreshes spots_left and status for today/tomorrow sessions only.
Reads lesson_ids from Supabase, calls lesson_players API, updates counts.
"""
import asyncio, json, os, httpx
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
from datetime import date, timedelta
from pathlib import Path
from extract_thejar import PlayByPointAPI

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://stwohmddmdwttasbyblt.supabase.co")
# The service key, under the name the rest of this server uses for it.
# `SUPABASE_KEY` held a service-role token here for a month while
# reading like the public anon key — the name is why it was pasted into
# a file that went to a public repo.
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

# Slugs come from venue_registry (the site's public/venues.json). A venue it
# does not know is skipped -- never sent with another venue's slug, which is
# what the old `.get(fid, "nplpickleball")` default did for 15 of 22 venues.
import venue_registry

def _load_cookies():
    raw = os.environ.get("PBP_COOKIES_JSON", "")
    if raw:
        try:
            data = json.loads(raw)
            return data.get("cookies", {}), data.get("user_id", 0)
        except Exception:
            pass
    for p in [Path(__file__).parent / ".pbp_cookies.json", Path.home() / ".pbp_cookies.json"]:
        if p.exists():
            try:
                data = json.loads(p.read_text())
                return data.get("cookies", {}), data.get("user_id", 0)
            except Exception:
                pass
    return {}, 0

async def refresh_venue(api, record, date_strs):
    data = record["data"]
    sessions = data.get("sessions", [])
    updated = 0
    for s in sessions:
        if s.get("date") not in date_strs:
            continue
        lid = s.get("lesson_id")
        if not lid:
            continue
        try:
            resp = await api._get_json('/api/public/clinics/lesson_players', params={'lesson_id': lid, 'rating_provider': 'dupr'})
            players = len((resp or {}).get("users", []))
            cap = s.get("capacity", 0)
            spots = max(0, cap - players) if cap else None
            is_full = cap > 0 and spots == 0
            s["spots_left"] = spots
            s["status"] = "Full" if is_full else "Available"
            updated += 1
            await asyncio.sleep(0.2)
        except Exception as e:
            print(f"  lesson {lid} error: {e}")
    return updated

async def main():
    today = date.today()
    tomorrow = today + timedelta(days=1)
    date_strs = {today.isoformat(), tomorrow.isoformat()}
    cookies, user_id = _load_cookies()

    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/availability_cache",
            params={"platform": "eq.playbypoint", "select": "id,data"},
            headers=headers,
        )
        records = resp.json()

    print(f"Refreshing session counts for {today} and {tomorrow}...")
    total_updated = 0

    for record in records:
        fid = record["data"].get("id")
        v = venue_registry.resolve(fid)
        if not isinstance(v, venue_registry.Venue) or v.status != "active":
            print(f"  skipping {fid}: not an active venue in the registry")
            continue
        slug = v.slug
        name = record["data"].get("name", str(fid))
        sessions_today = [s for s in record["data"].get("sessions", []) if s.get("date") in date_strs]
        if not sessions_today:
            continue
        try:
            async with PlayByPointAPI(cookies=cookies, club_slug=slug) as api:
                api._user_id = user_id
                updated = await refresh_venue(api, record, date_strs)
                total_updated += updated
                print(f"  {name}: {updated} sessions updated")
        except Exception as e:
            print(f"  {name} error: {e}")
            continue
        await asyncio.sleep(1)

    # Push updates back to Supabase
    async with httpx.AsyncClient() as client:
        for record in records:
            fid = record["data"].get("id")
            sessions_today = [s for s in record["data"].get("sessions", []) if s.get("date") in date_strs]
            if not sessions_today:
                continue
            await client.patch(
                f"{SUPABASE_URL}/rest/v1/availability_cache",
                params={"id": f"eq.{record['id']}"},
                headers=headers,
                json={"data": record["data"]},
            )

    print(f"Done. {total_updated} sessions refreshed.")

if __name__ == "__main__":
    asyncio.run(main())
