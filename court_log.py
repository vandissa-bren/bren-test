"""
court_log.py -- the court-hire history, shared by the court jobs.

  court_slot_rows      one venue-day's rows for the court-hire log
                       (court_slot_states, via record_court_slots)
  send_court_slots     posts those rows, in chunks
  record_court_rates   logs the court-hire rates now stored in
                       availability_cache (court_rate_states, 20261016110000)

Used by fetch_court_blocks.py (most venues, GitHub Actions) and
fetch_sportswell.py (SportsWell, Raya and Pickle4Real, from the Sydney
server). No settings are read at import, so either job can use it.
"""
from datetime import date, datetime
from zoneinfo import ZoneInfo

# Court-hire log rows per record_court_slots call.
COURT_LOG_CHUNK = 2000


def sec_to_hhmm(sec: int) -> str:
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}"


def court_slot_rows(facility_id, target_date: date, universe: dict, court_slots: dict,
                    valid_ids: set, observed_at: datetime, now_local: datetime = None) -> list:
    """The court-hire log rows for one venue-day: one per 30-minute slot.

    universe     {sec: shift} -- every slot PlayByPoint listed, free or taken
    court_slots  {"court_id|name": [secs free]} -- from fetch_blocks_for_surface
    valid_ids    the venue's bookable court ids (its inventory)

    Slots that have already started are left out: PlayByPoint stops offering a
    started slot, and logging it would record a booking that never happened.
    Courts outside the inventory are left out, the same rule the blocks follow.
    """
    now_local = now_local or datetime.now(ZoneInfo("Australia/Melbourne"))
    free_by_sec: dict = {}
    for court_key, secs in court_slots.items():
        court_id = court_key.split("|", 1)[0]
        if str(court_id) not in valid_ids:
            continue
        for sec in secs:
            free_by_sec.setdefault(int(sec), set()).add(str(court_id))

    rows = []
    for sec in sorted(set(universe) | set(free_by_sec)):
        if target_date == now_local.date():
            now_sec = now_local.hour * 3600 + now_local.minute * 60 + now_local.second
            if sec <= now_sec:
                continue
        elif target_date < now_local.date():
            continue
        rows.append({
            "facility_id": int(facility_id),
            "slot_date": target_date.isoformat(),
            "slot_start": sec_to_hhmm(sec),
            "free_court_ids": sorted(free_by_sec.get(sec, set())),
            "n_courts": len(valid_ids),
            "shift": universe.get(sec),
            "observed_at": observed_at.isoformat(),
        })
    return rows


async def send_court_slots(client, base_url: str, headers: dict, rows: list,
                           chunk: int = COURT_LOG_CHUNK) -> dict:
    """Post court-log rows through record_court_slots. Returns
    {"extended", "inserted", "error"}; error is None when every chunk saved."""
    out = {"extended": 0, "inserted": 0, "error": None}
    for i in range(0, len(rows), chunk):
        try:
            r = await client.post(f"{base_url}/rest/v1/rpc/record_court_slots",
                                  headers=headers, json={"p_rows": rows[i:i + chunk]})
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"
            return out
        if r.status_code != 200:
            out["error"] = f"HTTP {r.status_code} {r.text[:200]}"
            return out
        body = r.json() or {}
        for k in ("extended", "inserted"):
            out[k] += int(body.get(k) or 0)
    return out


async def record_court_rates(client, base_url: str, headers: dict) -> str:
    """Log the court-hire rates now in availability_cache (record_court_rates).
    Returns a one-line summary. Never raises: a missed rate read is picked up
    by the next run, so it must not fail the job that called it."""
    try:
        r = await client.post(f"{base_url}/rest/v1/rpc/record_court_rates", headers=headers, json={})
    except Exception as e:
        return f"Court rates NOT LOGGED: {type(e).__name__}: {e}"
    if r.status_code != 200:
        return f"Court rates NOT LOGGED: HTTP {r.status_code} {r.text[:200]}"
    body = r.json() or {}
    return (f"Court rates: {int(body.get('inserted') or 0)} new, "
            f"{int(body.get('extended') or 0)} unchanged")
