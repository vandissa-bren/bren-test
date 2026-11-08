"""
fetch_member_prices.py
Fetches member court-hire prices with Esta's PlayByPoint account and stores
them in availability_cache.data.member_court_prices, keyed exactly like
court_prices ("<court id>_<shift>", e.g. "15020_primetime_weekend"), so
record_court_rates() (migration 20261016110000) logs them into
court_rate_states.member_price beside the casual rate.

Which venues: every venue where Esta's PlayByPoint account holds a valid
membership (pbp_credentials.memberships, is_valid), falling back to
Dink & Drive (1557) when no memberships are stored.

Fixed 8 Nov 2026 (0 member rows logged since 6 Oct):
  * a run that priced nothing exited 0 ("No prices fetched -- skipping
    update"), so an expired PlayByPoint session or a Cloudflare block looked
    like a green run. It now exits 1 and says which.
  * weekend keys ("<id>_<shift>_weekend") were split as court id
    "<id>_<shift>" and skipped; weekday keys were priced on whatever day
    tomorrow was, at a guessed hour. Each key is now priced at a real free
    slot of that court and shift from by_date (guessed only when none).
  * the date sent to PlayByPoint was midnight UTC on the runner, not
    midnight Melbourne.
  * prices once cached were never refetched; they now refresh every run.
  * the save downloaded and PATCHed back the venue's whole `data` (sessions,
    rosters, court days), undoing any other job's write in between. It now
    sends only member_court_prices through merge_availability.
  * (review) prices returned with no member affiliation on any court are
    casual rates (membership lapsed or not recognised): the venue is not
    saved and the run fails, instead of logging them as member prices.
  * (review) member_court_prices is replaced with this run's prices, not
    merged into the old ones, so a key that failed to price or a court that
    no longer exists doesn't linger as a "current" member rate.
  * (review) this repo is public: logs show HTTP statuses and error types
    only, never user ids, response bodies or exception text.

Run via GitHub Actions (fetch_member_prices.yml).
Secrets: SUPABASE_URL, SUPABASE_SERVICE_KEY, ESTA_SUPABASE_USER_ID.
"""
import asyncio
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

MEL = ZoneInfo("Australia/Melbourne")
APP = "https://app.playbypoint.com"

# Used only when Esta's stored memberships list is empty or missing.
FALLBACK_VENUES = {1557: "dinkndrivepickleballclub"}   # Dink & Drive

# Last resort when by_date has no free slot for a court/shift.
SHIFT_FALLBACK_SEC = {"lowtime": 8 * 3600, "day": 13 * 3600, "primetime": 18 * 3600}

# Stop a venue early after this many sign-in refusals in a row.
MAX_AUTH_FAILS = 3


# ── pure helpers (tested in test_member_prices.py) ────────────────────────────

def split_key(key: str):
    """'15020_primetime_weekend' -> ('15020', 'primetime_weekend'), the same
    split record_court_rates() uses. None when the key has no shift."""
    court_id, sep, shift = str(key).partition("_")
    if not sep or not court_id or not shift:
        return None
    return court_id, shift


def melbourne_midnight_unix(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=MEL).timestamp())


def hhmm_to_sec(s) -> int | None:
    try:
        h, m = str(s).split(":")[:2]
        return int(h) * 3600 + int(m) * 60
    except (ValueError, AttributeError):
        return None


def member_venues(memberships) -> dict:
    """{facility_id: None} for each valid membership; FALLBACK_VENUES when the
    account has none stored. Slugs are filled from availability_cache later."""
    if isinstance(memberships, str):
        try:
            memberships = json.loads(memberships)
        except ValueError:
            memberships = None
    out = {}
    for m in memberships or []:
        if isinstance(m, dict) and m.get("is_valid") and m.get("facility_id") is not None:
            try:
                out[int(m["facility_id"])] = None
            except (TypeError, ValueError):
                pass
    return out or dict(FALLBACK_VENUES)


def price_windows(court_prices: dict, by_date: dict, first_day: date) -> dict:
    """For each court_prices key, a (date, start_sec) at which to ask
    PlayByPoint for the member price: the earliest free block from first_day
    on with that court and shift (blocks carry the tagged shift, so weekend
    keys land on a weekend day). Otherwise a guessed hour on the next day of
    the right kind (weekday / weekend)."""
    first = {}
    for day in sorted((by_date or {}).keys()):
        try:
            d = date.fromisoformat(str(day))
        except ValueError:
            continue
        if d < first_day:
            continue
        for b in by_date[day] or []:
            if not isinstance(b, dict):
                continue
            sec = b.get("start_sec")
            if not isinstance(sec, (int, float)):
                sec = hhmm_to_sec(b.get("start"))
            k = f"{b.get('court_id')}_{b.get('shift')}"
            if sec is not None and b.get("shift") and k not in first:
                first[k] = (d, int(sec))

    out = {}
    for key in court_prices or {}:
        parts = split_key(key)
        if not parts or not parts[0].isdigit():
            continue
        if key in first:
            out[key] = first[key]
            continue
        base, _, tag = parts[1].partition("_")
        want_weekend = tag == "weekend"
        d = first_day
        while (d.weekday() >= 5) != want_weekend:
            d += timedelta(days=1)
        out[key] = (d, SHIFT_FALLBACK_SEC.get(base, 11 * 3600))
    return out


def parse_member_price(data, pbp_user_id: int):
    """(price, affiliation) from /api/courts/<id>/price. Same rule as the
    casual price in fetch_court_blocks: total.original_reservation_fare, and
    a 0 fare with no shift tariff is no price (None), not a free court."""
    if not isinstance(data, dict):
        return None, None
    users = data.get("prices_per_user") or []
    mine = next((u for u in users if isinstance(u, dict) and u.get("id") == pbp_user_id),
                users[0] if users and isinstance(users[0], dict) else {})
    p = mine.get("price") or {}
    affiliation = p.get("affiliation")
    fare = (data.get("total") or {}).get("original_reservation_fare")
    try:
        price = round(float(fare), 2) if fare is not None else None
    except (TypeError, ValueError):
        price = None
    if price == 0 and not p.get("shift_prices"):
        price = None
    return price, affiliation


def has_member_affiliation(affiliations) -> bool:
    """True if at least one price came back with a member affiliation. All
    None means PlayByPoint priced Esta as a casual player."""
    return any(a not in (None, "", {}, []) for a in affiliations)


def member_prices_to_save(prices: dict, court_prices: dict) -> dict:
    """What replaces member_court_prices: only keys priced this run that are
    still in the venue's court_prices."""
    keep = set(court_prices or {})
    return {k: v for k, v in (prices or {}).items() if k in keep}


def refusal(status: int, body: str, headers) -> str | None:
    """Why PlayByPoint refused, or None if it didn't."""
    if status == 401 or (status == 403 and (body or "").strip() in ("", "null")):
        return "signed out: Esta's PlayByPoint session has expired (reconnect it in the app)"
    if status == 403 and (headers.get("cf-mitigated") or "Just a moment" in (body or "")):
        return "Cloudflare challenge: GitHub's runner is blocked, new cookies won't help"
    return None


# ── I/O ──────────────────────────────────────────────────────────────────────

def _env():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    esta = os.environ.get("ESTA_SUPABASE_USER_ID")
    missing = [n for n, v in (("SUPABASE_URL", url), ("SUPABASE_SERVICE_KEY", key),
                              ("ESTA_SUPABASE_USER_ID", esta)) if not v]
    if missing:
        sys.exit(f"Missing secret(s): {', '.join(missing)} -- set them in GitHub > Settings > Secrets > Actions")
    headers = {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    return url, headers, esta


async def get_credentials(client, base, headers, user_id):
    r = await client.get(f"{base}/rest/v1/pbp_credentials", headers=headers,
                         params={"user_id": f"eq.{user_id}",
                                 "select": "pbp_cookies,pbp_user_id,memberships"})
    if r.status_code != 200:
        sys.exit(f"pbp_credentials read failed: HTTP {r.status_code}")
    rows = r.json()
    if not rows or not rows[0].get("pbp_cookies") or not rows[0].get("pbp_user_id"):
        sys.exit("No PlayByPoint session stored for ESTA_SUPABASE_USER_ID -- "
                 "Esta must connect her PlayByPoint account in the app (Profile)")
    row = rows[0]
    cookies = row["pbp_cookies"]
    if isinstance(cookies, str):
        cookies = json.loads(cookies)
    if isinstance(cookies, dict) and isinstance(cookies.get("cookies"), (list, dict)):
        cookies = cookies["cookies"]
    if isinstance(cookies, list):
        cookies = {c["name"]: c["value"] for c in cookies}
    return cookies, int(row["pbp_user_id"]), row.get("memberships")


async def price_venue(pbp, fid, slug, row, pbp_user_id):
    """Returns (prices {key: price|None}, error or None)."""
    first_day = datetime.now(MEL).date() + timedelta(days=1)
    windows = price_windows(row.get("court_prices") or {}, row.get("by_date") or {}, first_day)
    print(f"  {len(windows)} court/shift rates to price")
    prices, auth_fails, last_refusal, affiliations = {}, 0, None, []
    for key, (d, sec) in sorted(windows.items()):
        court_id = split_key(key)[0]
        try:
            r = await pbp.get(
                f"{APP}/api/courts/{court_id}/price",
                params={"date": melbourne_midnight_unix(d), "admin_book": "false",
                        "hour_start": sec, "hour_end": sec + 3600,
                        "players_reservation_type": "1", "payment_method": "card",
                        "user_ids[]": pbp_user_id, "user_who_is_paying": pbp_user_id,
                        "users_fees[player0][fees][]": "", "kind": "reservation",
                        "coupon_code": ""},
                headers={"Referer": f"{APP}/book/{slug}"} if slug else None,
                timeout=15,
            )
        except Exception as e:
            print(f"    {key}: {type(e).__name__}")
            continue
        why = refusal(r.status_code, r.text, r.headers)
        if why:
            auth_fails += 1
            last_refusal = f"HTTP {r.status_code} -- {why}"
            print(f"    {key}: {last_refusal}")
            if auth_fails >= MAX_AUTH_FAILS and not prices:
                return prices, last_refusal
            continue
        auth_fails = 0
        if r.status_code != 200:
            print(f"    {key}: HTTP {r.status_code}")
            continue
        try:
            price, aff = parse_member_price(r.json(), pbp_user_id)
        except ValueError:
            print(f"    {key}: response wasn't JSON")
            continue
        affiliations.append(aff)
        prices[key] = price
        print(f"    {key} {d} {sec // 3600:02d}:00 -> {price} ({'member' if aff else 'no affiliation'})")
        await asyncio.sleep(0.4)
    if affiliations and not has_member_affiliation(affiliations):
        return prices, ("PlayByPoint priced Esta as a casual player on every court (no member "
                        "affiliation): is her membership here still current? Not saved")
    if not any(v is not None for v in prices.values()):
        return prices, last_refusal or "no member price returned for any court"
    return prices, None


async def main():
    import httpx
    from curl_cffi.requests import AsyncSession
    import court_log

    base, headers, esta = _env()
    failed = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        cookies, pbp_user_id, memberships = await get_credentials(client, base, headers, esta)
        venues = member_venues(memberships)
        print(f"Member venues: {sorted(venues)}")

        ids = ",".join(f"pbp-{fid}" for fid in venues)
        r = await client.get(f"{base}/rest/v1/availability_cache", headers=headers,
                             params={"id": f"in.({ids})",
                                     "select": "id,slug:data->slug,name:data->name,"
                                               "court_prices:data->court_prices,"
                                               "member_court_prices:data->member_court_prices,"
                                               "by_date:data->by_date"})
        if r.status_code != 200:
            sys.exit(f"availability_cache read failed: HTTP {r.status_code}")
        rows = {row["id"]: row for row in r.json()}

        async with AsyncSession(impersonate="chrome", cookies=cookies, headers={
                "Accept": "application/json, text/plain, */*", "Origin": APP,
                "X-Requested-With": "XMLHttpRequest"}) as pbp:
            for fid in sorted(venues):
                row = rows.get(f"pbp-{fid}")
                if not row or not row.get("court_prices"):
                    print(f"── facility {fid}: no court_prices cached yet -- skipped")
                    continue
                slug = row.get("slug") or venues[fid] or FALLBACK_VENUES.get(fid)
                print(f"── {row.get('name') or fid} (facility {fid}) ──")
                prices, err = await price_venue(pbp, fid, slug, row, pbp_user_id)
                if err:
                    failed.append(f"{fid}: {err}")
                    print(f"  NOT SAVED: {err}")
                    continue
                merged = member_prices_to_save(prices, row.get("court_prices"))
                s = await client.post(f"{base}/rest/v1/rpc/merge_availability", headers=headers,
                                      json={"p_id": row["id"], "p_set": {
                                          "member_court_prices": merged,
                                          "member_court_prices_at": datetime.now(timezone.utc).isoformat()}})
                if s.status_code != 200:
                    failed.append(f"{fid}: save HTTP {s.status_code}")
                    print(f"  SAVE FAILED: HTTP {s.status_code}")
                else:
                    print(f"  Saved {sum(v is not None for v in merged.values())} member prices")

        print(await court_log.record_court_rates(client, base, headers))

    if failed:
        sys.exit("Member prices failed for " + "; ".join(failed))
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
