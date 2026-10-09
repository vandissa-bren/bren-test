"""
push_to_supabase.py — Scrape PBP venues and push to Supabase cache.
Runs on the DO server every hour via cron.
Requires .pbp_cookies.json (uploaded by refresh_cookies.py from Windows machine).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
from rich.console import Console

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from extract_thejar import PlayByPointAPI, _extract_react_props_from_html


console = Console()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://stwohmddmdwttasbyblt.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
PROXY_URL = os.environ.get("PROXY_URL") or None
DAYS_AHEAD = 14
# Sessions further out than DAYS_AHEAD are on the program pages this job
# already reads. They aren't stored for the app, but their places taken go to
# the fill log, so insights can see how full sessions are weeks ahead
# (publish-versus-book). LOG_FAR_SESSIONS=0 turns it off.
LOG_FAR_SESSIONS = os.environ.get("LOG_FAR_SESSIONS", "1") != "0"
# Rosters for far-out competitive sessions (tournament divisions, league
# nights), for Discover's opened division. Within DAYS_AHEAD every session's
# roster is already read (sessions[].roster). Beyond it, only programmes that
# read as competitive, only lessons with someone entered, and only up to
# FAR_ROSTER_DAYS ahead. Stored in the venue's cache row as data.far_rosters,
# with full names like sessions[].roster (the reader shortens them for
# signed-out players). LOG_FAR_ROSTERS=0 turns it off.
LOG_FAR_ROSTERS = os.environ.get("LOG_FAR_ROSTERS", "1") != "0"
FAR_ROSTER_DAYS = int(os.environ.get("FAR_ROSTER_DAYS", "120"))
COMPETITIVE = re.compile(r"tournament|league|ladder|championship|competition|slam|classic|showdown|throwdown|invitational", re.I)

# The venues to scrape: every active Play By Point venue in the one list the
# site publishes (frontend public/venues.json), read through venue_registry.
# To add a venue, add it there -- nothing here needs editing.
import venue_registry as _registry
PBP_SLUG_MAP: dict[int, str] = {v.facility_id: v.slug for v in _registry.active_venues()}
VENUE_NAMES: dict[int, str] = {v.facility_id: v.name for v in _registry.active_venues()}


def _sec_to_hhmm(sec: int) -> str:
    h = int(sec) // 3600
    m = (int(sec) % 3600) // 60
    return f"{h:02d}:{m:02d}"


def _load_cookies() -> tuple[dict, int]:
    """Load PBP cookies from env var or local cache file."""
    raw = os.environ.get("PBP_COOKIES_JSON", "")
    if raw:
        try:
            data = json.loads(raw)
            cookies = data.get("cookies", {})
            if cookies:
                return cookies, data.get("user_id", 0)
        except Exception:
            pass
    for cache_path in [
        Path(__file__).parent / ".pbp_cookies.json",
        Path.home() / ".pbp_cookies.json",
    ]:
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text())
                cookies = data.get("cookies", {})
                if cookies:
                    return cookies, data.get("user_id", 0)
            except Exception:
                pass
    return {}, 0


def _pricing_tiers(records):
    """
    Keep the pricing records PBP publishes, including member tiers.

    Deliberately NOT flattened to price_member/price_non_member. Membership
    is not binary: The Rally publishes $20 for "V2 - Rally Member" and
    $12.50 for "VIP Rally", PicklePlex $20 for Essentials and $18.75 for
    Community+, both sets sharing player_category "member" and separable
    only by allowed_affiliations. A two-field model would silently collapse
    those into one number and show members the wrong tier.

    Only non-hidden records with a usable numeric price are kept. Zero is
    valid; None is not.
    """
    out = []
    for r in (records or []):
        if not isinstance(r, dict) or r.get("hidden"):
            continue
        raw = r.get("price")
        if raw is None or isinstance(raw, bool):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        tier = {"price": value, "player_category": r.get("player_category")}
        # lesson_unit states what the price buys. PBP writes "session" for a
        # single occurrence and something else for a commitment -- Dink &
        # Drive's league is "per_week_per_next_sessions" at $125/$100, which
        # is entry to the whole league rather than the cost of one Tuesday.
        # Only records PBP labels as a session may become a session price.
        #
        # Note lessons==1 does NOT distinguish them: that league is lessons=1
        # too. Across 230 sampled records, 227 said session, 3 were that
        # league, and none were unlabelled.
        for key in ("allowed_affiliations", "lessons", "lesson_unit",
                    "lesson_details", "time_unit",
                    "time_range_start", "time_range_end"):
            if r.get(key) is not None:
                tier[key] = r.get(key)
        out.append(tier)
    return out


async def supabase_upsert(records: list[dict]) -> None:
    if not records:
        return
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Fetch existing records to preserve court_prices, shift_map, court blocks
        row_ids = [r["id"] for r in records if "id" in r]
        existing_by_id = {}
        if row_ids:
            fetch_resp = await client.get(
                f"{SUPABASE_URL}/rest/v1/availability_cache",
                params={"id": f"in.({','.join(row_ids)})", "select": "id,data"},
                headers=headers,
            )
            if fetch_resp.status_code != 200:
                # Not cosmetic: existing_by_id feeds the merge below, which
                # preserves court_prices, shift_map and previously-fetched
                # by_date blocks. Continuing with an empty dict would skip
                # that merge and overwrite live court data with nothing.
                raise RuntimeError(
                    f"Supabase read failed ({fetch_resp.status_code}): "
                    f"{fetch_resp.text[:200]}"
                )
            for row in fetch_resp.json():
                existing_by_id[row["id"]] = row["data"]

        # Merge court_prices, shift_map, by_date into record["data"]
        today_iso = date.today().isoformat()
        for record in records:
            row_id = record.get("id")
            # Not a column: the programs this run could not read, whose
            # stored sessions are carried over rather than dropped.
            keep_slugs = set(record.pop("_keep_slugs", ()) or ())
            keep_rosters = set(record.pop("_keep_rosters", ()) or ())
            existing_data = existing_by_id.get(row_id, {})
            # Far rosters this run couldn't read (programme page unreadable, or
            # the roster read itself failed): keep the stored one until its date.
            if existing_data and "data" in record:
                new_fr = record["data"].setdefault("far_rosters", {})
                for lid, fr in (existing_data.get("far_rosters") or {}).items():
                    if (lid not in new_fr and isinstance(fr, dict) and str(fr.get("date") or "") >= today_iso
                            and (lid in keep_rosters or fr.get("program_slug") in keep_slugs)):
                        new_fr[lid] = fr
            if existing_data and keep_slugs and "data" in record:
                kept = [s for s in (existing_data.get("sessions") or [])
                        if s.get("program_slug") in keep_slugs
                        and str(s.get("date") or "") >= today_iso]
                record["data"]["sessions"] = (record["data"].get("sessions") or []) + kept
            if existing_data and "data" in record:
                inner = record["data"]
                if "court_prices" not in inner and "court_prices" in existing_data:
                    inner["court_prices"] = existing_data["court_prices"]
                if "shift_map" not in inner and "shift_map" in existing_data:
                    inner["shift_map"] = existing_data["shift_map"]
                existing_by_date = existing_data.get("by_date", {})
                new_by_date = inner.get("by_date", {})
                for date_str, existing_blocks in existing_by_date.items():
                    if date_str not in new_by_date or not new_by_date[date_str]:
                        if existing_blocks:
                            new_by_date[date_str] = existing_blocks
                inner["by_date"] = new_by_date

        resp = await client.post(
            f"{SUPABASE_URL}/rest/v1/availability_cache",
            json=records,
            headers=headers,
        )
        if resp.status_code not in (200, 201):
            # Must raise, not just print. This previously returned normally,
            # so the caller's "Pushed N venues" message printed regardless
            # and the workflow exited 0. An RLS rejection therefore went
            # unnoticed for four days while every scheduled scrape discarded
            # its results and reported success.
            raise RuntimeError(
                f"Supabase write failed ({resp.status_code}): {resp.text[:300]}"
            )

async def record_listing_horizon(rows: list[dict]) -> None:
    """How far ahead each program is listed, one row per program per day
    (listing_horizon, 20261016130000). Never fails the run: a missed day is
    a gap in a trend, not lost sessions."""
    if not rows:
        return
    from zoneinfo import ZoneInfo as _ZI
    today = datetime.now(_ZI("Australia/Melbourne")).date().isoformat()
    # One row per venue and program: a program read twice keeps its furthest date.
    best = {}
    for r in rows:
        k = (r["facility_id"], r["program_slug"])
        if k not in best or (r["furthest_date"], r["lessons_ahead"]) > (best[k]["furthest_date"], best[k]["lessons_ahead"]):
            best[k] = r
    body = [{**r, "observed_on": today} for r in best.values()]
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
               "Content-Type": "application/json",
               "Prefer": "resolution=merge-duplicates,return=minimal"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{SUPABASE_URL}/rest/v1/listing_horizon?on_conflict=facility_id,program_slug,observed_on",
                json=body, headers=headers)
        if resp.status_code in (200, 201, 204):
            console.print(f"  Listing horizon: {len(body)} programs recorded")
        else:
            console.print(f"  [yellow]Listing horizon NOT recorded: HTTP {resp.status_code} {resp.text[:200]}[/yellow]")
    except Exception as e:
        console.print(f"  [yellow]Listing horizon NOT recorded: {type(e).__name__}: {e}[/yellow]")


def is_competitive(stub: dict) -> bool:
    """A tournament or league programme, by PlayByPoint's category or its name."""
    return bool(COMPETITIVE.search(f"{stub.get('category') or ''} {stub.get('name') or ''}"))


async def read_roster(api, lesson_id) -> list[dict] | None:
    """One far-out session's entrants, the same shape as sessions[].roster
    (full names; get_division_roster shortens them for signed-out players).
    None when the read failed, so the caller can keep the last good roster
    instead."""
    try:
        rd = await api._get_json(
            "/api/public/clinics/lesson_players",
            params={"lesson_id": lesson_id, "rating_provider": "dupr"},
        )
    except Exception:
        return None
    return [
        {"id": u.get("id"), "name": u.get("name"), "initials": u.get("name_initials"),
         "avatar": u.get("avatar") or "", "rating": u.get("rating")}
        for u in (rd or {}).get("users", []) if isinstance(u, dict)
    ]


CLINIC_PAGE_SIZE = 50   # what PlayByPoint was always asked for
CLINIC_PAGES_MAX = 10   # 500 programmes; no venue is near that


async def fetch_clinic_stubs(api, facility_id: int) -> tuple[list[dict], int, bool]:
    """Every programme a venue lists, page by page.

    Until 9 Oct this asked for ONE page of 50 and never the next, so a venue
    with more than 50 programmes silently lost the rest: their sessions never
    reached the app or the fill log. Eastern Indoor was reading 43.

    Stops when a page comes back short, or when a page adds nothing new (in
    case PlayByPoint ignores `page` and repeats page 1: de-duplicated by id, so
    the result is never worse than the single page it replaces). A failure on
    ANY page raises, so the caller treats the list as unreadable and keeps the
    venue's stored sessions rather than saving a partial list as if it were all.

    Returns (stubs, pages read, maybe_more): maybe_more is True when a full page
    added nothing new, i.e. there may be more that paging couldn't reach.
    """
    stubs: list[dict] = []
    seen: set = set()
    pages, maybe_more = 0, False
    for page in range(1, CLINIC_PAGES_MAX + 1):
        resp = await api._get_json(
            "/api/public/clinics",
            params={"search": "", "facility_id": facility_id,
                    "per_page": CLINIC_PAGE_SIZE, "page": page},
        )
        batch = ((resp or {}).get("clinics") or []) if isinstance(resp, dict) else (resp or [])
        pages += 1
        fresh = []
        for s in batch:
            key = s.get("id") if isinstance(s, dict) else None
            if key is None:
                key = (s.get("url") if isinstance(s, dict) else None) or id(s)
            if key not in seen:
                seen.add(key)
                fresh.append(s)
        stubs.extend(fresh)
        if len(batch) < CLINIC_PAGE_SIZE:
            break
        if not fresh:
            maybe_more = page > 1
            break
    return stubs, pages, maybe_more


def _far_observation(lesson: dict, stub: dict, facility_id: int, program_slug: str, price: str, props: dict) -> dict | None:
    """A fill-log reading for a session beyond the app's window (see
    LOG_FAR_SESSIONS): the same fields the roster job sends, from the program
    page already read. None when the lesson can't be placed."""
    lid, ld = lesson.get("id"), lesson.get("lesson_date")
    cap = lesson.get("capacity") or stub.get("capacity") or 0
    if not lid or not isinstance(ld, str) or not cap:
        return None
    pc = lesson.get("player_count", 0) or 0
    left = max(0, int(cap) - int(pc))
    hs = lesson.get("hour_start", 0)
    he = lesson.get("hour_end", hs + 3600)
    lp = price
    for ip in (lesson.get("individual_prices") or []):
        if ip.get("price") and ip.get("player_category") != "member":
            p = float(ip["price"])
            lp = f"${p:.0f}" if p == int(p) else f"${p:.2f}"
            break
    wl_off = props.get("enableWaitlist") is False or props.get("waitlist") is False
    wl = lesson.get("waitlist_count")
    sl = stub.get("ntrp_str") or ""
    return {
        "session_key": f"pbp-{lid}",
        "spots_left": left,
        "capacity": int(cap),
        "session_date": ld,
        "start_time": _sec_to_hhmm(hs),
        "end_time": _sec_to_hhmm(he),
        "price": lp,
        "status": "Full" if left == 0 else "Available",
        "venue_id": f"pbp-{facility_id}",
        "session_type": stub.get("category") or "Session",
        "title": stub.get("name"),
        "skill_level": sl or None,
        "program_slug": program_slug,
        "price_tiers": _pricing_tiers(lesson.get("individual_prices")) or None,
        "coaches": [str(n).strip() for n in (lesson.get("teacher_names") or []) if n and str(n).strip()] or None,
        "waitlist": None if wl_off or not isinstance(wl, int) or wl < 0 else wl,
    }


async def record_far_sessions(rows: list[dict]) -> None:
    """Send the far-out sessions to the fill log (record_inventory_snapshot,
    which writes only what changed, plus a reading every 6 hours). Never fails
    the run: a missed night is a gap in a trend, not lost sessions."""
    if not rows:
        return
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"}
    written, failed = 0, 0
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            for i in range(0, len(rows), 500):
                resp = await client.post(f"{SUPABASE_URL}/rest/v1/rpc/record_inventory_snapshot", headers=headers,
                                         json={"p_observations": rows[i:i + 500], "p_source": "pbp"})
                if resp.status_code == 200:
                    try:
                        written += int(resp.json() or 0)
                    except Exception:
                        pass
                else:
                    failed += 1
                    console.print(f"  [yellow]Far sessions batch NOT logged: HTTP {resp.status_code} {resp.text[:200]}[/yellow]")
        console.print(f"  Far sessions: {len(rows)} read beyond {DAYS_AHEAD} days, {written} fill-log rows written"
                      + (f", {failed} batch(es) failed" if failed else ""))
    except Exception as e:
        console.print(f"  [yellow]Far sessions NOT logged: {type(e).__name__}: {e}[/yellow]")


async def scrape_pbp_venue(
    cookies: dict,
    user_id: int,
    facility_id: int,
    name: str,
    slug: str,
    dates: list[date],
) -> dict:
    result = {
        "id": facility_id,
        "name": name,
        "slug": slug,
        "platform": "playbypoint",
        "by_date": {d.isoformat(): [] for d in dates},
        "sessions": [],
        # Program-level pricing (prices/packages), keyed by program_slug.
        # Kept OUT of the session objects deliberately: a package price is
        # the cost of the package, not of one occurrence, so copying it onto
        # each session would misrepresent it as a per-session price.
        "program_pricing": {},
        # Rosters of far-out competitive sessions, keyed by lesson id:
        # {"program_slug", "date", "read_at", "roster": [...]} (LOG_FAR_ROSTERS).
        "far_rosters": {},
        # ── private bookkeeping, removed before anything is written ──
        # A scrape that could not read PlayByPoint must not be mistaken for
        # a venue with nothing on. See run_once.
        "_list_failed": False,      # the clinic list itself could not be read
        "_clinics_tried": 0,        # clinics in range that we tried to read
        "_failed_slugs": [],        # program slugs whose page could not be read
        "_listing": [],             # how far ahead each program is listed (see run_once)
        "_far": [],                 # sessions past the window, for the fill log (LOG_FAR_SESSIONS)
        "_roster_failed": [],       # far lesson ids whose roster read failed: keep the stored one
    }

    date_strs = {d.isoformat() for d in dates}
    last_day = max(date_strs) if date_strs else ""
    far_seen: set = set()
    roster_until = (date.today() + timedelta(days=FAR_ROSTER_DAYS)).isoformat()

    try:
        async with PlayByPointAPI(cookies=cookies, club_slug=slug, proxy=PROXY_URL) as api:
            api._user_id = user_id

            # Get clinic list: every page of it (see fetch_clinic_stubs)
            try:
                stubs, pages, maybe_more = await fetch_clinic_stubs(api, facility_id)
            except Exception as e:
                console.print(f"    [yellow]clinic list error for {name}: {e}[/yellow]")
                result["_list_failed"] = True
                return result
            if pages > 1 or maybe_more:
                console.print(f"    {name}: {len(stubs)} programmes over {pages} page(s)"
                              + (" -- [yellow]the last page was full and the next added nothing new; "
                                 "PlayByPoint may be ignoring 'page'[/yellow]" if maybe_more else ""))

            for stub in stubs:
                clinic_id = stub.get("id")
                program_url = stub.get("url") or ""
                program_slug = program_url.split("/programs/")[-1] if "/programs/" in program_url else ""
                if not clinic_id or not program_slug:
                    continue

                # Skip clinics with no upcoming sessions in our date range
                week_days = stub.get("future_week_days") or []
                # Use Melbourne timezone to avoid UTC date mismatch
                from zoneinfo import ZoneInfo
                melb = ZoneInfo("Australia/Melbourne")
                from datetime import datetime as _dt
                has_upcoming = any(
                    ((_dt.combine(d, _dt.min.time()).replace(tzinfo=melb).weekday() + 1) % 7) in week_days
                    for d in dates
                ) if week_days else True
                if not has_upcoming:
                    continue

                result["_clinics_tried"] += 1
                try:
                    # Fetch HTML page — only source for lesson dates/times
                    html = await api.program_detail_html(program_slug)
                    if not html:
                        # Non-200 (a 403 from Cloudflare, a redirect to sign
                        # in). Said plainly rather than as a NoneType error.
                        raise RuntimeError("program page could not be read")
                    props = _extract_react_props_from_html(html)
                    if props is None:
                        raise RuntimeError("program page had no session data (signed out or challenged?)")
                    lessons_raw = props.get("sessions") or props.get("clinic_lessons") or []

                    # How far ahead the venue lists this program: every future
                    # lesson date it publishes, not just the ones in our window
                    # (listing_horizon, 20261016130000). Melbourne's date, since
                    # the job runs overnight UTC.
                    from zoneinfo import ZoneInfo as _ZI
                    _today = datetime.now(_ZI("Australia/Melbourne")).date().isoformat()
                    _future = sorted(str(l.get("lesson_date")) for l in lessons_raw
                                     if isinstance(l, dict) and re.match(r"^\d{4}-\d{2}-\d{2}$", str(l.get("lesson_date") or ""))
                                     and str(l.get("lesson_date")) >= _today)
                    if _future:
                        result["_listing"].append({"program_slug": program_slug, "furthest_date": _future[-1],
                                                   "lessons_ahead": len(_future)})

                    # Metadata
                    raw_desc = props.get("description") or ""
                    desc_html = ""
                    if not raw_desc and html:
                        import re as _re
                        start_m = _re.search(r'class="program-description">', html)
                        if start_m:
                            chunk = html[start_m.end():start_m.end() + 8000]
                            # Find the end by locating closing row div
                            end_m = _re.search(r'</div>\s*</div>\s*</div>', chunk, _re.S)
                            raw_desc = chunk[:end_m.start()].strip() if end_m else chunk.strip()
                    desc_html = raw_desc[:5000]
                    desc = re.sub(r"<[^>]+>", " ", raw_desc).strip()
                    desc = desc.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&#8203;", "").replace("\u200b", "")
                    desc = re.sub(r"\s+", " ", desc).strip()[:1000]
                    sl = stub.get("ntrp_str") or ""
                    if not sl:
                        mn = props.get("min_rating")
                        mx = props.get("max_rating")
                        if mn and mx:
                            sl = f"{mn} / {mx}"
                        elif mn:
                            sl = f"{mn}+"
                    # `price` stays exactly as before: the public, non-member
                    # figure. It is what the whole app already renders, and
                    # nothing downstream should have to change.
                    price = ""
                    for pl in (props.get("prices") or props.get("packages") or []):
                        if not pl.get("hidden") and pl.get("price") and pl.get("player_category") != "member":
                            p = float(pl["price"])
                            price = f"${p:.0f}" if p == int(p) else f"${p:.2f}"
                            break

                    # Preserve every non-hidden tier alongside it. The member
                    # records were previously discarded by the filter above,
                    # which is why a member could never be shown their price.
                    #
                    # Stored as the records PBP publishes rather than flattened
                    # into price_member/price_non_member: membership is not
                    # binary. PicklePlex prices Essentials at $20 and
                    # Community+ at $18.75, both player_category "member",
                    # separable only by allowed_affiliations.
                    program_tiers = _pricing_tiers(
                        (props.get("prices") or []) + (props.get("packages") or []))
                    if program_tiers:
                        result["program_pricing"][program_slug] = program_tiers

                    for lesson in lessons_raw:
                        ld = lesson.get("lesson_date")
                        if ld not in date_strs:
                            # beyond the app's window: to the fill log only
                            if LOG_FAR_SESSIONS and isinstance(ld, str) and ld > last_day and lesson.get("id") not in far_seen:
                                obs = _far_observation(lesson, stub, facility_id, program_slug, price, props)
                                if obs:
                                    far_seen.add(lesson.get("id"))
                                    result["_far"].append(obs)
                            # ...and, for competitive programmes, its roster
                            if (LOG_FAR_ROSTERS and isinstance(ld, str) and last_day < ld <= roster_until
                                    and lesson.get("id") and str(lesson.get("id")) not in result["far_rosters"]
                                    and (lesson.get("player_count") or 0) > 0 and is_competitive(stub)):
                                roster = await read_roster(api, lesson.get("id"))
                                if roster is None:
                                    result["_roster_failed"].append(str(lesson.get("id")))
                                else:
                                    result["far_rosters"][str(lesson.get("id"))] = {
                                        "program_slug": program_slug, "date": ld,
                                        "read_at": datetime.utcnow().isoformat(), "roster": roster}
                            continue
                        lid = lesson.get("id")
                        cap = lesson.get("capacity") or stub.get("capacity") or 0
                        pc = lesson.get("player_count", 0)
                        spots = max(0, cap - pc) if cap else None
                        is_full = cap > 0 and spots == 0
                        hs = lesson.get("hour_start", 0)
                        he = lesson.get("hour_end", hs + 3600)

                        lp = price
                        for ip in (lesson.get("individual_prices") or []):
                            if ip.get("price") and ip.get("player_category") != "member":
                                p = float(ip["price"])
                                lp = f"${p:.0f}" if p == int(p) else f"${p:.2f}"
                                break

                        # Per-lesson tiers, including the member records the
                        # loop above skips. These ARE per-occurrence prices,
                        # so unlike package pricing they belong on the session.
                        lesson_tiers = _pricing_tiers(lesson.get("individual_prices"))

                        # Who coaches it, as PlayByPoint lists it (teacher_names).
                        # Kept for the fill log and insights: per-coach fill.
                        coaches = [str(n).strip() for n in (lesson.get("teacher_names") or [])
                                   if n and str(n).strip()]

                        # Roster
                        roster = []
                        if lid:
                            try:
                                rd = await api._get_json(
                                    "/api/public/clinics/lesson_players",
                                    params={"lesson_id": lid, "rating_provider": "dupr"},
                                )
                                roster = [
                                    {
                                        "id": u.get("id"),
                                        "name": u.get("name"),
                                        "initials": u.get("name_initials"),
                                        "avatar": u.get("avatar") or "",
                                        "rating": u.get("rating"),
                                    }
                                    for u in (rd or {}).get("users", [])
                                ]
                            except Exception:
                                pass

                        result["sessions"].append({
                            "title": stub.get("name", "Session"),
                            "type": stub.get("category") or "Session",
                            "date": ld,
                            "start": _sec_to_hhmm(hs),
                            "end": _sec_to_hhmm(he),
                            "price": lp,
                            "spots_left": spots,
                            "status": "Full" if is_full else "Available",
                            "capacity": cap,
                            "description": desc,
                            "description_html": desc_html,
                            "skill_level": sl,
                            "roster": roster,
                            "lesson_id": lid,
                            "program_slug": program_slug,
                            # Empty when the venue publishes no tiers, so the
                            # field is always present and callers need no
                            # special-casing. `price` above is unchanged.
                            "price_tiers": lesson_tiers,
                            "coaches": coaches,
                        })
                except Exception as e:
                    console.print(f"    [yellow]clinic {clinic_id} error for {name}: {e}[/yellow]")
                    result["_failed_slugs"].append(program_slug)

    except Exception as e:
        console.print(f"    [red]error for {name}: {e}[/red]")
        result["_list_failed"] = True

    return result








async def run_once():
    console.print(f"\n[bold]🏓 PickleMatch → Supabase sync[/bold] · {datetime.now().strftime('%H:%M:%S')}\n")

    dates = [date.today() + timedelta(days=i) for i in range(DAYS_AHEAD)]
    cookies, user_id = _load_cookies()

    # ── PlayByPoint ───────────────────────────────────────────────────────────
    if not cookies:
        console.print("[red]No PBP cookies. Run refresh_cookies.py first.[/red]")
    else:
        console.print(f"Scraping {len(PBP_SLUG_MAP)} PBP venues × {DAYS_AHEAD} days…")
        pbp_results = []
        for fid, slug in PBP_SLUG_MAP.items():
            r = await scrape_pbp_venue(cookies, user_id, fid, VENUE_NAMES.get(fid, f"Venue {fid}"), slug, dates)
            pbp_results.append(r)

        # ══ A FAILED READ IS NOT AN EMPTY VENUE ══════════════════════════
        # On 6 Oct every program page came back refused (403s; PBP sign-in
        # or Cloudflare) while the public clinic list still answered. Each
        # venue then held zero sessions, the run printed ✓ and pushed them,
        # and the stored catalogue was replaced with nothing: the site's
        # sessions went, and the roster capture found none to read.
        #
        # Now: a venue whose list failed, or whose every program failed, is
        # NOT SAVED and its stored row is left as it was. A venue where only
        # some programs failed is saved with those programs' stored sessions
        # carried over. And when no venue could be read the run exits
        # non-zero, so the workflow goes red instead of green.
        records, not_saved, partial = [], [], []
        listing_rows = []   # listing_horizon rows, written after the venues
        far_rows = []       # sessions beyond the window, for the fill log
        for r in pbp_results:
            if not isinstance(r, dict):
                continue
            list_failed = r.pop("_list_failed", False)
            tried = r.pop("_clinics_tried", 0)
            failed_slugs = sorted(set(r.pop("_failed_slugs", [])))
            for row in r.pop("_listing", []):
                listing_rows.append({"facility_id": int(r["id"]), **row})
            far_rows.extend(r.pop("_far", []))
            roster_failed = r.pop("_roster_failed", [])
            if list_failed or (tried and len(failed_slugs) >= tried):
                not_saved.append(r["name"])
                console.print(f"  [red]✗ NOT SAVED[/red] {r['name']} · "
                              f"{'clinic list unreadable' if list_failed else f'all {tried} programs unreadable'}"
                              " -- stored sessions kept")
                continue
            if failed_slugs:
                partial.append(r["name"])
            records.append({
                "id": f"pbp-{r['id']}",
                "venue_name": VENUE_NAMES.get(r["id"], r["name"]),
                "platform": "playbypoint",
                "date": date.today().isoformat(),
                "data": r,
                "updated_at": datetime.utcnow().isoformat(),
                "_keep_slugs": failed_slugs,
                "_keep_rosters": roster_failed,
            })
            note = f" · {len(failed_slugs)} of {tried} programs unreadable, their stored sessions kept" if failed_slugs else ""
            if r.get("far_rosters"):
                note += f" · {len(r['far_rosters'])} far-out rosters"
            console.print(f"  [green]✓[/green] {r['name']} · {sum(len(v) for v in r['by_date'].values())} blocks · {len(r['sessions'])} sessions{note}")

        # Any failure propagates: a scrape that cannot persist is a failed
        # run, and the workflow must go red rather than green.
        await supabase_upsert(records)
        if records:
            console.print(f"[green]✓ Pushed {len(records)} PBP venues to Supabase[/green]\n")
        if not_saved:
            console.print(f"[red]{len(not_saved)} venue(s) NOT SAVED: {', '.join(not_saved)}[/red]")
        if partial:
            console.print(f"[yellow]{len(partial)} venue(s) partly read: {', '.join(partial)}[/yellow]")
        await record_listing_horizon(listing_rows)
        await record_far_sessions(far_rows)
        if pbp_results and not records:
            console.print("[red]EVERY VENUE FAILED -- nothing written; exiting non-zero so the run shows as failed[/red]")
            sys.exit(1)


    console.print(f"Sync complete · {datetime.now().strftime('%H:%M:%S')}")


async def watch(interval_minutes: int = 60):
    while True:
        await run_once()
        console.print(f"[dim]Next sync in {interval_minutes} minutes…[/dim]")
        await asyncio.sleep(interval_minutes * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    if args.watch:
        asyncio.run(watch(args.interval))
    else:
        asyncio.run(run_once())
