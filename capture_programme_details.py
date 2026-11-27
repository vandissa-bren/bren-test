#!/usr/bin/env python3
"""
capture_programme_details.py -- 27 Nov 2026

Once a day, what each PlayByPoint programme says about itself, so Insights can
use more than the title:

  * from the public catalogue (/api/public/clinics): the programme's category
    (Event, Clinic, ...), its rating range (ntrp_str) and rating provider;
  * from the programme's page: its description (div.program-description), as
    plain text: level ranges ("DUPR 3.0-3.5"), the coach or host, the format,
    "10-week programme", "bring a partner".

Written to programme_details by record_programme_details (migration
20270223100000); a changed description keeps the old one in
programme_description_history. Insights reads them in the nightly build.

Venue facts only: never a player, a roster, a cookie or a token. Logs print
counts.

    python capture_programme_details.py            # every active PBP venue
    PROGRAMME_VENUES=1841,1919 python ...          # just these
"""
from __future__ import annotations

import asyncio
import html as H
import json
import os
import re
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
ONLY = {int(x) for x in os.environ.get("PROGRAMME_VENUES", "").split(",") if x.strip().isdigit()}
MAX_PAGES = int(os.environ.get("PROGRAMME_MAX_PAGES", "600"))
DESC = re.compile(r'<div class="program-description">(.*?)</div>\s*(?:<div class="ui divider|$)', re.S)


def text_of(fragment: str) -> str:
    """HTML to plain text: paragraphs and line breaks kept as new lines."""
    t = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</h\d>", "\n", fragment)
    t = re.sub(r"(?i)<li[^>]*>", "- ", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = H.unescape(t).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", x).strip() for x in t.splitlines()]
    return "\n".join(x for x in lines if x)[:4000]


def description_from(page: str | None) -> str | None:
    if not page:
        return None
    m = DESC.search(page)
    if not m:
        return None
    return text_of(m.group(1)) or ""


def slug_of(url: str | None) -> str | None:
    if not url or "/programs/" not in url:
        return None
    return url.split("/programs/")[-1].split("?")[0].strip("/") or None


async def main() -> int:
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("SUPABASE_URL / SUPABASE_KEY missing")
        return 1
    import venue_registry as VR
    from extract_thejar import PlayByPointAPI
    raw = os.environ.get("PBP_COOKIES_JSON", "")
    try:
        d = json.loads(raw) if raw else {}
        cookies, user_id = d.get("cookies", {}) or {}, d.get("user_id", 0) or 0
    except Exception:
        cookies, user_id = {}, 0

    venues = [v for v in VR.active_venues() if v.platform == "playbypoint" and (not ONLY or v.facility_id in ONLY)]
    rows, pages, no_desc, failed = [], 0, 0, 0
    async with PlayByPointAPI(cookies=cookies, club_slug="", proxy=os.environ.get("PROXY_URL") or None) as api:
        api._user_id = user_id
        for v in venues:
            try:
                progs = await api.programs(v.facility_id)
            except Exception as e:
                print(f"  venue {v.facility_id}: catalogue unreadable ({type(e).__name__})")
                failed += 1
                continue
            got = 0
            for p in progs:
                slug = slug_of(p.get("url"))
                if not slug:
                    continue
                desc = None
                if pages < MAX_PAGES:
                    pages += 1
                    try:
                        desc = description_from(await api.program_detail_html(slug))
                    except Exception:
                        desc = None
                if desc is None:
                    no_desc += 1
                rows.append({"slug": slug, "fid": v.facility_id, "clinic_id": p.get("id"), "name": p.get("name"),
                             "category": p.get("category"), "ntrp": p.get("ntrp_str") or None,
                             "provider": p.get("rating_provider"), "description": desc})
                got += 1
            print(f"  venue {v.facility_id}: {got} programmes")

    print(f"Programmes: {len(rows)} across {len(venues)} venues; {pages} pages read; "
          f"{no_desc} without a description; {failed} venues unreadable")
    if not rows:
        return 0
    async with httpx.AsyncClient(timeout=120) as client:
        saved = 0
        for i in range(0, len(rows), 200):
            r = await client.post(f"{SUPABASE_URL}/rest/v1/rpc/record_programme_details",
                                  headers={"apikey": SUPABASE_KEY, "Content-Type": "application/json"},
                                  json={"p_rows": rows[i:i + 200]})
            if r.status_code >= 300:
                print(f"  save failed: HTTP {r.status_code}")
                return 1
            saved += int(r.json() or 0)
        print(f"Saved: {saved} programmes ({sum(1 for x in rows if x['description'])} with a description)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
