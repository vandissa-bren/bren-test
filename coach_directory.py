"""
coach_directory.py -- PlayByPoint's "Book a Pro" list, for coaches' full names.

PlayByPoint names a session's coach as an initial and surname ("S. Rivera").
Its pros list (/api/teachers?facility_id=N, public, read signed out) gives
each pro's full name. This reads that list for every active venue and saves
it to coach_directory (20261107100000_coach_directory.sql), where the Coaches
tab's reader matches "S. Rivera" to the one pro it fits and shows the full
name to signed-in players.

Saved per pro per venue: PlayByPoint's id, name, title, page and photo
address. Packages and prices are not kept. Each run marks who it saw
(seen_at); a pro gone from the list for 21 days stops being used.

The log prints counts only, never a name.

    python coach_directory.py                 (needs SUPABASE_URL, SUPABASE_KEY)
    DRY_RUN=1 python coach_directory.py       (reads, saves nothing)
    COACH_FACILITIES=885,1379 python coach_directory.py

Exits 1 only when no venue could be read at all, or the save failed.
"""
from __future__ import annotations

import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from curl_cffi import requests as cr

sys.path.insert(0, str(Path(__file__).parent))

APP = "https://app.playbypoint.com"
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"
MAX_PAGES = 10


def facilities() -> list[int]:
    only = os.environ.get("COACH_FACILITIES", "")
    if only.strip():
        return [int(x) for x in only.split(",") if x.strip()]
    import venue_registry
    return sorted({v.facility_id for v in venue_registry.active_venues()})


def pbp_session():
    s = cr.Session(impersonate="chrome", timeout=30)
    s.headers.update({"Accept": "application/json, text/plain, */*", "X-Requested-With": "XMLHttpRequest",
                      "Origin": APP, "Referer": f"{APP}/pros"})
    return s


def text_of(v) -> str | None:
    """A plain string field, or a string inside a small object ({url: ...})."""
    if isinstance(v, str):
        return v.strip() or None
    if isinstance(v, dict):
        for k in ("url", "original", "large", "medium", "thumb", "src"):
            if isinstance(v.get(k), str) and v[k].strip():
                return v[k].strip()
    return None


def pages_left(meta, page: int) -> bool:
    if not isinstance(meta, dict):
        return False
    for k in ("total_pages", "pages", "last_page", "page_count"):
        if isinstance(meta.get(k), int):
            return page < meta[k]
    if meta.get("next_page"):
        return True
    return False


def read_venue(s, fid: int) -> tuple[list[dict], str]:
    """The pros listed at a venue, as rows to save; and how the read went."""
    rows, seen = [], set()
    skipped = 0
    for page in range(1, MAX_PAGES + 1):
        try:
            r = s.get(f"{APP}/api/teachers", params={"facility_id": fid, "page": page, "per_page": 100},
                      allow_redirects=False)
        except Exception as e:
            return rows, f"error {type(e).__name__}"
        if r.status_code != 200:
            return rows, f"HTTP {r.status_code}"
        try:
            data = r.json()
        except Exception:
            return rows, "not json" + (" (Cloudflare)" if "Just a moment" in (r.text or "") else "")
        teachers = data.get("teachers") if isinstance(data, dict) else data
        if not isinstance(teachers, list):
            return rows, "no list"
        new = 0
        for t in teachers:
            if not isinstance(t, dict):
                continue
            try:
                tid = int(t.get("id"))
            except (TypeError, ValueError):
                continue
            name = re.sub(r"\s+", " ", str(t.get("name") or "")).strip()
            if not name or tid in seen:
                continue
            # If the list names the venues a pro is at, they must include this
            # one (a guard in case the facility filter is ever ignored).
            ids = {f.get("id") for f in (t.get("facilities") or []) if isinstance(f, dict) and f.get("id") is not None}
            if ids and fid not in {int(i) for i in ids if str(i).isdigit()}:
                skipped += 1
                continue
            seen.add(tid)
            new += 1
            rows.append({"facility_id": fid, "teacher_id": tid, "name": name,
                         "title": text_of(t.get("title")), "url": text_of(t.get("url")),
                         "avatar": text_of(t.get("avatar"))})
        if not new or not pages_left(data.get("meta") if isinstance(data, dict) else None, page):
            break
        time.sleep(0.5)
    return rows, "ok" + (f", {skipped} listed for other venues" if skipped else "")


def save(rows: list[dict]) -> None:
    now = datetime.now(timezone.utc).isoformat()
    body = [{**r, "seen_at": now} for r in rows]
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
               "Content-Type": "application/json",
               "Prefer": "resolution=merge-duplicates,return=minimal"}
    r = httpx.post(f"{SUPABASE_URL}/rest/v1/coach_directory?on_conflict=facility_id,teacher_id",
                   json=body, headers=headers, timeout=30.0)
    if r.status_code not in (200, 201, 204):
        raise RuntimeError(f"Supabase write failed ({r.status_code}): {r.text[:300]}")


def main() -> int:
    if not DRY_RUN and not (SUPABASE_URL and SUPABASE_KEY):
        print("SUPABASE_URL and SUPABASE_KEY are needed (or DRY_RUN=1).")
        return 1
    s = pbp_session()
    fids = facilities()
    print(f"Reading the pros list for {len(fids)} venues{' (dry run)' if DRY_RUN else ''}")
    all_rows, read_ok = [], 0
    for fid in fids:
        rows, how = read_venue(s, fid)
        if how.startswith("ok"):
            read_ok += 1
        print(f"  venue {fid}: {len(rows)} pros ({how})")
        all_rows += rows
        time.sleep(0.5)
    uniq = {(r["facility_id"], r["teacher_id"]): r for r in all_rows}
    print(f"{len(uniq)} pro listings at {read_ok} of {len(fids)} venues; "
          f"{len({r['teacher_id'] for r in uniq.values()})} different pros")
    if not read_ok:
        print("No venue could be read: nothing saved.")
        return 1
    if DRY_RUN or not uniq:
        print("Nothing saved." if uniq else "No pros listed: nothing saved.")
        return 0
    save(list(uniq.values()))
    print("Saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
