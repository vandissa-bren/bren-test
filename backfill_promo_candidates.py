#!/usr/bin/env python3
"""
backfill_promo_candidates.py — add promo CANDIDATES for announcements already stored
(the Gmail job does this for new ones from now on). F134.

    python3 backfill_promo_candidates.py              # DRY RUN, last 60 days: lists what it would add
    python3 backfill_promo_candidates.py --days 90    # a different window
    python3 backfill_promo_candidates.py --write      # add them (one per announcement; re-running adds nothing)

Candidates are invisible to the app until confirmed:  select * from promo_review;
Reads SUPABASE_URL / SUPABASE_SERVICE_KEY from /app/.env. Put it in /app next to promo_detect.py.
"""
import os, sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv("/app/.env")
except Exception:
    pass

import httpx
from promo_detect import detect, store_candidate

URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")


def main():
    a = sys.argv[1:]
    write = "--write" in a
    days = int(a[a.index("--days") + 1]) if "--days" in a else 60
    for x in a:
        if x.startswith("--") and x not in ("--write", "--days"):
            sys.exit(f"STOP: unknown option {x}. Known: --days N, --write")
    if not URL or not KEY:
        sys.exit("STOP: SUPABASE_URL / SUPABASE_SERVICE_KEY not found in /app/.env")
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    hdr = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
    with httpx.Client(timeout=30) as c:
        r = c.get(f"{URL}/rest/v1/announcements", headers=hdr, params={
            "select": "id,facility_id,facility_name,title,body_text,fetched_at",
            "fetched_at": f"gte.{since}", "order": "fetched_at.desc"})
        if r.status_code != 200:
            sys.exit(f"STOP: could not read announcements (HTTP {r.status_code})")
        rows = r.json()
        have = c.get(f"{URL}/rest/v1/promos", headers=hdr,
                     params={"select": "announcement_id", "announcement_id": "not.is.null"})
        existing = {x["announcement_id"] for x in (have.json() if have.status_code == 200 else [])}

        print(f"== {'WRITE' if write else 'DRY RUN (nothing is added)'} · {len(rows)} announcements in the last {days} days")
        n = 0
        for x in rows:
            d = detect(x.get("title") or "", x.get("body_text") or "")
            if not d or not x.get("facility_id"):
                continue
            n += 1
            tag = "already a candidate" if x["id"] in existing else ""
            line = (f"  {x['fetched_at'][:10]}  {(x.get('facility_name') or '')[:22]:<22} {d['kind']:<14} "
                    f"{(x.get('title') or '')[:50]:<50} code={d['code'] or '-'}  {tag}")
            if write and not tag:
                line += "  → " + store_candidate(URL, KEY, x["id"], x["facility_id"], x.get("facility_name") or "",
                                                 x.get("title") or "", x.get("body_text") or "",
                                                 x["fetched_at"], client=c)
            print(line)
        print(f"\n{n} look like offers.")
        if not write:
            print("DRY RUN — nothing added. Re-run with --write, then: select * from promo_review;")


if __name__ == "__main__":
    main()
