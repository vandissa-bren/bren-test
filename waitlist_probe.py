"""
waitlist_probe.py -- where does PlayByPoint say how long a session's waitlist is?

The catalogue's overflow and waitlist metrics need waitlist length over time.
Before the fill log can keep it, we need to know which field carries it. This
reads, for every active PlayByPoint venue, the programme pages the nightly sync
reads, and for full sessions the roster endpoint the roster job reads, and
reports:

  · every field name on a session (lesson) object, and how often it is filled
  · any field whose name mentions "wait" (or "queue"), with its values
  · for a few full sessions: the roster response's top-level fields, and any
    "wait" field in it, as counts

Prints field names, counts and numbers only: never a name, an email or a
cookie. Saves nothing.

    python waitlist_probe.py            (needs PBP_COOKIES_JSON)
"""
from __future__ import annotations

import asyncio
import collections
import json
import os
import re
import sys
from datetime import date

from extract_thejar import PlayByPointAPI, _extract_react_props_from_html

WAIT = re.compile(r"wait|queue", re.I)
FULL_SAMPLES = 6           # full sessions whose roster response is inspected
PROGRAMS_PER_VENUE = 4     # programme pages read per venue


def venues() -> list[tuple[int, str, str]]:
    reg = json.load(open(os.path.join(os.path.dirname(__file__), "venues.json")))["venues"]
    return [(int(v["facilityId"]), v["slug"], v["name"]) for v in reg
            if v.get("platform") == "playbypoint" and v.get("status") == "active" and v.get("slug")]


def scalar(v):
    """Numbers and booleans as they are; anything else only as its type."""
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    if isinstance(v, list):
        return f"list[{len(v)}]"
    if isinstance(v, dict):
        return f"dict{sorted(v.keys())[:8]}"
    return "text"


async def main() -> int:
    raw = os.environ.get("PBP_COOKIES_JSON", "")
    try:
        data = json.loads(raw) or {}
    except Exception:
        data = {}
    cookies, user_id = data.get("cookies") or {}, data.get("user_id") or 0
    if not cookies:
        print("PBP_COOKIES_JSON is missing or has no cookies.")
        return 1

    fields = collections.Counter()      # lesson field -> lessons where it is non-empty
    seen = collections.Counter()        # lesson field -> lessons where present at all
    wait_values = collections.defaultdict(collections.Counter)
    stub_wait = collections.defaultdict(collections.Counter)
    prop_wait = collections.defaultdict(collections.Counter)
    lessons = 0
    full = []                           # (facility, slug, lesson id) of full sessions
    today = date.today().isoformat()

    for fid, slug, name in venues():
        async with PlayByPointAPI(cookies=cookies, club_slug=slug) as api:
            api._user_id = user_id
            try:
                stubs = await api.programs(fid)
            except Exception as e:
                print(f"{name}: programme list failed ({type(e).__name__})")
                continue
            for st in stubs:
                for k, v in st.items():
                    if WAIT.search(k):
                        stub_wait[k][json.dumps(scalar(v))] += 1
            read = 0
            for st in stubs:
                url = st.get("url") or ""
                pslug = url.split("/programs/")[-1] if "/programs/" in url else ""
                if not pslug or read >= PROGRAMS_PER_VENUE:
                    continue
                html = await api.program_detail_html(pslug)
                props = _extract_react_props_from_html(html) if html else None
                if not props:
                    continue
                read += 1
                for k, v in props.items():
                    if WAIT.search(k):
                        prop_wait[k][json.dumps(scalar(v))] += 1
                for ls in (props.get("sessions") or props.get("clinic_lessons") or []):
                    if not isinstance(ls, dict) or str(ls.get("lesson_date") or "") < today:
                        continue
                    lessons += 1
                    for k, v in ls.items():
                        seen[k] += 1
                        if v not in (None, "", [], {}, 0, False):
                            fields[k] += 1
                        if WAIT.search(k):
                            wait_values[k][json.dumps(scalar(v))] += 1
                    cap, pc = ls.get("capacity") or 0, ls.get("player_count") or 0
                    if cap and pc >= cap and ls.get("id") and len(full) < FULL_SAMPLES:
                        full.append((fid, slug, ls.get("id")))
        print(f"{name}: {read} programme pages read")

    print(f"\nSESSION (LESSON) FIELDS, {lessons} upcoming sessions read")
    for k, n in sorted(seen.items(), key=lambda x: -x[1]):
        print(f"  {k:<34} present {n:>4}  filled {fields[k]:>4}")
    print("\nFIELDS MENTIONING WAIT OR QUEUE")
    for label, src in (("on a session", wait_values), ("on the programme page", prop_wait), ("in the programme list", stub_wait)):
        if not src:
            print(f"  {label}: none")
        for k, vals in src.items():
            print(f"  {label}: {k} -> " + ", ".join(f"{v} ×{n}" for v, n in vals.most_common(8)))

    print(f"\nROSTER RESPONSE FOR {len(full)} FULL SESSIONS")
    for fid, slug, lid in full:
        async with PlayByPointAPI(cookies=cookies, club_slug=slug) as api:
            api._user_id = user_id
            try:
                rd = await api._get_json("/api/public/clinics/lesson_players", params={"lesson_id": lid, "rating_provider": "dupr"})
            except Exception as e:
                print(f"  lesson at {fid}: failed ({type(e).__name__})")
                continue
            if not isinstance(rd, dict):
                print(f"  lesson at {fid}: {type(rd).__name__}")
                continue
            top = {k: scalar(v) for k, v in rd.items()}
            waits = {k: scalar(v) for k, v in rd.items() if WAIT.search(k)}
            users = rd.get("users") or []
            ukeys = sorted({k for u in users if isinstance(u, dict) for k in u.keys()})
            uwait = collections.Counter(f"{k}={json.dumps(scalar(u[k]))}" for u in users if isinstance(u, dict) for k in u if WAIT.search(k) or k in ("status", "state"))
            print(f"  lesson at {fid}: top {top}")
            print(f"    wait fields: {waits or 'none'}; user fields: {ukeys}")
            if uwait:
                print(f"    user wait/status values: {dict(uwait)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
