"""
roster_ratings_probe.py -- how many league players can we get a DUPR for?

Phase 2 of host leagues (14 Oct). PlayByPoint's public player list shows no
DUPR at venues that use another rating (PicklePlex uses NRP), but a signed-in
call per player (/api/users/<id>/ratings) returned DUPR entries in phase 0.
This measures it across whole rosters, before phase 2 is built on it:

  - for each programme: nights 1 to 3, how many players, what fields the
    public list gives (to see whether guests or season-vs-night bookings show)
  - for every player on those nights: their ratings entries, counted by
    provider -- how many have DUPR doubles, DUPR singles, a DUPR id, and what
    other fields an entry carries (reliability? last updated?)

It prints ONLY counts, field names and value SHAPES ("d.ddd"), never names,
emails, photos, ids or rating values, so the log is safe to share.

    python roster_ratings_probe.py
    PROBE_SLUGS="slug-one,slug-two" python roster_ratings_probe.py

Needs PBP_COOKIES_JSON (the scraper's signed-in session). Exit code is always
0: this is a fact-finding read, not a check.
"""
from __future__ import annotations

import html
import json
import os
import re
import time
from collections import Counter, defaultdict

from curl_cffi import requests as cr

APP = "https://app.playbypoint.com"
SLUGS = [s.strip() for s in os.environ.get(
    "PROBE_SLUGS", "intermediateleague2026,intermediate-competition").split(",") if s.strip()]
NIGHTS = 3
PAUSE = 0.35          # between per-player calls; ~40 players per programme


def session(cookies: dict | None):
    s = cr.Session(impersonate="chrome", timeout=30)
    s.headers.update({"Accept": "application/json, text/plain, */*", "X-Requested-With": "XMLHttpRequest",
                      "Origin": APP, "Referer": f"{APP}/home"})
    for k, v in (cookies or {}).items():
        s.cookies.set(k, v, domain=".playbypoint.com")
    return s


def shape(v) -> str:
    """A value's form without the value: 3.412 -> 'd.ddd', 'NR' -> 'NR', None -> 'none'."""
    if v is None or v == "":
        return "none"
    t = str(v).strip()
    if t.upper() in ("NR", "N/R", "NOT RATED"):
        return "NR"
    return re.sub(r"\d", "d", re.sub(r"[A-Za-z]", "a", t))[:12]


def page_props(text: str) -> list[dict]:
    props = []
    for m in re.finditer(r'data-react-props="([^"]*)"', text or ""):
        try:
            props.append(json.loads(html.unescape(m.group(1))))
        except Exception:
            pass
    return props


def get_json(s, path, params=None):
    try:
        r = s.get(APP + path, params=params, allow_redirects=False)
    except Exception as e:
        return None, f"ERROR {type(e).__name__}"
    try:
        return (r.json() if (r.text or "").lstrip().startswith(("{", "[")) else None), r.status_code
    except Exception:
        return None, r.status_code


def probe(s, slug: str) -> None:
    print(f"\n=== {slug} ===")
    r = s.get(f"{APP}/programs/{slug}", allow_redirects=False)
    lessons = []
    for p in page_props(r.text):
        for key in ("sessions", "clinic_lessons"):
            lessons += [l for l in (p.get(key) or []) if isinstance(l, dict) and l.get("id") and not l.get("deleted_at")]
    if not lessons:
        print(f"Programme page {r.status_code}: no nights (signed out, or the page changed). Skipping.")
        return

    # The season to look at: the newest batch (a season sold up front already
    # has its players booked on every night, even before night 1).
    by_batch: dict = defaultdict(list)
    for l in lessons:
        by_batch[l.get("batch_id") or "no batch"].append(l)
    batches = sorted(by_batch.values(), key=lambda ls: min(str(l.get("lesson_date")) for l in ls))
    season = batches[-1]
    season.sort(key=lambda l: str(l.get("lesson_date")))
    print(f"Programme page {r.status_code}: {len(lessons)} nights in {len(batches)} batch(es); "
          f"looking at the batch starting {season[0].get('lesson_date')} ({len(season)} nights)")

    list_fields: set = set()
    per_night = []
    players: dict = {}                    # id -> first night index seen
    for i, l in enumerate(season[:NIGHTS], 1):
        body, status = get_json(s, "/api/public/clinics/lesson_players",
                                {"lesson_id": l["id"], "rating_provider": "dupr"})
        users = (body or {}).get("users") if isinstance(body, dict) else None
        if not isinstance(users, list):
            print(f"  night {i} ({l.get('lesson_date')}): player list {status}, no list")
            continue
        for u in users:
            if isinstance(u, dict):
                list_fields |= set(u.keys())
                if u.get("id") is not None:
                    players.setdefault(u["id"], i)
        no_id = sum(1 for u in users if isinstance(u, dict) and u.get("id") is None)
        rated = Counter(shape(u.get("rating")) for u in users if isinstance(u, dict))
        other = sorted(k for k in body.keys() if k != "users") if isinstance(body, dict) else []
        per_night.append(i)
        print(f"  night {i} ({l.get('lesson_date')}): {len(users)} players, {no_id} without an id, "
              f"rating shown on the list {dict(rated)}" + (f", other fields {other}" if other else ""))
    print(f"  fields on the public list: {sorted(list_fields)}")
    joined_late = sum(1 for n in players.values() if n > 1)
    print(f"  different players across those nights: {len(players)} ({joined_late} first seen after night 1)")

    # Every player's ratings entries
    tally = Counter()
    entry_fields: dict = defaultdict(set)
    value_shapes: dict = defaultdict(Counter)
    top_fields: set = set()
    fails = Counter()
    for uid in players:
        body, status = get_json(s, f"/api/users/{uid}/ratings")
        time.sleep(PAUSE)
        if not isinstance(body, dict):
            fails[str(status)] += 1
            continue
        top_fields |= set(body.keys())
        provs = set()
        for e in body.get("ratings") or []:
            if not isinstance(e, dict):
                continue
            prov = str(e.get("provider") or "?").lower()
            provs.add(prov)
            entry_fields[prov] |= set(e.keys())
            for k in ("single", "double"):
                value_shapes[f"{prov}.{k}"][shape(e.get(k))] += 1
            if prov == "dupr":
                tally["dupr entry"] += 1
                tally["dupr doubles set"] += shape(e.get("double")) not in ("none", "NR")
                tally["dupr singles set"] += shape(e.get("single")) not in ("none", "NR")
                tally["dupr singles only (no doubles)"] += (shape(e.get("single")) not in ("none", "NR")
                                                            and shape(e.get("double")) in ("none", "NR"))
                tally["dupr id present"] += any(isinstance(v, str) and re.match(r"^[A-Z0-9]{5,10}$", v)
                                                for k, v in e.items() if k != "provider")
        tally["no ratings at all"] += not provs
        for p in provs:
            tally[f"has {p}"] += 1
    n = len(players)
    print(f"\n  Ratings calls: {n - sum(fails.values())} of {n} answered" + (f", failures {dict(fails)}" if fails else ""))
    for k in sorted(tally):
        print(f"    {k:<34} {tally[k]:>3} of {n}")
    print(f"  top-level fields of the ratings reply: {sorted(top_fields)}")
    for prov, keys in sorted(entry_fields.items()):
        print(f"  {prov} entry fields: {sorted(keys)}")
    for k, c in sorted(value_shapes.items()):
        print(f"  value shapes {k}: {dict(c)}")


def main() -> int:
    raw = os.environ.get("PBP_COOKIES_JSON", "")
    cookies = {}
    if raw:
        try:
            cookies = json.loads(raw).get("cookies") or {}
        except Exception:
            print("PBP_COOKIES_JSON is set but isn't valid JSON.")
    if not cookies:
        print("No signed-in session (PBP_COOKIES_JSON); the ratings calls need one. Stopping.")
        return 0
    s = session(cookies)
    for slug in SLUGS:
        try:
            probe(s, slug)
        except Exception as e:
            print(f"\n=== {slug} === stopped: {type(e).__name__}: {e}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
