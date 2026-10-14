"""
dupr_id_probe.py -- does PlayByPoint show a player's DUPR ID anywhere we can read?

Phase 0 of host leagues (14 Oct). If PlayByPoint exposes a player's DUPR ID,
a PickleMatch player who has already saved their DUPR ID could be matched to
their PlayByPoint roster spot with no claim step.

It reads one league programme's page (public), takes up to two of its nights,
and asks PlayByPoint for each night's player list with each rating provider.
Then, for up to three of those players, it tries the player-level endpoints.

It prints ONLY:
  - status codes
  - the field names PlayByPoint returns (never names, emails or photos)
  - which fields mention DUPR, and how many players have a value in them
  - a DUPR-looking value masked to its first two characters ("AB****")
So the log is safe to share.

    python dupr_id_probe.py                  (PROBE_SLUG defaults to the PicklePlex league)
    PROBE_SLUG=<program slug> python dupr_id_probe.py

Uses PBP_COOKIES_JSON when set, for endpoints that need a signed-in caller.
Exit code is always 0: this is a fact-finding read, not a check.
"""
from __future__ import annotations

import html
import json
import os
import re

from curl_cffi import requests as cr

APP = "https://app.playbypoint.com"
SLUG = os.environ.get("PROBE_SLUG", "intermediateleague2026")
DUPR_ID = re.compile(r"^[A-Z0-9]{5,10}$")


def session(cookies: dict | None):
    s = cr.Session(impersonate="chrome", timeout=30)
    s.headers.update({"Accept": "application/json, text/plain, */*", "X-Requested-With": "XMLHttpRequest",
                      "Origin": APP, "Referer": f"{APP}/home"})
    for k, v in (cookies or {}).items():
        s.cookies.set(k, v, domain=".playbypoint.com")
    return s


def mask(v) -> str:
    t = str(v)
    return t[:2] + "*" * max(len(t) - 2, 0)


def walk(obj, path=""):
    """Every (path, value) inside a JSON value, keys joined with dots."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for v in obj[:50]:
            yield from walk(v, f"{path}[]")
    else:
        yield path, obj


def dupr_fields(obj) -> list[str]:
    """Paths whose name mentions dupr, with a masked sample when it looks like an ID."""
    out = []
    for p, v in walk(obj):
        if "dupr" in p.lower():
            look = f" = {mask(v)} (looks like a DUPR ID)" if isinstance(v, str) and DUPR_ID.match(v) else \
                   f" = <{type(v).__name__}{', empty' if v in (None, '', []) else ''}>"
            out.append(p + look)
    return sorted(set(out))


def page_props(text: str) -> list[dict]:
    props = []
    for m in re.finditer(r'data-react-props="([^"]*)"', text or ""):
        try:
            props.append(json.loads(html.unescape(m.group(1))))
        except Exception:
            pass
    return props


def get_json(s, label, path, params=None):
    try:
        r = s.get(APP + path, params=params, allow_redirects=False)
    except Exception as e:
        print(f"  {label:<44} ERROR {type(e).__name__}")
        return None
    body = None
    if (r.text or "").lstrip().startswith(("{", "[")):
        try:
            body = r.json()
        except Exception:
            pass
    shape = "json" if body is not None else ("empty" if not (r.text or "").strip() or r.text.strip() == "null" else "other")
    print(f"  {label:<44} {r.status_code}  {shape}")
    return body


def main() -> int:
    raw = os.environ.get("PBP_COOKIES_JSON", "")
    cookies = {}
    if raw:
        try:
            cookies = json.loads(raw).get("cookies") or {}
        except Exception:
            print("PBP_COOKIES_JSON is set but isn't valid JSON; carrying on signed out.")
    s = session(cookies)
    print(f"Programme: {SLUG}   signed in: {'yes' if cookies else 'no'}\n")

    # 1. the programme page: its nights, and any DUPR fields in the page data
    r = s.get(f"{APP}/programs/{SLUG}", allow_redirects=False)
    props = page_props(r.text)
    print(f"Programme page: {r.status_code}, {len(props)} data blocks")
    lessons = []
    for p in props:
        for d in dupr_fields(p):
            print(f"  page field: {d}")
        for key in ("sessions", "clinic_lessons"):
            for l in p.get(key) or []:
                if isinstance(l, dict) and l.get("id"):
                    lessons.append(l)
    lessons.sort(key=lambda l: str(l.get("lesson_date") or ""))
    print(f"  nights listed: {len(lessons)}"
          + (f", first {lessons[0].get('lesson_date')}, last {lessons[-1].get('lesson_date')}" if lessons else ""))
    if lessons:
        print(f"  night fields: {sorted(lessons[0].keys())}")
    if not lessons:
        print("No nights found on the page; nothing more to probe.")
        return 0

    # 2. each night's player list, with each rating provider
    picks = [lessons[0]] + ([lessons[-1]] if len(lessons) > 1 else [])
    user_ids: list = []
    for l in picks:
        print(f"\nNight {l.get('lesson_date')}:")
        for provider in ("dupr", "DUPR", "nrp", None):
            params = {"lesson_id": l["id"]}
            if provider:
                params["rating_provider"] = provider
            body = get_json(s, f"lesson_players rating_provider={provider}", "/api/public/clinics/lesson_players", params)
            users = (body or {}).get("users") if isinstance(body, dict) else None
            if not isinstance(users, list):
                continue
            keys = sorted({k for u in users if isinstance(u, dict) for k in u.keys()})
            rated = sum(1 for u in users if isinstance(u, dict) and u.get("rating") not in (None, "", "NR"))
            print(f"    players {len(users)}, with a rating {rated}, fields {keys}")
            for d in dupr_fields(users):
                print(f"    field: {d}")
            if body and isinstance(body, dict):
                other = sorted(k for k in body.keys() if k != "users")
                if other:
                    print(f"    other top-level fields: {other}")
            user_ids += [u["id"] for u in users if isinstance(u, dict) and u.get("id")]

    # 3. player-level endpoints, for up to three players
    seen = list(dict.fromkeys(user_ids))[:3]
    print(f"\nPlayer endpoints, {len(seen)} players:")
    for n, uid in enumerate(seen, 1):
        for path in (f"/api/users/{uid}", f"/api/public/users/{uid}", f"/api/users/{uid}/ratings",
                     f"/api/public/users/{uid}/ratings", f"/api/users/{uid}/profile"):
            body = get_json(s, f"player {n}: {path.replace(str(uid), '<id>')}", path)
            if body is None:
                continue
            top = sorted(body.keys()) if isinstance(body, dict) else f"list of {len(body)}"
            print(f"      fields: {top}")
            for d in dupr_fields(body):
                print(f"      field: {d}")

    print("\nDone. A line ending '(looks like a DUPR ID)' means PlayByPoint shows it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
