"""
pbp_probe.py -- why is PlayByPoint refusing the scrapers, and what still works?

Since about midday 7 Nov 2026 the court-block, session, geo and roster jobs get
a 403 with Cloudflare's "you have been blocked" page, while coach_names_probe
(same library, same runners) still gets answers. The two differ in:

  * the browser label: the scrapers' PlayByPointAPI forces a
    "Chrome/124" User-Agent on top of curl_cffi's current-Chrome handshake;
    the coach probe sends curl_cffi's own matching label;
  * the addresses: the scrapers read signed-in /api/facilities/... endpoints;
    the coach probe mostly reads public ones.

This asks each address four ways (label matching / forced Chrome 124, signed
out / signed in) and prints what came back, so we can tell which of the two it
is. Then it lists the /api/ addresses PlayByPoint's own web app calls today
(read from its JavaScript), the way we found them in the first place, so a
moved endpoint shows up.

It prints status codes, Cloudflare's error code and ray id, and addresses
only -- never a cookie or anyone's data -- so its log is safe to share.
About 30 requests, a second apart.

    python pbp_probe.py               (uses PBP_COOKIES_JSON if set)
    PROBE_FACILITY=755 python pbp_probe.py

Exit code 0 always: it's a look, not a check.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import curl_cffi
from curl_cffi import requests as cr

APP = "https://app.playbypoint.com"
FACILITY = int(os.environ.get("PROBE_FACILITY", "597"))   # The Jar | South Melbourne
TZ = os.environ.get("PROBE_TZ", "Australia/Melbourne")
OLD_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")   # what PlayByPointAPI sends
PAUSE = 1.0


def kind(r) -> str:
    t = r.text or ""
    if r.headers.get("cf-mitigated") or "Just a moment" in t or "challenges.cloudflare.com/turnstile" in t:
        return "CF CHALLENGE (Just a moment)"
    if "cf-error-details" in t or "Sorry, you have been blocked" in t or "oldie" in t[:400]:
        code = re.search(r"Error\s*(?:code)?\s*:?\s*(10\d\d)", t)
        return f"CF BLOCK PAGE{' error ' + code.group(1) if code else ''}"
    if "data-react-props" in t:
        return "page with data"
    if t.strip() in ("null", ""):
        return f"empty ({t.strip() or 'no body'})"
    if t.lstrip().startswith(("{", "[")):
        return "json"
    return "html/other"


def session(cookies: dict | None, old_ua: bool):
    s = cr.Session(impersonate="chrome", timeout=30, verify=os.environ.get("PROBE_CA_BUNDLE") or True)
    h = {"Accept": "application/json, text/plain, */*", "X-Requested-With": "XMLHttpRequest",
         "Origin": APP, "Referer": f"{APP}/home"}
    if old_ua:
        h.update({"User-Agent": OLD_UA, "Accept-Language": "en-AU,en;q=0.9"})
    s.headers.update(h)
    for k, v in (cookies or {}).items():
        s.cookies.set(k, v, domain=".playbypoint.com")
    return s


def get(s, url, html=False):
    time.sleep(PAUSE)
    try:
        return s.get(url, headers={"Accept": "text/html"} if html else None, allow_redirects=False)
    except Exception as e:
        return e


def line(r) -> str:
    if isinstance(r, Exception):
        return f"ERROR {type(r).__name__}: {str(r)[:80]}"
    ray = r.headers.get("cf-ray") or "-"
    loc = r.headers.get("location")
    return f"{r.status_code} {kind(r)}  ray {ray}{'  -> ' + loc if loc else ''}"


def main() -> int:
    raw = os.environ.get("PBP_COOKIES_JSON", "")
    cookies = {}
    if raw:
        try:
            cookies = (json.loads(raw) or {}).get("cookies") or {}
        except Exception:
            print("PBP_COOKIES_JSON is set but isn't valid JSON.")
    print(f"curl_cffi {curl_cffi.__version__} · facility {FACILITY} · cookies: "
          f"{str(len(cookies)) + (' incl. cf_clearance' if 'cf_clearance' in cookies else '') if cookies else 'none'}")

    # Probe this machine's address once (country and network only, no data sent).
    try:
        ip = cr.get("https://www.cloudflare.com/cdn-cgi/trace", timeout=10).text
        print("This machine: " + " ".join(l for l in ip.splitlines() if l.startswith(("ip=", "loc=", "http="))))
    except Exception:
        pass

    midnight = datetime.now(ZoneInfo(TZ)).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    ts = int(midnight.timestamp())

    # A program slug for the program-page test (public list, signed out).
    slug = None
    r = get(session(None, False), f"{APP}/api/public/clinics?search=&facility_id={FACILITY}&per_page=5")
    try:
        for c in (r.json() or {}).get("clinics") or []:
            if "/programs/" in (c.get("url") or ""):
                slug = c["url"].split("/programs/")[-1]
                break
    except Exception:
        pass

    targets = [
        ("home page", f"{APP}/", True),
        ("public clinic list", f"{APP}/api/public/clinics?search=&facility_id={FACILITY}&per_page=5", False),
        ("a program page", f"{APP}/programs/{slug}" if slug else None, True),
        ("facilities list", f"{APP}/api/facilities/", False),
        ("court_types", f"{APP}/api/facilities/{FACILITY}/court_types?kind=reservation", False),
        ("available_hours", f"{APP}/api/facilities/{FACILITY}/available_hours?timestamp={ts}"
                            "&surface=pickleball&kind=reservation&courts_for_pros=false", False),
    ]
    variants = [("label matching, signed out", None, False),
                ("Chrome 124 label, signed out", None, True)]
    if cookies:
        variants += [("label matching, signed in", cookies, False),
                     ("Chrome 124 label, signed in", cookies, True)]

    for vname, ck, old in variants:
        s = session(ck, old)
        print(f"\n── {vname} ──")
        for label, url, html in targets:
            if not url:
                print(f"  {label:<20} skipped (no program found)")
                continue
            print(f"  {label:<20} {line(get(s, url, html))}")

    # The addresses PlayByPoint's web app calls today, read from its JavaScript.
    print("\n── /api/ addresses in PlayByPoint's JavaScript today ──")
    s = session(cookies or None, False)
    page = get(s, f"{APP}/book/{os.environ.get('PROBE_BOOK_SLUG', 'nplpickleball')}", html=True)
    if isinstance(page, Exception) or page.status_code != 200:
        page = get(s, f"{APP}/home", html=True)
    if isinstance(page, Exception) or page.status_code != 200:
        print(f"  couldn't load a page to read scripts from: {line(page)}")
        return 0
    found: set[str] = set()
    for src in re.findall(r'<script[^>]+src="([^"]+)"', page.text or "")[:15]:
        js = get(s, src if src.startswith("http") else f"{APP}{src}")
        if isinstance(js, Exception) or js.status_code != 200:
            continue
        for m in re.findall(r'["\'`](/api/[^"\'`\s]{0,120})["\'`]', js.text or ""):
            if re.search(r"facilit|availab|court|clinic|program|reserv|roster|participant|v1|v2|v3", m, re.I):
                found.add(re.sub(r"\$\{[^}]*\}", "{x}", m))
    known = ("court_types", "available_hours", "available_courts", "/price", "public/clinics")
    for p in sorted(found):
        print(f"  {'  ' if any(k in p for k in known) else '* '}{p}")
    print(f"  ({len(found)} addresses; * = not one the scrapers use)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
