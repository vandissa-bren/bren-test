"""
pbp_probe.py -- why is PlayByPoint refusing the scrapers?

Two different things produce a 403 from PlayByPoint, and they need opposite
fixes:

  Cloudflare challenge   Cloudflare decided the caller looks like a bot (by IP
                         or fingerprint). Answers with an HTML "Just a moment"
                         page and a `cf-mitigated: challenge` header. New
                         cookies don't help; running from somewhere else does.

  Signed out             PlayByPoint itself says no: the login cookies in
                         PBP_COOKIES_JSON have expired. Answers `null` with no
                         Cloudflare header. Program pages still load (they're
                         public); signed-in endpoints like court_types don't.

This reads five things and says which it is. It prints status codes and
headers only -- never a cookie -- so its log is safe to share.

    python pbp_probe.py               (uses PBP_COOKIES_JSON if set)

Exit code: 0 when everything answered, 1 when something was refused.
"""
from __future__ import annotations

import json
import os
import sys

from curl_cffi import requests as cr

APP = "https://app.playbypoint.com"
FACILITY = int(os.environ.get("PROBE_FACILITY", "597"))   # The Jar | South Melbourne


def kind(resp) -> str:
    t = resp.text or ""
    if resp.headers.get("cf-mitigated") or "Just a moment" in t or "challenges.cloudflare.com" in t:
        return "CLOUDFLARE CHALLENGE"
    if "data-react-props" in t:
        return "page with session data"
    if t.strip() in ("null", ""):
        return f"empty ({t.strip() or 'no body'})"
    if t.lstrip().startswith(("{", "[")):
        return "json"
    return "html/other"


def session(cookies: dict | None):
    s = cr.Session(impersonate="chrome", timeout=30,
                   verify=os.environ.get("PROBE_CA_BUNDLE") or True)
    s.headers.update({"Accept": "application/json, text/plain, */*", "X-Requested-With": "XMLHttpRequest",
                      "Origin": APP, "Referer": f"{APP}/home"})
    for k, v in (cookies or {}).items():
        s.cookies.set(k, v, domain=".playbypoint.com")
    return s


def get(s, label, url):
    try:
        r = s.get(url, allow_redirects=False)
        k = kind(r)
        loc = r.headers.get("location")
        print(f"  {label:<34} {r.status_code}  {k}{f'  -> {loc}' if loc else ''}")
        return r.status_code, k, r
    except Exception as e:
        print(f"  {label:<34} ERROR {type(e).__name__}: {str(e)[:120]}")
        return None, "error", None


def main() -> int:
    raw = os.environ.get("PBP_COOKIES_JSON", "")
    cookies, user_id = {}, None
    if raw:
        try:
            data = json.loads(raw)
            cookies, user_id = data.get("cookies") or {}, data.get("user_id")
        except Exception:
            print("PBP_COOKIES_JSON is set but isn't valid JSON.")
    print(f"Cookies supplied: {'yes, ' + str(len(cookies)) + ' cookies' if cookies else 'no'}"
          f"{' (includes cf_clearance)' if 'cf_clearance' in cookies else ''}\n")

    anon = session(None)
    print("Signed out:")
    _, clinics_kind, r = get(anon, "public clinic list (JSON)",
                             f"{APP}/api/public/clinics?search=&facility_id={FACILITY}&per_page=5")
    slug = None
    try:
        stubs = (r.json() or {}).get("clinics") or [] if r is not None and clinics_kind == "json" else []
        slug = next(((s.get("url") or "").split("/programs/")[-1] for s in stubs if "/programs/" in (s.get("url") or "")), None)
    except Exception:
        pass
    page_anon = get(anon, "a program page", f"{APP}/programs/{slug}")[1] if slug else "skipped"
    court_anon = get(anon, "court_types", f"{APP}/api/facilities/{FACILITY}/court_types")[1]

    page_auth = court_auth = me_kind = "skipped"
    me_status = court_status = None
    if cookies:
        auth = session(cookies)
        print("\nWith PBP_COOKIES_JSON:")
        if slug:
            page_auth = get(auth, "a program page", f"{APP}/programs/{slug}")[1]
        court_status, court_auth, _ = get(auth, "court_types", f"{APP}/api/facilities/{FACILITY}/court_types")
        me_status, me_kind, rm = get(auth, "who am I (/api/users/current)", f"{APP}/api/users/current")
        if me_status == 200 and rm is not None:
            try:
                d = rm.json() or {}
                same = user_id is None or str(d.get("id")) == str(user_id)
                print(f"  signed in as user {d.get('id')}{'' if same else ' (NOT the user_id in the secret)'}")
            except Exception:
                pass

    print("\nVERDICT")
    challenged = [k for k in (clinics_kind, page_anon, court_anon, page_auth, court_auth, me_kind) if k == "CLOUDFLARE CHALLENGE"]
    if challenged:
        print("  Cloudflare is challenging requests from this machine. New cookies won't fix it;")
        print("  the scrapers need to run from a machine Cloudflare lets through (e.g. the PickleMatch server).")
        return 1
    if cookies and court_status == 200:
        print("  Everything answered, signed in. If the scheduled runs still fail, it was transient.")
        return 0
    if cookies and court_status == 403 and court_auth.startswith("empty"):
        print("  PlayByPoint says these cookies are SIGNED OUT: PBP_COOKIES_JSON has expired.")
        print("  Program pages still load, so sessions can be read, but court blocks need a fresh login.")
        print("  Refresh it with refresh_pbp_cookies.py and paste the result into the secret.")
        return 1
    if not cookies:
        print("  No cookies supplied: signed-out reads only. Set PBP_COOKIES_JSON to test the login.")
        return 0 if page_anon == "page with session data" else 1
    print("  Something else answered; read the lines above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
