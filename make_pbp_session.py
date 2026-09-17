#!/usr/bin/env python3
"""
make_pbp_session.py — C14, 17 Sep 2026
Log in to PlayByPoint the BROWSERLESS way (curl_cffi, no Playwright) and print a
PBP_COOKIES_JSON value ready to paste into the GitHub Actions secret.

Reads PBP_EMAIL / PBP_PASSWORD from /app/.env (already the new password).
Prints ONLY the JSON line to stdout; all status goes to stderr, so you can:
    python3 make_pbp_session.py > /tmp/pbp.json    # capture just the JSON
or copy the single JSON line it prints.
"""
import asyncio, json, os, re, sys

try:
    from dotenv import load_dotenv
    load_dotenv("/app/.env")
except Exception:
    pass

PBP = os.environ.get("PBP_BASE_OVERRIDE") or "https://app.playbypoint.com"
EMAIL = os.environ.get("PBP_EMAIL")
PASSWORD = os.environ.get("PBP_PASSWORD")

def err(*a): print(*a, file=sys.stderr)

async def main():
    if not EMAIL or not PASSWORD:
        err("ERROR: PBP_EMAIL / PBP_PASSWORD not set in /app/.env"); sys.exit(2)
    from curl_cffi.requests import AsyncSession
    async with AsyncSession(impersonate="chrome124") as s:
        r = await s.get(f"{PBP}/users/sign_in")
        m = re.search(r'<meta name="csrf-token" content="([^"]+)"', r.text or "")
        if not m:
            err(f"ERROR: could not load login page (HTTP {r.status_code})"); sys.exit(1)
        r2 = await s.post(f"{PBP}/users/sign_in",
            json={"user": {"email": EMAIL, "password": PASSWORD, "remember_me": "1"}},
            headers={"Accept": "application/json", "Content-Type": "application/json",
                     "X-CSRF-Token": m.group(1), "X-Requested-With": "XMLHttpRequest",
                     "Referer": f"{PBP}/users/sign_in"})
        ok = False
        try: ok = bool(r2.json().get("success"))
        except Exception: pass
        jar = {k: v for k, v in s.cookies.items()}
        if not (ok and jar.get("_paybycourt_session")):
            err(f"ERROR: login not accepted (HTTP {r2.status_code}). Check the password in /app/.env."); sys.exit(1)
        err("login: accepted")
        # user_id via the browserless client (whoami), same as the geo job.
        user_id = 0
        try:
            sys.path.insert(0, "/app")
            from extract_thejar import PlayByPointAPI
            async with PlayByPointAPI(cookies=jar, club_slug="thejar") as api:
                uid = await api.whoami()
                if uid: user_id = int(uid)
        except Exception as e:
            err(f"note: whoami failed ({type(e).__name__}); user_id=0 (prices may be limited)")
        err(f"user_id: {user_id if user_id else 'not found (0)'}")
        payload = {"cookies": jar, "user_id": user_id, "email": EMAIL}
        print(json.dumps(payload))     # the ONLY thing on stdout
        err(f"cookies captured: {len(jar)} — the JSON line above is your PBP_COOKIES_JSON")

asyncio.run(main())
