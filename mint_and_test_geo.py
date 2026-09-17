#!/usr/bin/env python3
"""
mint_and_test_geo.py — C14, run on the geo droplet (170.64.183.118) as pmjobs.
Mints a PlayByPoint session using the account already in /home/pmjobs/secrets.json
(the announcements account, which PBP accepts here), then TESTS it against a few of
the 18 GitHub court venues before we trust it for the whole job.

Prints the session JSON to stdout ONLY if the test passes; all status to stderr.
Usage (as pmjobs):
    cd /home/pmjobs/picklematch && \
      env -i PATH=/usr/bin:/bin HOME=/home/pmjobs /home/pmjobs/venv/bin/python \
      /home/pmjobs/make_pbp_session_test.py > /tmp/pbp.json
"""
import asyncio, json, os, re, sys
def err(*a): print(*a, file=sys.stderr)

SECRETS = os.environ.get("GEO_SECRETS", "/home/pmjobs/secrets.json")
PBP = "https://app.playbypoint.com"
# a few of the 18 GitHub venues to prove access (The Jar, Runway, Dink & Drive)
TEST_VENUES = [(597, "The Jar"), (1714, "Runway"), (1557, "Dink & Drive")]

async def main():
    s = json.load(open(SECRETS))
    email, password = s.get("pbp_email"), s.get("pbp_password")
    if not (email and password):
        err("ERROR: no pbp_email/pbp_password in secrets.json"); sys.exit(2)
    from curl_cffi.requests import AsyncSession
    async with AsyncSession(impersonate="chrome124") as sess:
        r = await sess.get(f"{PBP}/users/sign_in")
        m = re.search(r'<meta name="csrf-token" content="([^"]+)"', r.text or "")
        if not m:
            err(f"ERROR: login page HTTP {r.status_code} (is THIS host blocked too?)"); sys.exit(1)
        r2 = await sess.post(f"{PBP}/users/sign_in",
            json={"user": {"email": email, "password": password, "remember_me": "1"}},
            headers={"Accept":"application/json","Content-Type":"application/json",
                     "X-CSRF-Token":m.group(1),"X-Requested-With":"XMLHttpRequest",
                     "Referer":f"{PBP}/users/sign_in"})
        ok=False
        try: ok=bool(r2.json().get("success"))
        except: pass
        jar={k:v for k,v in sess.cookies.items()}
        if not (ok and jar.get("_paybycourt_session")):
            err(f"ERROR: login not accepted (HTTP {r2.status_code})"); sys.exit(1)
        err("login: accepted (announcements account)")
    # test the session against the GitHub venues via the real client
    sys.path.insert(0, "/home/pmjobs/picklematch")
    from extract_thejar import PlayByPointAPI
    uid = 0
    async with PlayByPointAPI(cookies=jar, club_slug="thejar") as api:
        try:
            u = await api.whoami()
            if u: uid = int(u)
        except Exception: pass
        passed, failed = [], []
        for fid, name in TEST_VENUES:
            try:
                ct = await api.court_types(fid, kind=None)
                passed.append(name)
                err(f"  test {name} ({fid}): OK ({len(ct or [])} surfaces)")
            except Exception as e:
                failed.append(name)
                err(f"  test {name} ({fid}): FAILED {type(e).__name__} {str(e)[:50]}")
            await asyncio.sleep(0.5)
    if failed:
        err(f"RESULT: announcements account CANNOT read {failed} — do NOT use for the GitHub job (fall back to Option A).")
        sys.exit(3)
    err(f"RESULT: announcements account reads all tested venues {passed}. Safe for the GitHub court job.")
    print(json.dumps({"cookies": jar, "user_id": uid, "email": email}))
    err("The JSON line above is your PBP_COOKIES_JSON.")

asyncio.run(main())
