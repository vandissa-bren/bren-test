"""
refresh_cookies_server.py — Refresh PBP cookies on the DO server.
Runs via cron every 6 hours.
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, '/app')
try:
    from dotenv import load_dotenv
    load_dotenv('/app/.env')
except Exception:
    pass
from extract_thejar import harvest_cookies_via_playwright

import os

# C13c, 17 Sep 2026: read credentials from the environment (/app/.env),
# not hardcoded here. The literal password used to sit in this file and
# was exposed while the repo was public.
PBP_EMAIL = os.environ.get("PBP_EMAIL")
PBP_PASSWORD = os.environ.get("PBP_PASSWORD")
COOKIE_PATH = os.environ.get("PBP_COOKIE_PATH", "/app/.pbp_cookies.json")


async def main():
    if not PBP_EMAIL or not PBP_PASSWORD:
        print("ERROR: PBP_EMAIL / PBP_PASSWORD not set in the environment (/app/.env)")
        sys.exit(2)
    print("Logging in to PlayByPoint...")
    cookies, user_id = await harvest_cookies_via_playwright(
        email=PBP_EMAIL,
        password=PBP_PASSWORD,
        headless=True,
    )
    if not cookies:
        print("ERROR: No cookies returned")
        sys.exit(1)

    data = {"cookies": cookies, "user_id": user_id, "email": PBP_EMAIL}
    Path(COOKIE_PATH).write_text(json.dumps(data))
    print(f"✓ Saved {len(cookies)} cookies to {COOKIE_PATH}")

    # Notify API server to reload cookies
    import httpx
    try:
        r = httpx.post(
            "http://localhost:8000/api/internal/refresh-cookies",
            json={"pbp_cookies_json": json.dumps(data)},
            timeout=5,
        )
        print(f"✓ API server notified: {r.status_code}")
    except Exception as e:
        print(f"API notify failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
