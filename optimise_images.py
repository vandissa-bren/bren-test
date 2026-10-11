"""
optimise_images.py -- small WebP copies of the site's heaviest images (11 Oct 2026).

The SEO audit (11 Oct) found the homepage loading about 11 MB of images on a
phone: venue banners as 1.6-3 MB PNGs, venue logos up to 1.2 MB shown at
62-82 px, promo banners at 0.4-0.7 MB. Supabase can't resize on request on
this plan ("FeatureNotEnabled"), so this makes the small copies once and
stores them beside the originals:

  venue banners  venues/banners/<id>-banner.webp   at most 1600 px wide
  venue logos    venues/logos/<id>.webp             at most 192 x 192
  promo banners  banners/<name>.webp                at most 1600 px wide
                 (promos.image_url is pointed at the copy)
  featured       banners/featured-<kind>.webp       1600 / 2000 / 1080 px wide
                 (featured_banner's three columns are pointed at the copies)

Originals are never deleted or overwritten, and a row is only repointed after
its copy has uploaded. A copy that wouldn't be smaller is skipped. Venue
banners and logos are listed in public/venues.json (frontend repo): its
updated copy points at these files and goes live after this has run.

    DRY_RUN=1 python optimise_images.py     (default: reads and reports, saves nothing)
    DRY_RUN=0 python optimise_images.py     (uploads, and repoints promos and featured)

Needs SUPABASE_URL and SUPABASE_KEY (service key). Prints names and sizes only.
"""
from __future__ import annotations

import io
import os
import re
import sys
import time

import httpx
from PIL import Image, ImageOps

SITE = "https://picklematch.com.au"
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
DRY_RUN = os.environ.get("DRY_RUN", "1") != "0"
UA = {"User-Agent": "PickleMatch image optimiser (+https://picklematch.com.au)"}


def headers(extra: dict | None = None) -> dict:
    """apikey for new-style keys; Bearer only for legacy eyJ... keys (as api_server.py)."""
    h = {"apikey": SUPABASE_KEY}
    if SUPABASE_KEY.startswith("eyJ"):
        h["Authorization"] = f"Bearer {SUPABASE_KEY}"
    return {**h, **(extra or {})}


def public_url(bucket: str, path: str) -> str:
    return f"{SUPABASE_URL}/storage/v1/object/public/{bucket}/{path}"


def fetch(client: httpx.Client, url: str) -> bytes | None:
    if url.startswith("/"):
        url = SITE + url  # a file on the site itself
    try:
        r = client.get(url, headers=UA, follow_redirects=True, timeout=60)
        return r.content if r.status_code == 200 and r.content else None
    except (httpx.HTTPError, ValueError):
        return None


def to_webp(data: bytes, max_w: int, max_h: int | None = None, quality: int = 80) -> bytes | None:
    try:
        im = Image.open(io.BytesIO(data))
        if getattr(im, "is_animated", False):
            return None  # an animation would lose its frames
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGBA") if im.mode in ("P", "LA", "RGBA") or "transparency" in im.info else im.convert("RGB")
        im.thumbnail((max_w, max_h or 10_000), Image.LANCZOS)
        out = io.BytesIO()
        im.save(out, "WEBP", quality=quality, method=6)
        return out.getvalue()
    except Exception:
        return None


def upload(client: httpx.Client, bucket: str, path: str, body: bytes) -> bool:
    r = client.post(f"{SUPABASE_URL}/storage/v1/object/{bucket}/{path}", content=body, timeout=60,
                    headers=headers({"Content-Type": "image/webp", "x-upsert": "true", "cache-control": "86400"}))
    return r.status_code in (200, 201)


def kb(n: int) -> str:
    return f"{n / 1024:,.0f} KB"


rows: list[tuple[str, str, int, int, str]] = []   # what, name, before, after, outcome


def make_copy(client, what, name, src, bucket, path, max_w, max_h=None, quality=80) -> str | None:
    """Download src, write bucket/path as WebP. Returns the copy's public address, or None."""
    data = fetch(client, src)
    if not data:
        rows.append((what, name, 0, 0, "could not download")); return None
    webp = to_webp(data, max_w, max_h, quality)
    if not webp:
        rows.append((what, name, len(data), 0, "could not convert")); return None
    if len(webp) >= len(data):
        rows.append((what, name, len(data), len(webp), "kept original (no smaller)")); return None
    if DRY_RUN:
        rows.append((what, name, len(data), len(webp), "dry run")); return public_url(bucket, path)
    if not upload(client, bucket, path, webp):
        rows.append((what, name, len(data), len(webp), "UPLOAD FAILED")); return None
    rows.append((what, name, len(data), len(webp), "uploaded"))
    return public_url(bucket, path)


def patch(client, table: str, where: str, body: dict) -> bool:
    r = client.patch(f"{SUPABASE_URL}/rest/v1/{table}?{where}", json=body, timeout=30,
                     headers=headers({"Content-Type": "application/json", "Prefer": "return=minimal"}))
    return r.status_code in (200, 204)


def main() -> int:
    if not (SUPABASE_URL and SUPABASE_KEY):
        print("SUPABASE_URL and SUPABASE_KEY are needed."); return 1
    print(f"{'DRY RUN: nothing will be saved' if DRY_RUN else 'Saving copies and repointing'}\n")
    with httpx.Client() as client:
        # ── venues: from the published venue list ──
        venues = client.get(f"{SITE}/venues.json", headers=UA, timeout=30).json()["venues"]
        active = [v for v in venues if v.get("facilityId") and v.get("status") != "delisted"]
        for v in active:
            if v.get("bannerUrl") and not v["bannerUrl"].split("?")[0].endswith(".webp"):
                make_copy(client, "venue banner", v["name"], v["bannerUrl"], "venues", f"banners/{v['id']}-banner.webp", 1600)
            if v.get("logoUrl") and not v["logoUrl"].split("?")[0].endswith(".webp"):
                make_copy(client, "venue logo", v["name"], v["logoUrl"], "venues", f"logos/{v['id']}.webp", 192, 192, 85)

        # ── promo banners: repoint each row once its copy is up ──
        r = client.get(f"{SUPABASE_URL}/rest/v1/promos?select=id,image_url&image_url=not.is.null", headers=headers(), timeout=30)
        promos = r.json() if r.status_code == 200 else []
        if r.status_code != 200:
            print(f"promos not read: HTTP {r.status_code}")
        repointed = 0
        for p in promos:
            src = p["image_url"]
            m = re.search(r"/storage/v1/object/public/banners/([^?]+)", src)
            if not m or m.group(1).lower().endswith((".webp", ".gif")):
                continue
            name = re.sub(r"\.[a-z0-9]+$", "", m.group(1), flags=re.I)
            url = make_copy(client, "promo banner", name[:40], src, "banners", f"{name}.webp", 1600)
            if url and not DRY_RUN and patch(client, "promos", f"id=eq.{p['id']}", {"image_url": f"{url}?v={int(time.time())}"}):
                repointed += 1

        # ── the featured banner ──
        r = client.get(f"{SUPABASE_URL}/rest/v1/featured_banner?select=*&id=eq.1", headers=headers(), timeout=30)
        fb = (r.json() or [None])[0] if r.status_code == 200 else None
        featured = 0
        if fb:
            for col, kind, w in (("condensed_image_url", "condensed", 1600), ("expanded_image_url", "expanded", 2000),
                                 ("expanded_mobile_image_url", "expanded-mobile", 1080)):
                src = fb.get(col)
                if not src or src.split("?")[0].endswith(".webp"):
                    continue
                url = make_copy(client, "featured", kind, src, "banners", f"featured-{kind}.webp", w)
                if url and not DRY_RUN and patch(client, "featured_banner", "id=eq.1", {col: f"{url}?v={int(time.time())}"}):
                    featured += 1

    before = sum(r[2] for r in rows if r[4] in ("uploaded", "dry run"))
    after = sum(r[3] for r in rows if r[4] in ("uploaded", "dry run"))
    print(f"{'what':<14} {'name':<42} {'before':>9} {'after':>9}  outcome")
    for what, name, b, a, outcome in rows:
        print(f"{what:<14} {name[:42]:<42} {kb(b):>9} {kb(a):>9}  {outcome}")
    print(f"\n{len([r for r in rows if r[4] in ('uploaded', 'dry run')])} images: {kb(before)} -> {kb(after)}"
          + (f" ({100 - after * 100 // max(before, 1)}% smaller)" if before else ""))
    if not DRY_RUN:
        print(f"promo rows repointed: {repointed} · featured banner columns repointed: {featured}")
        print("Next: publish the updated public/venues.json so venues use their new banners and logos.")
    failed = [r for r in rows if r[4] == "UPLOAD FAILED"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
