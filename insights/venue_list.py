"""
venue_list.py -- the venue list Insights reads (27 Nov 2026).

The scrapers read the published list (picklematch.com.au/venues.json, through
venue_registry), but Insights read the repo's own venues.json, which had fallen
behind: Dink & Drive Box Hill and Club Pickled were scraped but never shown.
Now Insights reads the published list too, once per run, and only falls back to
the repo's copy if the site can't be read.
"""
from __future__ import annotations

import json
import os
import urllib.request
from functools import lru_cache
from pathlib import Path

SOURCE_URL = os.environ.get("VENUE_SOURCE_URL", "https://picklematch.com.au/venues.json")
LOCAL = Path(__file__).resolve().parent.parent / "venues.json"


@lru_cache(maxsize=1)
def raw() -> dict:
    """The whole venues file: {"venues": [...], ...}."""
    # tests read the repo's copy (fixed data); so does a run with VENUE_SOURCE_URL=""
    if SOURCE_URL and "PYTEST_CURRENT_TEST" not in os.environ:
        try:
            req = urllib.request.Request(SOURCE_URL, headers={"Accept": "application/json",
                                                              "User-Agent": "picklematch-insights"})
            with urllib.request.urlopen(req, timeout=20) as r:
                d = json.loads(r.read().decode("utf-8"))
            if isinstance(d, dict) and isinstance(d.get("venues"), list) and d["venues"]:
                print(f"Venues: {len(d['venues'])} from the published list")
                return d
        except Exception as e:
            print(f"Venues: published list unreadable ({type(e).__name__}); using the repo's venues.json")
    return json.loads(LOCAL.read_text())


def venues() -> list:
    return raw()["venues"]
