"""guide.py (28 Nov 2026): the new venue guide's figures on a small market."""
import os, sys
from datetime import date
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import guide as GD

SITE = {"id": 9, "name": "New Club", "lat": -37.7329, "lng": 144.8568, "courts": 5}
A = {"id": 1, "name": "Runway", "lat": -37.700, "lng": 144.880, "courts": 8}      # ~4 km
B = {"id": 2, "name": "Real Dill", "lat": -37.778, "lng": 144.780, "courts": 4}   # ~8 km
FAR = {"id": 3, "name": "Far Away", "lat": -37.80, "lng": 145.30, "courts": 6}    # ~40 km


def _market():
    rows, k = [], 0
    for fid, title, hm, fill, fg, lvl in [(1, "Advanced social", "18:00", 0.95, "Social & open play", "advanced"),
                                          (1, "Learn to play", "10:00", 0.15, "Learn to play", "beginner"),
                                          (2, "Intermediate social", "18:00", 0.92, "Social & open play", "intermediate"),
                                          (3, "Far social", "18:00", 0.99, "Social & open play", "intermediate")]:
        for w in range(10):
            d = pd.Timestamp("2026-09-22") + pd.Timedelta(weeks=w)
            k += 1
            rows.append({"session_key": f"s{k}", "fid": fid, "title": title, "lvl": lvl, "fgroup": fg,
                         "starts_at": pd.Timestamp(f"{d.date()} {hm}", tz="Australia/Melbourne").tz_convert("UTC"),
                         "capacity": 16, "fill": fill, "sold_out": fill >= 0.95, "so_h": 48.0, "price_hr": 12.5,
                         "dur_h": 2.0, "courts": 4, "hire_rate": 45.0, "waitlist_max": 2 if fill > 0.9 else 0})
    f = pd.DataFrame(rows)
    okv = []
    for p in range(40):
        venue = 1 if p < 20 else 2
        for s in f[f["fid"] == venue]["session_key"].head(3 + p % 3):
            okv.append({"pbp_user_id": p, "session_key": s, "fid": venue, "rating": 3.6 if p % 2 else 3.0})
        if p < 6:   # Real Dill players who also play at Runway
            pass
    for p in range(20, 27):
        for s in f[f["fid"] == 1]["session_key"].head(2):
            okv.append({"pbp_user_id": p, "session_key": s, "fid": 1, "rating": 3.0})
    okv = pd.DataFrame(okv)
    firsts = okv.drop_duplicates(["pbp_user_id", "fid"]).assign(first_date="2026-11-10", first_key=lambda d: d["session_key"])
    return f, okv, firsts


def test_guide_for_a_new_venue():
    f, okv, firsts = _market()
    vmap = {9: SITE, 1: A, 2: B, 3: FAR}
    g = GD.build(f, okv, firsts, vmap, date(2026, 11, 30), {"venues": {}})["venues"]["9"]
    names = [c["name"] for c in g["competitors"]]
    assert names == ["Runway", "Real Dill"]                  # nearest first; the far venue is out of range
    assert 5 <= g["range"] <= 15
    assert g["competitors"][0]["status"] in ("Full", "Busy", "Room")
    assert g["players"]["home"] == 40
    assert any(t["from"] == "Real Dill" and t["to"] == "Runway" and t["players"] == 7 for t in g["players"]["travel"])
    full = g["full"]
    assert [r["dow"] for r in full] == sorted(r["dow"] for r in full)     # Monday to Sunday
    assert all(r["fill"] >= 90 for r in full) and not any(r["venue"] == "Far Away" for r in full)
    w = g["week"]
    assert w["courts"] == 5 and w["sessions"]
    s = w["sessions"][0]
    assert s["courts"] <= 4 and s["price"] == 25 and s["hours"] == 2.0      # weeknight evening keeps a court for hire


def test_a_venue_with_nothing_nearby_gets_no_guide():
    f, okv, firsts = _market()
    vmap = {3: FAR, 1: A}
    out = GD.build(f[f["fid"] == 1], okv, firsts, vmap, date(2026, 11, 30), {"venues": {}})["venues"]
    assert "3" not in out
