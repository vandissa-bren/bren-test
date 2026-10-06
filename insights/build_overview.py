"""
build_overview.py -- the dataset behind the Melbourne overview prototype.

Same inputs and rules as report_dd.py / report_more.py (imported). Venues are
named; players appear only as counts, and no group under 5 players is shown.
Returns (data, network); build_insights.py publishes them.
"""
import json
from math import radians, sin, cos, atan2, degrees
from pathlib import Path

import numpy as np
import pandas as pd

import report_dd as R
import report_more as M

HERE = Path(__file__).resolve().parent
CBD = (-37.8136, 144.9631)
MIN_GROUP = 5


def registry_all():
    d = json.loads((HERE.parent / "venues.json").read_text())["venues"]
    return d


def area_of(lat, lng):
    """Inner city within 6 km of the CBD; otherwise the compass sector."""
    dl, dn = radians(lat - CBD[0]), radians(lng - CBD[1])
    a = sin(dl / 2) ** 2 + cos(radians(CBD[0])) * cos(radians(lat)) * sin(dn / 2) ** 2
    km = 2 * 6371 * np.arcsin(np.sqrt(a))
    if km <= 6:
        return "Inner city", km
    y = sin(dn) * cos(radians(lat))
    x = cos(radians(CBD[0])) * sin(radians(lat)) - sin(radians(CBD[0])) * cos(radians(lat)) * cos(dn)
    b = (degrees(atan2(y, x)) + 360) % 360
    if b >= 315 or b < 45:
        return "North", km
    if b < 135:
        return "East", km
    if b < 225:
        return "South & bayside", km
    return "West", km


def clean_name(n):
    return n.replace(" | ", " ").replace("Dink & Drive Pickleball Club", "Dink & Drive")


def r1(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 1)


def main():
    s, r, c = R.load()
    R.NOW = max(s["last_obs"].max(), r["last_seen"].max())
    s = R.attach_titles(s, c)
    fin = R.finished(s)
    m = R.rosters(s, r)
    rated = m[m["rating"].notna()]
    five = rated.assign(f=rated["rating"] == 5.0).groupby("fid")["f"].mean()
    m.loc[m["fid"].isin(set(five[five > 0.5].index)) & (m["rating"] == 5.0), "rating"] = np.nan
    ok = m[~m["cancelled"] & m["fid"].notna()]

    start = fin["starts_at"].min()
    end = fin["starts_at"].max()
    days = (end - start).days + 1
    weeks = days / 7

    # ── venues (registry: all 43, tracked or not) ──
    reg = registry_all()
    venues = []
    for v in reg:
        if not v.get("lat"):
            continue
        area, km = area_of(v["lat"], v["lng"])
        fid = int(v["id"]) if str(v.get("id", "")).isdigit() else None
        venues.append({"id": fid if fid else v.get("id"), "name": clean_name(v["name"]), "area": area,
                       "kmCbd": round(km, 1), "lat": v["lat"], "lng": v["lng"],
                       "courts": v.get("courtCount"), "suburb": v.get("city"),
                       "pbp": v.get("platform") == "playbypoint", "status": v.get("status")})
    vmap = {v["id"]: v for v in venues if isinstance(v["id"], int)}

    # ── session rows ──
    f = fin[fin["fid"].notna()].copy()
    f["fid"] = f["fid"].astype(int)
    f = f[f["fid"].isin(vmap)]
    f["fgroup"] = f["fmt"].map(M.FORMAT_GROUP).fillna("Unknown")
    f["lvl"] = f["level_class"].fillna("no level stated")
    loc = f["starts_at"].dt.tz_convert(R.MEL)
    f["date"] = loc.dt.strftime("%Y-%m-%d")
    f["slot"] = np.where(f["weekend"], "Weekend", "Weekday " + f["daypart"].map(
        {"early": "before 9am", "day": "9am–5pm", "evening": "after 5pm"}))
    # watched from 7+ days out: fill at 7/3/1 days
    w7 = (f["first_obs"] <= f["starts_at"] - pd.Timedelta(days=7)) & \
         f[["spots_7d", "spots_3d", "spots_1d"]].le(f["capacity"], axis=0).all(axis=1)
    sess = []
    for i, x in enumerate(f.itertuples()):
        sess.append({
            "i": i, "v": int(x.fid), "date": x.date, "dow": int(x.dow), "hm": x.hm, "hour": int(x.hour),
            "slot": x.slot, "fmt": x.fgroup, "lvl": x.lvl, "label": x.label if isinstance(x.label, str) else None,
            "title": x.title if isinstance(x.title, str) else None,
            "cap": int(x.capacity), "booked": int(x.booked), "fill": round(100 * x.fill, 1),
            "so": bool(x.sold_out), "soH": r1(x.so_h),
            "price": r1(x.price_n), "pph": r1(x.price_hr),
            "f7": round(100 * (1 - x.spots_7d / x.capacity), 1) if w7.loc[x.Index] else None,
            "f3": round(100 * (1 - x.spots_3d / x.capacity), 1) if w7.loc[x.Index] else None,
            "f1": round(100 * (1 - x.spots_1d / x.capacity), 1) if w7.loc[x.Index] else None,
        })

    # ── per-venue player facts ──
    okv = ok[ok["fid"].isin(vmap)].copy()
    okv["fid"] = okv["fid"].astype(int)
    per = okv.groupby(["pbp_user_id", "fid"]).size().reset_index(name="n")
    tot = per.groupby("pbp_user_id")["n"].sum()
    per["share"] = per["n"] / per["pbp_user_id"].map(tot)
    home = per.sort_values(["pbp_user_id", "n", "fid"], ascending=[True, False, True]).drop_duplicates("pbp_user_id").set_index("pbp_user_id")["fid"]
    canc = m[m["fid"].isin(vmap)].copy()
    canc["fid"] = canc["fid"].astype(int)

    # lead time and cancellations: the timing preview's own rules (same as the report)
    import timing_preview as TP
    raw = R.load()[1]
    tl = TP.build(s, raw, R.NOW)

    # publish horizon: sessions first seen 3+ days after capture began (so first sighting ~ listing)
    cap0 = s["first_obs"].min()
    pub = s[s["fid"].notna() & (s["first_obs"] > cap0 + pd.Timedelta(days=3))].copy()
    pub["ahead"] = (pub["starts_at"] - pub["first_obs"]).dt.total_seconds() / 86400

    for fid, v in vmap.items():
        g = f[f["fid"] == fid]
        p = per[per["fid"] == fid]
        v["tracked"] = len(g) > 0
        if not len(g):
            continue
        logged = s[(s["fid"] == fid) & (s["starts_at"] >= start) & (s["starts_at"] <= end)]
        lv = okv[okv["fid"] == fid]
        cc = canc[canc["fid"] == fid]
        rr = lv["rating"].dropna()
        tlv = tl[tl["fid"] == fid]
        lp = TP.lead_profile(tlv)
        pb = pub[pub["fid"] == fid]["ahead"]
        homes_here = home[home == fid].index
        away = [km for km in (R.km(fid, h) for h in home[home.index.isin(p["pbp_user_id"])]) if not np.isnan(km)]
        v.update({
            "sessions": int(len(g)), "perWeek": round(len(logged) / weeks, 1),
            "offered": int(g["capacity"].sum()), "taken": int(g["booked"].sum()),
            "fill": round(100 * g["fill"].mean(), 1), "soldOut": round(100 * g["sold_out"].mean(), 1),
            "soH": r1(g["so_h"].median()) if g["so_h"].notna().any() else None,
            "empty": int(g["spots_at_start"].sum()),
            "pph": r1(g["price_hr"].median()) if g["price_hr"].notna().any() else None,
            "spend": int((g["price_n"] * g["booked"]).sum()) if g["price_n"].notna().any() else None,
            "players": int(p["pbp_user_id"].nunique()),
            "repeat": round(100 * (p["n"] >= 2).mean(), 1) if len(p) else None,
            "loyal": round(100 * (p["share"] > 0.8).mean(), 1) if len(p) else None,
            "homeHere": int(len(homes_here)),
            "fromAfar": round(100 * np.mean([km > 5 for km in away]), 1) if away else None,
            "cancel": round(100 * tlv["cancelled"].mean(), 1) if len(tlv) else None,
            "lead": (lp["median_days"] if isinstance(lp["median_days"], (int, float)) else 8.0) if lp["n"] >= 20 else None,
            "leadN": int(lp["n"]),
            "lastMinute": lp["last_minute_share"] if lp["n"] >= 20 else None,
            "listed": r1(pb.median()) if len(pb) >= 5 else None,
            "rated": int(len(rr)), "avgRating": round(float(rr.mean()), 2) if len(rr) >= 10 else None,
            "titled": round(100 * g["title"].notna().mean(), 1),
        })

    # ── network: players shared between venue pairs ──
    vs = okv.groupby("pbp_user_id")["fid"].apply(lambda x: sorted(set(x)))
    pairs = {}
    for lst in vs:
        for a_i in range(len(lst)):
            for b_i in range(a_i + 1, len(lst)):
                k = (lst[a_i], lst[b_i])
                pairs[k] = pairs.get(k, 0) + 1
    links = [{"a": a, "b": b, "n": n} for (a, b), n in pairs.items() if n >= MIN_GROUP]

    # ── city players ──
    pl = okv.groupby("pbp_user_id").agg(n=("session_key", "size"), venues=("fid", "nunique"), rating=("rating", "median"))
    city_players = {
        "total": int(len(pl)),
        "freq": {k: int(v) for k, v in {"1 session": (pl.n == 1).sum(), "2 sessions": (pl.n == 2).sum(),
                                         "3–4 sessions": pl.n.between(3, 4).sum(), "5+ sessions": (pl.n >= 5).sum()}.items()},
        "venues": {k: int(v) for k, v in {"1 venue": (pl.venues == 1).sum(), "2 venues": (pl.venues == 2).sum(),
                                           "3+ venues": (pl.venues >= 3).sum()}.items()},
        "level": {k: int(v) for k, v in pl["rating"].map(M.rating_class).value_counts().items()},
    }

    # ── reuse level ranges and travel groups ──
    more = json.loads((R.WORK / "report_more.json").read_text())
    ranges = [dict(x, venue=clean_name(R.name(x["fid"]))) for x in more["level_ranges"]]
    for x in ranges:
        x["p25"], x["p75"] = round(x["p25"], 2), round(x["p75"], 2)
    travel = [x for x in more["travel"]["rows"] if x["dim"] != "dd"]

    city = {
        "from": str(start.tz_convert(R.MEL).date()), "to": str(end.tz_convert(R.MEL).date()), "days": days,
        "venuesTracked": int(sum(1 for v in vmap.values() if v.get("tracked"))),
        "venuesKnown": len(venues), "sessions": len(f), "offered": int(f["capacity"].sum()),
        "taken": int(f["booked"].sum()), "fill": round(100 * f["fill"].mean(), 1),
        "soldOut": round(100 * f["sold_out"].mean(), 1), "empty": int(f["spots_at_start"].sum()),
        "spend": int((f["price_n"] * f["booked"]).sum()), "pph": r1(f["price_hr"].median()),
        "players": int(len(pl)), "bookings": int(len(okv)),
        "titled": round(100 * f["title"].notna().mean(), 1),
        "unplaced": int(fin["fid"].isna().sum()),
    }
    # ── co-play network (players as random codes; see network.py) ──
    import network as NET
    f2 = f.assign(i=np.arange(len(f)), fmt=f["fgroup"])
    net = NET.build(f2, m, ok, vmap, clean_name, lambda fid: vmap[fid]["area"])
    for fid, extra in net["venues"].items():
        vmap[fid].update(extra)
    net = {k: v for k, v in net.items() if k != "venues"}
    data = {"city": city, "venues": venues, "sessions": sess, "links": links, "players": city_players,
            "ranges": ranges, "travel": travel}
    return data, net


if __name__ == "__main__":
    d, _ = main()
    print(json.dumps(d["city"], indent=1))
    print(len(d["sessions"]), "sessions;", len(d["links"]), "links;", len(d["venues"]), "venues")
