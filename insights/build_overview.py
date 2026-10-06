"""
build_overview.py -- the dataset behind the Melbourne overview prototype.

Same inputs and rules as venue_report.py / report_more.py (imported). Venues are
named; players appear only as counts, and no group under 5 players is shown.
Returns (data, network); build_insights.py publishes them.
"""
import json
from math import radians, sin, cos, atan2, degrees
from pathlib import Path

import numpy as np
import pandas as pd

import venue_report as R
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


clean_name = R.short_name


def r1(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 1)


def _verified(x) -> bool:
    """A rating that looks like a real DUPR: two or three decimals that aren't a
    half step. PlayByPoint fills whole and half numbers (2.5, 3.0) when a player
    has no DUPR linked: those are self-rated."""
    try:
        c = round(float(x) * 1000)
    except (TypeError, ValueError):
        return False
    return c % 500 != 0


def level_fit(okv, vmap, cls=None, min_players=MIN_GROUP):
    """What each level a venue runs actually draws: for every venue and every
    level its sessions state (the label as read from the title, e.g. "3.0-3.5",
    "3.25+", "Beginner"), the ratings of the players booked into them.

    One rating per player per group (their median there), so a regular doesn't
    count ten times. Verified = looks like a real DUPR (see _verified); the rest
    are PlayByPoint self-ratings. Groups with fewer than `min_players` rated
    players are left out (the 5-player rule). `cls` maps session_key to the
    session's level class (beginner / intermediate / advanced / all levels), so
    the page can filter these by level like everything else."""
    rows = []
    g = okv[okv["rating"].notna() & okv["label"].notna()].copy()
    if g.empty:
        return rows
    for (fid, label), grp in g.groupby(["fid", "label"]):
        per = grp.groupby("pbp_user_id")["rating"].median()
        if len(per) < min_players:
            continue
        ver = grp.groupby("pbp_user_id")["rating"].apply(lambda s: any(_verified(x) for x in s))
        lo, hi = grp["band_lo"].iloc[0], grp["band_hi"].iloc[0]
        kinds = pd.Series([cls.get(k) for k in grp["session_key"].unique()] if cls else [], dtype=object).dropna()
        has_band = not (isinstance(lo, float) and np.isnan(lo))
        inside = None
        if has_band:
            inside = round(100 * float(((per >= lo - 0.01) & (per <= hi + 0.01)).mean()), 1)
        rows.append({
            "v": int(fid), "label": str(label),
            "lo": round(float(lo), 2) if has_band else None, "hi": round(float(min(hi, 6.0)), 2) if has_band else None,
            "sessions": int(grp["session_key"].nunique()), "players": int(len(per)),
            "verified": int(ver.sum()),
            "p10": round(float(per.quantile(.10)), 2), "p25": round(float(per.quantile(.25)), 2),
            "median": round(float(per.median()), 2),
            "p75": round(float(per.quantile(.75)), 2), "p90": round(float(per.quantile(.90)), 2),
            "inside": inside,
            "cls": str(kinds.mode().iloc[0]) if len(kinds) else None,
        })
    return sorted(rows, key=lambda x: (x["v"], x["median"]))


def _spread(per: pd.Series) -> dict:
    return {"p10": round(float(per.quantile(.10)), 2), "p25": round(float(per.quantile(.25)), 2),
            "median": round(float(per.median()), 2),
            "p75": round(float(per.quantile(.75)), 2), "p90": round(float(per.quantile(.90)), 2)}


def level_by_class(okv, cls, min_players=MIN_GROUP):
    """What "Beginner", "Intermediate" and "Advanced" mean at each venue: for every
    venue and every level class its sessions state (read from the title, so a
    session called "Beginner Social" counts as well as one called "2.0-2.5"),
    the ratings of the players booked into them. One rating per player (their
    median across those sessions). Also the same for the whole city (v = null),
    as the reference row. The stated range is the venue's most common numeric
    range for that class, when its titles give one. Under `min_players` rated
    players: left out."""
    g = okv[okv["rating"].notna()].copy()
    g["cls"] = g["session_key"].map(cls)
    g = g[g["cls"].isin(["beginner", "intermediate", "advanced", "all levels"])]
    rows = []
    for (fid, c), grp in list(g.groupby(["fid", "cls"])) + [((None, c), x) for c, x in g.groupby("cls")]:
        per = grp.groupby("pbp_user_id")["rating"].median()
        if len(per) < min_players:
            continue
        ver = grp.groupby("pbp_user_id")["rating"].apply(lambda s: any(_verified(x) for x in s))
        banded = grp.dropna(subset=["band_lo", "band_hi"]).drop_duplicates("session_key")
        lo = hi = inside = None
        if fid is not None and len(banded):
            lo, hi = banded.groupby(["band_lo", "band_hi"]).size().sort_values(ascending=False).index[0]
            lo, hi = round(float(lo), 2), round(float(min(hi, 6.0)), 2)
            inside = round(100 * float(((per >= lo - 0.01) & (per <= hi + 0.01)).mean()), 1)
        labels = grp.drop_duplicates("session_key")["label"].dropna().value_counts()
        rows.append({"v": None if fid is None else int(fid), "cls": str(c),
                     "sessions": int(grp["session_key"].nunique()), "players": int(len(per)),
                     "verified": int(ver.sum()), **_spread(per),
                     "lo": lo, "hi": hi, "inside": inside,
                     "labels": [str(x) for x in labels.index[:3]] if fid is not None else []})
    return sorted(rows, key=lambda x: (x["cls"], x["v"] is not None, x["median"]))


def prev_summary(data_dir: Path) -> dict:
    """The period before, for "compared with": each finished session in compact
    form (same groups as the main sessions list, so the page can filter both the
    same way), plus distinct players for the city and each venue."""
    old = R.DATA
    R.DATA = Path(data_dir)
    try:
        s, r, c = R.load()
        R.NOW = max(s["last_obs"].max(), r["last_seen"].max())
        s = R.attach_titles(s, c)
        fin = R.finished(s)
        m = R.rosters(s, r)
    finally:
        R.DATA = old
    f = fin[fin["fid"].notna()].copy()
    f["fid"] = f["fid"].astype(int)
    f["fgroup"] = f["fmt"].map(M.FORMAT_GROUP).fillna("Unknown")
    f["lvl"] = f["level_class"].fillna("no level stated")
    if "waitlist_max" not in f:
        f["waitlist_max"] = np.nan
    loc = f["starts_at"].dt.tz_convert(R.MEL)
    # rows rather than objects: a 90-day window is a few thousand sessions
    # price / mprice: per-session casual and member price (session_prices), for takings
    cols = ["v", "date", "dow", "hour", "fmt", "lvl", "cap", "booked", "so", "price", "mprice"]
    cash = lambda v: None if v is None or v != v else round(float(v), 2)
    sess = [[int(x.fid), d, int(x.dow), int(x.hour), x.fgroup, x.lvl, int(x.capacity), int(x.booked), int(bool(x.sold_out)),
             cash(x.price_n), cash(x.mprice_n)]
            for x, d in zip(f.itertuples(), loc.dt.strftime("%Y-%m-%d"))]
    ok = m[~m["cancelled"] & m["fid"].notna()]
    per_v = ok.assign(fid=ok["fid"].astype(int)).groupby("fid")["pbp_user_id"].nunique()
    return {"from": str(fin["starts_at"].min().tz_convert(R.MEL).date()) if len(fin) else None,
            "to": str(fin["starts_at"].max().tz_convert(R.MEL).date()) if len(fin) else None,
            "cols": cols, "sessions": sess, "players": int(ok["pbp_user_id"].nunique()),
            "venuePlayers": {str(int(k)): int(v) for k, v in per_v.items() if v >= MIN_GROUP}}


def money(g) -> dict:
    """Takings and the value of empty places, from per-session prices
    (venue_report.session_prices). Listed prices, so an upper bound: members
    may pay less. spendLow is the same if everyone paid the member price where
    one is published. Sessions whose season price couldn't be split are left
    out and counted. $ per court-hour assumes a court per 4 places."""
    p = g[g["price_n"].notna()]
    low = p["mprice_n"].where(p["mprice_n"].notna(), p["price_n"])
    both = p[p["mprice_n"].notna() & (p["price_n"] > 0)]
    hrs = p["dur_h"] * np.ceil(p["capacity"].clip(lower=1) / 4)
    timed = p[hrs.notna() & (hrs > 0)]
    return {
        "spend": int((p["price_n"] * p["booked"]).sum()) if len(p) else None,
        "spendLow": int((low * p["booked"]).sum()) if len(both) else None,
        "emptyValue": int((p["price_n"] * p["spots_at_start"]).sum()) if len(p) else None,
        "priced": int(len(p)),
        "memberPriced": int(len(both)),
        "memberDisc": round(100 * float((1 - both["mprice_n"] / both["price_n"]).median()), 1) if len(both) >= 3 else None,
        "seasonLeftOut": int((g["price_basis"] == "season").sum()),
        "seasonSplit": int(g["price_basis"].isin(["season-weeks", "season-listing", "package"]).sum()),
        "perCourtHour": r1(float((timed["price_n"] * timed["booked"]).sum() / hrs[timed.index].sum())) if len(timed) >= 5 else None,
        "waitlisted": int((g["waitlist_max"].fillna(0) > 0).sum()) if "waitlist_max" in g else 0,
    }


# ── timing (Sessions › Timing, the venue page): the timing preview's rules ──
CLOCK = [("early", 0, 9), ("morning", 9, 12), ("midday", 12, 17), ("evening", 17, 21), ("late", 21, 24)]
LEAD_EDGES = [1, 2, 4, 8]          # same day · 1 day · 2–3 · 4–7 · 8+ (timing_preview.LEAD_BUCKETS)
CANCEL_EDGES = [2, 24, 72]         # under 2 h · 2–24 h · 1–3 days · 3+ days before the start


def timing(tl, f):
    """Bookings in the window's finished sessions, summed per venue × kind ×
    level so the page's filters apply. Counts of bookings, never players:
      b   bookings (roster places held at some point)   c   dropped out
      cb  when they dropped out, hours before the start (CANCEL_EDGES)
      fr  places freed in sessions that had sold out    rb  of those, rebooked
      l   how far ahead, for bookings we watched long enough to know (LEAD_EDGES)
      k   when they booked: day × part of day (CLOCK), bookings we saw happen
    Cancellations already exclude gaps in our own capture (timing_preview)."""
    key = f.set_index("session_key")[["fid", "fgroup", "lvl"]]
    x = tl[tl["session_key"].isin(key.index)].drop(columns=["fid"], errors="ignore").join(key, on="session_key")
    if x.empty:
        return []
    x["fid"] = x["fid"].astype(int)
    last_booking = x.groupby("session_key")["first_seen"].transform("max")
    x["_freed"] = x["cancelled"] & x["ever_full"]
    x["_rebooked"] = x["_freed"] & (last_booking > x["last_seen"])
    lead = x["lead_days"].where(x["exact"], np.maximum(x["lead_days"], 8))
    x["_lb"] = np.where(x["lead_ok"] & (x["lead_days"] > 0), np.searchsorted(LEAD_EDGES, lead, side="right"), -1)
    x["_cb"] = np.where(x["cancelled"], np.searchsorted(CANCEL_EDGES, x["cancel_hours_before"].fillna(0), side="right"), -1)
    loc = x["first_seen"].dt.tz_convert(R.MEL)
    part = np.searchsorted([b[2] for b in CLOCK], loc.dt.hour, side="right")
    x["_k"] = np.where(x["exact"] & (x["lead_days"] > 0), loc.dt.dayofweek * len(CLOCK) + part, -1)
    out = []
    for (v, fm, lv), g in x.groupby(["fid", "fgroup", "lvl"]):
        cnt = lambda col, n: [int(c) for c in np.bincount(g[col][g[col] >= 0].astype(int), minlength=n)[:n]]
        out.append({"v": int(v), "fmt": fm, "lvl": lv, "b": int(len(g)), "c": int(g["cancelled"].sum()),
                    "cb": cnt("_cb", len(CANCEL_EDGES) + 1), "fr": int(g["_freed"].sum()), "rb": int(g["_rebooked"].sum()),
                    "l": cnt("_lb", len(LEAD_EDGES) + 1), "k": cnt("_k", 7 * len(CLOCK))})
    return out


def quality(s, f, tl, start, end, vmap):
    """What each venue's listings say and leave out, and how well we read
    them. Counts; the page turns them into shares. Listing: sessions logged
    in the window, pulled before they ran, moved, repriced; of the finished
    ones with a title, those stating no level; fixtures (same day and time,
    3+ runs) and those whose title varies. Our data: finished sessions with a
    title, an end time, a roster read, a roster that matches places taken
    (within 1), and a last reading within 6 h of the start."""
    gone = R.pulled(s)
    w = s[s["fid"].notna() & (s["starts_at"] >= start) & (s["starts_at"] <= end) & (s["starts_at"] < R.NOW)].copy()
    w["fid"] = w["fid"].astype(int)
    w["_gone"] = gone.reindex(w.index).fillna(False)
    lob = pd.to_datetime(f["last_obs_before"], utc=True, errors="coerce", format="mixed") if "last_obs_before" in f else pd.Series(pd.NaT, index=f.index)
    fresh = (f["starts_at"] - lob) <= pd.Timedelta(hours=6)
    held = tl[~tl["cancelled"]].groupby("session_key").size()
    roster = f["session_key"].map(held)
    out = {}
    for fid in vmap:
        ww, ff = w[w["fid"] == fid], f[f["fid"] == fid]
        if not len(ff):
            continue
        titled = ff[ff["title"].notna()]
        fx = titled.groupby(["dow", "hm"])["title"].agg(["size", "nunique"])
        fx = fx[fx["size"] >= 3]
        rr = roster[ff.index]
        has = rr.notna()
        out[str(fid)] = {
            "logged": int(len(ww)), "pulled": int(ww["_gone"].sum()),
            "moved": int((ww["n_start_times"] > 1).sum()), "repriced": int((ww["n_prices"] > 1).sum()),
            "fin": int(len(ff)), "titled": int(len(titled)), "noLevel": int((titled["lvl"] == "no level stated").sum()),
            "fixtures": int(len(fx)), "mixed": int((fx["nunique"] > 1).sum()),
            "withEnd": int(ff["end_time"].notna().sum()) if "end_time" in ff else 0,
            "rostered": int(has.sum()), "matched": int(((ff["booked"] - rr).abs() <= 1)[has].sum()),
            "fresh": int(fresh[ff.index].sum()),
        }
    return out


DAYPARTS = [("early", 0, 9), ("morning", 9, 12), ("midday", 12, 17), ("evening", 17, 20), ("late", 20, 24)]


def market(m, s, fin, ok, fid) -> dict:
    """A venue's market, for its insights page: who it really competes with,
    why travellers come, its prices against the market, and when its players
    play elsewhere. The venue report's own functions, for every venue. Counts
    of players under MIN_GROUP are held back (None), as everywhere."""
    hold = lambda n: int(n) if n >= MIN_GROUP else None
    rv = R.rivals(m, s, fin, fid)
    rivals = [{"v": x["fid"], "km": x["km"], "players": x["shared_players"], "sharePlay": x["shared_play_pct"],
               "headToHead": x["head_to_head_pct"], "fill": x["avg_fill"]}
              for x in rv["venues"] if x["shared_players"] >= MIN_GROUP]
    tr = R.travellers(m, s, fid)
    homes = sorted(((k, n) for k, n in tr["homes"].items() if n >= MIN_GROUP), key=lambda x: -x[1])
    draws = sorted(((k, n) for k, n in tr["product_draw_sessions"].items() if n >= MIN_GROUP), key=lambda x: -x[1])
    travellers = {"players": tr["players"], "travellers": hold(tr["travellers"]), "kmMedian": tr["km_median"],
                  "classes": {k: hold(n) for k, n in tr["classes"].items()},
                  "homes": [[k, n] for k, n in homes[:6]], "draws": [[k, n] for k, n in draws[:5]]}
    price = [{"family": x["family"], "level": x["level"], "here": x["venue_price_hr"], "market": x["market_price_hr"],
              "p25": x["market_p25"], "p75": x["market_p75"], "n": x["market_n"], "fill": x["venue_fill"], "marketFill": x["market_fill"]}
             for x in R.price_position(fin, fid)]
    # when its players play somewhere else: bookings at other venues by day and part of day
    ids = ok[ok["fid"] == fid]["pbp_user_id"].unique()
    allp = ok[ok["pbp_user_id"].isin(ids)]
    away = allp[allp["fid"] != fid]
    cells = []
    for d in range(7):
        for name, lo, hi in DAYPARTS:
            x = away[(away["dow"] == d) & (away["hour"] >= lo) & (away["hour"] < hi)]
            n = x["pbp_user_id"].nunique()
            if n >= MIN_GROUP:
                cells.append([d, name, int(len(x)), int(n)])
    return {"rivals": rivals, "travellers": travellers, "price": price,
            "away": {"bookings": int(len(away)), "of": int(len(allp)), "cells": cells}}


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
    if "waitlist_max" not in f:
        f["waitlist_max"] = np.nan
    loc = f["starts_at"].dt.tz_convert(R.MEL)
    f["date"] = loc.dt.strftime("%Y-%m-%d")
    f["slot"] = np.where(f["weekend"], "Weekend", "Weekday " + f["daypart"].map(
        {"early": "before 9am", "day": "9am–5pm", "evening": "after 5pm"}))
    # fill 7 / 3 / 1 days out, each where we were watching the session by then
    # (a session listed 5 days ahead has a 3- and 1-day figure, not a 7-day one)
    def stage(col, days):
        ok = (f["first_obs"] <= f["starts_at"] - pd.Timedelta(days=days)) & f[col].notna() & f[col].le(f["capacity"])
        return pd.Series(np.where(ok, (100 * (1 - f[col] / f["capacity"])).round(1), np.nan), index=f.index)
    f7, f3, f1 = stage("spots_7d", 7), stage("spots_3d", 3), stage("spots_1d", 1)
    opt = lambda v: None if v != v else float(v)
    sess = []
    for i, x in enumerate(f.itertuples()):
        sess.append({
            "i": i, "v": int(x.fid), "date": x.date, "dow": int(x.dow), "hm": x.hm, "hour": int(x.hour),
            "slot": x.slot, "fmt": x.fgroup, "lvl": x.lvl, "label": x.label if isinstance(x.label, str) else None,
            "title": x.title if isinstance(x.title, str) else None,
            "cap": int(x.capacity), "booked": int(x.booked), "fill": round(100 * x.fill, 1),
            "so": bool(x.sold_out), "soH": r1(x.so_h),
            "price": r1(x.price_n), "pph": r1(x.price_hr),
            # per-session member price where published; length in hours; how the
            # price was worked out (venue_report.session_prices); longest waitlist
            "mprice": r1(x.mprice_n), "dur": r1(x.dur_h), "pb": x.price_basis,
            "wl": int(x.waitlist_max) if x.waitlist_max == x.waitlist_max and x.waitlist_max is not None else None,
            "f7": opt(f7.loc[x.Index]), "f3": opt(f3.loc[x.Index]), "f1": opt(f1.loc[x.Index]),
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
            **money(g),
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
    travel = [x for x in more["travel"]["rows"] if x["dim"] != "venue"]

    city = {
        "from": str(start.tz_convert(R.MEL).date()), "to": str(end.tz_convert(R.MEL).date()), "days": days,
        "venuesTracked": int(sum(1 for v in vmap.values() if v.get("tracked"))),
        "venuesKnown": len(venues), "sessions": len(f), "offered": int(f["capacity"].sum()),
        "taken": int(f["booked"].sum()), "fill": round(100 * f["fill"].mean(), 1),
        "soldOut": round(100 * f["sold_out"].mean(), 1), "empty": int(f["spots_at_start"].sum()),
        **money(f), "pph": r1(f["price_hr"].median()),
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
            "ranges": ranges, "travel": travel, "levelFit": level_fit(okv, vmap, dict(zip(s["session_key"], s["level_class"]))),
            "levelClass": level_by_class(okv, dict(zip(s["session_key"], s["level_class"]))),
            "market": {str(fid): market(m, s, fin, okv, fid) for fid, v in vmap.items() if v.get("tracked")},
            "timing": timing(tl, f), "quality": quality(s, f, tl, start, end, vmap)}
    return data, net


if __name__ == "__main__":
    d, _ = main()
    print(json.dumps(d["city"], indent=1))
    print(len(d["sessions"]), "sessions;", len(d["links"]), "links;", len(d["venues"]), "venues")
