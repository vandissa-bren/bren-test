"""
guide.py -- the figures behind the new venue guide (28 Nov 2026).

For every venue on the published list (tracked or not, so a venue that has
just opened gets one too), the market within its player range:

  range        the player range: start from the tracked venues within 10 km,
               then the distance from the venue that covers 80% of the bookings
               made by players whose own venue is one of them (5-15 km)
  competitors  each venue in range: distance, how full (and weekly over the
               last 6 full weeks), sold out, how many of its players play
               nowhere else, a status (Full 80%+, Busy 60-79%, Room under 60%),
               its two fullest kinds and its fullest regular session
  moving       kinds whose fill moved 15+ points, last 28 days against the 28 before
  prices       per kind: median, lowest and highest price per player-hour; court hire
  players      players whose own venue is in range: how many, their level, how
               often they play, new to the area each week, groups by level
               (where they play, when, how full their sessions run), players who
               travel 9 km+ between venues in range, first-timers who came back
  groups       social groups (more_figures.groups) at venues in range
  grid         day x part of day: how full, sessions; the five fullest and emptiest
  kinds        kind x level: sessions a week, how full, sold out, 4-week change
  full         regular sessions 90%+ full, Monday to Sunday by start time
  week         an example week on the venue's own courts (see example_week)

Player counts follow privacy.py (0 or 5+); its COUNTS list names these fields.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, timedelta
from math import asin, cos, radians, sin, sqrt

import numpy as np
import pandas as pd

import figures as FG

SEED_KM = 10.0
RANGE_MIN, RANGE_MAX = 5.0, 15.0
TRIP_SHARE = 0.80
FULL, BUSY = 80.0, 60.0
TRAVEL_KM = 9.0
MIN_N = 5
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
PARTS = (("Morning", 0, 12), ("Afternoon", 12, 17), ("Evening", 17, 24))


def _km(a, b):
    if not a.get("lat") or not b.get("lat"):
        return np.nan
    dl, dn = radians(b["lat"] - a["lat"]), radians(b["lng"] - a["lng"])
    h = sin(dl / 2) ** 2 + cos(radians(a["lat"])) * cos(radians(b["lat"])) * sin(dn / 2) ** 2
    return 2 * 6371 * asin(sqrt(h))


def _hold(n):
    n = int(n)
    return n if n == 0 or n >= MIN_N else None


def _r1(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 1)


def _part(h):
    for name, a, b in PARTS:
        if a <= h < b:
            return name
    return "Evening"


def kind_of(fgroup, lvl):
    fg = fgroup if isinstance(fgroup, str) and fgroup != "Unknown" else "Other sessions"
    if isinstance(lvl, str) and lvl in ("beginner", "intermediate", "advanced"):
        return f"{lvl.capitalize()} {fg[0].lower() + fg[1:]}"
    if isinstance(lvl, str) and lvl == "all levels":
        return f"All-levels {fg[0].lower() + fg[1:]}"
    return fg


def _rating_class(x):
    if x != x:
        return "unrated"
    return "beginner" if x < 2.75 else "intermediate" if x < 3.5 else "advanced"


def build(f: pd.DataFrame, okv: pd.DataFrame, firsts, vmap: dict, end_day: date, groups: dict,
          oneoff: set = frozenset()) -> dict:
    """{"venues": {fid: guide}} for every venue in vmap with a location."""
    x = FG._series(f)
    x["_part"] = x["_h"].map(_part)
    x["kind"] = [kind_of(a, b) for a, b in zip(x["fgroup"], x["lvl"])]
    tracked = sorted(set(x["fid"].astype(int)))
    weeks = max(1.0, (x["_date"].max() - x["_date"].min()).days / 7) if len(x) else 1.0
    end_ts = pd.Timestamp(end_day)

    o = okv.copy()
    per = o.groupby(["pbp_user_id", "fid"]).size().reset_index(name="n")
    home = (per.sort_values(["pbp_user_id", "n", "fid"], ascending=[True, False, True])
               .drop_duplicates("pbp_user_id").set_index("pbp_user_id")["fid"].astype(int))
    rating = o.groupby("pbp_user_id")["rating"].median() if "rating" in o else pd.Series(dtype=float)
    nb = o.groupby("pbp_user_id").size()

    fv = None
    if firsts is not None and len(firsts):
        fv = firsts.dropna(subset=["pbp_user_id", "fid", "first_date"]).copy()
        if oneoff and "first_key" in fv:
            fv = fv[~fv["first_key"].isin(oneoff)]
        fv["fid"] = fv["fid"].astype(int)
        fv["first_date"] = pd.to_datetime(fv["first_date"], errors="coerce")

    out, comp = {}, {}
    for fid, site in vmap.items():
        if not site.get("lat"):
            continue
        dist = {v: _km(site, vmap[v]) for v in tracked if v in vmap and v != fid}
        seed = [v for v, k in dist.items() if k <= SEED_KM]
        if not seed:
            continue
        # the player range: where the players of the venues nearby actually play
        hp = home[home.isin(seed)].index
        trips = o[o["pbp_user_id"].isin(hp)]["fid"].map(lambda v: dist.get(int(v), 0.0 if int(v) == fid else np.nan)).dropna()
        rng = float(np.clip(round(np.quantile(trips, TRIP_SHARE) * 2) / 2, RANGE_MIN, RANGE_MAX)) if len(trips) else SEED_KM
        near = sorted([v for v, k in dist.items() if k <= rng], key=lambda v: dist[v])
        if not near:
            continue
        g = x[x["fid"].isin(near)]
        out[str(fid)] = {
            "range": rng, "seedKm": SEED_KM,
            "tripsInRange": _r1(100 * np.mean(trips <= rng)) if len(trips) else None,
            "competitors": [dict(comp.setdefault(v, _competitor(x, o, v, vmap[v], end_ts)), km=_r1(dist[v])) for v in near],
            "moving": _moving(g, end_ts),
            "prices": _prices(g),
            "players": _players(o, g, near, home, rating, nb, fv, vmap, dist, end_ts),
            "groups": _groups(groups, near, vmap),
            **_grid(g),
            "kinds": _kinds(g, weeks, end_ts),
            "full": _full(g, vmap),
            "week": example_week(g, site, vmap),
        }
    return {"venues": out}


def _competitor(x, o, v, meta, end_ts):
    g = x[x["fid"] == v]
    p = o[o["fid"] == v]
    per = o[o["pbp_user_id"].isin(p["pbp_user_id"].unique())].groupby("pbp_user_id")["fid"].agg(lambda s: (s == v).mean())
    fill = 100 * g["fill"].mean() if len(g) else None
    wk = []
    last = end_ts - timedelta(days=end_ts.dayofweek + 7)        # Monday of the last full week
    for i in range(5, -1, -1):
        a = (last - timedelta(weeks=i)).date()
        w = g[(g["_date"] >= a) & (g["_date"] < a + timedelta(days=7))]
        wk.append(_r1(100 * w["fill"].mean()) if len(w) else None)
    kinds = (g.groupby("kind").agg(n=("fill", "size"), fill=("fill", "mean"))
               .query("n >= 4").sort_values("fill", ascending=False).head(2))
    ser = g.groupby("sid").agg(n=("fill", "size"), fill=("fill", "mean"), title=("title", "last"),
                               dow=("_dow", "first"), hm=("_hm", "first")).query("n >= 3")
    anchor = None
    if len(ser):
        a = ser.sort_values("fill", ascending=False).iloc[0]
        anchor = {"title": a["title"], "dow": int(a["dow"]), "hm": a["hm"], "fill": _r1(100 * a["fill"])}
    return {"v": int(v), "name": meta.get("name"), "fill": _r1(fill), "weekly": wk,
            "soldOut": _r1(100 * g["sold_out"].mean()) if len(g) else None,
            # a rate of players: only when its players and the rest are each 0 or 5+
            "onlyHere": _r1(100 * (per > 0.8).mean()) if len(per) >= MIN_N and _hold((per > 0.8).sum()) is not None
                        and _hold((per <= 0.8).sum()) is not None else None,
            "status": None if fill is None else "Full" if fill >= FULL else "Busy" if fill >= BUSY else "Room",
            "knownFor": [{"kind": k, "fill": _r1(100 * r["fill"])} for k, r in kinds.iterrows()],
            "anchor": anchor, "perWeek": _r1(len(g) / max(1.0, (g["_date"].max() - g["_date"].min()).days / 7)) if len(g) else None}


def _window(g, end_ts, a, b):
    return g[(g["_date"] > (end_ts - timedelta(days=a)).date()) & (g["_date"] <= (end_ts - timedelta(days=b)).date())]


def _moving(g, end_ts):
    now, before = _window(g, end_ts, 28, 0), _window(g, end_ts, 56, 28)
    rows = []
    for (v, k), a in now.groupby(["fid", "kind"]):
        b = before[(before["fid"] == v) & (before["kind"] == k)]
        if len(a) >= 4 and len(b) >= 4:
            d = 100 * (a["fill"].mean() - b["fill"].mean())
            if abs(d) >= 15:
                rows.append({"v": int(v), "kind": k, "before": _r1(100 * b["fill"].mean()), "now": _r1(100 * a["fill"].mean())})
    return sorted(rows, key=lambda r: -abs(r["now"] - r["before"]))[:6]


def _prices(g):
    rows = []
    for k, a in g.dropna(subset=["price_hr"]).groupby("fgroup"):
        if len(a) >= 5 and k != "Unknown":
            rows.append({"kind": k, "median": _r1(a["price_hr"].median()), "low": _r1(a["price_hr"].quantile(0.1)),
                         "high": _r1(a["price_hr"].quantile(0.9)), "sessions": int(len(a)),
                         "length": _r1(a["dur_h"].median()) if "dur_h" in a and a["dur_h"].notna().any() else None})
    h = g["hire_rate"].dropna() if "hire_rate" in g else pd.Series(dtype=float)
    hire = {"median": _r1(h.median()), "low": _r1(h.min()), "high": _r1(h.max())} if len(h) else None
    return {"perPlayerHour": sorted(rows, key=lambda r: -r["sessions"]), "courtHire": hire}


def _players(o, g, near, home, rating, nb, fv, vmap, dist, end_ts):
    hp = home[home.isin(near)].index
    cls = rating.reindex(hp).map(_rating_class)
    freq = nb.reindex(hp)
    oo = o[o["pbp_user_id"].isin(hp) & o["fid"].isin(near)]
    segs = []
    sk = g.set_index("session_key")
    for c in ("advanced", "intermediate", "beginner", "unrated"):
        ids = cls[cls == c].index
        b = oo[oo["pbp_user_id"].isin(ids)]
        if not len(b):
            continue
        where = b["fid"].value_counts().head(3)
        ss = sk.reindex(b["session_key"].unique()).dropna(subset=["fill"])
        when = Counter(f"{DAYS[int(d)]} {p.lower()}" for d, p in zip(ss["_dow"], ss["_part"])).most_common(2)
        segs.append({"level": c, "players": _hold(len(ids)),
                     "playsAt": [vmap[int(v)]["name"] for v in where.index if int(v) in vmap],
                     "when": [w for w, _ in when], "fill": _r1(100 * ss["fill"].mean()) if len(ss) else None})
    # new to the area: first booking at any venue in range, per week (last 8 full weeks)
    weekly = []
    if fv is not None:
        firsts = fv[fv["fid"].isin(near)].groupby("pbp_user_id")["first_date"].min().dropna()
        last = end_ts - timedelta(days=end_ts.dayofweek + 7)
        for i in range(7, -1, -1):
            a = last - timedelta(weeks=i)
            weekly.append({"week": str(a.date()), "players": _hold(((firsts >= a) & (firsts < a + timedelta(days=7))).sum())})
    # players who travel 9 km+ from their own venue to another in range
    trav = []
    for (h, v), n in (oo[oo["fid"] != oo["pbp_user_id"].map(home)]
                      .assign(h=lambda d: d["pbp_user_id"].map(home))
                      .groupby(["h", "fid"])["pbp_user_id"].nunique().items()):
        k = _km(vmap[int(h)], vmap[int(v)]) if int(h) in vmap and int(v) in vmap else np.nan
        if k >= TRAVEL_KM and n >= MIN_N:
            b = g[(g["fid"] == v) & g["session_key"].isin(oo[(oo["fid"] == v) & (oo["pbp_user_id"].map(home) == h)]["session_key"])]
            top = b["kind"].mode().iloc[0] if len(b) else None
            trav.append({"from": vmap[int(h)]["name"], "to": vmap[int(v)]["name"], "km": _r1(k), "players": _hold(n), "for": top})
    trav.sort(key=lambda r: -(r["players"] or 0))
    return {"home": _hold(len(hp)),
            "levels": {c: _hold((cls == c).sum()) for c in ("advanced", "intermediate", "beginner", "unrated")},
            "freq": {"1": _hold((freq == 1).sum()), "2": _hold((freq == 2).sum()),
                     "3-4": _hold(freq.between(3, 4).sum()), "5+": _hold((freq >= 5).sum())},
            "segments": segs, "newWeekly": weekly, "travel": trav[:8],
            "cameBack": [{"v": int(v), "name": vmap[v]["name"], "pct": vmap[v].get("cameBack")} for v in near
                         if vmap[v].get("cameBack") is not None]}


def _groups(groups, near, vmap):
    gv = (groups or {}).get("venues", {})
    rows, n, players = [], 0, 0
    for v in near:
        x = gv.get(str(v))
        if not x:
            continue
        n += x.get("groups") or 0
        players += x.get("players") or 0
        for g in x.get("list") or []:
            rows.append(dict(g, venue=vmap[v]["name"]))
    return {"groups": n, "players": _hold(players), "list": sorted(rows, key=lambda r: -r["size"])[:6]}


def _grid(g):
    cells = []
    for (d, p), a in g.groupby(["_dow", "_part"]):
        cells.append({"dow": int(d), "part": p, "fill": _r1(100 * a["fill"].mean()), "sessions": int(len(a))})
    ranked = sorted([c for c in cells if c["sessions"] >= 8], key=lambda c: -c["fill"])
    lab = lambda c: dict(c, label=f"{DAYS[c['dow']]} {c['part'].lower()}")
    return {"grid": cells, "fullest": [lab(c) for c in ranked[:5]], "emptiest": [lab(c) for c in ranked[::-1][:5]]}


def _kinds(g, weeks, end_ts):
    now, before = _window(g, end_ts, 28, 0), _window(g, end_ts, 56, 28)
    rows = []
    for k, a in g.groupby("kind"):
        pw = len(a) / weeks
        if pw < 0.5:
            continue
        n1, b1 = now[now["kind"] == k], before[before["kind"] == k]
        ch = _r1(100 * (n1["fill"].mean() - b1["fill"].mean())) if len(n1) >= 4 and len(b1) >= 4 else None
        rows.append({"kind": k, "perWeek": _r1(pw), "fill": _r1(100 * a["fill"].mean()),
                     "soldOut": _r1(100 * a["sold_out"].mean()), "change": ch,
                     "fullAndFew": bool(100 * a["fill"].mean() >= FULL and pw <= 25)})
    return sorted(rows, key=lambda r: -r["fill"])


def _full(g, vmap):
    rows = []
    for sid, a in g.groupby("sid"):
        if len(a) >= 3 and a["fill"].mean() >= 0.9:
            so = a["so_h"].dropna() if "so_h" in a else pd.Series(dtype=float)
            wl = a["waitlist_max"].dropna() if "waitlist_max" in a else pd.Series(dtype=float)
            rows.append({"dow": int(a["_dow"].iloc[0]), "hm": a["_hm"].iloc[0], "min": int(a["_min"].iloc[0]),
                         "venue": vmap[int(a["fid"].iloc[0])]["name"], "title": a["title"].dropna().iloc[-1] if a["title"].notna().any() else None,
                         "kind": a["kind"].mode().iloc[0], "fill": _r1(100 * a["fill"].mean()),
                         "soldOutDaysAhead": _r1(so.median() / 24) if len(so) >= 2 else None,
                         "waitlist": bool(len(wl) and wl.max() > 0)})
    rows.sort(key=lambda r: (r["dow"], r["min"]))
    for r in rows:
        r.pop("min")
    return rows


def example_week(g, site, vmap, max_sessions=8, per_kind=2):
    """An example first week on the venue's courts, from the sessions nearby
    that run 90%+ full: one session per day and time, at most two of a kind,
    courts from the usual places per court for that kind nearby (one court left
    for hire on weeknights while the venue has 3+), priced at the kind's
    median per player-hour nearby times its usual length."""
    courts = site.get("courts")
    if not courts or g.empty:
        return None
    courts = int(courts)
    full = _full(g, vmap)
    ppc = (g["capacity"] / g["courts"].clip(lower=1)).groupby(g["kind"]).median()
    cap = g.groupby("kind")["capacity"].median()
    dur = g.groupby("kind")["dur_h"].median() if "dur_h" in g else pd.Series(dtype=float)
    pph = g.groupby("kind")["price_hr"].median()
    used, kinds, rows = set(), Counter(), []
    for r in sorted(full, key=lambda r: -r["fill"]):
        slot = (r["dow"], r["hm"][:2])
        k = r["kind"]
        if slot in used or kinds[k] >= per_kind:
            continue
        evening_weekday = r["dow"] < 5 and int(r["hm"][:2]) >= 17
        top = courts - 1 if evening_weekday and courts >= 3 else courts
        per = float(ppc.get(k, 4) or 4)
        n_courts = int(max(1, min(top, round(float(cap.get(k, per * top)) / per))))
        length = float(dur.get(k, 2.0)) if dur.get(k, np.nan) == dur.get(k, np.nan) else 2.0
        price = pph.get(k, np.nan)
        rows.append({"dow": r["dow"], "hm": r["hm"], "kind": k, "courts": n_courts, "of": courts,
                     "places": int(round(n_courts * per)), "hours": _r1(length),
                     "price": int(round(price * length)) if price == price else None,
                     "because": {"venue": r["venue"], "title": r["title"], "fill": r["fill"],
                                 "waitlist": r["waitlist"], "soldOutDaysAhead": r["soldOutDaysAhead"]}})
        used.add(slot)
        kinds[k] += 1
        if len(rows) >= max_sessions:
            break
    rows.sort(key=lambda r: (r["dow"], r["hm"]))
    return {"courts": courts, "sessions": rows, "places": sum(r["places"] for r in rows),
            "courtHours": _r1(sum(r["courts"] * (r["hours"] or 2) for r in rows))}
