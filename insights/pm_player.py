"""
pm_player.py -- each player's PM level page, for staff (14 Nov 2026)

Built each night from the PM level v2 fit (pm_v2.fit(..., detail=...)) and
stored by record_pm_players (migration 20270216100000); admin Insights >
Staff > PM level > Look up a player reads it through admin_pm_player. Never
in the insights file, never printed: ids only here, names are joined in the
database for staff.

Per player:
  * terms: the evidence behind their level and each one's share of the weight.
    The level is the weighted average: level ~= sum(share x value).
  * sessions: each weekly session they played in the window, with where they
    sit in it (pulled up / inside / above / open to all / ...).
  * partners: players they deliberately play with (below).
  * recs: weekly sessions around their level near where they play, and ones
    their partners play that fit them.
  * dupr: their DUPR trail (each value when it changed, and where it came from).

Partners: being at the same weekly session a lot is not enough (two regulars
of Tuesday 7pm share every run without knowing each other). For player i and j,
over each weekly session S that i played:

    expected(S) = runs i played in S x q
    q           = chance j is at a run of S that i ISN'T at, shrunk to a prior
                = (j's runs of S without i + a.prior) / (runs of S without i + a)
    prior       = j's share of S's runs   (S runs weekly: a regular is a regular)
                  blended, for one-offs and new sessions, towards
                  j's share of that venue's sessions in the window

So two regulars who both never miss are expected to share every run (no
excess); j turning up only when i does, or the two of them at the same
one-off, or at several venues, is excess. A pair plays together when they
shared 3+ sessions, at least 2 more than expected and at least twice as many,
with Poisson chance <= 0.1% of that many by coincidence, both ways round
(each counted from their own sessions). Likely coaches are left out; so are
sessions with over 40 players (events).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

CFG = dict(
    alpha=2.0,             # strength of q's prior, in runs
    min_together=3, min_excess=2.0, max_p=0.001, min_runs=3,
    max_players=40,        # sessions bigger than this (events) don't count towards partners
    partners_max=12,
    active_days=21,        # a weekly session still running: a run in the last 3 weeks
    active_runs=3,         # ... and 3+ runs in the window
    suits=0.25, step_up=0.50, below=0.50,
    rec_max=6,
    range_visits=2,        # a venue counts towards how far they travel once they've played there twice
)


# ── partners ──────────────────────────────────────────────────────────────────
def partners(d: pd.DataFrame, sess: pd.DataFrame, coaches=frozenset(), cfg=CFG) -> pd.DataFrame:
    """Pairs who play together on purpose, both ways round:
    columns i, j, together, expected, venues, series, p (from i's sessions)."""
    c = cfg
    V = d[["pbp_user_id", "session_key"]].copy()
    V = V[~V["pbp_user_id"].isin(set(coaches))]
    V["series"] = V["session_key"].map(sess["series"])
    V["fid"] = V["session_key"].map(sess["fid"]).astype("int64")
    V = V.dropna(subset=["series"]).drop_duplicates(["pbp_user_id", "session_key"])
    size = V.groupby("session_key")["pbp_user_id"].transform("size")
    small = V[size <= c["max_players"]]

    # shared sessions
    P = small.merge(small, on=["session_key", "series", "fid"], suffixes=("_i", "_j"))
    P = P[P["pbp_user_id_i"] < P["pbp_user_id_j"]].rename(columns={"pbp_user_id_i": "i", "pbp_user_id_j": "j"})
    if P.empty:
        return pd.DataFrame(columns=["i", "j", "together", "expected", "venues", "series", "p"])
    K = P.groupby(["i", "j"]).agg(together=("session_key", "size"), venues=("fid", "nunique"),
                                  series=("series", "nunique")).reset_index()
    K = K[K["together"] >= c["min_together"]]
    if K.empty:
        return pd.DataFrame(columns=["i", "j", "together", "expected", "venues", "series", "p"])

    # strata: each weekly session with 3+ runs on its own; every other session
    # (one-offs, new ones) pooled by venue. Under "they book independently",
    # how many of a stratum's runs they share is hypergeometric (runs, i's
    # runs, j's runs): regulars who never miss share every run and that's
    # expected; turning up on the same runs when both often miss is not.
    runs_s = small.groupby("series")["session_key"].nunique()
    stratum = small["series"].where(small["series"].map(runs_s) >= c["min_runs"], "venue:" + small["fid"].astype(str))
    small = small.assign(st=stratum.to_numpy())
    N = small.groupby("st")["session_key"].nunique()
    NI = small.groupby(["pbp_user_id", "st"]).size().rename("n").reset_index()
    # every stratum both of them played in, shared runs or not
    R = K[["i", "j"]].merge(NI.rename(columns={"pbp_user_id": "i", "n": "n_i"}), on="i")
    R = R.merge(NI.rename(columns={"pbp_user_id": "j", "n": "n_j"}), on=["j", "st"])
    R["N"] = R["st"].map(N).to_numpy()
    R["e"] = R["n_i"] * R["n_j"] / R["N"]
    out = K.merge(R.groupby(["i", "j"])["e"].sum().rename("expected").reset_index(), on=["i", "j"])
    out = out[out["together"] - out["expected"] >= c["min_excess"]]
    if out.empty:
        return pd.DataFrame(columns=["i", "j", "together", "expected", "venues", "series", "p"])
    # chance of sharing this many by coincidence: the strata's hypergeometrics summed (exact)
    from scipy.stats import hypergeom
    keys = set(map(tuple, out[["i", "j"]].to_numpy()))
    R = R[[t in keys for t in zip(R["i"], R["j"])]]
    pv = {}
    for key, g in R.groupby(["i", "j"], sort=False):
        pmf = np.array([1.0])
        for Nn, ni, nj in zip(g["N"], g["n_i"], g["n_j"]):
            x = np.arange(0, min(ni, nj) + 1)
            pmf = np.convolve(pmf, hypergeom.pmf(x, Nn, ni, nj))
        pv[key] = pmf
    out["p"] = [float(pv[(a, b)][k:].sum()) if k < len(pv[(a, b)]) else 0.0
                for a, b, k in zip(out["i"], out["j"], out["together"])]
    out = pd.concat([out, out.rename(columns={"i": "j", "j": "i"})], ignore_index=True)
    ok = out["p"] <= c["max_p"]
    return out[ok].reset_index(drop=True)


# ── weekly sessions ───────────────────────────────────────────────────────────
def series_table(d: pd.DataFrame, sess: pd.DataFrame, rooms: list, end, cfg=CFG) -> dict:
    """Every weekly session in the window: where and when, its label, its room
    level (pm_v2.rooms) and whether it's still running."""
    from pm_v2 import _hold, _r
    rm = {r["series"]: r for r in rooms or []}
    end = pd.Timestamp(end)
    out = {}
    for ser, g in sess.reset_index().groupby("series"):
        g = g.sort_values("starts_at")
        last = g.iloc[-1]
        st = pd.Timestamp(last["starts_at"]).tz_convert("Australia/Melbourne")
        lo, hi = last.get("band_lo"), last.get("band_hi")
        r = rm.get(ser)
        runs = int(len(g))
        out[ser] = {
            "v": int(last["fid"]), "title": last["title"] if isinstance(last.get("title"), str) else None,
            "label": last["label"] if isinstance(last.get("label"), str) else None,
            "dow": int(st.dayofweek), "hm": st.strftime("%H:%M"), "family": last.get("family"),
            "coached": bool(last.get("coached")), "weekly": not str(ser).startswith("one:"),
            "labelLo": _r(lo) if lo == lo else None, "labelHi": _r(min(hi, 5.0)) if hi == hi else None,
            "runs": runs, "last": str(st.date()),
            "active": bool(runs >= cfg["active_runs"] and (end - pd.Timestamp(last["starts_at"])).days <= cfg["active_days"]),
            "level": r["level"] if r else None, "lo": r["lo"] if r else None, "hi": r["hi"] if r else None,
            "players": _hold(r["players"]) if r else None,
        }
    return out


# ── the maths behind each level ───────────────────────────────────────────────
def _place(x, k, has_d, theta):
    """Where a player sits in one weekly session (k: index into the pair arrays)."""
    if np.isnan(x["pRoom"][k]):
        return "noroom"                               # too few others with a level
    if x["pOpen"][k]:
        return "open"                                 # all levels / no floor: no pull up
    if x["wE"][k] > 0:
        return "pulled"                               # below its lower edge, 3+ visits
    if theta < x["pLo"][k]:
        return "below"                                # below the edge, under 3 visits
    if theta > 2 * x["pVal"][k] - x["pLo"][k]:
        return "above"
    return "inside"


def breakdown(meta: dict, T: pd.DataFrame) -> dict:
    """pid -> {terms, sessions}: each piece of evidence's value and share of the weight."""
    from pm_v2 import _r
    x = meta["x"]
    pids = meta["pids"]
    out = {}
    by_player = pd.Series(np.arange(len(x["pPlayer"]))).groupby(x["pPlayer"]).apply(list).to_dict()
    for i, pid in enumerate(pids):
        tot = float(x["prec"][i])
        theta = float(T.at[pid, "level"])
        terms = []
        if x["wD"][i] > 0:
            terms.append({"k": "dupr", "x": _r(x["dT"][i]), "share": _r(100 * x["wD"][i] / tot, 1), "sd": _r(x["sD"][i])})
        sessions = []
        for k in by_player.get(i, []):
            wc, we = float(x["wC"][k]), float(x["wE"][k])
            w = wc + we
            val = (wc * np.nan_to_num(x["pVal"][k]) + we * np.nan_to_num(x["pEdge"][k])) / w if w > 0 else None
            place = _place(x, k, x["hasD"][i], theta)
            row = {"s": str(x["pSeries"][k]), "visits": int(x["pVisits"][k]), "place": place,
                   "room": _r(x["pRoom"][k]) if not np.isnan(x["pRoom"][k]) else None,
                   "edge": _r(x["pLo"][k]) if np.isfinite(x["pLo"][k]) else None,
                   "x": _r(val) if val is not None else None, "share": _r(100 * w / tot, 1)}
            sessions.append(row)
            if w > 0:
                terms.append({"k": "room", "s": row["s"], "x": row["x"], "share": row["share"],
                              "mode": "edge" if we > wc else "centre", "visits": row["visits"],
                              "room": row["room"], "edge": row["edge"]})
        if x["wL"][i] > 0:
            terms.append({"k": "labels" if x["labOwn"][i] else "venue", "x": _r(x["lab"][i]),
                          "share": _r(100 * x["wL"][i] / tot, 1)})
        if x["wS"][i] > 0:
            terms.append({"k": "self", "x": _r(x["self"][i]), "raw": _r(x["selfRaw"][i]),
                          "share": _r(100 * x["wS"][i] / tot, 1)})
        terms.sort(key=lambda t: -(t["share"] or 0))
        sessions.sort(key=lambda r: (-r["visits"], r["s"]))
        out[pid] = {"terms": terms, "sessions": sessions, "sum": _r(x["new"][i], 3)}
    return out


# ── DUPR trail ────────────────────────────────────────────────────────────────
def dupr_trail(m: pd.DataFrame, ratelog, reads) -> dict:
    """pid -> [{at, v, src}]: each DUPR value when first seen and each change,
    newest first, plus the last time it was seen unchanged."""
    from ratings import dupr
    parts = [m[["pbp_user_id", "last_seen", "rating_at_time"]].set_axis(["p", "at", "v"], axis=1).assign(src="roster")]
    if ratelog is not None and len(ratelog):
        parts.append(ratelog[["pbp_user_id", "seen_at", "rating"]].set_axis(["p", "at", "v"], axis=1).assign(src="roster"))
    if reads is not None and len(reads):
        parts.append(reads[["pbp_user_id", "read_at", "doubles"]].set_axis(["p", "at", "v"], axis=1).assign(src="read"))
    h = pd.concat(parts, ignore_index=True)
    h["v"] = dupr(h["v"])
    h["at"] = pd.to_datetime(h["at"], utc=True, errors="coerce")
    h["p"] = pd.to_numeric(h["p"], errors="coerce")
    h = h.dropna(subset=["v", "at", "p"]).sort_values(["p", "at"])
    h["p"] = h["p"].astype("int64")
    out = {}
    for pid, g in h.groupby("p", sort=False):
        v = g["v"].to_numpy()
        keep = np.r_[True, np.abs(np.diff(v)) >= 0.005]
        rows = [{"at": str(r.at.date()), "v": round(float(r.v), 3), "src": r.src} for r in g[keep].itertuples()]
        out[int(pid)] = {"changes": rows[::-1][:12], "lastSeen": str(g["at"].iloc[-1].date())}
    return out


# ── recommendations ───────────────────────────────────────────────────────────
def candidates(S: dict, vmap: dict) -> pd.DataFrame:
    """Weekly sessions that can be recommended: still running, not coached,
    with a level (the room's, else the label's middle) and a place on the map."""
    rows = []
    for s, info in S.items():
        if not (info["active"] and info["weekly"] and not info["coached"]):
            continue
        lv, by = info.get("level"), "room"
        if lv is None and info.get("labelLo") is not None:
            hi = info["labelHi"] if info.get("labelHi") is not None else info["labelLo"] + 0.5
            lv, by = (info["labelLo"] + hi) / 2, "label"
        v = vmap.get(info["v"]) or {}
        if lv is None or v.get("lat") is None:
            continue
        rows.append((s, lv, by, info.get("labelLo"), math.radians(v["lat"]), math.radians(v["lng"])))
    return pd.DataFrame(rows, columns=["s", "lv", "by", "labelLo", "lat", "lng"])


def _fits(gap, label_lo, level, cfg=CFG):
    f = np.where(np.abs(gap) <= cfg["suits"], "suits",
                 np.where((gap > 0) & (gap <= cfg["step_up"]), "up",
                          np.where((gap < 0) & (gap >= -cfg["below"]), "below", "no")))
    floor = np.nan_to_num(np.asarray(label_lo, dtype=float), nan=-9)
    return np.where(floor > level + cfg["step_up"], "no", f)


def _km_from(C: pd.DataFrame, places: list) -> np.ndarray:
    """Distance (km) from each candidate to the nearest of `places` [(lat, lng) radians]."""
    if not places or C.empty:
        return np.full(len(C), np.nan)
    la = C["lat"].to_numpy()[:, None]
    lo = C["lng"].to_numpy()[:, None]
    pla = np.array([p[0] for p in places])[None, :]
    plo = np.array([p[1] for p in places])[None, :]
    h = np.sin((pla - la) / 2) ** 2 + np.cos(la) * np.cos(pla) * np.sin((plo - lo) / 2) ** 2
    return (2 * 6371 * np.arcsin(np.sqrt(h))).min(axis=1)


def _venue_km(a, b, vmap):
    pa, pb = vmap.get(a) or {}, vmap.get(b) or {}
    if pa.get("lat") is None or pb.get("lat") is None:
        return None
    la1, lo1, la2, lo2 = map(math.radians, (pa["lat"], pa["lng"], pb["lat"], pb["lng"]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def travel(played: dict, S: dict, vmap: dict, cfg=CFG):
    """(home venue, km) from how they actually play: home is the venue they
    play most; km is the furthest venue from it they've played at least twice.
    None when they've only played regularly at one venue."""
    from collections import Counter
    by_v = Counter()
    for s, n in played.items():
        if s in S:
            by_v[S[s]["v"]] += n
    if not by_v:
        return None, None
    home = by_v.most_common(1)[0][0]
    ds = [_venue_km(home, v, vmap) for v, n in by_v.items() if v != home and n >= cfg["range_visits"]]
    ds = [x for x in ds if x is not None]
    return home, (max(ds) if ds else None)


def recommend(level, played: dict, partner_ids, played_by: dict, S: dict, C: pd.DataFrame, vmap: dict,
              reach: dict, cfg=CFG):
    """Two short lists of weekly sessions for one player.
    near: around their level (suits them, or a step up), no further from their
    home venue than they travel (reach: {"home", "km", "how"}), that they don't
    already play; ranked by fit (a room level beats a label), then distance
    as a share of how far they go.
    partners: ones their partners play (2+ visits) and they don't, that fit
    them (suits, a step up, or a bit below); ranked by how many partners go."""
    from collections import Counter
    home, R = reach.get("home"), reach.get("km")
    hv = vmap.get(home) or {}
    places = [(math.radians(hv["lat"]), math.radians(hv["lng"]))] if hv.get("lat") is not None else []
    if C.empty:
        return {"near": [], "partners": []}
    gap = C["lv"].to_numpy() - level
    fit = _fits(gap, C["labelLo"], level, cfg)
    km = _km_from(C, places)
    new = ~C["s"].isin(set(played)).to_numpy()
    reach_km = R if R is not None else 0.0
    sel = new & np.isin(fit, ["suits", "up"]) & (np.nan_to_num(km, nan=np.inf) <= reach_km + 0.05)
    far = np.nan_to_num(km, nan=0) / reach_km if reach_km > 0 else np.zeros(len(C))
    rank = (C["by"].to_numpy() != "room") * 0.3 + np.abs(gap - 0.1) + 0.5 * far
    near = [{"s": C["s"].iat[k], "fit": str(fit[k]), "gap": round(float(gap[k]), 2), "by": C["by"].iat[k],
             "km": round(float(km[k]), 1)} for k in np.nonzero(sel)[0][np.argsort(rank[sel], kind="stable")][:cfg["rec_max"]]]

    cnt, who = Counter(), {}
    for pid in partner_ids:
        for s, n in played_by.get(pid, {}).items():
            if n >= 2 and s not in played:
                cnt[s] += 1
                who.setdefault(s, []).append(int(pid))
    pos = {s: k for k, s in enumerate(C["s"])}
    order = {"suits": 0, "up": 1, "below": 2}
    pals = []
    for s, n in cnt.items():
        k = pos.get(s)
        if k is None or fit[k] == "no":
            continue
        pals.append(({"s": s, "fit": str(fit[k]), "gap": round(float(gap[k]), 2), "by": C["by"].iat[k],
                      "km": None if np.isnan(km[k]) else round(float(km[k]), 1), "partners": who[s]},
                     (-n, order[str(fit[k])], np.nan_to_num(km[k], nan=99))))
    pals.sort(key=lambda t: t[1])
    return {"near": near, "partners": [p for p, _ in pals[:cfg["rec_max"]]]}


# ── the build's entry point ───────────────────────────────────────────────────
def build(det: dict, vmap: dict, m: pd.DataFrame, ratelog=None, reads=None, cfg=CFG):
    """(players, series) for record_pm_players. players: one row per player
    with a PM level; series: every weekly session any of them played or is
    recommended."""
    from pm_v2 import _r
    d, sess, T, meta, dn = det["d"], det["sess"], det["T"], det["meta"], det["dn"]
    S = series_table(d, sess, det["rooms"], det["end"], cfg)
    C = candidates(S, vmap)
    B = breakdown(meta, T)
    trail = dupr_trail(m, ratelog, reads)
    pr = partners(d, sess, det.get("coaches", set()), cfg)
    visits = d.assign(series=d["session_key"].map(sess["series"])).groupby(["pbp_user_id", "series"]).size()
    played_by = {int(p): g.droplevel(0).to_dict() for p, g in visits.groupby(level=0)}
    pr_by = {int(p): g.sort_values("together", ascending=False) for p, g in pr.groupby("i")}
    n_by = d.groupby("pbp_user_id").size()

    # how far each player travels: their own furthest regular venue from home;
    # for players who only play at one venue, how far that venue's other
    # regulars go when they play elsewhere (median), or just their own venue
    trav = {int(pid): travel(played_by.get(int(pid), {}), S, vmap, cfg) for pid in T.index}
    by_home = {}
    for pid, (h, k) in trav.items():
        if h is not None and k is not None and n_by.get(pid, 0) >= 4:
            by_home.setdefault(h, []).append(k)
    peer = {h: float(np.median(v)) for h, v in by_home.items() if len(v) >= 5}

    def reach_of(pid):
        h, k = trav.get(pid, (None, None))
        if k is not None:
            return {"home": h, "km": round(k, 1), "how": "own"}
        if h in peer:
            return {"home": h, "km": round(peer[h], 1), "how": "peers"}
        return {"home": h, "km": 0.0, "how": "venue"}

    players = []
    for pid, t in T.iterrows():
        pid = int(pid)
        played = played_by.get(pid, {})
        g = pr_by.get(pid)
        pals = [] if g is None else [
            {"id": int(r.j), "together": int(r.together), "expected": round(float(r.expected), 1),
             "venues": int(r.venues), "series": int(r.series),
             "share": round(100 * r.together / max(int(n_by.get(pid, 1)), 1))}
            for r in g.head(cfg["partners_max"]).itertuples()]
        dd = dn.loc[pid] if pid in dn.index else None
        b = B.get(pid, {"terms": [], "sessions": [], "sum": None})
        players.append({
            "id": pid, "level": _r(t.level), "sd": _r(t.sd), "kind": t.kind, "n": int(t.n), "coach": bool(t.coach),
            "home": int(t.home) if t.home == t.home else None,
            "dupr": None if dd is None or dd["dupr"] != dd["dupr"] else {
                "v": _r(dd["dupr"], 3), "months": _r(dd["months"], 1), "lean": _r(dd["lean"], 3), "target": _r(dd["target"], 3)},
            "trail": trail.get(pid),
            "terms": b["terms"], "sum": b["sum"], "sessions": b["sessions"], "partners": pals,
            "reach": (reach := reach_of(pid)),
            "recs": recommend(float(t.level), played, [p["id"] for p in pals], played_by, S, C, vmap, reach, cfg),
        })
    used = {s["s"] for p in players for s in p["sessions"]} | {
        r["s"] for p in players for k in ("near", "partners") for r in p["recs"][k]}
    return players, {s: v for s, v in S.items() if s in used}
