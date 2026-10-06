"""
report_more.py -- the wider comparisons for the Dink & Drive example report
(2 Oct 2026 request): how fast each kind of session fills, how far each player
group travels, the player-group tree, the real rating range of each level at
each venue, and level x format / time-slot fill matrices.

Same inputs and rules as report_dd.py (it is imported). Other venues are
lettered with the report's letters (A-F = the six sharing the most Dink & Drive
players, as in the lead-time and rivals charts); G onwards are the rest, in no
meaningful order. Output: report_more.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import report_dd as R

HERE = Path(__file__).resolve().parent
FID = R.FID
LETTERS = {597: "A", 1883: "B", 1461: "C", 1664: "D", 1714: "E", 1485: "F"}
MIN_GROUP = 5          # never show a player group smaller than this
MIN_RATINGS = 10       # rated places needed to show a venue's level range

FORMAT_GROUP = {"social": "Social & open play", "open_play": "Social & open play",
                "clinic": "Coaching & clinics", "coaching": "Coaching & clinics",
                "learn_to_play": "Learn to play", "round_robin": "Round robin",
                "league": "League & ladder", "ladder": "League & ladder", "match_play": "Match play"}
FORMAT_ORDER = ["Social & open play", "Coaching & clinics", "Learn to play", "Round robin",
                "League & ladder", "Match play"]
LEVEL_ORDER = ["beginner", "intermediate", "advanced", "all levels"]


def letter(fid, extra):
    fid = int(fid)
    if fid == FID:
        return "Dink & Drive"
    if fid in LETTERS:
        return LETTERS[fid]
    if fid not in extra:
        extra[fid] = chr(ord("G") + len(extra))
    return extra[fid]


def rating_class(x):
    if np.isnan(x):
        return "unrated"
    return "beginner" if x < 2.75 else "intermediate" if x < 3.5 else "advanced"


# ── 1. how fast sessions fill ────────────────────────────────────────────────
def fill_speed(fin):
    """% full at 7 days, 3 days, 1 day before and at start, for sessions we
    watched from at least 7 days out (so every point is a real reading)."""
    w = fin[(fin["first_obs"] <= fin["starts_at"] - pd.Timedelta(days=7))
            & (fin[["spots_7d", "spots_3d", "spots_1d"]].le(fin["capacity"], axis=0).all(axis=1))].copy()
    for k in ("7d", "3d", "1d"):
        w[f"f{k}"] = 1 - w[f"spots_{k}"] / w["capacity"]
    w["fgroup"] = w["fmt"].map(FORMAT_GROUP)
    w["who"] = np.where(w["fid"] == FID, "dd", "market")

    def curve(g):
        return {"n": len(g), "d7": round(100 * g["f7d"].mean(), 1), "d3": round(100 * g["f3d"].mean(), 1),
                "d1": round(100 * g["f1d"].mean(), 1), "start": round(100 * g["fill"].mean(), 1),
                "sold_out": R.pct(int(g["sold_out"].sum()), len(g))}

    out = {"by_format": [], "by_level": [], "watched": len(w)}
    for fg in FORMAT_ORDER:
        for who in ("dd", "market"):
            g = w[(w["fgroup"] == fg) & (w["who"] == who)]
            if len(g):
                out["by_format"].append({"group": fg, "who": who, **curve(g)})
    for lv in LEVEL_ORDER:
        for who in ("dd", "market"):
            g = w[(w["level_class"] == lv) & (w["who"] == who)]
            if len(g):
                out["by_level"].append({"group": lv, "who": who, **curve(g)})
    # sell-out speed: hours before start that sold-out sessions filled (seen filling)
    so = fin[fin["so_h"].notna()].copy()
    so["fgroup"] = so["fmt"].map(FORMAT_GROUP)
    out["sellout_hours"] = []
    for fg in FORMAT_ORDER:
        for who, gg in (("dd", so[so["fid"] == FID]), ("market", so[so["fid"] != FID])):
            g = gg[gg["fgroup"] == fg]
            if len(g):
                out["sellout_hours"].append({"group": fg, "who": who, "n": len(g),
                                             "median_h": round(float(g["so_h"].median()), 1)})
    return out


# ── 2. fill matrices ─────────────────────────────────────────────────────────
def matrices(fin):
    f = fin[fin["fid"].notna()].copy()
    f["fgroup"] = f["fmt"].map(FORMAT_GROUP)
    f["who"] = np.where(f["fid"] == FID, "dd", "market")
    f["slot"] = np.where(f["weekend"], "Weekend", "Weekday " + f["daypart"].map(
        {"early": "before 9am", "day": "9am–5pm", "evening": "after 5pm"}))
    cells_fl, cells_sl = [], []
    for (fg, lv, who), g in f[f["fgroup"].notna() & f["level_class"].notna()].groupby(["fgroup", "level_class", "who"]):
        cells_fl.append({"format": fg, "level": lv, "who": who, "n": len(g), "fill": round(100 * g["fill"].mean(), 1),
                         "venues": int(g["fid"].nunique())})
    for (sl, lv, who), g in f[f["level_class"].notna()].groupby(["slot", "level_class", "who"]):
        cells_sl.append({"slot": sl, "level": lv, "who": who, "n": len(g), "fill": round(100 * g["fill"].mean(), 1),
                         "venues": int(g["fid"].nunique())})
    return {"format_level": cells_fl, "slot_level": cells_sl}


# ── 3. level ranges by venue ────────────────────────────────────────────────
def level_ranges(m, extra):
    ok = m[~m["cancelled"] & m["rating"].notna()].copy()
    ok["level_class"] = [R.level_class({"all_levels": False, "band": (lo, hi)}) if not np.isnan(lo) else None
                         for lo, hi in zip(ok["band_lo"], ok["band_hi"])]
    out = []
    for (fid, lv), g in ok[ok["level_class"].notna()].groupby(["fid", "level_class"]):
        if len(g) < MIN_RATINGS or g["pbp_user_id"].nunique() < MIN_GROUP:
            continue
        stated = g.groupby(["band_lo", "band_hi"]).size().sort_values(ascending=False)
        lo, hi = stated.index[0]
        out.append({"venue": letter(fid, extra), "fid": int(fid), "level": lv, "n": len(g),
                    "players": int(g["pbp_user_id"].nunique()), "sessions": int(g["session_key"].nunique()),
                    "mean": round(float(g["rating"].mean()), 2), "median": float(g["rating"].median()),
                    "p25": float(g["rating"].quantile(.25)), "p75": float(g["rating"].quantile(.75)),
                    "p10": float(g["rating"].quantile(.10)), "p90": float(g["rating"].quantile(.90)),
                    "stated_lo": float(lo), "stated_hi": float(min(hi, 5.0)),
                    "inside": R.pct(int(((g["rating"] >= g["band_lo"] - .01) & (g["rating"] <= g["band_hi"] + .01)).sum()), len(g))})
    return sorted(out, key=lambda x: (LEVEL_ORDER.index(x["level"]), x["mean"]))


# ── 4. how far player groups travel ─────────────────────────────────────────
def player_table(m):
    ok = m[~m["cancelled"] & m["fid"].notna()].copy()
    n = ok.groupby(["pbp_user_id", "fid"]).size().reset_index(name="n")
    home = n.sort_values(["pbp_user_id", "n"], ascending=[True, False]).drop_duplicates("pbp_user_id")
    home = home.set_index("pbp_user_id")["fid"]
    ok["home"] = ok["pbp_user_id"].map(home)
    pairs = {(a, b): R.km(a, b) for a, b in set(zip(ok["home"], ok["fid"]))}
    ok["trip_km"] = [0.0 if a == b else pairs[(a, b)] for a, b in zip(ok["home"], ok["fid"])]
    p = ok.groupby("pbp_user_id").agg(sessions=("session_key", "size"), venues=("fid", "nunique"),
                                      rating=("rating", "median"), home=("home", "first"),
                                      max_km=("trip_km", "max"), mean_km=("trip_km", "mean"))
    fam = (ok.dropna(subset=["family"]).groupby(["pbp_user_id", "family"]).size().reset_index(name="k")
             .sort_values(["pbp_user_id", "k"], ascending=[True, False]))
    top = fam.drop_duplicates("pbp_user_id").set_index("pbp_user_id")
    nfam = fam.groupby("pbp_user_id")["family"].nunique()
    share = fam.merge(fam.groupby("pbp_user_id")["k"].sum().rename("t"), on="pbp_user_id")
    share = share.assign(s=share["k"] / share["t"]).sort_values("s", ascending=False).drop_duplicates("pbp_user_id").set_index("pbp_user_id")["s"]
    p["plays"] = [("mixed" if share.get(i, 1) < 0.6 else top.loc[i, "family"]) if i in top.index else "unknown"
                  for i in p.index]
    p["level"] = p["rating"].map(rating_class)
    p["freq"] = pd.cut(p["sessions"], [0, 1, 2, 4, 999], labels=["1 session", "2 sessions", "3–4 sessions", "5+ sessions"]).astype(str)
    return ok, p


def travel(ok, p):
    """Each booking's distance from the player's home venue (the venue they play
    at most), for players with 2+ sessions -- one session has no trip."""
    p2 = p[p["sessions"] >= 2]
    b = ok[ok["pbp_user_id"].isin(p2.index)].merge(p2[["level", "plays", "freq"]], left_on="pbp_user_id", right_index=True)
    b = b[b["trip_km"].notna()]
    bands = [(-1, 0.001, "At home venue"), (0.001, 5, "Under 5 km"), (5, 10, "5–10 km"), (10, 20, "10–20 km"), (20, 999, "Over 20 km")]

    def dist(g, key, val):
        tot = len(g)
        row = {"dim": key, "group": val, "players": int(g["pbp_user_id"].nunique()), "bookings": tot}
        for lo, hi, lab in bands:
            row[lab] = R.pct(int(((g["trip_km"] > lo) & (g["trip_km"] <= hi)).sum()), tot)
        pl = p2.loc[g["pbp_user_id"].unique()]
        row["travelled_5km_share"] = R.pct(int((pl["max_km"] > 5).sum()), len(pl))
        away = g[g["trip_km"] > 0]
        row["away_median_km"] = None if away.empty else round(float(away["trip_km"].median()), 1)
        return row

    out = [dist(b, "all", "All players with 2+ sessions")]
    for lv in ("beginner", "intermediate", "advanced", "unrated"):
        g = b[b["level"] == lv]
        if g["pbp_user_id"].nunique() >= MIN_GROUP:
            out.append(dist(g, "level", lv))
    for pl in ("social", "learning", "competitive", "mixed"):
        g = b[b["plays"] == pl]
        if g["pbp_user_id"].nunique() >= MIN_GROUP:
            out.append(dist(g, "plays", pl))
    for fq in ("2 sessions", "3–4 sessions", "5+ sessions"):
        g = b[b["freq"] == fq]
        if g["pbp_user_id"].nunique() >= MIN_GROUP:
            out.append(dist(g, "freq", fq))
    # Dink & Drive's own players with 2+ sessions anywhere
    dd_ids = ok[ok["fid"] == FID]["pbp_user_id"].unique()
    g = b[b["pbp_user_id"].isin(dd_ids)]
    out.append(dist(g, "dd", "Dink & Drive players"))
    return {"players_2plus": int(len(p2)), "rows": out, "bands": [x[2] for x in bands]}


# ── 5. player-group tree for Dink & Drive ───────────────────────────────────
def tree(ok, p, trav):
    dd = ok[ok["fid"] == FID]
    ids = dd["pbp_user_id"].unique()
    f = dd.groupby("pbp_user_id").size()
    allp = ok[ok["pbp_user_id"].isin(ids)]
    share = allp.assign(here=allp["fid"] == FID).groupby("pbp_user_id")["here"].mean()
    q = p.loc[ids]
    # level from each player's own median rating across all their sessions
    lv = q["level"].value_counts().to_dict()
    plays = q["plays"].value_counts().to_dict()
    home_km = q["home"].map(lambda h: 0.0 if h == FID else R.km(FID, h))
    where = {"home venue is Dink & Drive": int((q["home"] == FID).sum()),
             "home venue within 5 km": int(((q["home"] != FID) & (home_km <= 5)).sum()),
             "home venue over 5 km away": int((home_km > 5).sum())}
    # combined segments: how often x where else
    seg = pd.DataFrame({"freq": f, "share": share})
    segments = {
        "Core regulars (3+ visits, over 80% of their play here)": int(((seg["freq"] >= 3) & (seg["share"] > .8)).sum()),
        "Occasional locals (1–2 visits, over 80% here)": int(((seg["freq"] <= 2) & (seg["share"] > .8)).sum()),
        "Shared players (half or more of their play elsewhere)": int((seg["share"] <= .5).sum()),
    }
    segments["Everyone else (between)"] = int(len(seg) - sum(segments.values()))
    return {"players": int(len(ids)),
            "how_often": {"1 visit": int((f == 1).sum()), "2 visits": int((f == 2).sum()),
                          "3–4 visits": int(f.between(3, 4).sum()), "5+ visits": int((f >= 5).sum())},
            "where_else": {"over 80% here": int((share > .8).sum()), "50–80% here": int(share.between(.5, .8).sum()),
                           "under 50% here": int((share < .5).sum())},
            "level": lv, "plays": plays, "home": where, "travellers": trav["classes"],
            "segments": segments}


def main():
    s, r, c = R.load()
    R.NOW = max(s["last_obs"].max(), r["last_seen"].max())
    s = R.attach_titles(s, c)
    fin = R.finished(s)
    m = R.rosters(s, r)
    # One venue lists most of its players at exactly 5.0 (227 of 271 ratings),
    # beginners included: a default, like 1.0 and 2.0. Where over half of a
    # venue's ratings are exactly 5.0, its 5.0s count as unrated.
    rated = m[m["rating"].notna()]
    five = rated.assign(f=rated["rating"] == 5.0).groupby("fid")["f"].mean()
    default5 = set(five[five > 0.5].index)
    m.loc[m["fid"].isin(default5) & (m["rating"] == 5.0), "rating"] = np.nan
    extra = {}
    ok, p = player_table(m)
    trav = R.travellers(m, s)
    out = {"fill_speed": fill_speed(fin), "matrices": matrices(fin),
           "level_ranges": level_ranges(m, extra), "travel": travel(ok, p),
           "tree": tree(ok, p, trav), "letters_extra": {v: k for k, v in extra.items()}}
    R.WORK.mkdir(parents=True, exist_ok=True)
    (R.WORK / "report_more.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    o = main()
    print(json.dumps(o, indent=1, default=str)[:12000])
