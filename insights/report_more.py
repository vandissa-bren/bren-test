"""
report_more.py -- the wider comparisons: how fast each kind of session fills,
how far each player group travels, the real rating range of each level at each
venue, and level x format / time-slot fill matrices. Given a venue, also that
venue against the market and its player-group tree.

    python insights/report_more.py               market-wide parts only
    python insights/report_more.py <facility_id> plus that venue's parts

The daily build (build_overview.py) calls main() with no venue. Same inputs and
rules as venue_report.py (it is imported). With a venue, other venues are
lettered A, B, C... by how much of the venue's players' play they take, so a
report can name them or not. Output: report_more.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import venue_report as R

HERE = Path(__file__).resolve().parent
MIN_GROUP = 5          # never show a player group smaller than this
MIN_RATINGS = 10       # rated places needed to show a venue's level range

FORMAT_GROUP = {"social": "Social & open play", "open_play": "Social & open play",
                "clinic": "Coaching & clinics", "coaching": "Coaching & clinics",
                "learn_to_play": "Learn to play", "round_robin": "Round robin",
                "league": "League & ladder", "ladder": "League & ladder", "match_play": "Match play"}
FORMAT_ORDER = ["Social & open play", "Coaching & clinics", "Learn to play", "Round robin",
                "League & ladder", "Match play"]
LEVEL_ORDER = ["beginner", "intermediate", "advanced", "all levels"]


def letters_for(ok, focus):
    """Letters for other venues, A first, by their share of the focus venue's
    players' sessions; venues none of them play at follow, by id."""
    ids = ok[ok["fid"] == focus]["pbp_user_id"].unique()
    rank = (ok[ok["pbp_user_id"].isin(ids) & (ok["fid"] != focus)]
            .groupby("fid").size().sort_values(ascending=False, kind="stable").index.astype(int).tolist())
    rest = sorted({int(f) for f in ok["fid"].dropna()} - set(rank) - {focus})
    out, n = {}, 0
    for f in rank + rest:
        k, s = n, ""
        while True:                      # A..Z, then AA, AB...
            s = chr(ord("A") + k % 26) + s
            k = k // 26 - 1
            if k < 0:
                break
        out[f] = s
        n += 1
    return out


def label(fid, focus, letters):
    """The focus venue by name; others by letter when there is a focus, else by name."""
    fid = int(fid)
    if focus is None:
        return R.name(fid)
    return R.name(fid) if fid == focus else letters.get(fid, str(fid))


def rating_class(x):
    if np.isnan(x):
        return "unrated"
    return "beginner" if x < 2.75 else "intermediate" if x < 3.5 else "advanced"


# ── 1. how fast sessions fill ────────────────────────────────────────────────
def fill_speed(fin, focus=None):
    """% full at 7 days, 3 days, 1 day before and at start, for sessions we
    watched from at least 7 days out (so every point is a real reading)."""
    w = fin[(fin["first_obs"] <= fin["starts_at"] - pd.Timedelta(days=7))
            & (fin[["spots_7d", "spots_3d", "spots_1d"]].le(fin["capacity"], axis=0).all(axis=1))].copy()
    for k in ("7d", "3d", "1d"):
        w[f"f{k}"] = 1 - w[f"spots_{k}"] / w["capacity"]
    w["fgroup"] = w["fmt"].map(FORMAT_GROUP)
    w["who"] = np.where(w["fid"] == focus, "venue", "market")

    def curve(g):
        return {"n": len(g), "d7": round(100 * g["f7d"].mean(), 1), "d3": round(100 * g["f3d"].mean(), 1),
                "d1": round(100 * g["f1d"].mean(), 1), "start": round(100 * g["fill"].mean(), 1),
                "sold_out": R.pct(int(g["sold_out"].sum()), len(g))}

    out = {"by_format": [], "by_level": [], "watched": len(w)}
    for fg in FORMAT_ORDER:
        for who in ("venue", "market"):
            g = w[(w["fgroup"] == fg) & (w["who"] == who)]
            if len(g):
                out["by_format"].append({"group": fg, "who": who, **curve(g)})
    for lv in LEVEL_ORDER:
        for who in ("venue", "market"):
            g = w[(w["level_class"] == lv) & (w["who"] == who)]
            if len(g):
                out["by_level"].append({"group": lv, "who": who, **curve(g)})
    # sell-out speed: hours before start that sold-out sessions filled (seen filling)
    so = fin[fin["so_h"].notna()].copy()
    so["fgroup"] = so["fmt"].map(FORMAT_GROUP)
    out["sellout_hours"] = []
    for fg in FORMAT_ORDER:
        for who, gg in (("venue", so[so["fid"] == focus]), ("market", so[so["fid"] != focus])):
            g = gg[gg["fgroup"] == fg]
            if len(g):
                out["sellout_hours"].append({"group": fg, "who": who, "n": len(g),
                                             "median_h": round(float(g["so_h"].median()), 1)})
    return out


# ── 2. fill matrices ─────────────────────────────────────────────────────────
def matrices(fin, focus=None):
    f = fin[fin["fid"].notna()].copy()
    f["fgroup"] = f["fmt"].map(FORMAT_GROUP)
    f["who"] = np.where(f["fid"] == focus, "venue", "market")
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
def level_ranges(m, focus=None, letters=None):
    ok = m[~m["cancelled"] & m["rating"].notna()].copy()
    ok["level_class"] = [R.level_class({"all_levels": False, "band": (lo, hi)}) if not np.isnan(lo) else None
                         for lo, hi in zip(ok["band_lo"], ok["band_hi"])]
    out = []
    for (fid, lv), g in ok[ok["level_class"].notna()].groupby(["fid", "level_class"]):
        if len(g) < MIN_RATINGS or g["pbp_user_id"].nunique() < MIN_GROUP:
            continue
        stated = g.groupby(["band_lo", "band_hi"]).size().sort_values(ascending=False)
        lo, hi = stated.index[0]
        out.append({"venue": label(fid, focus, letters or {}), "fid": int(fid), "level": lv, "n": len(g),
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


def travel(ok, p, focus=None):
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
    # the focus venue's own players with 2+ sessions anywhere
    if focus is not None:
        ids = ok[ok["fid"] == focus]["pbp_user_id"].unique()
        g = b[b["pbp_user_id"].isin(ids)]
        if g["pbp_user_id"].nunique() >= MIN_GROUP:
            out.append(dist(g, "venue", f"{R.name(focus)} players"))
    return {"players_2plus": int(len(p2)), "rows": out, "bands": [x[2] for x in bands]}


# ── 5. player-group tree for one venue ──────────────────────────────────────
def tree(ok, p, trav, focus):
    here = ok[ok["fid"] == focus]
    ids = here["pbp_user_id"].unique()
    f = here.groupby("pbp_user_id").size()
    allp = ok[ok["pbp_user_id"].isin(ids)]
    share = allp.assign(here=allp["fid"] == focus).groupby("pbp_user_id")["here"].mean()
    q = p.loc[ids]
    # level from each player's own median rating across all their sessions
    lv = q["level"].value_counts().to_dict()
    plays = q["plays"].value_counts().to_dict()
    home_km = q["home"].map(lambda h: 0.0 if h == focus else R.km(focus, h))
    where = {"home venue is this venue": int((q["home"] == focus).sum()),
             "home venue within 5 km": int(((q["home"] != focus) & (home_km <= 5)).sum()),
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


def main(focus=None):
    """Market-wide comparisons; with a focus venue id, also that venue's parts."""
    focus = None if focus is None else int(focus)
    s, r, c = R.load()
    R.NOW = max(s["last_obs"].max(), r["last_seen"].max())
    s = R.attach_titles(s, c)
    fin = R.finished(s)
    m = R.rosters(s, r)
    # Some venues list most of their players at exactly 5.0, beginners included:
    # a default, like 1.0 and 2.0. Where over half of a venue's ratings are
    # exactly 5.0, its 5.0s count as unrated.
    rated = m[m["rating"].notna()]
    five = rated.assign(f=rated["rating"] == 5.0).groupby("fid")["f"].mean()
    default5 = set(five[five > 0.5].index)
    m.loc[m["fid"].isin(default5) & (m["rating"] == 5.0), "rating"] = np.nan
    ok, p = player_table(m)
    letters = letters_for(ok, focus) if focus is not None else {}
    out = {"focus": focus, "fill_speed": fill_speed(fin, focus), "matrices": matrices(fin, focus),
           "level_ranges": level_ranges(m, focus, letters), "travel": travel(ok, p, focus)}
    if focus is not None:
        trav = R.travellers(m, s, focus)
        out["tree"] = tree(ok, p, trav, focus)
        out["letters"] = {v: k for k, v in letters.items()}
    R.WORK.mkdir(parents=True, exist_ok=True)
    (R.WORK / "report_more.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg is not None and not arg.isdigit():
        sys.exit("usage: python insights/report_more.py [facility_id]")
    o = main(arg)
    print(json.dumps(o, indent=1, default=str)[:12000])
