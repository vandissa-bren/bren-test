"""
more_figures.py -- four more figures for Insights (27 Nov 2026), admin only.

  prices   price changes: weekly sessions whose price changed (and how full they
           ran before and after), and court-hire rate changes (court_rate_states)
  groups   social groups: players who deliberately play together (the "plays
           with" maths, pm_player.partners) joined into groups of 3 or more
  paths    the player path at each venue: of the players who first came in a
           period with 8 weeks to show what they'd do, how many came once, came
           back, became regulars, are still regulars, or lapsed (and whether
           they now play somewhere else)
  health   a health score for each weekly session: how full it runs, which way
           it's heading, whether first-timers come back, whether its regulars stay,
           each against the city's other weekly sessions; graded A-E by fifths

Privacy as everywhere (privacy.py): a count of players is 0 or 5+; a share of
players is published only when both the players it counts and the rest are 0
or 5+. Groups smaller than 5 are only ever counted, never described.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

import numpy as np
import pandas as pd

import figures as FG
from figures import DATA_START, MEL, MIN_GROUP, _hold, _r1

# price changes
PRICE_STEP = 0.5            # prices compared to the nearest 50c
PRICE_HOLD = 2              # the old and the new price each held for 2+ runs
PRICE_FILL_RUNS = 4         # fill compared over up to 4 runs either side
HIRE_HOLD_DAYS = 7          # a court-hire rate counts once it held a week
# groups
GROUP_MIN = 3
# paths
PATH_DAYS = 56              # 8 weeks to show what a first-timer does ...
PATH_DAYS_SHORT = 28        # ... 4 weeks while the data is younger than that (until ~21 Dec 2026)
PATH_ACTIVE_DAYS = 28       # "still" = played there in the last 4 weeks
REGULAR_SESSIONS = 4        # sessions in the 8 weeks to count as a regular (3 in 4 weeks)
# health
HEALTH_RUNS = 8
WEIGHTS = {"fill": 0.4, "trend": 0.2, "newcomers": 0.2, "regulars": 0.2}


def _ok_split(a: int, b: int) -> bool:
    return all(x == 0 or x >= MIN_GROUP for x in (a, b))


def _pct(a: int, n: int):
    return _r1(100 * a / n) if n and _ok_split(a, n - a) else None


def _day(ts) -> date:
    return ts.tz_convert(MEL).date()


# ── price changes ────────────────────────────────────────────────────────────
def prices(x: pd.DataFrame, rates, end_day: date) -> dict:
    out = {"city": {}, "venues": defaultdict(lambda: {"sessions": [], "hire": []})}
    lab = FG._labels(x.dropna(subset=["sid"]))
    changes = []
    for sid, g in x.dropna(subset=["sid"]).sort_values("starts_at").groupby("sid"):
        p = (g["price_n"] / PRICE_STEP).round() * PRICE_STEP
        p = p.where(g["price_n"].notna())
        g = g.assign(_p=p.to_numpy()).dropna(subset=["_p"])
        if len(g) < 2 * PRICE_HOLD:
            continue
        ps, fills, dates = g["_p"].to_numpy(), g["fill"].to_numpy(), g["_date"].to_numpy()
        for i in range(PRICE_HOLD, len(ps) - PRICE_HOLD + 1):
            if ps[i] == ps[i - 1]:
                continue
            if not (all(ps[i - k] == ps[i - 1] for k in range(1, PRICE_HOLD + 1))
                    and all(ps[i + k] == ps[i] for k in range(PRICE_HOLD) if i + k < len(ps))):
                continue
            before = fills[max(0, i - PRICE_FILL_RUNS):i]
            after = fills[i:i + PRICE_FILL_RUNS]
            l = lab.get(sid, {})
            changes.append({"v": int(g["fid"].iloc[0]), "sid": sid, "title": l.get("title"), "dow": l.get("dow"),
                            "hm": l.get("hm"), "date": str(dates[i]), "from": float(ps[i - 1]), "to": float(ps[i]),
                            "fillBefore": _r1(100 * np.mean(before)),
                            "fillAfter": _r1(100 * np.mean(after)) if len(after) >= 2 else None,
                            "runsAfter": int(len(after))})
    for c in sorted(changes, key=lambda c: c["date"], reverse=True):
        v = out["venues"][str(c.pop("v"))]
        if len(v["sessions"]) < 10:
            v["sessions"].append(c)
    up = [c for c in changes if c["to"] > c["from"]]
    down = [c for c in changes if c["to"] < c["from"]]
    shift = lambda cs: _r1(np.mean([c["fillAfter"] - c["fillBefore"] for c in cs if c["fillAfter"] is not None])) \
        if sum(c["fillAfter"] is not None for c in cs) >= 5 else None
    out["city"] = {"changes": len(changes), "up": len(up), "down": len(down),
                   "medianChangePct": _r1(np.median([100 * (c["to"] - c["from"]) / c["from"] for c in changes if c["from"]]))
                   if changes else None,
                   "fillShiftAfterRise": shift(up), "fillShiftAfterCut": shift(down)}

    # court hire: each state of a court's casual rate per PlayByPoint shift
    if rates is not None and len(rates):
        r = rates.dropna(subset=["price"]).copy()
        r["first_seen"] = pd.to_datetime(r["first_seen"], utc=True, errors="coerce", format="mixed")
        r = r.sort_values("first_seen")
        hire = defaultdict(int)
        r["last_seen"] = pd.to_datetime(r["last_seen"], utc=True, errors="coerce", format="mixed")
        for (fid, court, shift), g in r.groupby(["facility_id", "court_id", "shift"]):
            # a rate counts once it held a week: the log also catches the price
            # flickering for a day or two (a promo, a one-off), which isn't a change
            g = g[(g["last_seen"] - g["first_seen"]) >= pd.Timedelta(days=HIRE_HOLD_DAYS)]
            pr, ts = g["price"].to_numpy(), g["first_seen"].to_list()
            for i in range(1, len(pr)):
                if pr[i] != pr[i - 1] and _day(ts[i]) > DATA_START:
                    hire[(int(fid), str(shift), str(_day(ts[i])), float(pr[i - 1]), float(pr[i]))] += 1
        for (fid, shift, d, a, b), n in sorted(hire.items(), key=lambda kv: kv[0][2], reverse=True):
            v = out["venues"][str(fid)]
            if len(v["hire"]) < 10:
                v["hire"].append({"shift": shift, "date": d, "from": a, "to": b, "courts": n})
        out["city"]["hireChanges"] = len({(k[0], k[1], k[2]) for k in hire})
    out["venues"] = dict(out["venues"])
    return out


# ── social groups ────────────────────────────────────────────────────────────
def groups(x: pd.DataFrame, okv: pd.DataFrame, end_day: date) -> dict:
    import pm_player as PP
    out = {"city": {"groups": 0}, "venues": {}}
    sess = x.dropna(subset=["sid"]).drop_duplicates("session_key").set_index("session_key")[["sid", "fid"]]
    sess = sess.rename(columns={"sid": "series"})
    d = okv[okv["session_key"].isin(sess.index)][["pbp_user_id", "session_key"]].drop_duplicates()
    if d.empty:
        return out
    pairs = PP.partners(d, sess)
    if pairs.empty:
        return out
    # groups: players joined by "plays with" links (union-find)
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for i, j in zip(pairs["i"], pairs["j"]):
        ra, rb = find(i), find(j)
        if ra != rb:
            parent[ra] = rb
    members = defaultdict(set)
    for p in list(parent):
        members[find(p)].add(p)
    G = [m for m in members.values() if len(m) >= GROUP_MIN]
    lab = FG._labels(x.dropna(subset=["sid"]))
    weeks = max(1.0, (x["_date"].max() - x["_date"].min()).days / 7) if len(x) else 1.0
    active = d.groupby("pbp_user_id").size()
    active = set(active[active >= 3].index)
    in_group = set().union(*G) if G else set()
    by_venue = defaultdict(list)
    sizes = {"3-4": 0, "5-9": 0, "10+": 0}
    moving = 0
    for m in G:
        n = len(m)
        sizes["3-4" if n < 5 else "5-9" if n < 10 else "10+"] += 1
        mine = d[d["pbp_user_id"].isin(m)]
        together = mine.groupby("session_key")["pbp_user_id"].nunique()
        together = together[together >= 2].index
        tv = sess.loc[together, "fid"]
        venues = int(tv.nunique())
        moving += venues >= 2
        home = int(mine["session_key"].map(sess["fid"]).mode().iloc[0])
        top = sess.loc[together, "series"].mode()
        l = lab.get(top.iloc[0], {}) if len(top) else {}
        by_venue[home].append({"size": n, "perWeek": _r1(len(together) / weeks), "venues": venues,
                               "title": l.get("title"), "dow": l.get("dow"), "hm": l.get("hm")})
    out["city"] = {"groups": len(G), "sizes": sizes, "players": _hold(len(in_group)),
                   "shareOfActive": _pct(len(in_group & active), len(active)), "acrossVenues": moving}
    for fid, gs in by_venue.items():
        here = d[d["session_key"].map(sess["fid"]) == fid]
        regs = here.groupby("pbp_user_id").size()
        regs = set(regs[regs >= 3].index)
        n_in = len({p for m in G for p in m} & regs)
        out["venues"][str(fid)] = {
            "groups": len(gs), "players": _hold(sum(g["size"] for g in gs)),
            "shareOfRegulars": _pct(n_in, len(regs)),
            # only groups of 5+ are described; smaller ones are only counted
            "list": sorted([g for g in gs if g["size"] >= MIN_GROUP], key=lambda g: -g["size"])[:8],
        }
    return out


# ── the player path ──────────────────────────────────────────────────────────
STAGES = ("once", "cameBack", "regular", "stillRegular", "lapsed", "nowElsewhere")


def paths(okv: pd.DataFrame, firsts, end_day: date, tracked) -> dict:
    o = okv.dropna(subset=["fid"]).copy()
    o["fid"] = o["fid"].astype(int)
    o["d"] = o["starts_at"].dt.tz_convert(MEL).dt.date
    known = {}
    if firsts is not None and len(firsts):
        f = firsts.dropna(subset=["pbp_user_id", "first_date", "fid"])
        known = {(int(p), int(v)): pd.Timestamp(x).date() for p, v, x in zip(f["pbp_user_id"], f["fid"], f["first_date"])}
    # 8 weeks to show what a first-timer does, once there's a fortnight of first-timers
    # that old; till then 4 weeks (and 3 sessions make a regular)
    days = PATH_DAYS if end_day - timedelta(days=PATH_DAYS) >= FG.COHORT_FROM + timedelta(days=14) else PATH_DAYS_SHORT
    regular_n = REGULAR_SESSIONS if days == PATH_DAYS else 3
    last_day = end_day - timedelta(days=days)
    recent = end_day - timedelta(days=PATH_ACTIVE_DAYS)
    plays_recent = o[o["d"] > recent].groupby("pbp_user_id")["fid"].agg(set)
    out = {"days": days, "regularSessions": regular_n, "city": {}, "venues": {}}
    city = defaultdict(int)
    for fid, g in o.groupby("fid"):
        if fid not in set(tracked):
            continue
        cnt = defaultdict(int)
        for p, h in g.groupby("pbp_user_id"):
            first = min(h["d"])
            k = known.get((int(p), int(fid)))
            if k is not None and k < first:
                continue                    # played there before the window: not a first-timer
            if not (FG.COHORT_FROM <= first <= last_day):
                continue
            n = int((h["d"] <= first + timedelta(days=days)).sum())
            still = fid in plays_recent.get(p, set())
            cnt["firsts"] += 1
            if n == 1:
                cnt["once"] += 1
            else:
                cnt["cameBack"] += 1
            if n >= regular_n:
                cnt["regular"] += 1
                if still:
                    cnt["stillRegular"] += 1
                else:
                    cnt["lapsed"] += 1
                    if plays_recent.get(p):
                        cnt["nowElsewhere"] += 1
        if not cnt["firsts"]:
            continue
        for k, v in cnt.items():
            city[k] += v
        out["venues"][str(fid)] = _path_shares(cnt)
    out["city"] = _path_shares(city) if city.get("firsts") else {}
    return out


def _path_shares(c: dict) -> dict:
    n = c.get("firsts", 0)
    reg = c.get("regular", 0)
    return {"firsts": _hold(n),
            "once": _pct(c.get("once", 0), n), "cameBack": _pct(c.get("cameBack", 0), n),
            "regular": _pct(reg, n),
            # of those who became regulars
            "stillRegular": _pct(c.get("stillRegular", 0), reg), "lapsed": _pct(c.get("lapsed", 0), reg),
            "nowElsewhere": _pct(c.get("nowElsewhere", 0), c.get("lapsed", 0)),
            "regularsN": _hold(reg)}


# ── session health ───────────────────────────────────────────────────────────
def health(x: pd.DataFrame, okv: pd.DataFrame, firsts, end_day: date) -> dict:
    out = {"weights": WEIGHTS, "city": {}, "venues": {}}
    lab = FG._labels(x.dropna(subset=["sid"]))
    o = okv.dropna(subset=["fid"]).copy()
    o["d"] = o["starts_at"].dt.tz_convert(MEL).dt.date
    sid_of = dict(zip(x["session_key"], x["sid"]))
    o["sid"] = o["session_key"].map(sid_of)
    # each player's first session at each venue (in the window, unless known earlier)
    known = {}
    if firsts is not None and len(firsts):
        f = firsts.dropna(subset=["pbp_user_id", "first_date", "fid"])
        known = {(int(p), int(v)): pd.Timestamp(d).date() for p, v, d in zip(f["pbp_user_id"], f["fid"], f["first_date"])}
    fo = o.sort_values("starts_at").drop_duplicates(["pbp_user_id", "fid"])
    fo = fo[[known.get((int(p), int(v))) in (None, d) or known.get((int(p), int(v))) >= d
             for p, v, d in zip(fo["pbp_user_id"], fo["fid"], fo["d"])]]
    fo = fo[(fo["d"] >= FG.COHORT_FROM) & (fo["d"] <= end_day - timedelta(days=28))]
    venue_days = o.groupby(["pbp_user_id", "fid"])["d"].agg(sorted)
    grades = defaultdict(int)
    rows = []
    for sid, g in x.dropna(subset=["sid"]).sort_values("starts_at").groupby("sid"):
        if len(g) < 4 or not FG._current(g, end_day):
            continue
        if g["capacity"].median() <= 2:         # private lessons and assessments
            continue
        g = g.tail(HEALTH_RUNS)
        fid = int(g["fid"].iloc[0])
        parts = {"fill": float(100 * g["fill"].mean())}
        if len(g) >= 6:
            half = len(g) // 2
            delta = 100 * (g["fill"].iloc[half:].mean() - g["fill"].iloc[:half].mean())
            parts["trend"] = float(np.clip(50 + 2.5 * delta, 0, 100))
        else:
            delta = None
        # first-timers whose first session here was this one, back within 28 days
        nf = fo[fo["sid"] == sid]
        back = sum(any(fd < d2 <= fd + timedelta(days=28) for d2 in venue_days.get((p, v), []))
                   for p, v, fd in zip(nf["pbp_user_id"], nf["fid"], nf["d"]))
        new_pct = _pct(back, len(nf)) if len(nf) >= 5 else None
        if new_pct is not None:
            parts["newcomers"] = new_pct
        # its regulars in the first half: still coming in the second?
        keys = list(g["session_key"])
        early, late = set(keys[:len(keys) // 2]), set(keys[len(keys) // 2:])
        oo = o[o["session_key"].isin(keys)]
        regs = oo[oo["session_key"].isin(early)].groupby("pbp_user_id").size()
        regs = set(regs[regs >= 2].index)
        kept = len(regs & set(oo.loc[oo["session_key"].isin(late), "pbp_user_id"]))
        kept_pct = _pct(kept, len(regs)) if len(regs) >= 5 else None
        if kept_pct is not None:
            parts["regulars"] = kept_pct
        l = lab.get(sid, {})
        rows.append((str(fid), parts, {
            "sid": sid, "title": l.get("title"), "dow": l.get("dow"), "hm": l.get("hm"), "runs": int(len(g)),
            "fill": _r1(parts["fill"]), "trend": _r1(delta),
            "newcomersBack": new_pct, "regularsKept": kept_pct,
            "players": _hold(oo["pbp_user_id"].nunique())}))
    # each part as where it sits among the city's weekly sessions (percentile,
    # 0-100), so a 45% city doesn't grade most sessions E; the score is the
    # weighted mean of the parts it has; grades by fifths of the city
    by_part = {k: sorted(p[k] for _, p, _ in rows if k in p) for k in WEIGHTS}
    def pct_rank(k, v):
        a = by_part[k]
        return 100 * (np.searchsorted(a, v, "left") + np.searchsorted(a, v, "right")) / (2 * len(a))
    scores = []
    for fid, parts, row in rows:
        w = sum(WEIGHTS[k] for k in parts)
        row["score"] = round(sum(WEIGHTS[k] * pct_rank(k, v) for k, v in parts.items()) / w) if w else None
        scores.append(row["score"])
    cuts = np.percentile([x for x in scores if x is not None], [20, 40, 60, 80]) if scores else []
    for fid, parts, row in rows:
        sc = row["score"]
        row["grade"] = None if sc is None else "ABCDE"[4 - int(np.searchsorted(cuts, sc, "right"))]
        grades[row["grade"]] += 1
        out["venues"].setdefault(fid, []).append(row)
    for v in out["venues"].values():
        v.sort(key=lambda r: -(r["score"] or 0))
    out["city"] = {"grades": dict(grades), "sessions": len(rows)}
    return out


def build(f: pd.DataFrame, okv: pd.DataFrame, firsts, end_day, tracked, rates=None) -> dict:
    if f.empty:
        return {}
    end_day = FG._end_day(f, end_day)
    x = FG._series(f)
    o = okv.dropna(subset=["fid"]).copy()
    o["fid"] = o["fid"].astype(int)
    return {"prices": prices(x, rates, end_day), "groups": groups(x, o, end_day),
            "paths": paths(o, firsts, end_day, [int(t) for t in tracked]),
            "health": health(x, o, firsts, end_day)}
