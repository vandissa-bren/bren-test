"""
venue_pages.py -- the pictures on a venue's Insights page (18 Nov 2026).

Summary tab:
  trend      fill by week, the venue against the city (8 weeks)
  week       its week hour by hour: how full its sessions run (the page adds
             idle courts from courtFree)
  anchors    its weekly sessions: how full, and the share of their players
             who book nothing else at the venue (the quadrant)
  (the page composes "five things worth acting on" from these and the other
  figures after privacy.py has run, so no hidden number is written into a sentence)
Market tab:
  supply     kinds of session within LOCAL_KM: sessions a week against fill
  travel     what its regulars travel more than TRAVEL_KM for, and where
  prices     price per player-hour of its sessions against similar ones nearby
  share      its share of local bookings by kind, the 28 days before against the last 28
Also:
  newcomers  per weekly session, how many first-timers came back within 28 days

Everything that counts players is 5 or more, or 0 (privacy.py lists the paths;
the builders hold them back here too). A percentage of players is published
only when both the players it counts and the rest are 0 or 5+.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

import figures as FG
from figures import MEL, MIN_GROUP, DOW, _hold, _r1, ampm, kind_name

WEEKS = 8                   # trend, anchors, travel: the last 8 weeks
LOCAL_KM = 10               # "nearby" for supply, prices and share
TRAVEL_KM = 8               # "out of the area"
REGULAR_MIN = 3             # bookings at the venue in the 8 weeks, to be one of its regulars
ANCHOR_RUNS = 3             # a weekly session needs 3 runs in the 8 weeks
NEW_DAYS = 28               # newcomers: came back within 28 days
NEW_MIN = 10                # ... with 10+ first-timers
SUPPLY_MIN = 4              # a kind needs 4 sessions nearby in 28 days
SHARE_DAYS = 28
SHARE_MIN = 20              # local bookings of a kind in each period
TREND_MIN = 2               # sessions in a week, for a weekly fill
HOURS = range(6, 23)
COACHED = FG.COACHED_GROUPS


def _ok_split(a: int, b: int) -> bool:
    """A percentage a/(a+b) of players is safe when both parts are 0 or 5+."""
    return all(x == 0 or x >= MIN_GROUP for x in (a, b))


def _pct(a: int, n: int):
    return _r1(100 * a / n) if n and _ok_split(a, n - a) else None


def daypart(h: int) -> str:
    return "mornings" if h < 12 else ("afternoons" if h < 17 else "evenings")


def _kind(fg, lv) -> str:
    k = kind_name(fg, lv)
    return k[:1].upper() + k[1:]


def _prep(f: pd.DataFrame, end_day: date) -> pd.DataFrame:
    x = FG._series(f)
    x["_wk"] = [(end_day - d).days // 7 for d in x["_date"]]
    x["_k"] = [_kind(a, b) for a, b in zip(x["fgroup"], x["lvl"])]
    x["_part"] = [daypart(h) for h in x["_h"]]
    return x


# ── Summary ────────────────────────────────────────────────────────────────
def trend(x: pd.DataFrame, fid: int) -> dict:
    """Mean fill per week, oldest first (index 0 = 8 weeks ago)."""
    here, city = [], []
    for w in range(WEEKS - 1, -1, -1):
        a = x[x["_wk"] == w]
        h = a[a["fid"] == fid]
        here.append(_r1(100 * h["fill"].mean()) if len(h) >= TREND_MIN else None)
        city.append(_r1(100 * a["fill"].mean()) if len(a) >= TREND_MIN else None)
    return {"here": here, "city": city}


def week(x: pd.DataFrame, fid: int) -> list:
    """[dow, hour, mean fill %, sessions] for each hour it runs sessions in
    the 8 weeks (sessions are counted by the hour they start)."""
    h = x[(x["fid"] == fid) & x["_h"].isin(HOURS)]
    out = []
    for (d, hr), g in h.groupby(["_dow", "_h"]):
        out.append([int(d), int(hr), _r1(100 * g["fill"].mean()), int(len(g))])
    return out


def anchors(x: pd.DataFrame, okv: pd.DataFrame, fid: int, end_day: date) -> list:
    """Its current weekly sessions: average fill, players, and the share of
    them who booked nothing else at the venue in the 8 weeks."""
    w = x[(x["fid"] == fid)].dropna(subset=["sid"])
    if w.empty:
        return []
    lab = FG._labels(x[x["fid"] == fid].dropna(subset=["sid"]))
    sid_of = dict(zip(w["session_key"], w["sid"]))
    o = okv[(okv["fid"] == fid) & okv["session_key"].isin(set(x.loc[x["fid"] == fid, "session_key"]))]
    o = o.assign(sid=o["session_key"].map(sid_of).fillna("_other"))
    per = o.groupby("pbp_user_id")["sid"].agg(set)
    rows = []
    for sid, g in w.sort_values("starts_at").groupby("sid", sort=False):
        if len(g) < ANCHOR_RUNS or not FG._current(g, end_day):
            continue
        ids = o.loc[o["sid"] == sid, "pbp_user_id"].unique()
        n = len(ids)
        only = int(sum(1 for p in ids if per.get(p) == {sid}))
        l = lab.get(sid, {})
        rows.append({"sid": sid, "title": l.get("title") or str(g["title"].iloc[-1]),
                     "dow": l.get("dow", int(g["_dow"].iloc[-1])), "hm": l.get("hm", g["_hm"].iloc[-1]),
                     "fill": _r1(100 * g["fill"].mean()), "runs": int(len(g)),
                     "players": _hold(n), "only": _pct(only, n)})
    rows.sort(key=lambda r: -(r["players"] or 0))
    return rows[:14]


def newcomers(x: pd.DataFrame, okv: pd.DataFrame, firsts, fid: int, end_day: date) -> dict:
    """First-timers at the venue (their first booking there from 12 Oct, at
    least 28 days ago): how many came back to anything there within 28 days,
    for the venue and per weekly session they first came to."""
    out = {"days": NEW_DAYS, "venue": None, "n": None, "rows": []}
    o = okv[okv["fid"] == fid][["pbp_user_id", "session_key", "starts_at"]].copy()
    if o.empty:
        return out
    o["d"] = o["starts_at"].dt.tz_convert(MEL).dt.date
    first = o.sort_values("starts_at").drop_duplicates("pbp_user_id").set_index("pbp_user_id")
    if firsts is not None and not firsts.empty:
        # played there before the 8 weeks: not a first-timer
        fv = firsts[firsts["fid"] == fid].dropna(subset=["pbp_user_id", "first_date"])
        known = {int(p): pd.Timestamp(d).date() for p, d in zip(fv["pbp_user_id"], fv["first_date"])}
        first = first[[known.get(int(p)) is None or known[int(p)] >= d for p, d in zip(first.index, first["d"])]]
    last_ok = end_day - timedelta(days=NEW_DAYS)
    first = first[(first["d"] >= FG.COHORT_FROM) & (first["d"] <= last_ok)]
    if first.empty:
        return out
    days = o.groupby("pbp_user_id")["d"].apply(sorted)
    back = {p: any(r.d < d2 <= r.d + timedelta(days=NEW_DAYS) for d2 in days[p]) for p, r in first.iterrows()}
    nb = sum(back.values())
    out["n"], out["venue"] = _hold(len(first)), _pct(nb, len(first))
    sid_of = dict(zip(x["session_key"], x["sid"]))
    lab = FG._labels(x[x["fid"] == fid].dropna(subset=["sid"]))
    first = first.assign(sid=first["session_key"].map(sid_of))
    for sid, g in first.dropna(subset=["sid"]).groupby("sid"):
        if len(g) < NEW_MIN:
            continue
        k = sum(back[p] for p in g.index)
        l = lab.get(sid, {})
        out["rows"].append({"sid": sid, "title": l.get("title"), "dow": l.get("dow"), "hm": l.get("hm"),
                            "n": _hold(len(g)), "back": _pct(k, len(g))})
    out["rows"].sort(key=lambda r: (r["back"] is None, r["back"] if r["back"] is not None else 0))
    return out


# ── Market ─────────────────────────────────────────────────────────────────
def _near(km, fid, venues, limit) -> set:
    out = {fid}
    for v in venues:
        d = km(fid, v)
        if d == d and d <= limit:
            out.add(int(v))
    return out


def supply(x: pd.DataFrame, fid: int, near: set) -> list:
    """Kinds of session (format, level, part of day) within LOCAL_KM in the
    last 28 days: sessions a week and average fill; mine = it runs some."""
    a = x[x["fid"].isin(near) & (x["_wk"] < 4) & x["fgroup"].map(lambda g: isinstance(g, str) and g not in ("", "Unknown"))]
    rows = []
    for (k, part), g in a.groupby(["_k", "_part"]):
        if len(g) < SUPPLY_MIN:
            continue
        rows.append({"kind": k, "part": part, "perWeek": _r1(len(g) / 4), "fill": _r1(100 * g["fill"].mean()),
                     "mine": bool((g["fid"] == fid).any())})
    mine = [r for r in rows if r["mine"]]
    rest = sorted((r for r in rows if not r["mine"]), key=lambda r: -r["perWeek"])
    return mine + rest[:max(0, 14 - len(mine))]


def travel(x: pd.DataFrame, okv: pd.DataFrame, fid: int, km) -> list:
    """What its regulars (3+ bookings there in the 8 weeks) book more than
    TRAVEL_KM away: per kind and venue, players (5+), with its distance and
    whether this venue runs that kind itself."""
    o = okv[okv["starts_at"].notna()]
    reg = o[o["fid"] == fid].groupby("pbp_user_id").size()
    reg = set(reg[reg >= REGULAR_MIN].index)
    if not reg:
        return []
    kind = dict(zip(x["session_key"], x["_k"]))
    away = o[o["pbp_user_id"].isin(reg) & (o["fid"] != fid)].copy()
    away["k"] = away["session_key"].map(kind)
    away = away.dropna(subset=["k", "fid"])
    dist = {int(v): km(fid, int(v)) for v in away["fid"].unique()}
    away["km"] = away["fid"].astype(int).map(dist)
    away = away[away["km"] > TRAVEL_KM]
    mine = set(x.loc[x["fid"] == fid, "_k"])
    rows = []
    for (k, v), g in away.groupby(["k", "fid"]):
        n = g["pbp_user_id"].nunique()
        if n >= MIN_GROUP:
            rows.append({"kind": k, "v": int(v), "players": int(n), "km": _r1(g["km"].iloc[0]),
                         "runsIt": k in mine})
    rows.sort(key=lambda r: -r["players"])
    return rows[:6]


def prices(x: pd.DataFrame, fid: int, near: set) -> dict:
    """Price per player-hour of every weekly session nearby (each session's
    median over its runs), coaching left out, against its own."""
    a = x[x["fid"].isin(near) & ~x["fgroup"].isin(COACHED) & x["price_hr"].notna() & (x["price_hr"] > 0)].dropna(subset=["sid"])
    if a.empty:
        return {"others": [], "mine": [], "median": None}
    per = a.groupby("sid").agg(v=("fid", "first"), p=("price_hr", "median"), fill=("fill", "mean"), runs=("fid", "size"))
    others = per[(per["v"] != fid) & (per["runs"] >= 2)]
    lab = FG._labels(x[x["fid"] == fid].dropna(subset=["sid"]))
    mine = []
    for sid, r in per[per["v"] == fid].sort_values("runs", ascending=False).head(5).iterrows():
        l = lab.get(sid, {})
        mine.append({"title": l.get("title"), "dow": l.get("dow"), "hm": l.get("hm"),
                     "price": _r1(r["p"]), "fill": _r1(100 * r["fill"])})
    return {"others": sorted(_r1(p) for p in others["p"]), "mine": mine,
            "median": _r1(float(others["p"].median())) if len(others) else None}


def share(x: pd.DataFrame, okv: pd.DataFrame, fid: int, near: set, end_day: date) -> dict:
    """Its share of the bookings within LOCAL_KM, by kind: the 28 days before
    against the last 28. Bookings are not players, but a share is only shown
    where 5+ players (or none) booked it here in each period."""
    now0 = end_day - timedelta(days=SHARE_DAYS - 1)
    b0 = now0 - timedelta(days=SHARE_DAYS)
    out = {"from": str(b0), "mid": str(now0), "to": str(end_day), "ready": b0 >= FG.DATA_START, "rows": []}
    if not out["ready"]:
        return out
    kind = dict(zip(x["session_key"], x["_k"]))
    o = okv[okv["fid"].isin(near)].copy()
    o["k"] = o["session_key"].map(kind)
    o = o.dropna(subset=["k"])
    o["d"] = o["starts_at"].dt.tz_convert(MEL).dt.date
    o = o[(o["d"] >= b0) & (o["d"] <= end_day)]
    o["now"] = o["d"] >= now0
    rows = []
    for k, g in o.groupby("k"):
        here = g[g["fid"] == fid]
        if here.empty:
            continue
        cell = []
        for now in (False, True):
            gg, hh = g[g["now"] == now], here[here["now"] == now]
            ph = hh["pbp_user_id"].nunique()
            if len(gg) < SHARE_MIN or not (ph == 0 or ph >= MIN_GROUP):
                cell = None
                break
            cell.append(_r1(100 * len(hh) / len(gg)))
        if cell:
            rows.append({"kind": k, "before": cell[0], "now": cell[1], "local": int(len(g[g["now"]]))})
    rows.sort(key=lambda r: -r["local"])
    out["rows"] = [{k: v for k, v in r.items() if k != "local"} for r in rows[:6]]
    return out


# ── all of it ──────────────────────────────────────────────────────────────
def build(f: pd.DataFrame, okv: pd.DataFrame, firsts, end_day, km, tracked) -> dict:
    out = {"weeks": WEEKS, "localKm": LOCAL_KM, "travelKm": TRAVEL_KM, "venues": {}}
    if f.empty:
        return out
    end_day = FG._end_day(f, end_day)
    x = _prep(f, end_day)
    x = x[(x["_wk"] >= 0) & (x["_wk"] < WEEKS)]
    o = okv.dropna(subset=["fid"]).copy()
    o["fid"] = o["fid"].astype(int)
    o = o[o["starts_at"].dt.tz_convert(MEL).dt.date > end_day - timedelta(days=7 * WEEKS)]
    tracked = [int(v) for v in tracked]
    for fid in tracked:
        if not (x["fid"] == fid).any():
            continue
        near = _near(km, fid, [v for v in tracked if v != fid], LOCAL_KM)
        out["venues"][str(fid)] = {
            "trend": trend(x, fid), "week": week(x, fid), "anchors": anchors(x, o, fid, end_day),
            "newcomers": newcomers(x, o, firsts, fid, end_day),
            "supply": supply(x, fid, near), "travel": travel(x, o, fid, km),
            "prices": prices(x, fid, near), "share": share(x, o, fid, near, end_day)}
    out["onlySplit"] = _r1(float(np.median([a["only"] for v in out["venues"].values() for a in v["anchors"]
                                            if a["only"] is not None]))) if out["venues"] else None
    return out
