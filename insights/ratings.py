"""
ratings.py -- Insights › Ratings: how players' DUPRs move (build: "ratings").

From the rating log (player_rating_log: every change in a player's DUPR as the
rosters show it, from 19 Sep) and the played-roster re-reads
(player_rating_checks: each player's DUPR 1, 3, 7 and 14 days after a session,
from 8 Oct), with the window's rosters.

Only DUPR-style ratings count (two or three decimals that aren't a half step:
venue_report._verified). Self-ratings never read as a change.

AFTER A SESSION: for each DUPR player in each finished session, their DUPR on
the day (the roster's reading) against their DUPR about two weeks later: the
reading nearest 14 days after the start, between 7 and 21 days after (re-reads,
later rosters, or the log). A change in those two weeks reflects all their
matches then, not only that session; across many players the rest averages
out. Each player counts once per group (the mean of their changes there), so a
regular doesn't count ten times. Groups under 5 players are held back (None).

  since       first day of the rating log
  city        dupr: DUPR players in the period; changes / players / up / down /
              avgAbs: DUPR changes logged in the period
  weeks       [monday, changes, players changed, up, down]
  after       {"rated": [players, mean change, % up, % down], "other": [...]}:
              two weeks after DUPR-rated sessions vs other sessions
  room        [[band, players, mean change]]: by where the player sat against
              the other DUPR players in the session (their DUPR minus the room's)
  venues      {fid: {"all": [players, mean, % up, % down], "visitors": [n, mean],
                     "regulars": [n, mean], "prog": [n, median per 4 weeks]}}
  progression city: [players, median change per 4 weeks, % up, % down] for DUPR
              players read 14+ days apart in the period
  programmes  [{v, title, label, lo, hi, fmt, rated, sessions, players (DUPR),
                median, p25, p75, inside, after: [n, mean] | None}]: each
              programme's measured level (its DUPR players' ratings on the day)
  coaches     {coach key: {"after": [n, mean], "dupr": median}}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_GROUP = 5
EPS = 0.005                      # a change smaller than this is "no change"
ROOM_BANDS = [(-9, -0.3, "0.3+ below the room"), (-0.3, -0.1, "0.1–0.3 below"), (-0.1, 0.1, "about level"),
              (0.1, 0.3, "0.1–0.3 above"), (0.3, 9, "0.3+ above the room")]


def dupr(x):
    """A DUPR-style value, else NaN (the build's _verified rule, 2 to 8)."""
    v = pd.to_numeric(x, errors="coerce")
    c = (v * 1000).round()
    return v.where(v.between(2, 8) & (c % 500 != 0)).round(3)


def _hold(n):
    return int(n) if n >= MIN_GROUP else None


def _r(x, k=3):
    return None if x is None or x != x else round(float(x), k)


def _summary(ev: pd.DataFrame):
    """[players, mean change, % up, % down] with each player counted once."""
    if ev.empty:
        return None
    per = ev.groupby("pbp_user_id")["delta"].mean()
    if len(per) < MIN_GROUP:
        return None
    return [int(len(per)), _r(per.mean()), round(100 * float((per > EPS).mean()), 1), round(100 * float((per < -EPS).mean()), 1)]


def _pair(ev: pd.DataFrame):
    if ev.empty:
        return None
    per = ev.groupby("pbp_user_id")["delta"].mean()
    return [int(len(per)), _r(per.mean())] if len(per) >= MIN_GROUP else None


def observations(r: pd.DataFrame, log: pd.DataFrame, checks: pd.DataFrame) -> pd.DataFrame:
    """Every time we read a player's DUPR: roster sightings, re-reads and the log."""
    parts = []
    if r is not None and len(r):
        x = r[["pbp_user_id", "last_seen"]].assign(v=dupr(r["rating_at_time"])).rename(columns={"last_seen": "time"})
        parts.append(x)
    if checks is not None and len(checks):
        parts.append(pd.DataFrame({"pbp_user_id": checks["pbp_user_id"], "time": checks["observed_at"],
                                   "v": dupr(checks["rating"])}))
    if log is not None and len(log):
        parts.append(pd.DataFrame({"pbp_user_id": log["pbp_user_id"], "time": log["seen_at"], "v": dupr(log["rating"])}))
        pl = log.dropna(subset=["prev_seen_at", "prev_rating"])
        parts.append(pd.DataFrame({"pbp_user_id": pl["pbp_user_id"], "time": pl["prev_seen_at"], "v": dupr(pl["prev_rating"])}))
    if not parts:
        return pd.DataFrame(columns=["pbp_user_id", "time", "v"])
    o = pd.concat(parts, ignore_index=True)
    o["time"] = pd.to_datetime(o["time"], utc=True, errors="coerce", format="mixed").astype("datetime64[ns, UTC]")
    o = o.dropna(subset=["time", "v"])
    o["pbp_user_id"] = o["pbp_user_id"].astype("int64")
    return o.sort_values("time").drop_duplicates(["pbp_user_id", "time", "v"])


def events(okv: pd.DataFrame, f: pd.DataFrame, obs: pd.DataFrame, rated_keys: set) -> pd.DataFrame:
    """One row per DUPR player per finished session: before, after (about two
    weeks on), and the session's facts."""
    meta = f.set_index("session_key")
    e = okv[okv["session_key"].isin(meta.index)][["pbp_user_id", "session_key", "fid", "starts_at", "rating_at_time"]].copy()
    e["before"] = dupr(e["rating_at_time"])
    e = e.dropna(subset=["before"])
    if e.empty:
        return e.assign(after=np.nan, delta=np.nan)
    e["pbp_user_id"] = e["pbp_user_id"].astype("int64")
    e["target"] = (e["starts_at"] + pd.Timedelta(days=14)).astype("datetime64[ns, UTC]")
    e = e.sort_values("target")
    o = obs.rename(columns={"time": "at", "v": "after"}).sort_values("at")
    e = pd.merge_asof(e, o, left_on="target", right_on="at", by="pbp_user_id",
                      direction="nearest", tolerance=pd.Timedelta(days=7))
    e["delta"] = (e["after"] - e["before"]).round(3)
    for col in ("title", "label", "band_lo", "band_hi", "fgroup", "co_list"):
        e[col] = e["session_key"].map(meta[col]) if col in meta else None
    e["rated"] = e["session_key"].isin(rated_keys)
    # where the player sat: their DUPR minus the other DUPR players' mean (needs 2+ others)
    g = e.groupby("session_key")["before"]
    tot, n = g.transform("sum"), g.transform("count")
    e["room"] = np.where(n >= 3, e["before"] - (tot - e["before"]) / (n - 1).clip(lower=1), np.nan)
    return e


def build(okv, f, r, log, checks, start, end, home: pd.Series, rated_keys: set, coach_key=lambda n: n.lower()):
    if log is None:
        log = pd.DataFrame(columns=["pbp_user_id", "rating", "prev_rating", "seen_at", "prev_seen_at", "session_key", "source"])
    log = log.copy()
    for col in ("seen_at", "prev_seen_at"):
        log[col] = pd.to_datetime(log[col], utc=True, errors="coerce", format="mixed")
    if checks is not None and len(checks):
        checks = checks.copy()
        checks["observed_at"] = pd.to_datetime(checks["observed_at"], utc=True, errors="coerce", format="mixed")
    obs = observations(r, log, checks)
    ev = events(okv, f, obs, rated_keys)
    if not len(log) and ev.empty:
        return None
    a = ev.dropna(subset=["delta"]) if len(ev) else ev

    # ── changes logged in the period ──
    end_x = end + pd.Timedelta(days=1)
    ch = log[log["prev_rating"].notna() & (log["seen_at"] >= start) & (log["seen_at"] < end_x)].copy()
    ch["d"] = pd.to_numeric(ch["rating"]) - pd.to_numeric(ch["prev_rating"])
    ch["week"] = ch["seen_at"].dt.tz_convert("Australia/Melbourne").dt.date.map(lambda d: d - pd.Timedelta(days=d.weekday()))
    weeks = [[str(w), int(len(g)), int(g["pbp_user_id"].nunique()), int((g["d"] > 0).sum()), int((g["d"] < 0).sum())]
             for w, g in ch.groupby("week")]
    dupr_players = int(ev["pbp_user_id"].nunique()) if len(ev) else 0
    city = {"dupr": dupr_players, "changes": int(len(ch)), "players": int(ch["pbp_user_id"].nunique()),
            "up": int((ch["d"] > 0).sum()), "down": int((ch["d"] < 0).sum()),
            "avgAbs": _r(ch["d"].abs().mean()) if len(ch) else None,
            "measured": int(a["pbp_user_id"].nunique()) if len(a) else 0}

    after = {"rated": _summary(a[a["rated"]]) if len(a) else None,
             "other": _summary(a[~a["rated"]]) if len(a) else None}

    room = []
    if len(a):
        rr = a.dropna(subset=["room"])
        for lo, hi, lab in ROOM_BANDS:
            g = rr[(rr["room"] >= lo) & (rr["room"] < hi)]
            p = _pair(g)
            room.append([lab, p[0] if p else None, p[1] if p else None])

    # ── regulars' progression: first and last reading 14+ days apart in the period ──
    ob = obs[(obs["time"] >= start) & (obs["time"] < end_x)]
    prog = pd.DataFrame(columns=["rate"])
    if len(ob):
        g = ob.groupby("pbp_user_id")
        span = (g["time"].max() - g["time"].min()).dt.total_seconds() / 86400
        first = g.apply(lambda x: x.sort_values("time")["v"].iloc[0])
        last = g.apply(lambda x: x.sort_values("time")["v"].iloc[-1])
        prog = pd.DataFrame({"span": span, "rate": (last - first) / span.clip(lower=1) * 28})
        prog = prog[prog["span"] >= 14]
    progression = None
    if len(prog) >= MIN_GROUP:
        progression = [int(len(prog)), _r(prog["rate"].median()), round(100 * float((prog["rate"] > EPS).mean()), 1),
                       round(100 * float((prog["rate"] < -EPS).mean()), 1)]

    # ── venues ──
    venues = {}
    if len(ev):
        for fid, g in a.groupby("fid") if len(a) else []:
            hv = g["pbp_user_id"].map(home)
            venues[str(int(fid))] = {"all": _summary(g), "visitors": _pair(g[hv != fid]), "regulars": _pair(g[hv == fid])}
        if len(prog):
            ph = prog.assign(home=prog.index.map(home)).dropna(subset=["home"])
            for fid, g in ph.groupby("home"):
                v = venues.setdefault(str(int(fid)), {"all": None, "visitors": None, "regulars": None})
                v["prog"] = [int(len(g)), _r(g["rate"].median())] if len(g) >= MIN_GROUP else None
        venues = {k: v for k, v in venues.items() if any(v.get(x) for x in ("all", "visitors", "regulars", "prog"))}

    # ── programmes: measured level and movement ──
    programmes = []
    if len(ev):
        t = ev[ev["title"].notna()]
        for (fid, title), g in t.groupby(["fid", "title"]):
            per = g.groupby("pbp_user_id")["before"].median()
            if len(per) < MIN_GROUP:
                continue
            lo, hi = g["band_lo"].iloc[0], g["band_hi"].iloc[0]
            has = lo == lo and lo is not None
            programmes.append({
                "v": int(fid), "title": str(title),
                "label": g["label"].dropna().iloc[0] if g["label"].notna().any() else None,
                "lo": _r(lo, 2) if has else None, "hi": _r(min(hi, 6.0), 2) if has else None,
                "fmt": g["fgroup"].dropna().mode().iloc[0] if g["fgroup"].notna().any() else None,
                "rated": bool(g["rated"].any()), "sessions": int(g["session_key"].nunique()),
                "players": int(len(per)), "median": _r(per.median(), 2),
                "p25": _r(per.quantile(.25), 2), "p75": _r(per.quantile(.75), 2),
                "inside": round(100 * float(((per >= lo - 0.01) & (per <= hi + 0.01)).mean()), 1) if has else None,
                "after": _pair(g.dropna(subset=["delta"])),
            })
        programmes.sort(key=lambda x: (x["v"], x["median"]))

    # ── coaches ──
    coaches = {}
    if len(ev) and "co_list" in ev:
        rows = [(coach_key(n), i) for i, lst in ev["co_list"].items() if isinstance(lst, list) for n in lst]
        if rows:
            ck = pd.DataFrame(rows, columns=["k", "i"])
            for k, g in ck.groupby("k"):
                sub = ev.loc[g["i"].unique()]
                per = sub.groupby("pbp_user_id")["before"].median()
                coaches[k] = {"after": _pair(sub.dropna(subset=["delta"])),
                              "dupr": _r(per.median(), 2) if len(per) >= MIN_GROUP else None}
            coaches = {k: v for k, v in coaches.items() if v["after"] or v["dupr"]}

    since = log["seen_at"].min() if len(log) else None
    return {"since": str(since.tz_convert("Australia/Melbourne").date()) if since is not None and since == since else None,
            "city": city, "weeks": weeks, "after": after, "room": room, "progression": progression,
            "venues": venues, "programmes": programmes, "coaches": coaches}
