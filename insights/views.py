"""
views.py -- the restructured insights page's own views (redesign, 9 Nov 2026).

Built from the same frames build_overview.main() already has, so every figure
here uses the build's one definition of each thing (GLOSSARY):

  shares(okv, ids)            venues that share players, both directions,
                              with how that compares with chance
  shared_regulars(...)        weekly sessions that share a regular crowd
                              (network.py calls it; it holds the series)
  digest(f, okv, ...)         "what changed": the last 7 days against the
                              4 weeks before, in plain sentences
  scorecard(f, okv, ...)      per venue and city: the five headline figures
                              with six weeks of weekly values
  hire_takings(d)             hire takings per court-hour, per venue
  scorecard_changes(d)        the scorecard's change against the period before
  GLOSSARY                    the page's one definition of every figure

Players are only ever counts of 5 or more. The builders here already hold
small counts back, and privacy.enforce checks every path again (privacy.py).
"""
from __future__ import annotations

from collections import Counter
from datetime import timedelta

import numpy as np
import pandas as pd

MIN_GROUP = 5
MEL = "Australia/Melbourne"
DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DOW3 = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# came back: played again at the same venue within this many days (one rule
# for every "came back" figure; 14 so a 28-day window still has a base)
CAME_BACK_DAYS = 14
# sold out early: full at least this many hours before the start
EARLY_H = 48
# a regular of a weekly session: at least this many visits to it in the window
REGULAR_VISITS = 2
# sessions that share regulars: at least this many shared, and this share of
# the smaller session's regulars
SHARED_MIN = MIN_GROUP
SHARED_PCT = 15
# digest: changes of at least this many points (rates) or this % (counts)
DIGEST_POINTS = 10
DIGEST_PCT = 20
DIGEST_MAX = 5
DIGEST_VENUE_MAX = 3
DIGEST_MIN_SESSIONS = 5
# a count is only compared with a baseline that's big enough to mean something:
# at least this many a week over the 4 weeks before
DIGEST_MIN_BASE = {"sessions": 8, "players": 20}
# above this rise a percentage misleads ("up 700%"): say the two counts instead
DIGEST_PCT_CAP = 100

GLOSSARY = {
    "players": {"name": "Players",
                "definition": "Different people booked into a finished session in the period, not counting cancellations. Shown only when 5 or more."},
    "newPlayers": {"name": "New players",
                   "definition": "Players whose first session we've ever seen falls in the period. Rosters start on 19 Sep 2026, so early on this includes people who already played."},
    "newHere": {"name": "New here",
                "definition": "Players whose first session at this venue falls in the period."},
    "cameBack": {"name": "Came back",
                 "definition": f"Of players whose first booking at the venue in the period left at least {CAME_BACK_DAYS} days to look, the share who booked there again within {CAME_BACK_DAYS} days."},
    "newCameBack": {"name": "New players who came back",
                    "definition": f"Of new players with at least {CAME_BACK_DAYS} days left in the period, the share who booked again within {CAME_BACK_DAYS} days."},
    "mostlyHere": {"name": "Do most of their playing here",
                   "definition": "Share of a venue's players who made more than 80% of their bookings in the period at that venue."},
    "travel5": {"name": "Travel 5 km or more",
                "definition": "Share of a venue's players whose own most-played venue is more than 5 km away."},
    "sessionLevel": {"name": "Session level",
                     "definition": "The level a session states in its title or listing: beginner, intermediate, advanced, all levels, or no level stated."},
    "playerLevel": {"name": "Player level",
                    "definition": "A player's typical (median) rating in the period, in bands: beginner under 2.75, intermediate 2.75 to 3.49, advanced 3.5 and up. Players without a usable rating are unrated."},
    "rating": {"name": "Rating",
               "definition": "The median of each player's own median rating. Only shown with enough rated players."},
    "fill": {"name": "Fill",
             "definition": "Places booked at the start as a share of places offered, averaged over sessions."},
    "soldOut": {"name": "Sold out",
                "definition": "Share of sessions with no places left at the start."},
    "soldEarly": {"name": "Sold out 2+ days early",
                  "definition": f"Share of sessions that were full at least {EARLY_H} hours before they started. Sessions already full when first listed aren't counted, as we can't tell when they filled."},
    "sessionsPerWeek": {"name": "Sessions a week",
                        "definition": "Finished sessions divided by the weeks the period covers."},
    "takings": {"name": "Takings",
                "definition": "Listed price times places booked. An upper estimate: members may pay less."},
    "emptyValue": {"name": "Empty places worth",
                   "definition": "Listed price times places left empty at the start."},
    "pricePerPlayerHour": {"name": "Price per player-hour",
                           "definition": "What one player pays for an hour of a session: the session's listed price divided by its length. The median over sessions."},
    "sessionTakingsPerCourtHour": {"name": "Session takings per court-hour",
                                   "definition": "What sessions took per court-hour they used: takings divided by length times courts. Courts are estimated from the format (5 players a court for social and open play, 8 for coaching, else 4)."},
    "sessionTakingsPerCourtHourIfFull": {"name": "Session takings per court-hour if full",
                                         "definition": "The same, if every place had been booked."},
    "hireRatePerCourtHour": {"name": "Court hire rate",
                             "definition": "The casual price to hire one court for an hour, at the same venue, day and time as the sessions. The median."},
    "hireTakingsPerCourtHour": {"name": "Hire takings per court-hour",
                                "definition": "The casual hire rate times the share of hireable court-hours that were hired. An estimate: members pay less."},
    "sharesPlayers": {"name": "Shares players",
                      "definition": "Of the venue's players who came twice or more, the share also seen at the other venue in the period. Pairs with fewer than 5 shared players are hidden."},
    "vsChance": {"name": "Compared with chance",
                 "definition": "Shared players divided by how many we'd expect if this venue's players were as likely to turn up at the other venue as any other player who came twice or more. 2.0 means twice as many as that."},
    "regular": {"name": "Regular",
                "definition": f"A player with at least {REGULAR_VISITS} visits to the same weekly session in the period."},
    "weeklySession": {"name": "Weekly session",
                      "definition": "The same session week to week: same venue and weekday, and the same programme or a similar title, at about the same time. League and competition fixtures aren't weekly sessions."},
    "sharedRegulars": {"name": "Sessions that share regulars",
                       "definition": f"Two weekly sessions with at least {SHARED_MIN} regulars in common, who make up at least {SHARED_PCT}% of the smaller session's regulars."},
    "cancelled": {"name": "Dropped out",
                  "definition": "A booking counts as cancelled only if a check at least 30 minutes before the start no longer found the player on the list."},
    # the seven figures (figures.py)
    "needCourt": {"name": "Sessions that need another court",
                  "definition": "Weekly sessions still running (a run in the last 10 days) that, in 3 of their last 4 weeks, were full at least 2 days before the start (even if a place freed up later) or had a waitlist of 2 or more seen at least a day before the start. Needs 4 weeks, 3 of them watched from 2 days out."},
    "quietRegulars": {"name": "Regulars who've gone quiet",
                      "definition": "Players who played at the venue in 4 of the 6 weeks up to their last visit, and haven't booked there for over twice their usual gap (at least 14 days). \"Booked at another tracked venue\" means one of the venues we follow; we can't see club or private play. Holiday weeks push this up."},
    "cohortBack": {"name": "New players coming back",
                   "definition": "Players grouped by the week of their first session (anywhere, or at this venue), from the week of 12 Oct. The share who booked again within 1, 2, 4 and 8 weeks, shown once that time has passed for the whole group. Weeks with fewer than 20 new players join their month."},
    "sameFaces": {"name": "Same faces each week",
                  "definition": "Of a weekly session's players, the share who also played the week before. The middle value of its last 4 such weeks, each with 8 to 50 players; a week after a skipped week isn't compared. Low isn't bad: beginner sessions should turn over."},
    "whereToRun": {"name": "Where to run a new session",
                   "definition": "A kind of session at a day and time. Medium: at least 2 courts free in 60% of the weeks read; no session of its own of that format, or under 70% full, within 90 minutes (its sessions in the last 4 weeks); and either its players who came twice or more already play that kind elsewhere then (5 or more), or a similar session within 6 km sells out early or runs in the top quarter for fill. High: both reasons, over 6 or more weeks of court records."},
    "refilled": {"name": "Freed places refilled",
                 "definition": "Of sessions that had a waitlist, the share where a place freed after the session sold out was taken again before the start: a new player appeared after it sold out, and more people were on the list over time than it has places. We can't tell whether they came off the waitlist. Shown with 10 or more such sessions."},
    "leaning": {"name": "Leaning on a few players",
                "definition": "The share of a venue's bookings over the last 28 days made by its busiest 20% of players. Shown with 50 or more players. Includes coaches booked as players: the lists don't say who's coaching."},
    "cutOrMove": {"name": "Sessions to cut or move",
                  "definition": "Weekly sessions still running that were under 40% full in 3 of their last 4 weeks, the latest included. A session that started after our data did isn't judged on its first 4 weeks. Where a weekly session of the same kind within 6 km at the same day and time fills 20 or more points better, it's named."},
}


def _r1(x):
    return None if x is None or x != x else round(float(x), 1)


def early_rate(g: pd.DataFrame):
    """% of sessions full at least EARLY_H hours before the start (GLOSSARY
    soldEarly). Sessions already full when we first saw them have no sell-out
    time: they're left out of the count altogether, not counted as "not early"."""
    if g is None or g.empty:
        return None
    if "sold_out" in g:
        x = g[~(g["sold_out"].fillna(False).astype(bool) & g["so_h"].isna())]
    else:
        x = g
    if x.empty:
        return None
    return 100 * float((x["so_h"] >= EARLY_H).mean())


def _hold(n):
    n = int(n)
    return n if n >= MIN_GROUP or n == 0 else None


# ── venues that share players ───────────────────────────────────────────────
def shares(okv: pd.DataFrame, ids) -> dict:
    """Both directions, A -> B:
      base     A's players with 2+ sessions at A in the period
      shared   of those, players seen at B at least once
      share    shared / base, %
      chance   shared / (base x the share of other players with 2+ sessions
               who were seen at B)
    rows: [a, b, shared, share, chance]; pairs with shared < 5 left out.
    base: {fid: n} (5+ only)."""
    ids = set(int(i) for i in ids)
    x = okv[okv["fid"].isin(ids)]
    if x.empty:
        return {"cols": ["a", "b", "shared", "share", "chance"], "rows": [], "base": {}, "players": 0}
    per = x.groupby(["pbp_user_id", "fid"]).size()
    seen = per.reset_index().groupby("fid")["pbp_user_id"].apply(set)
    total = x["pbp_user_id"].nunique()
    regs = per[per >= 2].reset_index().groupby("fid")["pbp_user_id"].apply(set)
    # chance: how often a player who plays as much as A's base (2+ sessions in
    # the period, anywhere) is seen at B. Against every player, keen players
    # would read above chance everywhere.
    keen = set(x.groupby("pbp_user_id").size().loc[lambda n: n >= 2].index)
    rows, base = [], {}
    for a, ra in regs.items():
        if len(ra) < MIN_GROUP:
            continue
        base[str(int(a))] = len(ra)
        for b, sb in seen.items():
            if b == a:
                continue
            n = len(ra & sb)
            if n < MIN_GROUP:
                continue
            pool = keen - ra                 # keen players other than A's own base
            q = len(pool & sb) / len(pool) if pool else 0
            exp = len(ra) * q
            rows.append([int(a), int(b), n, round(100 * n / len(ra), 1),
                         round(n / exp, 2) if exp > 0 else None])
    rows.sort(key=lambda r: (r[0], -r[2]))
    return {"cols": ["a", "b", "shared", "share", "chance"], "rows": rows, "base": base, "players": int(total)}


# ── sessions that share regulars ────────────────────────────────────────────
def series_label(s: dict, vname) -> str:
    h, m = (int(p) for p in str(s["hm"]).split(":")[:2])
    t = f"{(h % 12) or 12}{'' if m == 0 else f':{m:02d}'}{'am' if h < 12 else 'pm'}"
    return f"{vname(s['v'])} · {DOW3[int(s['dow'])]} {t} · {s['t']}"


def shared_regulars(series: list, ser_players: dict, vname, top=20, per_venue=5) -> dict:
    """series: network.py's published series (j, v, t, dow, hm ...).
    ser_players: {j: Counter(player -> visits)}.
    pairs: [{a, b, shared, pct, label_a, label_b}], pct = shared / the smaller
    session's regulars. City: the top `top` by shared; venues: {fid: top
    `per_venue` pairs with either session there}."""
    regs = {j: {p for p, c in pc.items() if c >= REGULAR_VISITS} for j, pc in ser_players.items()}
    regs = {j: r for j, r in regs.items() if len(r) >= SHARED_MIN}
    by_j = {s["j"]: s for s in series}
    js = sorted(regs)
    pairs = []
    for i, a in enumerate(js):
        ra = regs[a]
        for b in js[i + 1:]:
            rb = regs[b]
            n = len(ra & rb)
            if n < SHARED_MIN:
                continue
            pct = 100 * n / min(len(ra), len(rb))
            if pct < SHARED_PCT:
                continue
            sa, sb = by_j.get(a), by_j.get(b)
            if not sa or not sb:
                continue
            pairs.append({"a": int(a), "b": int(b), "shared": int(n), "pct": round(pct, 1),
                          "labelA": series_label(sa, vname), "labelB": series_label(sb, vname),
                          "va": int(sa["v"]), "vb": int(sb["v"])})
    pairs.sort(key=lambda p: (-p["shared"], -p["pct"], p["a"], p["b"]))
    venues = {}
    for p in pairs:
        for v in {p["va"], p["vb"]}:
            lst = venues.setdefault(str(v), [])
            if len(lst) < per_venue:
                lst.append(p)
    return {"city": pairs[:top], "venues": venues}


# ── what changed ────────────────────────────────────────────────────────────
def _periods(f: pd.DataFrame):
    day = f["starts_at"].dt.tz_convert(MEL).dt.date
    end = day.max()
    start = day.min()
    last0 = end - timedelta(days=6)
    base0, base1 = last0 - timedelta(days=28), last0 - timedelta(days=1)
    return day, start, end, last0, base0, base1


def _measure(g: pd.DataFrame, gb: pd.DataFrame, players_now=None, players_base=None):
    """Rates in points, counts per week."""
    out = {}
    if len(g) >= DIGEST_MIN_SESSIONS and len(gb) >= DIGEST_MIN_SESSIONS:
        out["fill"] = (100 * g["fill"].mean(), 100 * gb["fill"].mean())
        out["soldEarly"] = (early_rate(g), early_rate(gb))
    out["sessions"] = (len(g), len(gb) / 4)
    if players_now is not None:
        out["players"] = (players_now, players_base)
    return out


def _items(meas: dict):
    """Candidate sentences from a measure dict, with a score for ranking."""
    out = []
    for k, (now, before) in meas.items():
        if now is None or before is None:
            continue
        if k in ("fill", "soldEarly"):
            diff = now - before
            if abs(diff) < DIGEST_POINTS:
                continue
            up = diff > 0
            if k == "fill":
                text = f"Average fill {'rose' if up else 'fell'} to {now:.0f}%, from {before:.0f}% over the 4 weeks before."
            else:
                text = (f"{now:.0f}% of sessions sold out at least 2 days early, "
                        f"{'up' if up else 'down'} from {before:.0f}%.")
            out.append({"metric": k, "dir": "up" if up else "down", "now": round(now, 1),
                        "before": round(before, 1), "text": text, "score": abs(diff) / DIGEST_POINTS})
        else:
            if before <= 0 or now < MIN_GROUP or before < max(MIN_GROUP, DIGEST_MIN_BASE.get(k, MIN_GROUP)):
                continue
            pct = 100 * (now - before) / before
            if abs(pct) < DIGEST_PCT or abs(now - before) < MIN_GROUP:
                continue
            up = pct > 0
            noun = "sessions ran" if k == "sessions" else "players booked"
            if pct > DIGEST_PCT_CAP:
                text = (f"{int(round(now))} {noun} in the last 7 days, against about {int(round(before))} "
                        f"a week over the 4 weeks before.")
            else:
                text = (f"{int(round(now))} {noun} in the last 7 days, {'up' if up else 'down'} "
                        f"{abs(pct):.0f}% on the weekly average of {int(round(before))}.")
            item = {"metric": k, "dir": "up" if up else "down", "now": int(round(now)),
                    "before": int(round(before)), "text": text, "score": min(abs(pct), 200) / DIGEST_PCT}
            out.append(item)
    return out


def _every_week(days, b0) -> bool:
    """True when something ran in each of the 4 baseline weeks: a venue that
    only started partway through has no fair "4 weeks before"."""
    if days is None or len(days) == 0:
        return False
    wk = {(d - b0).days // 7 for d in days}
    return {0, 1, 2, 3} <= wk


def _weekly_players(okv, day_of, b0, b1):
    """Average distinct players a week over the 4 baseline weeks."""
    x = okv[(day_of >= b0) & (day_of <= b1)]
    if x.empty:
        return 0
    wk = ((pd.to_datetime(day_of[x.index]) - pd.Timestamp(b0)).dt.days // 7)
    return float(x.groupby(wk)["pbp_user_id"].nunique().reindex(range(4), fill_value=0).mean())


def digest(f: pd.DataFrame, okv: pd.DataFrame, vname) -> dict:
    """The last 7 days against the 4 weeks before (city and per venue). Needs
    35 days in the window; otherwise items = [] with a reason. Every count a
    sentence names is 5 or more; changes of 10+ points or 20%+."""
    empty = lambda why: {"items": [], "venues": {}, "reason": why}
    if f.empty:
        return empty("No finished sessions in the period.")
    day, start, end, last0, b0, b1 = _periods(f)
    if start > b0:
        return empty("Needs 5 weeks of data: the last 7 days and the 4 weeks before.")
    now_f = f[(day >= last0) & (day <= end)]
    base_f = f[(day >= b0) & (day <= b1)]
    oday = okv["starts_at"].dt.tz_convert(MEL).dt.date
    o_now = okv[(oday >= last0) & (oday <= end)]

    def players(x_now, x_all):
        return x_now["pbp_user_id"].nunique(), _weekly_players(x_all, oday[x_all.index], b0, b1)

    full = lambda gb: _every_week(day[gb.index], b0)
    pn, pb = players(o_now, okv)
    cand = _items(_measure(now_f, base_f, pn, pb)) if full(base_f) else []
    # by kind and session level (fill only)
    for col, noun in (("fgroup", None), ("lvl", "level")):
        for k, g in now_f.groupby(col):
            gb = base_f[base_f[col] == k]
            if (k in ("Unknown", "no level stated") or len(g) < DIGEST_MIN_SESSIONS
                    or len(gb) < DIGEST_MIN_SESSIONS or not full(gb)):
                continue
            now, before = 100 * g["fill"].mean(), 100 * gb["fill"].mean()
            if abs(now - before) < DIGEST_POINTS:
                continue
            name = f"{str(k)[:1].upper()}{str(k)[1:]} sessions"
            up = now > before
            cand.append({"metric": "fill", "dir": "up" if up else "down", "now": round(now, 1), "before": round(before, 1),
                         "scope": {col: k},
                         "text": f"{name} filled to {now:.0f}% on average, {'up' if up else 'down'} from {before:.0f}%.",
                         "score": abs(now - before) / DIGEST_POINTS})
    # biggest venue moves in fill
    for v, g in now_f.groupby("fid"):
        gb = base_f[base_f["fid"] == v]
        if len(g) < DIGEST_MIN_SESSIONS or len(gb) < DIGEST_MIN_SESSIONS or not full(gb):
            continue
        now, before = 100 * g["fill"].mean(), 100 * gb["fill"].mean()
        if abs(now - before) < DIGEST_POINTS:
            continue
        up = now > before
        cand.append({"metric": "fill", "dir": "up" if up else "down", "now": round(now, 1), "before": round(before, 1),
                     "scope": {"v": int(v)},
                     "text": f"At {vname(int(v))}, sessions filled to {now:.0f}% on average, {'up' if up else 'down'} from {before:.0f}%.",
                     "score": abs(now - before) / DIGEST_POINTS * 0.8})
    # a kind, level or venue that only repeats the city's own move adds nothing
    city_fill = next((c for c in cand if c["metric"] == "fill" and "scope" not in c), None)
    if city_fill:
        cand = [c for c in cand if not ("scope" in c and c["metric"] == "fill"
                                        and abs(c["now"] - city_fill["now"]) < 2
                                        and abs(c["before"] - city_fill["before"]) < 2)]
    items = sorted(cand, key=lambda c: -c["score"])[:DIGEST_MAX]
    venues = {}
    for v in sorted(set(f["fid"].astype(int))):
        g, gb = now_f[now_f["fid"] == v], base_f[base_f["fid"] == v]
        if not full(gb):
            continue
        ov = okv[okv["fid"] == v]
        vp_now, vp_base = players(o_now[o_now["fid"] == v], ov)
        vi = sorted(_items(_measure(g, gb, vp_now, vp_base)), key=lambda c: -c["score"])[:DIGEST_VENUE_MAX]
        if vi:
            venues[str(v)] = [{k: x[k] for k in x if k != "score"} for x in vi]
    clean = [{k: x[k] for k in x if k != "score"} for x in items]
    reason = None
    if not clean:
        reason = ("Nothing moved by 10 points or 20% in the last 7 days." if full(base_f)
                  else "Sessions didn't run in every one of the 4 weeks before, so there's no fair comparison yet.")
    return {"from": str(last0), "to": str(end), "baseFrom": str(b0), "baseTo": str(b1),
            "items": clean, "venues": venues, "reason": reason}


# ── scorecard ───────────────────────────────────────────────────────────────
def _weeks(f: pd.DataFrame, n=6):
    """Mondays of the last n whole weeks inside the window."""
    day = f["starts_at"].dt.tz_convert(MEL).dt.date
    start, end = day.min(), day.max()
    first_mon = start + timedelta(days=(7 - start.weekday()) % 7)
    mons = []
    m = first_mon
    while m + timedelta(days=6) <= end:
        mons.append(m)
        m += timedelta(days=7)
    return mons[-n:], day


def scorecard(f: pd.DataFrame, okv: pd.DataFrame, firsts: pd.DataFrame | None, venues: list, city: dict,
              weeks: float | None = None) -> dict:
    """Per venue and for the city: sessions a week, fill, sold out 2+ days
    early, new here, takings; and the same per whole week (last 6) for the
    small trend lines. Counts of players under 5 held back."""
    if f.empty:
        return {"weeks": [], "city": None, "venues": {}}
    mons, day = _weeks(f)
    first_at = {}
    if firsts is not None and not firsts.empty:
        F = firsts.dropna(subset=["pbp_user_id", "fid", "first_date"])
        for p, v, d in F[["pbp_user_id", "fid", "first_date"]].itertuples(index=False):
            try:
                first_at[(int(p), int(v))] = pd.Timestamp(d).date()
            except Exception:
                pass
    oday = okv["starts_at"].dt.tz_convert(MEL).dt.date
    takings = lambda g: int((g["price_n"].fillna(0) * g["booked"]).sum())

    first_any = {}
    for (p, v), d in first_at.items():
        if p not in first_any or d < first_any[p]:
            first_any[p] = d

    def new_here(o, d0, d1, anywhere=False):
        """At a venue: first session there in [d0, d1]. City (anywhere):
        first session we've ever seen (newPlayers)."""
        ps = set()
        for p, v in zip(o["pbp_user_id"], o["fid"]):
            d = first_any.get(int(p)) if anywhere else first_at.get((int(p), int(v)))
            if d is not None and d0 <= d <= d1:
                ps.add(p)
        return len(ps)

    def card(g, o, v=None):
        weeks_cov = weeks or max((day.max() - day.min()).days + 1, 1) / 7
        series = {"sessions": [], "fill": [], "soldEarly": [], "newHere": [], "takings": []}
        for m in mons:
            m1 = m + timedelta(days=6)
            gw = g[(day[g.index] >= m) & (day[g.index] <= m1)]
            ow = o[(oday[o.index] >= m) & (oday[o.index] <= m1)]
            series["sessions"].append(int(len(gw)))
            series["fill"].append(_r1(100 * gw["fill"].mean()) if len(gw) else None)
            series["soldEarly"].append(_r1(early_rate(gw)) if len(gw) else None)
            series["newHere"].append(_hold(new_here(ow, m, m1, v is None)))
            series["takings"].append(takings(gw) if len(gw) else None)
        d0, d1 = day.min(), day.max()
        return {"perWeek": _r1(len(g) / weeks_cov),
                "fill": _r1(100 * g["fill"].mean()) if len(g) else None,
                "soldEarly": _r1(early_rate(g)) if len(g) else None,
                "newHere": _hold(new_here(o, d0, d1, v is None)),
                "takings": takings(g) if len(g) else None,
                "series": series, "change": None}

    out = {"weeks": [str(m) for m in mons], "city": card(f, okv), "venues": {}}
    for v in venues:
        if not v.get("tracked") or not isinstance(v.get("id"), int):
            continue
        g = f[f["fid"] == v["id"]]
        if g.empty:
            continue
        out["venues"][str(v["id"])] = card(g, okv[okv["fid"] == v["id"]], v["id"])
    return out


def scorecard_changes(d: dict) -> dict:
    """Fills scorecard[...].change from d['prev'] (the period before, same
    length): perWeek (sessions a week, difference), fill (points), takings (%).
    sold out early and new here aren't in prev: None."""
    sc, prev = d.get("scorecard"), d.get("prev")
    if not sc or not prev or not prev.get("sessions"):
        return d
    cols = prev["cols"]
    P = pd.DataFrame(prev["sessions"], columns=cols)
    # the dates the period before was asked for, not its first and last session:
    # sessions a week must divide by the same length on both sides
    rng = prev.get("asked") or [prev.get("from"), prev.get("to")]
    try:
        days = (pd.Timestamp(rng[1]) - pd.Timestamp(rng[0])).days + 1
    except Exception:
        days = None
    weeks = (days or 7) / 7

    def chg(card, g):
        if card is None or g.empty:
            return None
        pw = len(g) / weeks
        fill = 100 * float((g["booked"] / g["cap"].clip(lower=1)).mean())
        tk = float((g["price"].fillna(0) * g["booked"]).sum())
        return {"perWeek": _r1(card["perWeek"] - pw) if card["perWeek"] is not None else None,
                "fill": _r1(card["fill"] - fill) if card["fill"] is not None else None,
                "takings": _r1(100 * (card["takings"] - tk) / tk) if card["takings"] is not None and tk > 0 else None,
                "soldEarly": None, "newHere": None}

    sc["city"]["change"] = chg(sc["city"], P)
    for k, card in sc["venues"].items():
        card["change"] = chg(card, P[P["v"] == int(k)])
    return d


# ── hire takings per court-hour ─────────────────────────────────────────────
def hire_takings(d: dict) -> dict:
    """venues[].hireTakingsPerCourtHour (and city): the venue's casual hire
    rate (hireRatePerCourtHour) x the share of hireable court-hours that were
    hired, from courtFree. Each hour counts by its hireable court-hours (courts
    not taken by the venue's own sessions), so hours mostly given to sessions
    don't weigh in. venues[].hireableCourtHours: those court-hours in the
    period. None without both."""
    cf = d.get("courtFree") or {}
    cols = cf.get("cols") or []
    if not cf.get("rows") or "hire" not in cols or "hireH" not in cols:
        for v in d.get("venues", []):
            v.setdefault("hireTakingsPerCourtHour", None)
            v.setdefault("hireableCourtHours", None)
        d.get("city", {}).setdefault("hireTakingsPerCourtHour", None)
        return d
    ix = {c: i for i, c in enumerate(cols)}
    acc = {}
    for r in cf["rows"]:
        h, w = r[ix["hire"]], r[ix["hireH"]] or 0
        if h is None or w <= 0:
            continue
        a = acc.setdefault(r[ix["v"]], [0.0, 0.0])
        a[0] += h * w
        a[1] += w
    vals = []
    for v in d.get("venues", []):
        rate = v.get("hireRatePerCourtHour")
        a = acc.get(v.get("id"))
        v["hireableCourtHours"] = int(round(a[1])) if a else None
        if rate is None or not a or a[1] <= 0:
            v["hireTakingsPerCourtHour"] = None
            continue
        v["hireTakingsPerCourtHour"] = _r1(rate * a[0] / a[1])
        vals.append((v["hireTakingsPerCourtHour"], a[1]))
    if "city" in d:
        d["city"]["hireTakingsPerCourtHour"] = (_r1(sum(x * w for x, w in vals) / sum(w for _, w in vals))
                                                if vals else None)
    return d


# ── canonical per-venue player figures ──────────────────────────────────────
def came_back(okv: pd.DataFrame, fid: int, end_day) -> tuple:
    """(rate %, base): of players whose first booking at the venue in the
    period left CAME_BACK_DAYS to look, the share who booked there again within
    CAME_BACK_DAYS. Rate None under 10 in the base (privacy.py checks 5+ too)."""
    x = okv[okv["fid"] == fid]
    if x.empty:
        return None, 0
    day = x["starts_at"].dt.tz_convert(MEL).dt.date
    x = x.assign(day=day.values)
    first = x.groupby("pbp_user_id")["day"].min()
    lim = end_day - timedelta(days=CAME_BACK_DAYS)
    eligible = first[first <= lim]
    if eligible.empty:
        return None, 0
    came = 0
    for p, d0 in eligible.items():
        ds = x.loc[x["pbp_user_id"] == p, "day"]
        if ((ds > d0) & (ds <= d0 + timedelta(days=CAME_BACK_DAYS))).any():
            came += 1
    n = len(eligible)
    return (round(100 * came / n, 1) if n >= 10 else None), n


def median_rating(lv: pd.DataFrame, min_players=10):
    """The median of each player's own median rating (GLOSSARY.rating)."""
    rr = lv.dropna(subset=["rating"]).groupby("pbp_user_id")["rating"].median()
    return (round(float(rr.median()), 2) if len(rr) >= min_players else None), int(len(rr))
