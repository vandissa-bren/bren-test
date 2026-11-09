"""
figures.py -- the seven new insights figures and "sessions to cut or move"
(redesign, 9 Nov 2026). Built from the frames build_overview.main() already
has, so they share its definitions (views.GLOSSARY):

  need_court(f)               sessions that need another court
  quiet_regulars(okv, f, end) regulars who've gone quiet (counts only)
  retention(okv, firsts, end) new players coming back, by weekly cohort
  same_faces(f, okv)          same faces each week, per weekly session
  where_candidates(...)       where to run a new session: everything but courts
  where_finish(d)             ... finished once the court log is in (build_insights)
  refilled(f, okv)            freed places refilled
  leaning(okv, end)           leaning on a few players
  cut_or_move(f, km)          sessions to cut or move

f: finished sessions at tracked venues (build_overview's f: fid, starts_at,
first_obs, first_full_at, sold_out, so_h, fill, capacity, booked, fgroup, lvl,
title, label, program_slug, session_type, waitlist_max, and from the export
waitlist_first_at / waitlist_first when present). okv: non-cancelled bookings
in those sessions (pbp_user_id, fid, session_key, starts_at, first_seen).

Players are only ever counts of 5 or more: the builders hold small counts back
and privacy.enforce checks every path again (FIG_COUNTS / FIG_RATES there).
"""
from __future__ import annotations

import math
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd

import series as SER

MIN_GROUP = 5
MEL = "Australia/Melbourne"
DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DOW3 = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# the same date build_insights cuts every window to
DATA_START = date(2026, 9, 21)

# another court
EARLY_H = 48                # sold out at least this many hours ahead
WAIT_MIN = 2                # a waitlist of at least this many
WAIT_AHEAD_H = 24           # ... first seen at least this long before the start, when we know
NEED_RUNS = 4               # look at the last 4 runs
NEED_FLAGS = 3              # under pressure in 3 of them
NEED_WATCHED = 3            # with at least 3 watched from 48 h out
NEED_FILL = 0.85            # ... and a pressure week only counts if it started 85%+ full: a session that
                            # sold out early but lost places before the start (term programmes, mass
                            # cancellations, places added) doesn't need another court
# coached sessions: another court needs a coach as well, so they're listed apart (needCoach)
COACHED_GROUPS = {"Coaching & clinics", "Learn to play"}
COACHED_WORDS = re.compile(r"\b(coach\w*|clinics?|lessons?|programs?|programmes?|kids|youth|juniors?|starter|"
                           r"learn to play|academy|drills?|skills)\b", re.I)
# regulars who've gone quiet
REG_WEEKS, REG_OF = 4, 6    # played in 4 of the 6 weeks up to their last visit
QUIET_X, QUIET_MIN_D = 2, 14
FIRM_DAYS = 70              # provisional until the data spans 10 weeks
# new players coming back
COHORT_FROM = date(2026, 10, 12)
COHORT_MIN = 20
BACK_WEEKS = (1, 2, 4, 8)
# same faces each week
FACES_MIN_PLAYERS, FACES_MAX_PLAYERS, FACES_RUNS = 8, 50, 4
# where to run a new session
NEAR_KM = 6
WHERE_TOL_H = 1             # elsewhere bookings within an hour of the slot
OWN_CLASH_MIN = 90          # nothing of the kind at the venue within 90 minutes
OWN_HALF_EMPTY = 70         # ... and no own session within 90 minutes under 70% full
COURTS_FREE, COURTS_SHARE, COURTS_WEEKS, HIGH_WEEKS = 2, 0.6, 4, 6
WHERE_TOP = 5
NEAR_MIN_RUNS = 3           # a nearby session counts as evidence only after 3 runs seen
WHERE_SPREAD_H = 2          # suggestions on the same day within 2 hours that rest on the same
                            # format or the same nearby session are one suggestion: the best is kept
# freed places refilled
REFILL_MIN_SESSIONS = 10
# leaning on a few players
LEAN_DAYS, LEAN_MIN_PLAYERS, LEAN_TOP = 28, 50, 0.2
# cut or move
CUT_FILL, CUT_FLAGS, CUT_BETTER = 40, 3, 20
NEW_SERIES_SKIP = 4         # a series that began after the data did: its first 4 runs are its start-up
# a weekly session is current only if it ran in the last 10 days
STALE_DAYS = 10
# where: only the venue's sessions in the last 4 weeks count as "what it runs"
RECENT_DAYS = 28


# ── small helpers ───────────────────────────────────────────────────────────
def _hold(n):
    n = int(n)
    return n if n >= MIN_GROUP or n == 0 else None


def _r1(x):
    return None if x is None or x != x else round(float(x), 1)


def ampm(hm) -> str:
    """'19:00' -> '7pm', '09:30' -> '9:30am'."""
    try:
        h, m = (int(x) for x in str(hm).split(":")[:2])
    except Exception:
        return str(hm)
    suf = "am" if h < 12 else "pm"
    h12 = h % 12 or 12
    return f"{h12}{suf}" if m == 0 else f"{h12}:{m:02d}{suf}"


def daypart(h: int) -> str:
    return "mornings" if h < 12 else ("afternoons" if h < 17 else "evenings")


def kind_name(fgroup, lvl) -> str:
    """('Social & open play', 'intermediate') -> 'intermediate social & open play'."""
    g = str(fgroup or "sessions").lower()
    lv = str(lvl or "")
    if not lv or lv == "no level stated":
        return g
    if lv == "all levels":
        return f"all-levels {g}"
    return f"{lv} {g}"


def _known(k) -> bool:
    """A (format, level) kind whose format is known."""
    return isinstance(k[0], str) and k[0] not in ("", "Unknown")


def _ts(x):
    return pd.to_datetime(x, utc=True, errors="coerce", format="mixed")


def _series(f: pd.DataFrame) -> pd.DataFrame:
    """f with sid (series.py), label rows and Melbourne dow/hm/date/week."""
    x = f.copy()
    x["sid"] = SER.assign_series(x)
    loc = x["starts_at"].dt.tz_convert(MEL)
    x["_dow"], x["_hm"], x["_h"] = loc.dt.dayofweek, loc.dt.strftime("%H:%M"), loc.dt.hour
    x["_min"] = loc.dt.hour * 60 + loc.dt.minute
    x["_date"] = loc.dt.date
    return x


def _labels(x: pd.DataFrame) -> dict:
    lab = SER.series_label(x.dropna(subset=["sid"]), sid_col="sid")
    return {r.sid: {"title": r.title, "dow": int(r.dow), "hm": r.hm, "runs": int(r.runs)} for r in lab.itertuples()}


def _label_text(l: dict) -> str:
    return f"{DOW3[l['dow']]} {ampm(l['hm'])} {l['title']}"


def _median(xs):
    xs = [x for x in xs if x is not None and x == x]
    return float(np.median(xs)) if xs else None


def _end_day(x: pd.DataFrame, end_day):
    """The build's last day; for the tests (no end_day), the frame's last day."""
    if end_day is not None:
        return end_day
    return x["starts_at"].dt.tz_convert(MEL).dt.date.max() if len(x) else date.today()


def _current(g: pd.DataFrame, end_day: date) -> bool:
    """A weekly session still running: its latest run within 10 days of the end."""
    return (end_day - g["_date"].iloc[-1]).days <= STALE_DAYS


def coached(fgroup, title) -> bool:
    """A session run by a coach: its format, or its title, says so."""
    return (isinstance(fgroup, str) and fgroup in COACHED_GROUPS) or bool(COACHED_WORDS.search(str(title or "")))


def _full_hours(x: pd.DataFrame):
    """(hours before the start it was first full while we watched, already
    full when first seen). Full at any reading counts, even if a late
    cancellation freed a place before the start."""
    ff = _ts(x["first_full_at"]) if "first_full_at" in x else pd.Series(pd.NaT, index=x.index)
    fo = _ts(x["first_obs"])
    hrs = (x["starts_at"] - ff).dt.total_seconds() / 3600
    became = ff.notna() & (ff > fo)
    already = ff.notna() & (ff <= fo)
    return hrs.where(became), already


# ── 1. sessions that need another court ────────────────────────────────────
def need_court(f: pd.DataFrame, end_day: date | None = None, top_city=20) -> dict:
    """A weekly session is under pressure in a week if it sold out at least
    48 h before the start (a session already full when we first saw it counts
    only if we first saw it 48 h or more ahead), or had a waitlist of 2 or more
    (first seen at least 24 h ahead, where we know when it appeared). Listed if
    under pressure in 3 of its last 4 runs, with at least 3 of those 4 watched
    from 48 h out. Waitlist lengths are people: only "had a waitlist" is shown."""
    out = {"city": [], "venues": {}}
    if f.empty:
        return out
    x = _series(f).dropna(subset=["sid"])
    if x.empty:
        return out
    end_day = _end_day(x, end_day)
    lab = _labels(x)
    ahead = (x["starts_at"] - _ts(x["first_obs"])).dt.total_seconds() / 3600
    x["_watched"] = ahead >= EARLY_H
    hrs_full, already = _full_hours(x)
    x["_fullH"] = hrs_full
    x["_early"] = (hrs_full >= EARLY_H).fillna(False) | (already & x["_watched"])
    wl = pd.to_numeric(x["waitlist_max"], errors="coerce").fillna(0) if "waitlist_max" in x else pd.Series(0, index=x.index)
    if "waitlist_first_at" in x:
        # waitlist_first_at is the first reading BEFORE the start with anyone
        # waiting (court_free.waitlist_times): a queue only seen after the start
        # has none, and doesn't count
        wfa = _ts(x["waitlist_first_at"])
        timely = wfa.notna() & ((x["starts_at"] - wfa).dt.total_seconds() / 3600 >= WAIT_AHEAD_H)
    else:                                                 # an old export: no timing at all
        timely = pd.Series(True, index=x.index)
    x["_wait"] = (wl >= WAIT_MIN) & timely
    started_full = pd.to_numeric(x["fill"], errors="coerce").fillna(0) >= NEED_FILL
    x["_press"] = (x["_early"] | x["_wait"]) & started_full
    rows = []
    for sid, g in x.sort_values("starts_at").groupby("sid", sort=False):
        if not _current(g, end_day):
            continue                                      # not run in the last 10 days
        last = g.tail(NEED_RUNS)
        if len(last) < NEED_RUNS or int(last["_watched"].sum()) < NEED_WATCHED:
            continue
        flags = [bool(b) for b in last["_press"]]
        if sum(flags) < NEED_FLAGS:
            continue
        hrs = _median(last.loc[last["_early"], "_fullH"].tolist())   # sold-out weeks only
        l = lab.get(sid, {})
        fg = g["fgroup"].iloc[-1] if "fgroup" in g else None
        rows.append({"sid": sid, "v": int(g["fid"].iloc[0]), "label": _label_text(l) if l else sid,
                     "title": l.get("title"), "dow": l.get("dow"), "hm": l.get("hm"),
                     "runs": flags, "hoursEarly": _r1(hrs), "waitlist": bool(last["_wait"].any()),
                     "fill": _r1(100 * last["fill"].mean()),
                     "coached": coached(fg, l.get("title") or (g["title"].iloc[-1] if "title" in g else None))})
    rows.sort(key=lambda r: (-sum(r["runs"]), -(r["hoursEarly"] or 0)))
    # coached sessions need a coach as well as a court: kept apart so the court list is courts only
    courts, coach = [r for r in rows if not r["coached"]], [r for r in rows if r["coached"]]
    out["city"] = courts[:top_city]
    out["needCoach"] = coach[:top_city]
    for r in courts:
        out["venues"].setdefault(str(r["v"]), []).append(r)
    return out


# ── 2. regulars who've gone quiet ──────────────────────────────────────────
def quiet_regulars(okv: pd.DataFrame, end_day: date, tracked=None) -> dict:
    """A regular at a venue played there in at least 4 of the 6 weeks up to
    their last visit. They've gone quiet when the days since that visit pass
    twice their usual gap there (and at least 14 days). Split by whether they've
    since booked at another tracked venue. Counts only."""
    firm = (end_day - DATA_START).days >= FIRM_DAYS
    out = {"provisional": not firm, "city": {"regulars": 0, "quiet": 0}, "venues": {}}
    if okv.empty:
        return out
    o = okv[["pbp_user_id", "fid", "starts_at"]].copy()
    o["d"] = o["starts_at"].dt.tz_convert(MEL).dt.date
    o = o[o["d"] <= end_day]
    city_reg, city_quiet = set(), set()
    by_p = {p: g for p, g in o.groupby("pbp_user_id")}
    for (p, v), g in o.groupby(["pbp_user_id", "fid"]):
        days = sorted(set(g["d"]))
        last = days[-1]
        lo = last - timedelta(days=7 * REG_OF - 1)
        weeks = {(d - lo).days // 7 for d in days if d >= lo}
        if len(weeks) < REG_WEEKS:
            continue
        gaps = [(b - a).days for a, b in zip(days, days[1:]) if (b - a).days > 0]
        usual = float(np.median(gaps)) if gaps else 7.0
        due = max(QUIET_MIN_D, QUIET_X * usual)
        vv = out["venues"].setdefault(str(int(v)), {"regulars": 0, "quiet": 0, "elsewhere": 0, "notSeen": 0, "to": {}})
        vv["regulars"] += 1
        city_reg.add(p)
        if (end_day - last).days <= due:
            continue
        vv["quiet"] += 1
        city_quiet.add(p)
        gp = by_p[p]
        later = gp[(gp["fid"] != v) & (gp["d"] > last)]
        if tracked is not None:
            later = later[later["fid"].isin(tracked)]
        if len(later):
            vv["elsewhere"] += 1
            for w in set(later["fid"].astype(int)):
                vv["to"][w] = vv["to"].get(w, 0) + 1
        else:
            vv["notSeen"] += 1
    for k, vv in out["venues"].items():
        vv["to"] = sorted(([int(w), n] for w, n in vv["to"].items() if n >= MIN_GROUP), key=lambda t: -t[1])[:3]
        for c in ("regulars", "quiet", "elsewhere", "notSeen"):
            vv[c] = _hold(vv[c])
        if vv["elsewhere"] is None or vv["notSeen"] is None:      # a split part under 5 hides the split
            vv["elsewhere"] = vv["notSeen"] = None
            vv["to"] = []
    out["city"] = {"regulars": _hold(len(city_reg)), "quiet": _hold(len(city_quiet))}
    return out


# ── 3. new players coming back ─────────────────────────────────────────────
def _monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _cohort_rows(first: dict, visits: dict, end_day: date) -> list:
    """first: player -> first date; visits: player -> sorted dates. Weekly
    cohorts from COHORT_FROM; a cohort under COHORT_MIN joins its calendar
    month; a month still under it is dropped. wK only once the window has
    passed for everyone in the cohort."""
    weekly = {}
    for p, d in first.items():
        if d is None or d < COHORT_FROM or d > end_day:
            continue
        weekly.setdefault(_monday(d), []).append(p)
    groups = []
    small = {}
    for wk in sorted(weekly):
        ps = weekly[wk]
        if len(ps) >= COHORT_MIN:
            groups.append(("week", wk, wk + timedelta(days=6), ps))
        else:
            small.setdefault((wk.year, wk.month), []).append((wk, ps))
    for (y, m), lst in small.items():
        ps = [p for _, w in lst for p in w]
        if len(ps) >= COHORT_MIN:
            groups.append(("month", lst[0][0], lst[-1][0] + timedelta(days=6), ps))
    groups.sort(key=lambda g: g[1])
    rows = []
    for kind, a, b, ps in groups:
        row = {"by": kind, "from": str(a), "to": str(b), "n": len(ps)}
        for k in BACK_WEEKS:
            if b + timedelta(days=7 * k) > end_day:
                row[f"w{k}"] = None
                continue
            back = 0
            for p in ps:
                d0 = first[p]
                d1 = d0 + timedelta(days=7 * k)
                if any(d0 < d <= d1 for d in visits.get(p, ())):
                    back += 1
            row[f"w{k}"] = round(100 * back / len(ps), 1)
        rows.append(row)
    return rows


def retention(okv: pd.DataFrame, firsts: pd.DataFrame | None, end_day: date) -> dict:
    """Of players first seen in a week (city: anywhere; venue: there), the share
    who booked again within 1, 2, 4 and 8 weeks. Cohorts from the week of
    12 Oct (anyone playing before rosters began looks new), 20 or more each."""
    out = {"from": str(COHORT_FROM), "city": [], "venues": {}}
    if okv.empty:
        return out
    o = okv[["pbp_user_id", "fid", "starts_at"]].copy()
    o["d"] = o["starts_at"].dt.tz_convert(MEL).dt.date
    first_at = {}
    if firsts is not None and not firsts.empty:
        F = firsts.dropna(subset=["pbp_user_id", "fid", "first_date"])
        for p, v, d in F[["pbp_user_id", "fid", "first_date"]].itertuples(index=False):
            try:
                first_at[(int(p), int(v))] = pd.Timestamp(d).date()
            except Exception:
                pass
    # bookings in the window fill any gaps (a player with no firsts row)
    for (p, v), d in o.groupby(["pbp_user_id", "fid"])["d"].min().items():
        k = (int(p), int(v))
        if k not in first_at or d < first_at[k]:
            first_at[k] = d
    first_any = {}
    for (p, v), d in first_at.items():
        if p not in first_any or d < first_any[p]:
            first_any[p] = d
    vis_any = {int(p): sorted(set(g)) for p, g in o.groupby("pbp_user_id")["d"]}
    out["city"] = _cohort_rows(first_any, vis_any, end_day)
    for v, ov in o.groupby("fid"):
        vis = {int(p): sorted(set(g)) for p, g in ov.groupby("pbp_user_id")["d"]}
        first = {p: d for (p, vv), d in first_at.items() if vv == int(v)}
        rows = _cohort_rows(first, vis, end_day)
        if rows:
            out["venues"][str(int(v))] = rows
    return out


# ── 4. same faces each week ────────────────────────────────────────────────
def same_faces(f: pd.DataFrame, okv: pd.DataFrame, end_day: date | None = None) -> dict:
    """Of a weekly session's players, the share who also played its previous
    week. The median of its last 4 such comparisons, each with 8 to 50 players.
    A run whose week before had no run (a skipped or pulled week) isn't
    compared: two weeks apart isn't "the week before"."""
    out = {"venues": {}}
    if f.empty or okv.empty:
        return out
    x = _series(f).dropna(subset=["sid"])
    if x.empty:
        return out
    end_day = _end_day(x, end_day)
    lab = _labels(x)
    who = okv.groupby("session_key")["pbp_user_id"].agg(lambda s: frozenset(s))
    for sid, g in x.sort_values("starts_at").groupby("sid", sort=False):
        if not _current(g, end_day):
            continue
        keys, dates = list(g["session_key"]), list(g["_date"])
        sets = [who.get(k, frozenset()) for k in keys]
        pairs = []
        for i in range(1, len(sets)):
            prev, cur = sets[i - 1], sets[i]
            if (dates[i] - dates[i - 1]).days > 8:
                continue                                  # a missing week in between
            if FACES_MIN_PLAYERS <= len(cur) <= FACES_MAX_PLAYERS and len(prev) <= FACES_MAX_PLAYERS:
                pairs.append((100 * len(cur & prev) / len(cur), len(cur)))
        last = pairs[-FACES_RUNS:]
        if len(last) < FACES_RUNS:
            continue
        l = lab.get(sid, {})
        out["venues"].setdefault(str(int(g["fid"].iloc[0])), []).append({
            "sid": sid, "label": _label_text(l) if l else sid, "fmt": g["fgroup"].iloc[-1] if "fgroup" in g else None,
            "pct": _r1(np.median([s for s, _ in last])), "players": int(round(np.median([n for _, n in last]))),
            "runs": FACES_RUNS})
    for v in out["venues"]:
        out["venues"][v].sort(key=lambda r: -r["pct"])
    return out


# ── 5. where to run a new session ──────────────────────────────────────────
def where_candidates(f: pd.DataFrame, okv: pd.DataFrame, km, vname, tracked, end_day: date | None = None) -> dict:
    """Everything but the court check (where_finish adds it once the court
    log is in). Candidates are (venue, weekday, hour, kind = format + level):
      a. the venue's players who came twice or more book that kind at other
         tracked venues at that weekday within an hour, while the venue runs
         nothing of that format (at any level, all-levels included) within 90
         minutes (5 or more players);
      b. sessions of that kind within 6 km at that time sold out early in at
         least half their runs, or run in the top quarter for fill;
      d. the venue has no session within 90 minutes under 70% full.
    "What the venue runs" is its sessions in the last 4 weeks: an old one-off
    or a session it has since dropped doesn't block a slot.
    A candidate needs a or b, and d."""
    out = {}
    if f.empty or okv.empty:
        return out
    x = _series(f)
    end_day = _end_day(x, end_day)
    x["_kind"] = list(zip(x["fgroup"], x["lvl"]))
    sk = x.set_index("session_key")
    o = okv[okv["session_key"].isin(sk.index)].copy()
    o["_dow"] = o["session_key"].map(sk["_dow"])
    o["_h"] = o["session_key"].map(sk["_h"])
    o["_kind"] = o["session_key"].map(sk["_kind"])
    q75 = float(x["fill"].quantile(0.75)) if len(x) else 1.0
    # nearby demand groups: venue x dow x hour x kind
    grp = {}
    for (v, d, h, k), g in x.groupby(["fid", "_dow", "_h", "_kind"]):
        if not _known(k) or len(g) < NEAR_MIN_RUNS:
            continue                                      # an unknown format, or too few runs to go on
        early = float(((g["so_h"] >= EARLY_H)).mean())
        grp[(int(v), int(d), int(h), k)] = {"runs": int(len(g)), "earlyRuns": int((g["so_h"] >= EARLY_H).sum()),
                                           "fill": float(g["fill"].mean()), "early": early,
                                           "name": vname(int(v)), "dow": int(d), "h": int(h), "kind": kind_name(*k),
                                           "label": f"{vname(int(v))} {DOW3[int(d)]} {ampm(f'{int(h):02d}:00')}"}
    # the venue's own recent sessions, for "it runs nothing like it" and the clash check
    recent = x[x["_date"] > end_day - timedelta(days=RECENT_DAYS)]
    own = recent.groupby(["fid", "_dow", "_min"]).agg(fill=("fill", "mean"),
                                                      fmts=("fgroup", lambda s: set(s))).reset_index()
    per = okv.groupby(["fid", "pbp_user_id"]).size()
    for v in tracked:
        v = int(v)
        base = set(per.loc[v][per.loc[v] >= 2].index) if v in per.index.get_level_values(0) else set()
        mine = own[own["fid"] == v]
        cands = {}
        if base:
            ob = o[o["pbp_user_id"].isin(base) & (o["fid"] != v) & o["fid"].isin(tracked)]
            for (d, h, k), g in ob.groupby(["_dow", "_h", "_kind"]):
                if not _known(k):
                    continue                              # "unknown" isn't a kind of session to suggest
                cands.setdefault((int(d), int(h), k), set()).update(g["pbp_user_id"])
        # widen each slot to its neighbouring hours
        a_count = {}
        for (d, h, k) in list(cands):
            ps = set()
            for hh in range(h - WHERE_TOL_H, h + WHERE_TOL_H + 1):
                ps |= cands.get((d, hh, k), set())
            a_count[(d, h, k)] = len(ps)
        # nearby demand for every slot of any kind seen within 6 km
        b_ev = {}
        for (w, d, h, k), gr in grp.items():
            if w == v:
                continue
            dist = km(v, w)
            if dist != dist or dist > NEAR_KM:
                continue
            if gr["early"] >= 0.5 or gr["fill"] >= q75:
                for hh in range(h - WHERE_TOL_H, h + WHERE_TOL_H + 1):
                    ev = b_ev.setdefault((d, hh, k), {})
                    cur = ev.get(w)                       # one entry per nearby venue: its best slot
                    if cur is None or (gr["early"], gr["fill"]) > (cur["early"], cur["fill"]):
                        ev[w] = dict(gr, v=w, km=round(dist, 1), key=f"{w}|{d}|{h}")   # one nearby venue and hour, whatever the kind
        slots = set(a_count) | set(b_ev)
        rows = []
        for (d, h, k) in slots:
            if not (6 <= h <= 21):
                continue
            m0 = h * 60
            near = mine[(mine["_dow"] == d) & ((mine["_min"] - m0).abs() <= OWN_CLASH_MIN)]
            if any(k[0] in fs for fs in near["fmts"]):
                continue                                     # it already runs this format then, at some level
            clash = near[near["fill"] < OWN_HALF_EMPTY / 100]
            if len(clash):
                continue                                     # d fails: it would compete with its own half-empty session
            a = a_count.get((d, h, k), 0)
            a = a if a >= MIN_GROUP else 0
            b = list(b_ev.get((d, h, k), {}).values())
            if not a and not b:
                continue
            rows.append({"dow": d, "h": h, "fmt": k[0], "lvl": k[1], "kind": kind_name(*k),
                         "players": a or None, "nearby": len(b),
                         "nearbyBest": max(b, key=lambda e: (e["early"], e["fill"])) if b else None})
        out[str(v)] = rows
    return out


def _where_text(r: dict, courts: dict | None) -> str:
    dname, hh = DOW[r["dow"]], ampm(f"{r['h']:02d}:00")
    parts = []
    if r.get("players"):
        parts.append(f"{r['players']} of your players who come twice or more book {r['kind']} elsewhere on "
                     f"{dname} {daypart(r['h'])} around {hh}, when you run nothing like it.")
    nb = r.get("nearbyBest")
    if nb:                                                # the best nearby session only
        nh = nb.get("h")
        when = f"{DOW[nb['dow']]}s at {ampm(f'{nh:02d}:00')}" if "dow" in nb and nh is not None else nb["label"]
        what = f"{nb['name']} ({nb['km']} km) runs {nb['kind']} on {when}" if "name" in nb else \
            f"{nb['label']} ({nb['km']} km)"
        if nb["earlyRuns"] and nb["early"] >= 0.5:
            how = f"it sold out at least 2 days early in {nb['earlyRuns']} of its last {nb['runs']} weeks"
        else:
            how = f"it was {100 * nb['fill']:.0f}% full on average over its last {nb['runs']} weeks"
        parts.append(f"Nearby, {what}, and {how}.")
    if courts:
        usual = courts.get("freeMed")
        tail = f" (usually {usual})" if isinstance(usual, int) and usual >= COURTS_FREE else ""
        parts.append(f"You have {COURTS_FREE} or more courts free on {dname}s at {hh} in {courts['free2']} of the "
                     f"last {courts['weeks']} weeks{tail}.")
    return " ".join(parts)


def _same_idea(a: dict, b: dict) -> bool:
    """Two suggestions for the same day within 2 hours that rest on the same
    format, or the same nearby session, are the same suggestion (the same
    players or the same competitor, an hour either way)."""
    if a["dow"] != b["dow"] or abs(a["h"] - b["h"]) > WHERE_SPREAD_H:
        return False
    if a["fmt"] == b["fmt"]:
        return True
    na, nb = a.get("near"), b.get("near")
    return bool(na and nb and na.get("key") == nb.get("key"))


def where_finish(d: dict) -> None:
    """Adds the court check (c) to d['_where'] and publishes d['figures']['where']:
    at least 2 courts free for the whole hour in 60% of the weeks read (4 or
    more weeks). High when all four checks hold over 6+ weeks of court records,
    else Medium. Top 5 per venue."""
    cand = d.pop("_where", None) or {}
    figs = d.setdefault("figures", {})
    cf = d.get("courtFree") or {}
    cols = cf.get("cols") or []
    if not cand:
        figs["where"] = {"venues": {}, "reason": "No sessions or players to work from."}
        return
    if not cf.get("rows") or not cols:
        figs["where"] = {"venues": {}, "reason": "Court records weren't available for this build."}
        return
    ix = {c: i for i, c in enumerate(cols)}
    # each venue's registered court count: a fallback when the court log has none
    owned = {}
    for vv in d.get("venues") or []:
        if isinstance(vv, dict) and isinstance(vv.get("id"), int) and isinstance(vv.get("courts"), (int, float)) \
                and vv["courts"] == vv["courts"] and vv["courts"] > 0:
            owned[vv["id"]] = int(vv["courts"])
    court = {}
    for row in cf["rows"]:
        v_, dw, hr = int(row[ix["v"]]), int(row[ix["dow"]]), int(row[ix["h"]])
        n_log = row[ix["courts"]] if "courts" in ix else None
        n_log = int(n_log) if isinstance(n_log, (int, float)) and n_log == n_log else None
        # the court log reads PlayByPoint's own courts, so it is the true count; the registry
        # (often stale) is only a fallback when the log has no count for this slot
        n = n_log if n_log is not None else owned.get(v_)
        fm = row[ix["freeMed"]]
        fm = int(round(fm)) if isinstance(fm, (int, float)) and fm == fm else None
        if fm is not None and n is not None:
            fm = min(fm, n)                               # never more courts free than it has
        court[(v_, dw, hr)] = {"weeks": int(row[ix["weeks"]]), "free2": int(row[ix["free2"]]), "freeMed": fm,
                               "courts": n}
    out = {}
    for v, rows in cand.items():
        keep = []
        for r in rows:
            c = court.get((int(v), r["dow"], r["h"]))
            if not c or c["weeks"] < COURTS_WEEKS or c["free2"] / c["weeks"] < COURTS_SHARE:
                continue
            if c["courts"] is not None and c["courts"] < COURTS_FREE:
                continue                                  # it can't have 2 courts free
            high = bool(r.get("players")) and r["nearby"] > 0 and c["weeks"] >= HIGH_WEEKS
            score = (2 if high else 1, (r.get("players") or 0) + 5 * r["nearby"], c["free2"] / c["weeks"])
            courts = dict(c)
            nb = r.get("nearbyBest")
            keep.append((score, {
                "dow": r["dow"], "h": r["h"], "fmt": r["fmt"], "lvl": r["lvl"], "kind": r["kind"],
                "confidence": "High" if high else "Medium", "players": r.get("players"),
                "nearby": r["nearby"], "near": ({"v": nb["v"], "km": nb["km"], "runs": nb["runs"],
                                                 "earlyRuns": nb["earlyRuns"], "fill": _r1(100 * nb["fill"]),
                                                 "key": nb.get("key")} if nb else None),
                "courts": courts, "text": _where_text(r, courts)}))
        keep.sort(key=lambda t: t[0], reverse=True)
        picked = []
        for _, k in keep:
            if any(_same_idea(k, p) for p in picked):
                continue                                  # a near-copy of a better suggestion
            picked.append(k)
            if len(picked) == WHERE_TOP:
                break
        if picked:
            out[v] = picked
    figs["where"] = {"venues": out, "reason": None if out else "No time passed all the checks yet."}


# ── 6. freed places refilled ───────────────────────────────────────────────
def refilled(f: pd.DataFrame, okv: pd.DataFrame, allr: pd.DataFrame | None = None) -> dict:
    """Of sessions that had a waitlist, the share where a place freed after
    the session first sold out was taken again before the start: a new player
    first appeared on the list after it sold out and before it began, AND more
    people were on the list over its life than it has places (someone left).
    Without the second test, places added with an extra court would look like
    refills. We can't tell whether the newcomer came off the waitlist. Shown
    with 10+ waitlisted sessions and 5+ refilled. allr: every roster row
    (cancelled included); without it the second test is skipped."""
    def calc(g):
        n = int(len(g))
        if n == 0:
            return None
        k = int(g["_refill"].sum())
        ok = n >= REFILL_MIN_SESSIONS and k >= MIN_GROUP
        return {"sessions": n, "refilled": k if ok else None, "pct": _r1(100 * k / n) if ok else None}
    out = {"city": None, "venues": {}}
    if f.empty or "waitlist_max" not in f:
        return out
    w = f[(pd.to_numeric(f["waitlist_max"], errors="coerce").fillna(0) > 0) & f["first_full_at"].notna()].copy()
    if w.empty:
        return out
    full_at = w.set_index("session_key")["first_full_at"]
    start = w.set_index("session_key")["starts_at"]
    o = okv[okv["session_key"].isin(full_at.index)]
    seen = _ts(o["first_seen"])
    late = o[(seen > _ts(o["session_key"].map(full_at))) & (seen < o["session_key"].map(start))]
    ok = w["session_key"].isin(set(late["session_key"]))
    if allr is not None and not allr.empty:
        ever = allr[allr["session_key"].isin(full_at.index)].groupby("session_key")["pbp_user_id"].nunique()
        cap = pd.to_numeric(w["capacity"], errors="coerce")
        ok &= w["session_key"].map(ever).fillna(0).to_numpy() > cap.fillna(1e9).to_numpy()
    w["_refill"] = ok
    out["city"] = calc(w)
    for v, g in w.groupby("fid"):
        out["venues"][str(int(v))] = calc(g)
    return out


# ── 7. leaning on a few players ────────────────────────────────────────────
def leaning(okv: pd.DataFrame, end_day: date) -> dict:
    """The share of a venue's bookings over the last 28 days made by its top
    20% of players. Shown with 50+ players. Coaches can't be told apart from
    players in the rosters, so none are left out."""
    out = {"days": LEAN_DAYS, "cityMedian": None, "venues": {}}
    if okv.empty:
        return out
    o = okv.copy()
    o["d"] = o["starts_at"].dt.tz_convert(MEL).dt.date
    o = o[(o["d"] > end_day - timedelta(days=LEAN_DAYS)) & (o["d"] <= end_day)]
    shares = []
    for v, g in o.groupby("fid"):
        n = g.groupby("pbp_user_id").size().sort_values(ascending=False)
        if len(n) < LEAN_MIN_PLAYERS:
            continue
        top = math.ceil(LEAN_TOP * len(n))
        pct = 100 * float(n.iloc[:top].sum()) / float(n.sum())
        shares.append(pct)
        out["venues"][str(int(v))] = {"players": int(len(n)), "top": int(top), "bookings": int(n.sum()),
                                      "pct": _r1(pct)}
    out["cityMedian"] = _r1(float(np.median(shares))) if shares else None
    return out


# ── 8. sessions to cut or move ─────────────────────────────────────────────
def cut_or_move(f: pd.DataFrame, km, vname, end_day: date | None = None) -> dict:
    """Weekly sessions under 40% full in 3 of their last 4 runs, the latest
    included, still running (a run in the last 10 days), with any weekly
    session of the same kind within 6 km at the same weekday and hour (within
    an hour) that fills 20+ points better over the same weeks. A session that
    began after the data did isn't judged on its first 4 runs: those are its
    start-up."""
    out = {"city": [], "venues": {}}
    if f.empty:
        return out
    x = _series(f)
    end_day = _end_day(x, end_day)
    lab = _labels(x.dropna(subset=["sid"]))
    x["_kind"] = list(zip(x["fgroup"], x["lvl"]))
    weekly_x = x.dropna(subset=["sid"])
    rows = []
    for sid, g in weekly_x.sort_values("starts_at").groupby("sid", sort=False):
        if not _current(g, end_day):
            continue                                      # already cut, or moved
        if (g["_date"].iloc[0] - DATA_START).days > 7:
            g = g.iloc[NEW_SERIES_SKIP:]                  # a new session: skip its start-up weeks
        last = g.tail(NEED_RUNS)
        if len(last) < NEED_RUNS:
            continue
        fills = [round(100 * v, 1) for v in last["fill"]]
        if sum(v < CUT_FILL for v in fills) < CUT_FLAGS or fills[-1] >= CUT_FILL:
            continue                                      # the latest week must be under 40% too
        v0, d0, h0, k0 = int(g["fid"].iloc[0]), int(g["_dow"].iloc[-1]), int(g["_h"].iloc[-1]), g["_kind"].iloc[-1]
        lo, hi = last["_date"].min(), last["_date"].max()
        same = weekly_x[(weekly_x["fid"] != v0) & (weekly_x["_dow"] == d0) & ((weekly_x["_h"] - h0).abs() <= 1)
                        & (weekly_x["_kind"] == k0) & (weekly_x["_date"] >= lo) & (weekly_x["_date"] <= hi)]
        better = None
        mean_here = float(np.mean(fills))
        for ws, gw in same.groupby("sid"):                # weekly sessions only: "runs ... on Mondays" must be true
            if len(gw) < 2:
                continue
            w = int(gw["fid"].iloc[0])
            dist = km(v0, w)
            if dist != dist or dist > NEAR_KM:
                continue
            fw = 100 * float(gw["fill"].mean())
            if fw - mean_here >= CUT_BETTER and (better is None or fw > better["fill"]):
                better = {"v": w, "name": vname(w), "km": round(dist, 1), "fill": round(fw, 1),
                          "hm": gw.sort_values("starts_at")["_hm"].iloc[-1]}
        l = lab.get(sid, {})
        txt = f"Under {CUT_FILL}% full in {sum(v < CUT_FILL for v in fills)} of its last {NEED_RUNS} weeks."
        if better:
            txt += (f" {better['name']} ({better['km']} km) runs {kind_name(*k0)} at {ampm(better['hm'])} "
                    f"on {DOW[d0]}s and fills {better['fill']:.0f}%.")
        rows.append({"sid": sid, "v": v0, "label": _label_text(l) if l else sid, "kind": kind_name(*k0),
                     "fills": fills, "better": better, "text": txt})
    rows.sort(key=lambda r: np.mean(r["fills"]))
    # Nobody booked in any of the 4 weeks (PlayByPoint's own count and the roster
    # agree, 9 Nov check). Venues list these in case someone books, and likely
    # release the court when nobody does, so "cut or move" means nothing for
    # them. Kept apart (neverBooked) so the list is sessions that ran half-empty.
    ran, never = [r for r in rows if any(v > 0 for v in r["fills"])], [r for r in rows if not any(v > 0 for v in r["fills"])]
    out["city"] = ran[:20]
    out["neverBooked"] = never
    for r in ran:
        out["venues"].setdefault(str(r["v"]), []).append(r)
    return out


# ── all of them ─────────────────────────────────────────────────────────────
def build(f, okv, firsts, end_day, km, vname, tracked, allr=None) -> tuple[dict, dict]:
    """(figures, where-candidates). where is finished in build_insights once the
    court log is in (where_finish). allr: every roster row, cancelled included
    (for freed places refilled)."""
    figs = {
        "needCourt": need_court(f, end_day),
        "quiet": quiet_regulars(okv, end_day, set(tracked)),
        "retention": retention(okv, firsts, end_day),
        "sameFaces": same_faces(f, okv, end_day),
        "refilled": refilled(f, okv, allr),
        "leaning": leaning(okv, end_day),
        "cutOrMove": cut_or_move(f, km, vname, end_day),
    }
    return figs, where_candidates(f, okv, km, vname, list(tracked), end_day)
