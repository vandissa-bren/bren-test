"""
venue_report.py -- every figure behind one venue's report, for any PlayByPoint
venue, plus the loaders and rules the daily insights build shares.

    python insights/venue_report.py <facility_id>

Inputs: data/sessions.csv, data/rosters.csv, data/catalogue.csv (the
insights_export_* functions) and backend/venues.json. Titles, levels and
formats are read with the feed model's own catalogue_terms, so the report and
the feed agree. Output: venue_report_<facility_id>.json.

No venue is built in: the venue is an argument, and the venues it is compared
with (rivals, the nearest venue) are found from the data.

Ratings on rosters are PlayByPoint's: 1.0 = unrated and 2.0 = an unconfirmed
default spike (the model's rule), both dropped; above 6.0 dropped as noise.
What's left is mostly self-rated half steps (2.5, 3.0, 3.5) plus a few DUPR.
"""
import json, os, re, sys
from math import radians, sin, cos, asin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from catalogue_terms import session_terms as parse

HERE = Path(__file__).resolve().parent
MEL = "Australia/Melbourne"
NOW = None  # set from the data
# Where the exports are read from and the working files go (build_insights.py sets both).
DATA = Path(os.environ.get("INSIGHTS_DATA", HERE / "data"))
WORK = Path(os.environ.get("INSIGHTS_WORK", HERE / "work"))


def registry():
    data = json.loads((HERE.parent / "venues.json").read_text())
    if isinstance(data, dict):
        data = data.get("venues", data)
    if isinstance(data, dict):
        data = list(data.values())
    return {int(v["id"]): v for v in data if isinstance(v, dict) and str(v.get("id", "")).isdigit()}


REG = registry()


def km(a, b):
    a, b = REG.get(int(a)), REG.get(int(b))
    if not a or not b or not a.get("lat") or not b.get("lat"):
        return np.nan
    dl, dn = radians(b["lat"] - a["lat"]), radians(b["lng"] - a["lng"])
    h = sin(dl / 2) ** 2 + cos(radians(a["lat"])) * cos(radians(b["lat"])) * sin(dn / 2) ** 2
    return 2 * 6371 * asin(sqrt(h))


def short_name(n):
    """A venue's display name: "The Jar | South Melbourne" -> "The Jar South
    Melbourne", "Raya Pickleball Club" -> "Raya". The same rule for every venue."""
    n = str(n).replace(" | ", " ").strip()
    return re.sub(r"\s+Pickleball Club$", "", n) or n


def name(fid):
    return short_name(REG.get(int(fid), {}).get("name", str(fid)))


def nearest(fid, k=1):
    """The k nearest active PlayByPoint venues to fid, by straight-line km."""
    cands = [(km(fid, f), f) for f, v in REG.items()
             if f != int(fid) and v.get("platform") == "playbypoint" and v.get("status") == "active"]
    return [f for d, f in sorted(c for c in cands if not np.isnan(c[0]))[:k]]


def pct(a, b):
    return None if not b else round(100.0 * a / b, 1)


def load():
    s = pd.read_csv(DATA / "sessions.csv", na_values=["null"])
    r = pd.read_csv(DATA / "rosters.csv", na_values=["null"])
    c = pd.read_csv(DATA / "catalogue.csv", na_values=["null"])
    for col in ("starts_at", "first_obs", "last_obs", "first_full_at"):
        s[col] = pd.to_datetime(s[col], utc=True, errors="coerce", format="mixed")
    for col in ("first_seen", "last_seen"):
        r[col] = pd.to_datetime(r[col], utc=True, errors="coerce", format="mixed")
    s["fid"] = s["venue_id"].str.extract(r"(\d+)$")[0].astype(float)
    c["key"] = "pbp-" + c["lesson_id"].astype(str)
    c["title"] = c["title"].str.strip()
    global PROGRAMS
    pf = DATA / "programs.csv"
    PROGRAMS = pd.read_csv(pf, na_values=["null"]) if pf.exists() else pd.DataFrame(
        columns=["venue_id", "program_slug", "tiers", "lessons_ahead_max"])
    return s, r, c


PROGRAMS = pd.DataFrame(columns=["venue_id", "program_slug", "tiers", "lessons_ahead_max"])


def attach_titles(s, c):
    """Title for every session: its own catalogue row, else the weekly fixture at
    the same venue, weekday, start time and type (venues repeat these)."""
    loc = s["starts_at"].dt.tz_convert(MEL)
    s["dow"], s["hm"] = loc.dt.dayofweek, loc.dt.strftime("%H:%M")
    s["hour"] = loc.dt.hour
    c["dow"] = pd.to_datetime(c["date"]).dt.dayofweek
    c["dur_h"] = (pd.to_datetime(c["end_time"], format="%H:%M", errors="coerce")
                  - pd.to_datetime(c["start"], format="%H:%M", errors="coerce")).dt.total_seconds() / 3600
    by_key = c.set_index("key")
    fx = (c.groupby(["venue_id", "dow", "start", "type"])
           .agg(titles=("title", lambda x: sorted(set(x))), skill=("skill_level", "first"), dur=("dur_h", "median"))
           .reset_index())
    fx = {(r.venue_id, r.dow, r.start, r.type): r for r in fx.itertuples()}
    # the same without the type: the app's titled rows carried no type and the
    # server's typed rows no title (20261017100000), so the typed match often missed
    fa = (c.groupby(["venue_id", "dow", "start"])
           .agg(titles=("title", lambda x: sorted(set(x.dropna()))), skill=("skill_level", "first"), dur=("dur_h", "median"))
           .reset_index())
    fa = {(r.venue_id, r.dow, r.start): r for r in fa.itertuples()}
    titles, skills, durs, how = [], [], [], []
    for r in s.itertuples():
        if r.session_key in by_key.index and isinstance(by_key.loc[r.session_key]["title"] if not isinstance(by_key.loc[r.session_key], pd.DataFrame) else by_key.loc[r.session_key].iloc[0]["title"], str):
            row = by_key.loc[r.session_key]
            row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
            titles.append(row["title"]); skills.append(row["skill_level"]); durs.append(row["dur_h"]); how.append("key")
            continue
        f = fx.get((r.venue_id, r.dow, r.hm, r.session_type))
        if f is None or len(f.titles) != 1:
            f = fa.get((r.venue_id, r.dow, r.hm))
        if f is not None and len(f.titles) == 1:
            titles.append(f.titles[0]); skills.append(f.skill); durs.append(f.dur); how.append("fixture")
        else:
            titles.append(None); skills.append(None); durs.append(np.nan); how.append(None)
    s["title"], s["skill_level"], s["dur_h"], s["title_how"] = titles, skills, durs, how
    # The session's own logged end time (20261016100000) beats the catalogue's.
    if "end_time" in s:
        end = pd.to_datetime(s["end_time"], format="%H:%M", errors="coerce")
        start = pd.to_datetime(s["hm"], format="%H:%M", errors="coerce")
        own = (end - start).dt.total_seconds() / 3600
        s["dur_h"] = own.where(own > 0, s["dur_h"])
    terms = [parse(t, ty if isinstance(ty, str) else None, sk if isinstance(sk, str) else None) if isinstance(t, str) and t else None
             for t, ty, sk in zip(s["title"], s["session_type"], s["skill_level"])]
    s["fmt"] = [session_format(t) if t else type_format(ty) for t, ty in zip(terms, s["session_type"])]
    s["family"] = [t["family"] if t else FORMAT_FAMILY.get(f) for t, f in zip(terms, s["fmt"])]
    s["label"] = [t["label"] if t else None for t in terms]
    s["band_lo"] = [t["band"][0] if t and t["band"] else np.nan for t in terms]
    s["band_hi"] = [t["band"][1] if t and t["band"] else np.nan for t in terms]
    s["level_class"] = [level_class(t) for t in terms]
    # the session's programme: its own logged one, else its listing's
    if "program_slug" not in s:
        s["program_slug"] = np.nan
    slug_by_key = c.dropna(subset=["program_slug"]).drop_duplicates("key").set_index("key")["program_slug"]
    s["program_slug"] = s["program_slug"].fillna(s["session_key"].map(slug_by_key))
    s = session_prices(s)
    s["price_hr"] = s["price_n"] / s["dur_h"]
    s["daypart"] = np.select([s["hour"] < 9, s["hour"] < 17], ["early", "day"], "evening")
    s["weekend"] = s["dow"] >= 5
    return s


FORMAT_FAMILY = {"round_robin": "competitive", "ladder": "competitive", "league": "competitive",
                 "match_play": "competitive", "learn_to_play": "learning", "coaching": "learning",
                 "clinic": "learning", "open_play": "social", "social": "social"}


def type_format(raw_type):
    """A session with no title at all: the venue's type, only where it says
    something specific ("DUPR Session", "Competitive Play" are match play).
    "Open Play" and "Event" alone stay unknown, since venues
    use them for match play and round robins too. Same rule as ins_family's
    untitled branch (20261017100000)."""
    ty = (raw_type if isinstance(raw_type, str) else "").strip().lower()
    if not ty:
        return None
    if "dupr" in ty or "competitive" in ty:
        return "match_play"
    if re.search(r"round\s*robin", ty):
        return "round_robin"
    if re.search(r"ladder|league|tournament|competition|championship", ty):
        return "league"
    if re.search(r"new to|learn|intro|beginner", ty):
        return "learn_to_play"
    if re.search(r"coach|lesson|clinic|academy|junior|skill|pathway|program", ty):
        return "clinic"
    if "social" in ty:
        return "social"
    return None


# ── what one session costs ─────────────────────────────────────────────────
SEASON = re.compile(r"\bleague\b|\bseason\b|\bseries\b|\bterm\b|\bcourse\b|\bprogram(?:me)?\b|\b\d+\s*-?\s*weeks?\b|\b\d+\s*-?\s*wks?\b", re.I)
WEEKS = re.compile(r"\b(\d{1,2})\s*-?\s*(?:weeks?|wks?|sessions|classes|lessons)\b", re.I)


def _tiers(raw):
    if isinstance(raw, list):
        return [t for t in raw if isinstance(t, dict)]
    if isinstance(raw, str) and raw.strip().startswith("["):
        try:
            return [t for t in json.loads(raw) if isinstance(t, dict)]
        except ValueError:
            return []
    return []


def _num(x):
    try:
        v = float(x)
        return v if v >= 0 else None
    except (TypeError, ValueError):
        return None


def _per_session(t):
    """A tier's price for one session: as it is when PlayByPoint says it buys a
    session, split across its lessons when it's a pack of several, else None
    (a commitment whose length the record doesn't say)."""
    p, unit, n = _num(t.get("price")), t.get("lesson_unit"), _num(t.get("lessons"))
    if p is None:
        return None
    if unit == "session" or (unit is None and (n is None or n <= 1)):
        return p
    if n and n > 1:
        return p / n
    return None


def session_prices(s):
    """Per-session prices, casual and member (insights only; the app keeps its
    own resolver). Some venues list a whole season as the price ($200 for a
    six-week league), which made takings and $/hour wrong.

    In order, for the casual price (anything not member-only):
      1. the session's own price records (price_tiers, logged from 6 Oct):
         the listed price if it's one of the per-session records, else the
         lowest per-session record, else a pack split into its sessions;
      2. the programme's published packages, split the same way;
      3. the listed price, when nothing marks it as a season. A season (title
         says league / season / series / term / N weeks, or the price is over
         3× the venue's typical session) is split by the weeks in its title,
         else by the most sessions its programme has listed; with neither, it
         is left out (price_n blank, price_basis "season").
    The member price is the lowest member record per session, the same way,
    only where the venue publishes member tiers. price_basis says which rule
    applied: tier, package, listed, season-weeks, season-listing, season."""
    listed = s["price"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)
    typical = listed.groupby(s["fid"]).transform("median")
    prog = {}
    for row in PROGRAMS.itertuples():
        prog[(str(row.venue_id), str(row.program_slug))] = (_tiers(row.tiers), _num(row.lessons_ahead_max))
    tiers_col = s["price_tiers"] if "price_tiers" in s else pd.Series([None] * len(s), index=s.index)
    out_p, out_m, basis = [], [], []
    for i, lp, typ, raw, vid, slug, title in zip(s.index, listed, typical, tiers_col, s["venue_id"], s["program_slug"],
                                                 s["title"] if "title" in s else [None] * len(s)):
        own = _tiers(raw)
        ptiers, ahead = prog.get((str(vid), str(slug)), ([], None))
        casual = [t for t in own if t.get("player_category") != "member"]
        member = [t for t in own + ptiers if t.get("player_category") == "member"]
        price, how = None, None
        per = [p for p in (_per_session(t) for t in casual if t.get("lesson_unit") in ("session", None)) if p is not None]
        if per:
            price, how = (lp if lp == lp and any(abs(lp - p) < 0.01 for p in per) else min(per)), "tier"
        if price is None:
            packs = [p for p in (_per_session(t) for t in casual + [t for t in ptiers if t.get("player_category") != "member"]
                                 if _num(t.get("lessons")) and _num(t.get("lessons")) > 1) if p is not None]
            if packs:
                price, how = min(packs), "package"
        if price is None and lp == lp:
            text = title if isinstance(title, str) else ""
            seasonal = bool(SEASON.search(text)) or (typ == typ and typ > 0 and lp > 3 * typ and lp > 40)
            if not seasonal:
                price, how = lp, "listed"
            else:
                m = WEEKS.search(text)
                if m and 2 <= int(m.group(1)) <= 52:
                    price, how = lp / int(m.group(1)), "season-weeks"
                elif ahead and ahead >= 2 and lp > 2 * (typ if typ == typ else 0):
                    price, how = lp / ahead, "season-listing"
                elif lp <= 40 and not (typ == typ and lp > 3 * typ):
                    price, how = lp, "listed"
                else:
                    how = "season"
        mp = [p for p in (_per_session(t) for t in member) if p is not None]
        mprice = min(mp) if mp else None
        if mprice is not None and price is not None and mprice > price:
            mprice = price
        out_p.append(price); out_m.append(mprice); basis.append(how)
    s["price_n"] = pd.Series(out_p, index=s.index, dtype=float)
    s["mprice_n"] = pd.Series(out_m, index=s.index, dtype=float)
    s["price_basis"] = basis
    return s


def session_format(t):
    """The format a session is grouped under. A social or open-play session whose
    results go to DUPR ("DUPR Doubles", "DUPR Recorded Session") is match play:
    its family is already competitive (catalogue_terms.play_family, F116), and
    the format now says so too. Same rule as ins_family in the live insights."""
    if not t:
        return None
    if t["family"] == "competitive" and t["format"] in ("social", "open_play"):
        return "match_play"
    return t["format"]


def level_class(t):
    if not t:
        return None
    if t["all_levels"]:
        return "all levels"
    if not t["band"]:
        return None
    lo, hi = t["band"]
    hi = min(hi, 4.5)
    mid = (lo + hi) / 2
    return "beginner" if mid < 2.75 else "intermediate" if mid < 3.5 else "advanced"


PULLED_GAP_H = 24      # last reading this long before the start ...
PULLED_SEEN_H = 12     # ... while the venue's other sessions were read this much later


def pulled(s):
    """Sessions the venue took down before they ran. The fill log's last
    reading of such a session is days before its start, while its venue's
    other sessions kept being read: the capture job was running, the session
    was no longer listed. (A capture outage stops every session at the venue,
    so it isn't counted.) Before 7 Oct these counted as finished, with the
    fill they had when they vanished."""
    out = pd.Series(False, index=s.index)
    if "last_obs_before" not in s:
        return out
    lob = pd.to_datetime(s["last_obs_before"], utc=True, errors="coerce", format="mixed")
    naive = lambda x: x.dt.tz_convert("UTC").dt.tz_localize(None)
    cand = s.assign(_lob=lob)[s["fid"].notna() & lob.notna() & ((s["starts_at"] - lob) > pd.Timedelta(hours=PULLED_GAP_H))]
    for fid, g in cand.groupby("fid"):
        v = s["fid"] == fid
        ts = np.sort(naive(pd.concat([s.loc[v, "first_obs"], s.loc[v, "last_obs"], lob[v]]).dropna()).values)
        if not len(ts):
            continue
        lo = naive(g["_lob"] + pd.Timedelta(hours=PULLED_SEEN_H)).values
        hi = naive(g["starts_at"]).values
        i = np.searchsorted(ts, lo, side="left")
        seen = (i < len(ts)) & (ts[np.minimum(i, len(ts) - 1)] < hi)
        out.loc[g.index[seen]] = True
    return out


def finished(s):
    gone = pulled(s)
    f = s[(s["starts_at"] < NOW) & s["capacity"].gt(0) & s["spots_at_start"].notna()
          & (s["spots_at_start"] <= s["capacity"]) & ~gone].copy()
    f["fill"] = 1 - f["spots_at_start"] / f["capacity"]
    f["booked"] = f["capacity"] - f["spots_at_start"]
    f["sold_out"] = f["spots_at_start"] == 0
    f["so_h"] = np.where(f["sold_out"] & (f["first_full_at"] > f["first_obs"]),
                         (f["starts_at"] - f["first_full_at"]).dt.total_seconds() / 3600, np.nan)
    return f


def rosters(s, r):
    m = r.merge(s[["session_key", "fid", "starts_at", "title", "family", "label", "band_lo", "band_hi",
                   "dow", "hour", "hm"]], on="session_key", how="left")
    m = m[m["fid"].notna()].copy()
    g = m.groupby("session_key")
    last_pass = g["last_seen"].transform("max")
    m["cancelled"] = (m["last_seen"] < last_pass) & (m["last_seen"] < m["starts_at"])
    m["rating"] = m["rating_at_time"].where(~m["rating_at_time"].isin([1.0, 2.0]) & (m["rating_at_time"] <= 6.0))
    return m


def headline(fin_v):
    days = (fin_v["starts_at"].max() - fin_v["starts_at"].min()).days + 1
    return {"finished": len(fin_v), "days": days,
            "avg_fill": round(100 * fin_v["fill"].mean(), 1),
            "sold_out": int(fin_v["sold_out"].sum()),
            "takings": round(float((fin_v["price_n"] * fin_v["booked"]).sum())),
            "empty_value": round(float((fin_v["price_n"] * fin_v["spots_at_start"]).sum())),
            "empty_places": int(fin_v["spots_at_start"].sum())}


def by_title(fin_v, m_v):
    out = []
    for t, g in fin_v.groupby(fin_v["title"].fillna("(title unknown)")):
        rr = m_v[(m_v["title"] == t) & ~m_v["cancelled"]]["rating"].dropna()
        lo, hi = g["band_lo"].iloc[0], g["band_hi"].iloc[0]
        within = None
        if len(rr) and not np.isnan(lo):
            within = pct(int(((rr >= lo - 0.01) & (rr <= hi + 0.01)).sum()), len(rr))
        out.append({"title": t, "label": g["label"].iloc[0], "family": g["family"].iloc[0],
                    "format": g["fmt"].iloc[0], "times": ", ".join(sorted(set(g["hm"]))),
                    "n": len(g), "fill": round(100 * g["fill"].mean(), 1), "sold_out": int(g["sold_out"].sum()),
                    "so_h_median": None if g["so_h"].dropna().empty else round(float(g["so_h"].median()), 1),
                    "capacity": float(g["capacity"].median()),
                    "price": float(g["price_n"].median()) if g["price_n"].notna().any() else None,
                    "price_hr": None if g["price_hr"].dropna().empty else round(float(g["price_hr"].median()), 2),
                    "takings": round(float((g["price_n"] * g["booked"]).sum())),
                    "empty_value": round(float((g["price_n"] * g["spots_at_start"]).sum())),
                    "ratings_n": int(len(rr)), "rating_median": None if rr.empty else float(rr.median()),
                    "rating_p25": None if rr.empty else float(rr.quantile(.25)),
                    "rating_p75": None if rr.empty else float(rr.quantile(.75)),
                    "rated_within_label": within})
    return sorted(out, key=lambda x: -x["n"])


def players(m, s, fid):
    here = m[(m["fid"] == fid) & ~m["cancelled"]]
    ids = here["pbp_user_id"].unique()
    f = here.groupby("pbp_user_id").size()
    buckets = {"1 session": int((f == 1).sum()), "2 sessions": int((f == 2).sum()),
               "3-4 sessions": int(f.between(3, 4).sum()), "5+ sessions": int((f >= 5).sum())}
    top = f.sort_values(ascending=False)
    allp = m[m["pbp_user_id"].isin(ids) & ~m["cancelled"]]
    share = allp.assign(here=allp["fid"] == fid).groupby("pbp_user_id")["here"].mean()
    loyalty = {"mostly here (over 80%)": int((share > 0.8).sum()),
               "split (50-80%)": int(share.between(0.5, 0.8).sum()),
               "mostly elsewhere (under 50%)": int((share < 0.5).sum())}
    # retention, low data: players whose first session here (since capture) was
    # in the window's first week; did they book another session here after it?
    first = here.sort_values("starts_at").drop_duplicates("pbp_user_id")
    w_from = s["starts_at"].min()
    w1 = first[(first["starts_at"] >= w_from) & (first["starts_at"] < w_from + pd.Timedelta(days=7))]
    later = here.merge(w1[["pbp_user_id", "starts_at"]].rename(columns={"starts_at": "first_at"}), on="pbp_user_id")
    came_back = later[later["starts_at"] > later["first_at"]]["pbp_user_id"].unique()
    # by what their first session was
    path = []
    for fam, g in w1.groupby(w1["family"].fillna("unknown")):
        back = g["pbp_user_id"].isin(came_back).sum()
        path.append({"first_session": fam, "players": len(g), "came_back": int(back), "share": pct(back, len(g))})
    rated = here["rating"].dropna()
    return {"players": len(ids), "bookings": int(len(here)), "frequency": buckets,
            "top20_share": pct(int(top.head(max(1, int(len(top) * 0.2))).sum()), int(top.sum())),
            "loyalty": loyalty,
            "retention": {"first_week_players": len(w1), "came_back": int(len(came_back)),
                          "share": pct(len(came_back), len(w1)), "by_first_session": path},
            "ratings": {"rows": int(len(here)), "rated": int(len(rated)), "share": pct(len(rated), len(here)),
                        "dist": {str(k): int(v) for k, v in rated.value_counts().sort_index().items()}}}


_DAY_IX: dict = {}


def _day_index(s):
    """Sessions by (venue, Melbourne date), built once per sessions frame: the
    traveller and rival checks look up one venue-day at a time."""
    key = id(s)
    if key not in _DAY_IX:
        x = s[s["fid"].notna() & s["family"].notna()].copy()
        x["_day"] = x["starts_at"].dt.tz_convert(MEL).dt.date
        _DAY_IX.clear()
        _DAY_IX[key] = {(int(f), d): g for (f, d), g in x.groupby([x["fid"].astype(int), "_day"])}
    return _DAY_IX[key]


def _clashes(ix, fids, v):
    """Is there a session of the same kind, and an overlapping level band, at
    one of `fids` on the same day within an hour of session `v`?"""
    day = v.starts_at.tz_convert(MEL).date()
    for f in fids:
        c = ix.get((int(f), day))
        if c is None:
            continue
        c = c[((c["starts_at"] - v.starts_at).abs() <= pd.Timedelta(hours=1)) & (c["family"] == v.family)]
        if not np.isnan(v.band_lo):
            c = c[((c["band_lo"] <= v.band_hi) & (c["band_hi"] >= v.band_lo)) | c["band_lo"].isna()]
        if len(c):
            return True
    return False


def _clashes_strict(ix, fids, v):
    """As _clashes, but a candidate with no level band doesn't count when the
    session states one (the traveller rule, unchanged)."""
    day = v.starts_at.tz_convert(MEL).date()
    for f in fids:
        c = ix.get((int(f), day))
        if c is None:
            continue
        c = c[((c["starts_at"] - v.starts_at).abs() <= pd.Timedelta(hours=1)) & (c["family"] == v.family)]
        if not np.isnan(v.band_lo):
            c = c[(c["band_lo"] <= v.band_hi) & (c["band_hi"] >= v.band_lo)]
        if len(c):
            return True
    return False


def travellers(m, s, fid):
    """The traveller tree from the catalogue doc: why players whose home venue is
    over 5 km away come to this venue."""
    ok = m[~m["cancelled"]]
    ids = ok[ok["fid"] == fid]["pbp_user_id"].unique()
    allp = ok[ok["pbp_user_id"].isin(ids)]
    home = (allp.groupby(["pbp_user_id", "fid"]).size().reset_index(name="n")
            .sort_values(["pbp_user_id", "n"], ascending=[True, False]).drop_duplicates("pbp_user_id"))
    home["km"] = home["fid"].map(lambda f: km(fid, f))
    trav = home[(home["fid"] != fid) & (home["km"] > 5)]
    near_here = {f for f in REG if f != fid and km(fid, f) <= 5}
    ix = _day_index(s)
    by_player = {k: g for k, g in allp.groupby("pbp_user_id")}
    out, classes = [], {"product draw": 0, "preference draw": 0, "location anchor": 0, "session draw": 0}
    for row in trav.itertuples():
        pid, hfid = row.pbp_user_id, row.fid
        home_area = {f for f in REG if km(hfid, f) <= 5} | {hfid}
        theirs = by_player.get(pid)
        mine = theirs[theirs["fid"] == fid]
        # Q1: was the same kind of session on near home, same day, within an hour?
        # (A session with no family can't be matched, as before.)
        same_near = any(_clashes_strict(ix, home_area, v) for v in mine.itertuples() if isinstance(v.family, str))
        slots = set(zip(mine["dow"], mine["hour"]))
        workhours = all(d < 5 and (6 <= h < 9 or 12 <= h < 14 or 17 <= h < 20) for d, h in slots)
        other_near = theirs["fid"].isin(near_here).any()
        if not same_near:
            cls = "product draw"
        elif len(slots) > 1:
            cls = "preference draw"
        elif workhours and not other_near:
            cls = "location anchor"
        else:
            cls = "session draw"
        classes[cls] += 1
        out.append({"home": name(hfid), "km": round(row.km, 1), "sessions_here": len(mine), "class": cls,
                    "titles": sorted(set(mine["title"].dropna()))})
    homes = pd.Series([o["home"] for o in out]).value_counts().to_dict()
    draws = pd.Series([t for o in out if o["class"] == "product draw" for t in o["titles"]]).value_counts().to_dict()
    return {"players": len(ids), "travellers": len(out), "classes": classes, "homes": homes,
            "product_draw_sessions": draws, "km_median": round(float(trav["km"].median()), 1) if len(trav) else None}


def rivals(m, s, fin, fid):
    ok = m[~m["cancelled"]]
    ids = ok[ok["fid"] == fid]["pbp_user_id"].unique()
    allp = ok[ok["pbp_user_id"].isin(ids)]
    total = len(allp)
    shared = (allp[allp["fid"] != fid].groupby("fid")
              .agg(sessions=("session_key", "size"), players=("pbp_user_id", "nunique")).reset_index())
    here = s[(s["fid"] == fid) & s["family"].notna()]
    ix = _day_index(s)
    out = []
    for row in shared.sort_values("sessions", ascending=False).head(6).itertuples():
        f = int(row.fid)
        clash = sum(_clashes(ix, [f], v) for v in here.itertuples())
        ff = fin[fin["fid"] == f]
        out.append({"fid": f, "name": name(f), "km": round(km(fid, f), 1),
                    "shared_players": int(row.players), "shared_sessions": int(row.sessions),
                    "shared_play_pct": pct(int(row.sessions), total),
                    "head_to_head_pct": pct(clash, len(here)),
                    "avg_fill": None if ff.empty else round(100 * ff["fill"].mean(), 1),
                    "sessions_logged": int((s["fid"] == f).sum())})
    return {"players": len(ids), "their_sessions": total, "venues": out}


def compare(s, fids):
    """Public facts, side by side: what each of these venues runs."""
    rows = []
    for f in fids:
        g = s[(s["fid"] == f) & s["title"].notna()]
        weeks = max(1.0, (g["starts_at"].max() - g["starts_at"].min()).days / 7)
        mix = (g.groupby(["family", "level_class"], dropna=False).size() / weeks).round(1)
        rows.append({"venue": name(f), "sessions_per_week": round(len(g) / weeks, 1),
                     "price_hr_median": round(float(g["price_hr"].median()), 2) if g["price_hr"].notna().any() else None,
                     "mix": {f"{a or 'unknown'} / {b or 'no level'}": float(v) for (a, b), v in mix.items()}})
    return rows


def similar(fin, fid, family, level, daypart, weekend):
    """Other venues' finished sessions of the same kind: the market's answer to
    'would this fill?'. The venue's own sessions are left out."""
    fin = fin[fin["fid"] != fid]
    g = fin[(fin["family"] == family) & (fin["level_class"] == level)
            & (fin["daypart"] == daypart) & (fin["weekend"] == weekend)]
    if g.empty:
        return {"n": 0}
    return {"n": len(g), "venues": int(g["fid"].nunique()), "median_fill": round(100 * g["fill"].median(), 1),
            "p25": round(100 * g["fill"].quantile(.25), 1), "p75": round(100 * g["fill"].quantile(.75), 1),
            "sold_out_share": pct(int(g["sold_out"].sum()), len(g))}


def early_social(fin, fid):
    g = fin[(fin["fid"] != fid) & (fin["family"] == "social") & (fin["daypart"] == "early") & ~fin["weekend"]]
    return {"n": len(g), "venues": int(g["fid"].nunique()),
            "median_fill": None if g.empty else round(100 * g["fill"].median(), 1)}


def price_position(fin, fid):
    out = []
    here = fin[(fin["fid"] == fid) & fin["price_hr"].notna()]
    for (fam, lvl), g in here.groupby(["family", "level_class"]):
        mk = fin[(fin["family"] == fam) & (fin["level_class"] == lvl) & fin["price_hr"].notna() & (fin["fid"] != fid)]
        if mk.empty:
            continue
        out.append({"family": fam, "level": lvl, "venue_price_hr": round(float(g["price_hr"].median()), 2),
                    "market_price_hr": round(float(mk["price_hr"].median()), 2),
                    "market_p25": round(float(mk["price_hr"].quantile(.25)), 2),
                    "market_p75": round(float(mk["price_hr"].quantile(.75)), 2),
                    "market_n": len(mk), "venue_fill": round(100 * g["fill"].mean(), 1),
                    "market_fill": round(100 * mk["fill"].mean(), 1)})
    return out


def main(fid: int):
    global NOW
    fid = int(fid)
    s, r, c = load()
    NOW = max(s["last_obs"].max(), r["last_seen"].max())
    s = attach_titles(s, c)
    fin = finished(s)
    fin_v = fin[fin["fid"] == fid]
    m = rosters(s, r)
    m_v = m[m["fid"] == fid]
    sim = lambda *a: similar(fin, fid, *a)
    out = {
        "as_of": str(NOW), "facility_id": fid, "venue": name(fid),
        "title_coverage": s[s["fid"] == fid]["title_how"].value_counts(dropna=False).to_dict(),
        "headline": headline(fin_v),
        "sessions": by_title(fin_v, m_v),
        "players": players(m, s, fid),
        "travellers": travellers(m, s, fid),
        "rivals": rivals(m, s, fin, fid),
        "nearest_compare": compare(s, [fid] + nearest(fid)),
        "price_position": price_position(fin, fid),
        "similar": {
            "beginner social, weekday evening": sim("social", "beginner", "evening", False),
            "beginner social, weekday day": sim("social", "beginner", "day", False),
            "intermediate social, weekday evening": sim("social", "intermediate", "evening", False),
            "intermediate social, weekday day": sim("social", "intermediate", "day", False),
            "advanced social, weekday evening": sim("social", "advanced", "evening", False),
            "advanced social, weekday day": sim("social", "advanced", "day", False),
            "competitive, weekday evening, intermediate": sim("competitive", "intermediate", "evening", False),
            "any social, early morning weekday": early_social(fin, fid),
            "learning, early morning weekday": sim("learning", None, "early", False),
        },
    }
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / f"venue_report_{fid}.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2 or not sys.argv[1].isdigit():
        sys.exit("usage: python insights/venue_report.py <facility_id>")
    o = main(int(sys.argv[1]))
    print(json.dumps({k: o[k] for k in ("venue", "title_coverage", "headline")}, indent=1, default=str))
