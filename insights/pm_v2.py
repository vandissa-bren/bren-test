"""
pm_v2.py -- the PM level, version 2: each player's TRUE level (13 Nov 2026).

Internal only (Insights > Staff > PM level). Nothing here reaches players, and
the build publishes only counts of 5 or more. The aim is the level a player
SHOULD be, not a copy of their DUPR: DUPR is the strongest single reading, but
a "3.0" who books 4.0 rooms week after week is probably not a 3.0, so the PM
level is free to disagree with it when the rooms say otherwise.

ONE FIT, PLAYERS AND ROOMS TOGETHER. A room's level comes from its players and
a player's level partly from their rooms, so both are solved at once, in
rounds, until nothing moves (about 10). Each round, for every player:

    level = (wD*dupr + sum over weekly sessions w*(room + offset) + wL*labels + wS*self)
            / (wD + sum w + wL + wS)

  DUPR      doubles DUPR from PlayByPoint (rosters and the rating log), moved
            the way it has been heading (at most 0.25), with a spread of 0.20
            plus 0.05 for every month since it last changed.
  rooms     for each weekly session they book: the level of everyone else in
            it, weighted by how sure we are of each of them. Each weekly
            session is ONE reading however often they go: a player's gap from
            their room is the same every week, so repeat visits add no new
            evidence about it. A session they tried once counts half; coming
            back counts in full; one they've stopped going to fades (half
            every 8 weeks). A room that disagrees wildly
            with the rest of a player's evidence counts less (robust weights).
  labels    what the sessions they book say they are (the label's midpoint,
            corrected for how high or low that venue's labels run), weak.
  self      their PlayByPoint self-rating, calibrated against real DUPRs (v1's
            calibration), weak.

The spread and offset for each kind of session (social, learning, competitive;
level-banded or open to all) are NOT guessed: each round re-measures them from
players with a DUPR, as the typical gap between their DUPR and the room. If
rooms turn out to say little about a player, their spread grows and DUPR wins;
if they say a lot, rooms win.

Coaches booked as players would pull a room up, so players who look like
coaches (in half or more of the runs of 2+ coached weekly sessions at a venue,
6+ coached sessions in all) are left out of every room's level.

A player's confidence is 1/sqrt(total weight): the "sd". A session's PM level
is the typical level of the players who booked its last 8 runs (5+ players
with sd <= 0.4), with its spread (10th to 90th percentile), next to its label.

TESTS (every run):
  forward   fit on everything up to 4 weeks ago, then see which DUPRs moved
            since. Where the PM level sat above a player's DUPR, did their DUPR
            rise? This is the real test of "what it should be".
  hidden    v1's check: hide a fifth of the DUPRs, estimate them from the
            rest. A sanity check only: v2 is meant to disagree with stale DUPRs.

fit(m, s, ratelog, end) -> (summary for the build, levels for record_pm_run)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ratings import dupr

VERSION = "v2"
MIN_GROUP = 5
FAMS = ("social", "learning", "competitive", "unknown")
CFG = dict(
    sD=0.20, sD_month=0.05, sD_max=0.60,       # DUPR spread, and its growth when it hasn't moved
    trend_weeks=4, trend_cap=0.25,             # lean the way DUPR is heading
    s_band=0.35, s_open=0.60,                  # room spreads to start (re-measured)
    s_floor=0.25, s_ceil=0.90, min_anchor=30,  # bounds on the re-measured spreads
    centre=0.35, s_edge=0.15, edge_k=0.75, edge_visits=3,      # a light pull to a room's middle; below its lower
                                               # edge (middle - edge_k x its spread), a firm pull up
    half_life_wk=8, visit_cap=1.0, first_visit=0.5,
    sL=0.50, sV=0.70, label_with_dupr=0.25, sS=0.50, city=3.1, sC=1.00,
    conf_k=10.0, damp=0.7, rounds=40, tol=0.003, nu=4.0,
    sure_sd=0.40, room_runs=8,
    centre_cap=2.0,                            # most the room middles together can weigh (see solve)
    label_anchor_n=3, label_shrink=5,          # venue label offset: sessions with 3+ DUPR players, shrunk to the city's
)
SETTINGS = dict(CFG)
BINS = [0, 2.5, 3.0, 3.5, 4.0, 9]
BIN_LABELS = ["below 2.5", "2.5–3.0", "3.0–3.5", "3.5–4.0", "4.0+"]
SELF_REAL = {2.5, 3.0, 3.5, 4.0, 4.5}


def _r(x, k=2):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), k)


def _hold(n):
    n = int(n)
    return n if n == 0 or n >= MIN_GROUP else None


# ── the inputs ────────────────────────────────────────────────────────────────
def prepare(m: pd.DataFrame, s: pd.DataFrame):
    """Visits (one per player per session played) and sessions, with what the
    model needs. m: the build's rosters (venue_report.rosters); s: sessions."""
    d = m[~m["cancelled"] & m["fid"].notna() & m["starts_at"].notna()].copy()
    d["pbp_user_id"] = d["pbp_user_id"].astype("int64")
    d = d.sort_values("last_seen").drop_duplicates(["pbp_user_id", "session_key"], keep="last")
    sess = s[s["session_key"].isin(set(d["session_key"]))].drop_duplicates("session_key").copy()
    sess = sess[sess["fid"].notna()]
    sess["family"] = sess["family"].where(sess["family"].isin(FAMS), "unknown")
    lo, hi = sess["band_lo"], sess["band_hi"].clip(upper=5.0)
    sess["banded"] = lo.notna() & ((hi - lo) <= 1.0)
    sess["label_mid"] = ((lo + hi) / 2).where(lo.notna())
    sess["coached"] = (sess["family"] == "learning") | sess.get("coaches", pd.Series(None, index=sess.index)).map(
        lambda v: bool(v) and str(v) not in ("[]", "nan", "None"))
    sess["series"] = _series(sess)
    d = d[d["session_key"].isin(set(sess["session_key"]))]
    return d, sess.set_index("session_key")


def _series(sess: pd.DataFrame) -> pd.Series:
    """The weekly session each one belongs to (series.py); a session with no
    series (a one-off, a competition) is its own."""
    try:
        import series as SR
        x = sess.assign(lvl=sess.get("level_class"))
        sid = SR.assign_series(x, kind="session_type", level="lvl")
    except Exception:
        sid = pd.Series([None] * len(sess), index=sess.index, dtype=object)
    return pd.Series([str(v) if v is not None and v == v else f"one:{k}" for v, k in zip(sid, sess["session_key"])],
                     index=sess.index)


def dupr_history(m: pd.DataFrame, ratelog) -> pd.DataFrame:
    """Every DUPR reading we have: (pbp_user_id, at, value). From the rosters
    (rating_at_time when the session was read) and the rating log."""
    a = m[["pbp_user_id", "last_seen", "rating_at_time"]].rename(columns={"last_seen": "at", "rating_at_time": "v"})
    parts = [a]
    if ratelog is not None and len(ratelog):
        b = ratelog[["pbp_user_id", "seen_at", "rating"]].rename(columns={"seen_at": "at", "rating": "v"})
        parts.append(b)
        if "prev_rating" in ratelog and "prev_seen_at" in ratelog:
            c = ratelog[["pbp_user_id", "prev_seen_at", "prev_rating"]].rename(
                columns={"prev_seen_at": "at", "prev_rating": "v"})
            parts.append(c)
    h = pd.concat(parts, ignore_index=True)
    h["v"] = dupr(h["v"])
    h["at"] = pd.to_datetime(h["at"], utc=True, errors="coerce")
    h = h.dropna(subset=["v", "at"])
    h["pbp_user_id"] = pd.to_numeric(h["pbp_user_id"], errors="coerce")
    h = h.dropna(subset=["pbp_user_id"])
    h["pbp_user_id"] = h["pbp_user_id"].astype("int64")
    return h.sort_values(["pbp_user_id", "at"]).drop_duplicates(["pbp_user_id", "at", "v"])


def dupr_now(h: pd.DataFrame, asof) -> pd.DataFrame:
    """Per player at `asof`: latest DUPR, months since it last changed, and
    the value moved the way it has been heading (capped)."""
    c = CFG
    h = h[h["at"] <= asof]
    if h.empty:
        return pd.DataFrame(columns=["dupr", "months", "lean", "target"])
    out = {}
    for pid, g in h.groupby("pbp_user_id", sort=False):
        v = g["v"].to_numpy()
        t = g["at"].to_numpy()
        last = v[-1]
        changed = np.nonzero(np.abs(np.diff(v)) >= 0.005)[0]
        since = t[changed[-1] + 1] if len(changed) else t[0]
        months = max(0.0, (pd.Timestamp(asof) - pd.Timestamp(since)).total_seconds() / (30.44 * 86400))
        lean = 0.0
        span_wk = (pd.Timestamp(t[-1]) - pd.Timestamp(t[0])).total_seconds() / (7 * 86400)
        if len(changed) and span_wk >= 3:
            slope = (last - v[0]) / span_wk
            k = len(changed)
            lean = float(np.clip(slope * c["trend_weeks"] * k / (k + 2), -c["trend_cap"], c["trend_cap"]))
        out[pid] = (last, months, lean, last + lean)
    return pd.DataFrame.from_dict(out, orient="index", columns=["dupr", "months", "lean", "target"])


def self_ratings(d: pd.DataFrame) -> pd.Series:
    raw = pd.to_numeric(d["rating_at_time"], errors="coerce")
    sv = raw.where(raw.isin(SELF_REAL) & dupr(d["rating_at_time"]).isna())
    x = d.assign(sv=sv).dropna(subset=["sv"])
    return x.groupby("pbp_user_id")["sv"].agg(lambda v: v.mode().iloc[0]) if len(x) else pd.Series(dtype=float)


def likely_coaches(d: pd.DataFrame, sess: pd.DataFrame) -> set:
    """Players in half or more of the runs of 2+ coached weekly sessions at a
    venue, with 6+ coached sessions in all."""
    co = sess[sess["coached"]]
    if co.empty:
        return set()
    runs = co.groupby("series").size()
    x = d[d["session_key"].isin(co.index)].assign(series=lambda z: z["session_key"].map(co["series"]))
    per = x.groupby(["pbp_user_id", "series"]).size().reset_index(name="n")
    per["share"] = per["n"] / per["series"].map(runs)
    many = per[(per["share"] >= 0.5) & (per["series"].map(runs) >= 3)].groupby("pbp_user_id").size()
    total = x.groupby("pbp_user_id").size()
    return set(many[(many >= 2)].index) & set(total[total >= 6].index)


# ── the fit ───────────────────────────────────────────────────────────────────
def solve(d, sess, dn: pd.DataFrame, self_cal, hide=frozenset(), asof=None, cfg=CFG):
    """One fit. d: visits; sess: sessions (indexed by key); dn: DUPR per player
    (dupr_now); hide: players whose DUPR is left out (tests). Returns a frame
    per player (level, sd, kind, n) and the measured groups and rooms."""
    c = cfg
    asof = pd.Timestamp(asof) if asof is not None else d["starts_at"].max()
    pids = np.array(sorted(d["pbp_user_id"].unique()))
    P = len(pids)
    pidx = pd.Series(np.arange(P), index=pids)
    skeys = sess.index.to_numpy()
    sidx = pd.Series(np.arange(len(skeys)), index=skeys)
    vi = pidx[d["pbp_user_id"].to_numpy()].to_numpy()
    vs = sidx[d["session_key"].to_numpy()].to_numpy()
    age = ((asof - d["starts_at"]).dt.total_seconds() / (7 * 86400)).clip(lower=0).to_numpy()
    rec = 0.5 ** (age / c["half_life_wk"])
    S = len(skeys)

    fam = sess["family"].to_numpy()
    banded = sess["banded"].to_numpy()
    grp_of_s = np.array([FAMS.index(f) * 2 + int(b) for f, b in zip(fam, banded)])     # 8 groups
    ser_codes, ser_of_s = np.unique(sess["series"].to_numpy(), return_inverse=True)
    venue_of_s = sess["fid"].astype(int).to_numpy()
    label_mid = sess["label_mid"].to_numpy()
    # a label with no real bottom ("Learn to play", "Beginner 2.0") has no
    # lower edge: nobody is pulled up by it
    # only a level-banded room with a real floor has a lower edge: one open to
    # all, or with no stated level, says nothing about how low its players go
    open_bot = (sess["band_lo"].fillna(0).to_numpy() <= 2.0) | ~sess["banded"].to_numpy()

    # pairs: (player, weekly session), with their visits' recency
    pair = vi.astype(np.int64) * len(ser_codes) + ser_of_s[vs]
    pkeys, pinv = np.unique(pair, return_inverse=True)
    pair_player = (pkeys // len(ser_codes)).astype(int)
    pair_visits = np.bincount(pinv, minlength=len(pkeys))
    pair_rho = np.bincount(pinv, weights=rec, minlength=len(pkeys))
    pair_rep = np.where(pair_visits >= 2, 1.0, c["first_visit"])
    pair_cnt = np.minimum(pair_rho, c["visit_cap"])
    # each pair's kind of session: its most common group
    pair_grp = np.zeros(len(pkeys), dtype=int)
    pair_grp[pinv] = grp_of_s[vs]                 # a weekly session keeps one kind; any visit's will do
    pair_coaching = np.zeros(len(pkeys), bool)
    pair_coaching[pinv] = (np.isin(pids, list(self_cal.get("coaches", set())))[vi]
                           & sess["coached"].fillna(False).to_numpy().astype(bool)[vs])

    # DUPR
    dn = dn.reindex(pids)
    has_d = dn["dupr"].notna().to_numpy() & ~np.isin(pids, list(hide))
    d_target = dn["target"].to_numpy()
    sD = np.minimum(c["sD"] + c["sD_month"] * dn["months"].fillna(0).to_numpy(), c["sD_max"])
    wD = np.where(has_d, 1 / sD ** 2, 0.0)

    # self-ratings, calibrated
    sv = self_cal["values"].reindex(pids).to_numpy()
    has_s = ~np.isnan(sv)
    s_val = self_cal["a"] + self_cal["b"] * (np.nan_to_num(sv) - 3.0)
    wS = np.where(has_s, 1 / c["sS"] ** 2, 0.0)

    coach = np.isin(pids, list(self_cal.get("coaches", set())))
    # a likely coach's coached sessions say nothing about how they play: they
    # run them. Their level comes from everything else (or their venue's).
    coached_s = sess["coached"].fillna(False).to_numpy().astype(bool)
    coaching = coach[vi] & coached_s[vs]

    # labels: per player, recency-weighted mean label midpoint of what they book
    lm = label_mid[vs]
    ok = ~np.isnan(lm) & ~coaching
    lab_w = np.bincount(vi[ok], weights=rec[ok], minlength=P)
    venue_ids = np.unique(venue_of_s)
    lab_off = pd.Series(0.0, index=venue_ids)

    # home venue (most visits): fallback start
    hv = pd.DataFrame({"p": vi, "v": venue_of_s[vs]}).value_counts().reset_index(name="n")
    home = hv.sort_values("n", ascending=False).drop_duplicates("p").set_index("p")["v"].reindex(range(P)).to_numpy()

    theta = np.where(has_d, d_target, np.where(has_s, s_val, c["city"]))
    th_lab = np.full(P, np.nan)
    grp_off = np.zeros(8)
    grp_sd = np.array([c["s_band"] if g % 2 else c["s_open"] for g in range(8)])
    prec = wD + wS + 1.0
    rounds = 0
    for rounds in range(1, c["rounds"] + 1):
        conf = np.where(coach, 0.0, prec / (prec + c["conf_k"]))
        # room sums per session, and each visit's leave-one-out room level
        Ssum = np.bincount(vs, weights=conf[vi] * theta[vi], minlength=S)
        Csum = np.bincount(vs, weights=conf[vi], minlength=S)
        Nsum = np.bincount(vs, weights=(conf[vi] > 0).astype(float), minlength=S)
        Qsum = np.bincount(vs, weights=conf[vi] * theta[vi] ** 2, minlength=S)
        s_mean = Ssum / np.maximum(Csum, 1e-9)
        s_sd = np.sqrt(np.maximum(Qsum / np.maximum(Csum, 1e-9) - s_mean ** 2, 0.0))
        loo_c = Csum[vs] - conf[vi]
        loo_n = Nsum[vs] - (conf[vi] > 0)
        room = np.where((loo_c > 0.3) & (loo_n >= 2), (Ssum[vs] - conf[vi] * theta[vi]) / np.maximum(loo_c, 1e-9), np.nan)
        # per pair: recency-weighted room level
        okr = ~np.isnan(room)
        pr_w = np.bincount(pinv[okr], weights=rec[okr], minlength=len(pkeys))
        pr_x = np.bincount(pinv[okr], weights=rec[okr] * room[okr], minlength=len(pkeys))
        pair_room = np.where(pr_w > 0, pr_x / np.maximum(pr_w, 1e-12), np.nan)
        pr_sd = np.bincount(pinv[okr], weights=rec[okr] * s_sd[vs][okr], minlength=len(pkeys))
        pair_sd = np.where(pr_w > 0, pr_sd / np.maximum(pr_w, 1e-12), np.nan)
        pair_obot = np.zeros(len(pkeys), bool)
        pair_obot[pinv] = open_bot[vs]

        # re-measure each kind of session's offset and spread from DUPR players
        if rounds >= 2:
            anc = has_d[pair_player] & ~np.isnan(pair_room)
            for g in range(8):
                sel = anc & (pair_grp == g)
                if sel.sum() >= c["min_anchor"]:
                    gap = d_target[pair_player[sel]] - pair_room[sel]
                    med = float(np.median(gap))
                    mad = float(np.median(np.abs(gap - med))) * 1.4826
                    grp_off[g] = med
                    grp_sd[g] = float(np.clip(mad, c["s_floor"], c["s_ceil"]))
            # how high or low each venue's labels run, measured ONLY from the
            # DUPR players in its sessions (a room level built from players with
            # no DUPR leans on the labels themselves, and at a venue with few
            # DUPRs that circle erased what the labels say). Each venue's
            # offset is shrunk to the city's by how many sessions back it.
            dsel = has_d[vi]
            Dsum = np.bincount(vs[dsel], weights=d_target[vi[dsel]], minlength=S)
            Dn = np.bincount(vs[dsel], minlength=S)
            dmean = np.where(Dn >= c["label_anchor_n"], Dsum / np.maximum(Dn, 1), np.nan)
            okl = ~np.isnan(dmean) & ~np.isnan(label_mid)
            if okl.any():
                gap_s = pd.Series(dmean[okl] - label_mid[okl])
                city_off = float(gap_s.median())
                per = gap_s.groupby(venue_of_s[okl]).agg(["median", "size"])
                k0 = c["label_shrink"]
                lab_off = ((per["median"] * per["size"] + city_off * k0) / (per["size"] + k0)).reindex(
                    venue_ids).fillna(city_off)

        # labels, corrected per venue
        corr = lab_off.reindex(venue_of_s[vs]).to_numpy()
        lab_x = np.bincount(vi[ok], weights=rec[ok] * (lm[ok] + corr[ok]), minlength=P)
        th_lab = np.where(lab_w > 0, lab_x / np.maximum(lab_w, 1e-12), np.nan)
        venue_lvl = pd.Series(theta).groupby(home).median()
        fallback = pd.Series(home).map(venue_lvl).fillna(c["city"]).to_numpy()
        wL = np.where(lab_w > 0, 1 / c["sL"] ** 2, 1 / c["sV"] ** 2) * np.where(has_d, c["label_with_dupr"], 1.0)
        lab_val = np.where(lab_w > 0, th_lab, fallback)

        # rooms: a room says a player's level is within its range. Inside it, a
        # light pull to its middle (all a player with nothing else has to go
        # on); outside it, a firm pull back to the edge they're past. Robust to
        # a room that disagrees wildly with the rest of their evidence.
        val = pair_room + grp_off[pair_grp]
        okp = ~np.isnan(val)
        base = pair_rep * pair_cnt
        z2 = ((val - theta[pair_player]) / grp_sd[pair_grp]) ** 2
        robust = (c["nu"] + 1) / (c["nu"] + np.nan_to_num(z2))
        half = c["edge_k"] * np.maximum(np.nan_to_num(pair_sd), grp_sd[pair_grp] * 0.5)
        # Only ever UP to a room's lower edge: holding a place in a stronger
        # room is evidence of level, but strong players often play down
        # (socials with friends), so being above a room says nothing.
        lo = np.where(pair_obot, -np.inf, val - half)
        th_p = theta[pair_player]
        edge = np.where(th_p < lo, lo, np.nan)
        w_c = np.where(okp, base * c["centre"] / grp_sd[pair_grp] ** 2 * robust, 0.0)
        # a player with a DUPR isn't pulled to a room's middle at all (the
        # weakest and strongest in any room sit away from it); only the firm
        # pull up below the edge moves them. A player without one has little
        # else to go on, so the middle pulls both ways.
        w_c = np.where(has_d[pair_player] | pair_coaching, 0.0, w_c)
        # the middles of the rooms a player shares with the same people aren't
        # separate pieces of evidence: cap their total, so a player in many
        # sessions isn't squeezed to their venue's middle (labels and their
        # DUPR-anchored offsets keep their say)
        if c["centre_cap"]:
            tot_c = np.bincount(pair_player, weights=w_c, minlength=P)
            w_c = w_c * np.minimum(1.0, c["centre_cap"] / np.maximum(tot_c, 1e-9))[pair_player]
        # the firm pull up is for players who KEEP COMING BACK to a stronger room
        w_e = np.where(okp & ~np.isnan(edge) & (pair_visits >= c["edge_visits"]) & ~pair_coaching,
                       base / c["s_edge"] ** 2, 0.0)
        wR = np.bincount(pair_player, weights=w_c + w_e, minlength=P)
        xR = np.bincount(pair_player, weights=w_c * np.nan_to_num(val) + w_e * np.nan_to_num(edge), minlength=P)

        num = wD * np.nan_to_num(d_target) + xR + wL * lab_val + wS * s_val
        prec = wD + wR + wL + wS
        new = num / prec
        step = new - theta
        theta = theta + c["damp"] * step
        if np.abs(step).max() < c["tol"]:
            break

    sd = 1 / np.sqrt(prec)
    n = np.bincount(vi, minlength=P)
    kind = np.where(has_d, "dupr", np.where(has_s, "self", "none"))
    T = pd.DataFrame({"level": theta, "sd": sd, "n": n, "kind": kind, "home": home, "coach": coach,
                      "room_w": wR}, index=pids)
    groups = [{"family": FAMS[g // 2], "banded": bool(g % 2), "offset": _r(grp_off[g]), "spread": _r(grp_sd[g]),
               "pairs": _hold((pair_grp == g).sum())} for g in range(8) if (pair_grp == g).any()]
    meta = {"rounds": rounds, "groups": groups, "labelOffsets": lab_off, "vi": vi, "vs": vs, "rec": rec,
            "pids": pids, "skeys": skeys,
            # the last round's evidence, for each player's breakdown (pm_player.py):
            # level ~= (wD*dT + sum over pairs (wC*val + wE*edge) + wL*lab + wS*self) / prec
            "x": {"wD": wD, "dT": d_target, "sD": sD, "hasD": has_d, "wS": wS, "self": s_val, "hasS": has_s,
                  "selfRaw": sv, "wL": wL, "lab": lab_val, "labOwn": lab_w > 0, "prec": prec, "new": new,
                  "pPlayer": pair_player, "pSeries": ser_codes[pkeys % len(ser_codes)], "pVisits": pair_visits,
                  "pRoom": pair_room, "pVal": val, "pLo": lo, "pEdge": edge, "wC": w_c, "wE": w_e,
                  "pOpen": pair_obot, "grpOff": grp_off[pair_grp], "pCoaching": pair_coaching}}
    return T, meta


def rooms(T: pd.DataFrame, d: pd.DataFrame, sess: pd.DataFrame, cfg=CFG):
    """Each weekly session's PM level: its last 8 runs' players with a sure
    level (5+), weighted median and 10th-90th percentile, against its label."""
    sure = T[(T["sd"] <= cfg["sure_sd"]) & ~T["coach"]]
    x = d[d["pbp_user_id"].isin(sure.index)].assign(series=lambda z: z["session_key"].map(sess["series"]))
    x = x.dropna(subset=["series"])
    out = []
    for ser, g in x.groupby("series"):
        keys = g.drop_duplicates("session_key").sort_values("starts_at")["session_key"].tail(cfg["room_runs"])
        g = g[g["session_key"].isin(set(keys))]
        lv = sure.loc[g["pbp_user_id"].unique(), "level"]
        if len(lv) < MIN_GROUP:
            continue
        last = sess.loc[keys.iloc[-1]]
        lo, hi = last.get("band_lo"), last.get("band_hi")
        mid = last.get("label_mid")
        lvl = float(lv.median())
        st = pd.Timestamp(last["starts_at"]).tz_convert("Australia/Melbourne")
        out.append({"series": ser, "v": int(last["fid"]), "title": last.get("title") if isinstance(last.get("title"), str) else None,
                    "label": last.get("label") if isinstance(last.get("label"), str) else None,
                    "dow": int(st.dayofweek), "hm": st.strftime("%H:%M"), "runs": int(len(keys)),
                    "players": int(len(lv)), "level": _r(lvl), "lo": _r(lv.quantile(.1)), "hi": _r(lv.quantile(.9)),
                    "labelLo": _r(lo) if lo == lo else None, "labelHi": _r(min(hi, 5.0)) if hi == hi else None,
                    "gap": _r(lvl - mid) if mid == mid else None,
                    "family": last.get("family")})
    return out


# ── tests ─────────────────────────────────────────────────────────────────────
def forward_test(d, sess, h, self_cal, end, weeks=4):
    """Fit as of `weeks` ago; did DUPRs then move the way the PM level said?"""
    cut = pd.Timestamp(end) - pd.Timedelta(weeks=weeks)
    d0 = d[d["starts_at"] <= cut]
    if d0["pbp_user_id"].nunique() < 50:
        return None
    s0 = sess[sess.index.isin(set(d0["session_key"]))]
    dn0 = dupr_now(h, cut)
    T0, _ = solve(d0, s0, dn0, self_cal, asof=cut)
    later = h[h["at"] > cut + pd.Timedelta(days=7)].groupby("pbp_user_id")["v"].last()
    both = T0.join(dn0[["dupr"]], how="inner").join(later.rename("after"), how="inner").dropna(subset=["dupr", "after"])
    if len(both) < 25:
        return None
    gap = both["level"] - both["dupr"]
    move = both["after"] - both["dupr"]
    movers = both[move.abs() >= 0.02]
    called = movers[(movers["level"] - movers["dupr"]).abs() >= 0.05]
    hit = (np.sign(called["level"] - called["dupr"]) == np.sign(called["after"] - called["dupr"]))
    up, down = gap >= 0.25, gap <= -0.25
    near = gap.abs() < 0.1
    return {
        "asOf": str(cut.date()), "weeks": weeks, "players": _hold(len(both)), "movers": _hold(len(movers)),
        "called": _hold(len(called)), "hitPct": _r(100 * hit.mean(), 1) if len(called) >= MIN_GROUP else None,
        "corr": _r(np.corrcoef(gap, move)[0, 1], 3) if len(both) >= 25 and gap.std() > 0 and move.std() > 0 else None,
        "above": {"players": _hold(up.sum()), "move": _r(move[up].mean(), 3) if up.sum() >= MIN_GROUP else None},
        "below": {"players": _hold(down.sum()), "move": _r(move[down].mean(), 3) if down.sum() >= MIN_GROUP else None},
        "near": {"players": _hold(near.sum()), "move": _r(move[near].mean(), 3) if near.sum() >= MIN_GROUP else None},
    }


def hidden_test(d, sess, dn, self_cal, folds=5, seed=13):
    keys = dn.index[dn["dupr"].notna()].intersection(d["pbp_user_id"].unique()).to_numpy()
    if len(keys) < 25:
        return None
    perm = np.random.default_rng(seed).permutation(keys)
    rows = []
    for f in range(folds):
        hold = frozenset(perm[f::folds].tolist())
        T, _ = solve(d, sess, dn, self_cal, hide=hold)
        for p in hold:
            rows.append((p, dn.at[p, "dupr"], T.at[p, "level"], dn.at[p, "months"]))
    o = pd.DataFrame(rows, columns=["p", "truth", "est", "months"])
    e = o["est"] - o["truth"]
    base = o["truth"] - o["truth"].mean()
    fresh = o["months"] <= 1.0
    pct = lambda s: _r(100 * float(s.mean()), 1) if len(s) else None
    return {"players": _hold(len(o)), "within50": pct(e.abs() <= 0.5), "within25": pct(e.abs() <= 0.25),
            "mae": _r(e.abs().mean(), 3), "baseline": {"within50": pct(base.abs() <= 0.5), "mae": _r(base.abs().mean(), 3)},
            "fresh": {"players": _hold(fresh.sum()), "within50": pct(e[fresh].abs() <= 0.5) if fresh.sum() >= MIN_GROUP else None}}


# ── privacy: no count of 1-4, shown or worked out by subtraction ─────────────
def _small(n):
    return isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 4


def _protect(total, parts: dict, exact=False):
    """parts: name -> count, all within `total`. Hides the smallest shown part
    until the total minus the shown parts isn't 1-4; for an exact split, one
    hidden part alone would be total minus the rest, so a second goes too."""
    if total is None:
        return {k: None for k in parts}
    out = dict(parts)
    for _ in range(len(out) + 1):
        shown = {k: v for k, v in out.items() if v is not None}
        rest = total - sum(shown.values())
        hidden = [k for k, v in out.items() if v is None]
        bad = _small(rest) if not exact or not hidden else (len(hidden) == 1 or _small(rest))
        if exact and not hidden and rest == 0:
            bad = False
        if not bad or not shown:
            break
        out[min(shown, key=shown.get)] = None
    return out


def _privacy(sm: dict) -> dict:
    sm["kinds"] = _protect(sm["players"], sm["kinds"], exact=True)
    sm.update(_protect(sm["players"], {"sure": sm["sure"], "regulars": sm["regulars"]}))
    sm.update(_protect(sm["regulars"], {"regularsSure": sm["regularsSure"]}))
    for k, row in sm["mix"].items():
        tot = sm["kinds"].get(k)
        cells = _protect(tot, {i: v for i, v in enumerate(row)}, exact=True)
        sm["mix"][k] = [cells[i] for i in range(len(row))]
    v = sm["vsDupr"]
    v.update(_protect(v["players"], {"above": v["above"], "below": v["below"], "within": v["within"]}, exact=True))
    f = sm["tests"].get("forward")
    if f:
        parts = _protect(f["players"], {"above": f["above"]["players"], "below": f["below"]["players"],
                                        "near": f["near"]["players"]})
        for k in ("above", "below", "near"):
            f[k]["players"] = parts[k]
            if parts[k] is None:
                f[k]["move"] = None
        f.update(_protect(f["players"], {"movers": f["movers"]}))
        f.update(_protect(f["movers"], {"called": f["called"]}))
        if f["called"] is None:
            f["hitPct"] = None
    hd = sm["tests"].get("hidden")
    if hd:
        hd["fresh"].update(_protect(hd["players"], {"players": hd["fresh"]["players"]}))
        if hd["fresh"]["players"] is None:
            hd["fresh"]["within50"] = None
    return sm


# ── the build's entry point ───────────────────────────────────────────────────
def fit(m: pd.DataFrame, s: pd.DataFrame, ratelog, frm, to, self_cal_ab=None, detail=None):
    """detail: a dict to fill with the fit's working (visits, sessions, players,
    evidence, DUPR history, rooms) for pm_player.py's per-player pages."""
    import time
    t0 = time.time()
    d, sess = prepare(m, s)
    if d.empty or len(sess) < 20:
        return None, []
    h = dupr_history(m, ratelog)
    end = d["starts_at"].max()
    dn = dupr_now(h, end)
    if dn["dupr"].notna().sum() < 25:
        return None, []
    sv = self_ratings(d)
    if self_cal_ab is None:
        import pm_level as V1
        both = pd.DataFrame({"dupr": dn["dupr"], "self": sv}).dropna()
        a, b, _ = V1.self_calibration(both)
    else:
        a, b = self_cal_ab
    coaches = likely_coaches(d, sess)
    cal = {"a": a, "b": b, "values": sv, "coaches": coaches}
    T, meta = solve(d, sess, dn, cal)
    room_list = rooms(T, d, sess)

    # how far the PM level sits from DUPR, for players who have one
    j = T.join(dn[["dupr", "months"]], how="left")
    jd = j[j["dupr"].notna()]
    gap = jd["level"] - jd["dupr"]
    T["bin"] = pd.cut(T["level"], BINS, labels=BIN_LABELS)
    T["sure"] = T["sd"] <= CFG["sure_sd"]
    mix = {k: [_hold(((T["bin"] == lab) & (T["kind"] == k)).sum()) for lab in BIN_LABELS] for k in ("dupr", "self", "none")}
    regular = T[T["n"] >= 4]
    venues = {}
    for v, g in pd.DataFrame(room_list).groupby("v") if room_list else []:
        gg = g["gap"].dropna()
        venues[str(int(v))] = {"sessions": int(len(g)), "labelGap": _r(gg.median()) if len(gg) else None}
    summary = {
        "version": VERSION, "status": "testing", "from": str(frm), "to": str(to), "rounds": meta["rounds"],
        "players": _hold(len(T)), "kinds": {k: _hold((T["kind"] == k).sum()) for k in ("dupr", "self", "none")},
        "sure": _hold(T["sure"].sum()), "regulars": _hold(len(regular)), "regularsSure": _hold(regular["sure"].sum()),
        "coaches": _hold(len(coaches)), "bins": BIN_LABELS, "mix": mix,
        "vsDupr": {"players": _hold(len(jd)), "above": _hold((gap >= 0.25).sum()), "below": _hold((gap <= -0.25).sum()),
                   "within": _hold((gap.abs() < 0.25).sum()), "medianGap": _r(gap.median()) if len(jd) else None,
                   "staleMonths": _r(jd["months"].median(), 1) if len(jd) else None},
        "groups": meta["groups"], "selfCal": {"a": _r(a, 3), "b": _r(b, 3)},
        "rooms": sorted(room_list, key=lambda r: -abs(r["gap"] or 0)),
        "venues": venues,
        "tests": {"forward": forward_test(d, sess, h, cal, end), "hidden": hidden_test(d, sess, dn, cal)},
        "settings": SETTINGS,
    }
    summary = _privacy(summary)
    summary["seconds"] = round(time.time() - t0)
    if detail is not None:
        detail.update(d=d, sess=sess, T=T, meta=meta, dn=dn, h=h, rooms=room_list, coaches=coaches, end=end)
    levels = [{"id": int(p), "level": round(float(r.level), 2), "sd": round(float(r.sd), 2), "n": int(r.n), "kind": r.kind}
              for p, r in T.iterrows()]
    return summary, levels
