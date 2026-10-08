"""
pm_level.py -- the PM level, fitted in the daily build (plan doc, 7-8 Oct).

A normalised doubles rating for every player, on DUPR's familiar 2-8 scale but
never called DUPR. Insights only, never shown per player. Version "v1":

  · every input is a reading of a player's level, each with its own noise:
      DUPR on the roster        level ~ DUPR                       (sd 0.15)
      the sessions they play    room = b*level + (1-b)*venue level  b: social 0.3,
                                clinics 0.2, competitive 0.35, DUPR-rated 0.5
                                (rooms only partly sort players; measured 7 Oct)
      session labels            the room ~ the label's midpoint, corrected per
                                session family and venue
      real self-ratings         level ~ a + b*(self-3), calibrated directly on
                                the players who show both a DUPR and a
                                self-rating; claims far from the rest count less
  · each player starts from their home venue's typical level, not the city's,
    so strong venues aren't squashed towards the middle
  · venue DUPR offsets are held at zero: sessions can't tell "better players"
    from "DUPRs running high" (7 Oct). They will come from DUPR movement.
  · recent sessions count more (half-life 8 weeks)

Each run tests itself: the DUPR players are split in five; each fifth has its
DUPR hidden and is estimated from everything else. Scores: share within
±0.5 and ±0.25, typical miss, the ends (below 2.75, above 3.75), how often the
80% range holds, against "everyone gets the average". The model status stays
"testing" until a bar is agreed and passed.

fit_and_test(m, s, end) -> (summary for the build, levels for record_pm_run)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from catalogue_terms import dupr_rated
from ratings import dupr

VERSION = "v1"
FAMS = ["social", "learning", "competitive", "unknown"]
SELF_REAL = {2.5, 3.0, 3.5, 4.0, 4.5}
CFG = dict(sD=0.15, sp=0.2, se=0.3, sPhi=0.5, sVen=0.8, s0=0.5, sL=0.45, sS=0.35, sLab=0.4,
           beta={"social": 0.3, "learning": 0.2, "competitive": 0.35, "unknown": 0.3, "rated": 0.5},
           half_life_wk=8, nu=4.0, irls=4)
BINS = [0, 2.75, 3.0, 3.25, 3.5, 3.75, 9]
BIN_LABELS = ["below 2.75", "2.75–3.0", "3.0–3.25", "3.25–3.5", "3.5–3.75", "3.75+"]
Z80 = 1.2816
MIN_GROUP = 5


def prepare(m: pd.DataFrame, s: pd.DataFrame):
    """Sightings that were played, one per player per session, with what we need."""
    d = m[~m["cancelled"] & m["fid"].notna()].copy()
    d["pbp_user_id"] = d["pbp_user_id"].astype("int64")
    d = d.sort_values("last_seen").drop_duplicates(["pbp_user_id", "session_key"], keep="last")
    d["dv"] = dupr(d["rating_at_time"])
    raw = pd.to_numeric(d["rating_at_time"], errors="coerce")
    d["sv"] = raw.where(raw.isin(SELF_REAL) & d["dv"].isna())
    sess = s[s["session_key"].isin(set(d["session_key"]))].drop_duplicates("session_key").set_index("session_key")
    txt = (sess["title"].fillna("") + " " + sess.get("session_type", pd.Series("", index=sess.index)).fillna("")).str.lower()
    sess = sess.assign(rated=[dupr_rated(t) or "dupr" in t for t in txt])
    pl = pd.DataFrame({
        "n": d.groupby("pbp_user_id").size(),
        "dupr": d.dropna(subset=["dv"]).groupby("pbp_user_id")["dv"].last(),       # latest reading
        "self": d.dropna(subset=["sv"]).groupby("pbp_user_id")["sv"].agg(lambda x: x.mode().iloc[0]),
        "home": d.groupby("pbp_user_id")["fid"].agg(lambda x: x.mode().iloc[0]),
    })
    return d, sess, pl


def self_calibration(pl: pd.DataFrame):
    """a, b in level ~ a + b*(self-3), from players who show both (ridge towards 3, 1)."""
    both = pl.dropna(subset=["dupr", "self"])
    if len(both) < 20:
        return 3.0, 1.0, int(len(both))
    X = np.c_[np.ones(len(both)), both["self"] - 3.0]
    y = both["dupr"].to_numpy()
    lam = np.diag([1.0, 4.0])
    prior = np.array([3.0, 1.0])
    a, b = np.linalg.solve(X.T @ X + lam, X.T @ y + lam @ prior)
    return float(a), float(np.clip(b, 0.3, 1.2)), int(len(both))


def solve(d, sess, pl, train: pd.Series, ab, cfg=CFG, var_idx=None):
    """One fit. train: DUPR values to use (others hidden). Returns level and sd per player."""
    c = cfg
    players = list(pl.index)
    pi = {k: i for i, k in enumerate(players)}
    sk = list(sess.index)
    si = {k: i for i, k in enumerate(sk)}
    vens = sorted(sess["fid"].dropna().astype(int).unique())
    vi = {v: i for i, v in enumerate(vens)}
    N, S, V = len(players), len(sk), len(vens)
    iMU = N + S
    iF = iMU + 1
    iLV = iF + 4               # label offset per venue
    iM = iLV + V               # typical level per venue
    K = iM + V
    rows, cols, vals, y, w0, kind = [], [], [], [], [], []
    t = 0

    def add(cs, vs, yy, ww, kd):
        nonlocal t
        rows.extend([t] * len(cs)); cols.extend(cs); vals.extend(vs)
        y.append(yy); w0.append(ww); kind.append(kd); t += 1

    fam = {k: (FAMS.index(f) if f in FAMS else 3) for k, f in sess["family"].fillna("unknown").items()}
    sven = {k: vi[int(v)] for k, v in sess["fid"].items() if v == v}
    # DUPR
    for pid, v in train.items():
        if pid in pi:
            add([pi[pid]], [1.0], float(v), 1 / c["sD"] ** 2, 0)
    # rooms, partial: phi_s - b*theta_i - (1-b)*venue_level = 0
    n_i = d.groupby("pbp_user_id").size()
    last = d["starts_at"].max()
    age = ((last - d["starts_at"]).dt.total_seconds() / 604800).to_numpy()
    rec = 0.5 ** (age / c["half_life_wk"])
    for k, (pid, key) in enumerate(zip(d["pbp_user_id"], d["session_key"])):
        if key not in si or pid not in pi:
            continue
        s_ = si[key]
        b = c["beta"]["rated"] if sess.at[key, "rated"] else c["beta"][FAMS[fam[key]]]
        ww = rec[k] / (n_i[pid] * c["sp"] ** 2 + c["se"] ** 2)
        add([N + s_, pi[pid], iM + sven[key]], [1.0, -b, -(1 - b)], 0.0, ww, 1)
    # labels: phi_s - family offset - venue label offset = band midpoint
    mid = (sess["band_lo"] + sess["band_hi"].clip(upper=5.0)) / 2
    for key, mv in mid.items():
        if mv == mv and key in sven:
            add([N + si[key], iF + fam[key], iLV + sven[key]], [1.0, -1.0, -1.0], float(mv), 1 / c["sL"] ** 2, 2)
    # self-ratings, calibrated (a, b fixed)
    a, b = ab
    for pid, sv in pl["self"].dropna().items():
        add([pi[pid]], [1.0], a + b * (float(sv) - 3.0), 1 / c["sS"] ** 2, 3)
    # priors: player ~ home venue level; session ~ its venue level; venue ~ city
    for pid, h in pl["home"].items():
        if h == h and int(h) in vi:
            add([pi[pid], iM + vi[int(h)]], [1.0, -1.0], 0.0, 1 / c["s0"] ** 2, 4)
        else:
            add([pi[pid], iMU], [1.0, -1.0], 0.0, 1 / c["s0"] ** 2, 4)
    for key, s_ in si.items():
        add([N + s_, iM + sven[key]] if key in sven else [N + s_, iMU], [1.0, -1.0], 0.0, 1 / c["sPhi"] ** 2, 4)
    for k in range(V):
        add([iM + k, iMU], [1.0, -1.0], 0.0, 1 / c["sVen"] ** 2, 4)
        add([iLV + k], [1.0], 0.0, 1 / c["sLab"] ** 2, 5)
    for k in range(4):
        add([iF + k], [1.0], 0.0, 1 / c["sLab"] ** 2, 5)
    add([iMU], [1.0], 3.1, 1.0, 5)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(t, K))
    y, w0, kind = np.array(y), np.array(w0), np.array(kind)
    w = w0.copy()
    soft = np.isin(kind, [1, 3])
    for _ in range(c["irls"]):
        lu = splu((A.T @ sp.diags(w) @ A).tocsc())
        x = lu.solve(A.T @ (w * y))
        z2 = (A @ x - y) ** 2 * w0
        w = np.where(soft, w0 * (c["nu"] + 1) / (c["nu"] + z2), w0)
    lu = splu((A.T @ sp.diags(w) @ A).tocsc())
    x = lu.solve(A.T @ (w * y))
    want = list(range(N)) if var_idx is None else [pi[p] for p in var_idx if p in pi]
    sd = pd.Series(np.nan, index=players)
    for j in range(0, len(want), 1500):          # in blocks, to keep memory small
        idx = want[j:j + 1500]
        E = np.zeros((K, len(idx)))
        E[idx, np.arange(len(idx))] = 1.0
        sd.iloc[idx] = np.sqrt(np.einsum("ij,ij->j", E, lu.solve(E)))
    return pd.Series(x[:N], index=players), sd


def test(d, sess, pl, ab, folds=5, seed=8):
    """Hide each fifth of the DUPR players in turn; estimate them from the rest."""
    keys = pl.index[pl["dupr"].notna()].to_numpy()
    if len(keys) < 25:
        return None
    rng = np.random.default_rng(seed)
    perm = rng.permutation(keys)
    rows = []
    for f in range(folds):
        hold = set(perm[f::folds])
        train = pl["dupr"].dropna()
        train = train[~train.index.isin(hold)]
        est, sd = solve(d, sess, pl, train, ab, var_idx=list(hold))
        for p in hold:
            rows.append((p, pl.at[p, "dupr"], est[p], sd[p], pl.at[p, "n"]))
    o = pd.DataFrame(rows, columns=["p", "truth", "est", "sd", "n"])
    e = o["est"] - o["truth"]
    base = o["truth"] - o["truth"].mean()
    psd = np.sqrt(o["sd"] ** 2 + CFG["sD"] ** 2)
    lo, hi = o["truth"] < 2.75, o["truth"] > 3.75
    pct = lambda m: round(100 * float(m.mean()), 1) if len(m) else None
    return {
        "players": int(len(o)),
        "within50": pct(e.abs() <= 0.5), "within25": pct(e.abs() <= 0.25), "mae": round(float(e.abs().mean()), 3),
        "baseline": {"within50": pct(base.abs() <= 0.5), "mae": round(float(base.abs().mean()), 3)},
        "low": {"n": int(lo.sum()), "within50": pct(e[lo].abs() <= 0.5), "bias": round(float(e[lo].mean()), 3) if lo.any() else None},
        "high": {"n": int(hi.sum()), "within50": pct(e[hi].abs() <= 0.5), "bias": round(float(e[hi].mean()), 3) if hi.any() else None},
        "range80": pct(e.abs() <= Z80 * psd),
    }


def fit_and_test(m: pd.DataFrame, s: pd.DataFrame, frm, to):
    d, sess, pl = prepare(m, s)
    if pl["dupr"].notna().sum() < 25 or len(sess) < 20:
        return None, []
    a, b, n_both = self_calibration(pl)
    est, sd = solve(d, sess, pl, pl["dupr"].dropna(), (a, b))
    tests = test(d, sess, pl, (a, b))
    kind = np.where(pl["dupr"].notna(), "dupr", np.where(pl["self"].notna(), "self", "none"))
    T = pd.DataFrame({"level": est, "sd": sd, "n": pl["n"], "kind": kind})
    T["bin"] = pd.cut(T["level"], BINS, labels=BIN_LABELS)
    mix = {k: [int(((T["bin"] == lab) & (T["kind"] == k)).sum()) for lab in BIN_LABELS] for k in ("dupr", "self", "none")}
    half = {k: round(float((Z80 * g["sd"]).median()), 2) for k, g in T.groupby("kind")}
    regular = T[T["n"] >= 4]
    by_home = {}
    for h, g in T.assign(home=pl["home"]).dropna(subset=["home"]).groupby("home"):
        if len(g) >= MIN_GROUP:
            by_home[str(int(h))] = [int(len(g)), round(float(g["level"].mean()), 2), round(float(g["level"].quantile(.1)), 2),
                                    round(float(g["level"].quantile(.9)), 2), round(100 * float((g["kind"] != "none").mean()), 1)]
    summary = {
        "version": VERSION, "status": "testing", "from": str(frm), "to": str(to),
        "players": int(len(T)), "kinds": {k: int((T["kind"] == k).sum()) for k in ("dupr", "self", "none")},
        "bins": BIN_LABELS, "mix": mix, "half80": half,
        "regulars": [int(len(regular)), round(100 * float((Z80 * regular["sd"] <= 0.5).mean()), 1) if len(regular) else None],
        "selfCal": {"a": round(a, 3), "b": round(b, 3), "players": n_both},
        "offsets": "held at zero", "tests": tests, "venues": by_home,
    }
    levels = [{"id": int(p), "level": round(float(r.level), 2), "sd": round(float(r.sd), 2), "n": int(r.n), "kind": r.kind}
              for p, r in T.iterrows()]
    return summary, levels


SETTINGS = {k: v for k, v in CFG.items()}
