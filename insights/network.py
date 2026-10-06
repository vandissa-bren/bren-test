"""
network.py -- the co-play network behind the overview's Network and Player tabs.

Two players are *partners* when they were booked into the same finished session.
Sessions with more than EVENT_CAP players (tournament-style events) are left out
of ties, because nobody plays with 120 people in one go.

Players are shown only as random 4-character codes. The salt is drawn fresh on
every build and never stored, so codes can't be traced back to PlayByPoint IDs
and change on each rebuild.
"""
import hashlib
import secrets
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import shortest_path, connected_components

EVENT_CAP = 50
SLOTS = ["Weekday before 9am", "Weekday 9am–5pm", "Weekday after 5pm", "Weekend"]
FMTS = ["Social & open play", "Coaching & clinics", "Learn to play", "Round robin", "League & ladder", "Match play", "Unknown"]
SLOT_SHORT = ["early weekdays", "weekday daytime", "weekday evenings", "weekends"]
CLASSES = ["beginner", "intermediate", "advanced", "unrated"]
ALPH = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def rating_class(x):
    if x is None or np.isnan(x):
        return "unrated"
    return "beginner" if x < 2.75 else "intermediate" if x < 3.5 else "advanced"


def codes(ids):
    salt = secrets.token_bytes(16)
    out, used = {}, set()
    for pid in ids:
        h = hashlib.sha256(salt + str(pid).encode()).digest()
        k = 4
        while True:
            c = "".join(ALPH[b % len(ALPH)] for b in h[:k])
            if c not in used:
                break
            k += 1
        used.add(c)
        out[pid] = c
    return out


def series_key(x):
    t = x.title if isinstance(x.title, str) else (x.label if isinstance(x.label, str) else "Untitled session")
    return (int(x.fid), t, int(x.dow), x.hm)


def build(f, m, ok, vmap, clean_name, area_of_fid):
    """f: finished sessions at tracked venues with column i (row in D.sessions).
    m: every roster row (cancelled flagged, rating fixed). ok: m without cancellations."""
    f = f.copy()
    sk2i = dict(zip(f["session_key"], f["i"]))
    fin_ok = ok[ok["session_key"].isin(sk2i)].copy()
    fin_ok["si"] = fin_ok["session_key"].map(sk2i)
    meta = f.set_index("i")

    # ── series: same venue, title, weekday and start time ──
    f["series"] = [series_key(x) for x in f.itertuples()]
    skeys = sorted(set(f["series"]), key=lambda k: (k[0], k[2], k[3], k[1]))
    sidx = {k: j for j, k in enumerate(skeys)}
    f["sj"] = f["series"].map(sidx)
    i2sj = dict(zip(f["i"], f["sj"]))
    fin_ok["sj"] = fin_ok["si"].map(i2sj)

    # ── players ──
    pids = sorted(fin_ok["pbp_user_id"].unique())
    pix = {p: k for k, p in enumerate(pids)}
    code = codes(pids)
    N = len(pids)
    fin_ok["pi"] = fin_ok["pbp_user_id"].map(pix)

    # ── ties ──
    size = fin_ok.groupby("si")["pi"].nunique()
    tie_rows = fin_ok[fin_ok["si"].map(size) <= EVENT_CAP]
    events = int((size > EVENT_CAP).sum())
    W = Counter()          # sessions shared
    NW = defaultdict(float)  # Newman weight: 1/(n-1) per shared session
    pair_venues = defaultdict(set)
    for si, g in tie_rows.groupby("si"):
        u = sorted(set(g["pi"]))
        n = len(u)
        if n < 2:
            continue
        fid = int(meta.at[si, "fid"])
        for a_ in range(n):
            for b_ in range(a_ + 1, n):
                k = (u[a_], u[b_])
                W[k] += 1
                NW[k] += 1 / (n - 1)
                pair_venues[k].add(fid)
    rows, cols = zip(*W.keys())
    A = sparse.coo_matrix((np.ones(len(W)), (rows, cols)), shape=(N, N)).tocsr()
    A = ((A + A.T) > 0).astype(np.int32)
    deg = np.asarray(A.sum(1)).ravel()
    A2 = (A @ A)
    A2.setdiag(0)
    A2.eliminate_zeros()
    within2 = ((A + A2) > 0)
    reach2 = np.asarray(within2.sum(1)).ravel() - deg  # second step only
    reg = Counter()
    for (a, b), w in W.items():
        if w >= 2:
            reg[a] += 1
            reg[b] += 1

    G = nx.Graph()
    G.add_nodes_from(range(N))
    for (a, b), w in W.items():
        G.add_edge(a, b, w=w, nw=NW[(a, b)])
    clus = nx.clustering(G)
    ncomp, lab = connected_components(A, directed=False)
    giant = np.bincount(lab).argmax()
    in_giant = lab == giant
    bt = nx.betweenness_centrality(G, k=min(500, N), seed=7)
    btv = np.array([bt[k] for k in range(N)])
    bt_pct = (pd.Series(btv).rank(pct=True) * 100).round().astype(int).values
    core = nx.core_number(G)

    # degrees of separation: exact, all pairs inside the largest connected group
    gi = np.where(in_giant)[0]
    D = shortest_path(A[gi][:, gi], method="D", unweighted=True, directed=False)
    iu = np.triu_indices(len(gi), 1)
    dvals = D[iu].astype(int)
    sep = Counter(np.minimum(dvals, 7))
    avg_sep = float(dvals.mean())
    closeness = np.zeros(N)
    closeness[gi] = (len(gi) - 1) / D.sum(1)
    steps_hist = {int(k): int(v) for k, v in sorted(sep.items())}
    del D

    # communities (Louvain on Newman weights)
    comms = nx.community.louvain_communities(G, weight="nw", seed=11, resolution=1.0)
    comms = sorted(comms, key=len, reverse=True)
    modularity = nx.community.modularity(G, comms, weight="nw")
    cm = np.full(N, -1)
    for c_i, cs in enumerate(comms):
        for p in cs:
            cm[p] = c_i

    # ── player facts ──
    rating = m.groupby("pbp_user_id")["rating"].median()
    canc = m[m["cancelled"] & m["session_key"].isin(sk2i)].groupby("pbp_user_id").size()
    sess_of = fin_ok.groupby("pi")["si"].apply(lambda x: sorted(set(int(v) for v in x)))
    # lead: bookings we saw happen (first on the roster after we first saw the session)
    fo = f.set_index("i")["first_obs"]
    st = f.set_index("i")["starts_at"]
    b = fin_ok.assign(fo=fin_ok["si"].map(fo), st=fin_ok["si"].map(st))
    seen = b[b["first_seen"] > b["fo"] + pd.Timedelta(hours=2)]
    lead = (seen.assign(d=(seen["st"] - seen["first_seen"]).dt.total_seconds() / 86400).groupby("pi")["d"].median())

    players = []
    for k, pid in enumerate(pids):
        ss = sess_of.get(k, [])
        rows_ = meta.loc[ss]
        vc = rows_["fid"].astype(int).value_counts()
        home = int(sorted(vc.items(), key=lambda t: (-t[1], t[0]))[0][0])
        r = rating.get(pid, np.nan)
        players.append({
            "c": code[pid], "n": len(ss), "vs": [[int(v), int(n)] for v, n in vc.items()],
            "home": home, "area": area_of_fid(home),
            "r": None if np.isnan(r) else round(float(r), 2), "lvl": rating_class(r),
            "deg": int(deg[k]), "reg": int(reg[k]), "r2": int(reach2[k]),
            "cl": round(float(clus[k]), 2), "bt": int(bt_pct[k]), "core": int(core[k]),
            "cm": int(cm[k]), "gc": bool(in_giant[k]), "clo": round(float(closeness[k]), 3),
            "sl": [int((rows_["slot"] == s).sum()) for s in SLOTS],
            "fm": [int((rows_["fmt"] == x).sum()) for x in FMTS],
            "dw": [int((rows_["dow"] == d).sum()) for d in range(7)],
            "lead": None if k not in lead.index else round(float(lead[k]), 1),
            "cx": int(canc.get(pid, 0)),
            "first": str(rows_["date"].min()), "last": str(rows_["date"].max()),
            "ss": ss,
        })

    # ── players like you: similar habits, never played together ──
    vids = sorted(vmap)
    vpos = {v: j for j, v in enumerate(vids)}
    F = np.zeros((N, len(vids) + 4 + 7 + 7 + 4))
    for k, p in enumerate(players):
        n = max(p["n"], 1)
        for v, c in p["vs"]:
            F[k, vpos[v]] = 1.6 * c / n
        o = len(vids)
        F[k, o:o + 4] = np.array(p["sl"]) / n
        F[k, o + 4:o + 11] = np.array(p["fm"]) / n
        F[k, o + 11:o + 18] = 0.6 * np.array(p["dw"]) / n
        F[k, o + 18 + CLASSES.index(p["lvl"])] = 0.8 if p["lvl"] != "unrated" else 0.3
    Fn = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-9)
    S = Fn @ Fn.T
    np.fill_diagonal(S, -1)
    Ad = A.toarray().astype(bool)
    S[Ad] = -1
    for k, p in enumerate(players):
        top = np.argsort(-S[k])[:5]
        p["like"] = [[int(j), int(round(100 * S[k, j]))] for j in top if S[k, j] > 0.5]
    del S, Ad

    # ── session series ──
    ser_players = {int(j): Counter(int(v) for v in g) for j, g in fin_ok.groupby("sj")["pi"]}
    series = []
    for j, (fid, title, dow, hm) in enumerate(skeys):
        g = f[f["sj"] == j]
        pc = ser_players.get(j, Counter())
        rr = [players[p]["r"] for p in pc if players[p]["r"] is not None]
        series.append({
            "j": j, "v": fid, "t": title, "dow": dow, "hm": hm,
            "lvl": g["lvl"].mode().iat[0], "fmt": g["fmt"].mode().iat[0], "slot": g["slot"].mode().iat[0],
            "runs": int(len(g)), "rost": int(g["i"].isin(fin_ok["si"]).sum()),
            "fill": round(100 * float(g["fill"].mean()), 1),
            "players": len(pc), "regulars": sum(1 for c in pc.values() if c >= 2),
            "r": round(float(np.median(rr)), 2) if len(rr) >= 5 else None,
            "cm": Counter(int(cm[p]) for p in pc).most_common(1)[0][0] if pc else None,
        })
    # who also plays: Jaccard on player sets
    sets = {j: set(ser_players.get(j, {}).keys()) for j in range(len(skeys))}
    for s_ in series:
        a = sets[s_["j"]]
        sims = []
        if len(a) >= 3:
            for j2, b2 in sets.items():
                if j2 == s_["j"] or len(b2) < 3:
                    continue
                inter = len(a & b2)
                if inter >= 3:
                    sims.append([j2, round(100 * inter / len(a | b2)), inter])
        s_["also"] = sorted(sims, key=lambda t: (-t[2], -t[1]))[:6]

    # ── sessions to try: series your partners play that you haven't ──
    nb = [list(G.neighbors(k)) for k in range(N)]
    p_series = {int(k_): set(int(v) for v in g) for k_, g in fin_ok.groupby("pi")["sj"]}
    for k, p in enumerate(players):
        mine = p_series.get(k, set())
        score = Counter()
        who = defaultdict(set)
        for q in nb[k]:
            w = G[k][q]["w"]
            for j in p_series.get(q, ()):
                if j in mine:
                    continue
                score[j] += w
                who[j].add(q)
        out = []
        for j, sc in score.most_common(40):
            if len(who[j]) < 2:
                continue
            sr = series[j]
            if sr["t"] == "Untitled session":
                continue
            fit = 0 if (p["r"] is None or sr["r"] is None) else abs(p["r"] - sr["r"])
            if fit > 0.6:
                continue
            out.append([j, len(who[j]), round(sc - 2 * fit, 1)])
        p["rec"] = sorted(out, key=lambda t: -t[2])[:5]

    # ── communities: profile + group graph ──
    vname = {v: clean_name(vmap[v]["name"]) for v in vmap}
    groups = []
    for c_i, cs in enumerate(comms):
        if len(cs) < 5:
            continue
        mem = [players[p] for p in cs]
        vc = Counter()
        for p in mem:
            for v, n in p["vs"]:
                vc[v] += n
        tot = sum(vc.values())
        lv = Counter(p["lvl"] for p in mem)
        sl = np.sum([p["sl"] for p in mem], axis=0)
        fm = np.sum([p["fm"] for p in mem], axis=0)
        dw = np.sum([p["dw"] for p in mem], axis=0)
        inside = sum(1 for (a, b2) in W if cm[a] == c_i and cm[b2] == c_i)
        touching = sum(1 for (a, b2) in W if (cm[a] == c_i) != (cm[b2] == c_i))
        rr = [p["r"] for p in mem if p["r"] is not None]
        top_v = vc.most_common(1)[0][0]
        rated = {k_: v for k_, v in lv.items() if k_ != "unrated"}
        main_lvl = max(rated, key=rated.get) if rated else "unrated"
        groups.append({
            "id": c_i, "size": len(cs), "venues": [[int(v), round(100 * n / tot)] for v, n in vc.most_common(5)],
            "top": int(top_v), "area": area_of_fid(top_v), "lvl": {k_: int(v) for k_, v in lv.items()}, "mainLvl": main_lvl,
            "r": round(float(np.median(rr)), 2) if len(rr) >= 5 else None,
            "sl": [int(x) for x in sl], "fm": [int(x) for x in fm], "dw": [int(x) for x in dw],
            "spp": round(float(np.mean([p["n"] for p in mem])), 1),
            "deg": float(np.median([p["deg"] for p in mem])),
            "inside": inside, "touching": touching,
            "selfShare": round(100 * inside / max(inside + touching, 1)),
            "name": vname[top_v], "when": SLOT_SHORT[int(np.argmax(sl))],
        })
    # label: G1… by size, then "home venue · when · level"; repeats get the runner-up venue
    seen = Counter((g["name"], g["when"]) for g in groups)
    for n_, g in enumerate(groups, 1):
        g["code"] = f"G{n_}"
        lvl = "" if g["mainLvl"] == "unrated" else f" · {g['mainLvl']}"
        extra = ""
        if seen[(g["name"], g["when"])] > 1 and len(g["venues"]) > 1:
            extra = f" + {vname[g['venues'][1][0]]}"
        g["label"] = f"{g['name']}{extra} · {g['when']}{lvl}"
    dup = Counter(g["label"] for g in groups)
    DAYN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for g in groups:
        if dup[g["label"]] > 1:
            d = DAYN[int(np.argmax(g["dw"]))]
            g["label"] = g["label"].replace(g["when"], f"{d}s" if g["when"] in ("weekends", "weekday daytime") else f"{d} evenings")
    gid = {g["id"] for g in groups}
    gl = Counter()
    for (a, b2), w in W.items():
        ca, cb = cm[a], cm[b2]
        if ca != cb and ca in gid and cb in gid:
            gl[tuple(sorted((int(ca), int(cb))))] += 1
    glinks = [{"a": a, "b": b2, "n": n} for (a, b2), n in gl.items() if n >= 5]
    # layout: groups by their ties, then players inside each group
    H = nx.Graph()
    for g in groups:
        H.add_node(g["id"], size=g["size"])
    for l in glinks:
        H.add_edge(l["a"], l["b"], weight=l["n"])
    pos0 = nx.spring_layout(H, weight="weight", seed=3, iterations=300)
    ids_ = [g["id"] for g in groups]
    gmax_ = max(g["size"] for g in groups)
    R_ = np.array([0.03 + 0.1 * np.sqrt(g["size"] / gmax_) for g in groups])
    X = np.array([pos0[i] for i in ids_], dtype=float)
    X = 0.5 + 0.35 * (X - X.mean(0)) / max(np.abs(X - X.mean(0)).max(), 1e-9)
    li = {i: k for k, i in enumerate(ids_)}
    L_ = [(li[l["a"]], li[l["b"]], l["n"]) for l in glinks]
    lw = max([n for *_, n in L_] or [1])
    for it in range(800):
        F_ = np.zeros_like(X)
        for a, b2, n in L_:  # ties pull groups together
            d = X[b2] - X[a]; dist = np.linalg.norm(d) + 1e-9
            f_ = 0.03 * np.sqrt(n / lw) * (dist - (R_[a] + R_[b2] + 0.03)) * d / dist
            F_[a] += f_; F_[b2] -= f_
        for a in range(len(X)):  # everyone pushes apart a little; overlaps push hard
            d = X - X[a]; dist = np.linalg.norm(d, axis=1) + 1e-9
            gap_ = R_ + R_[a] + 0.025 - dist
            push = np.where(gap_ > 0, 0.5 * gap_, 0.0004 / dist ** 2)
            push[a] = 0
            F_[a] -= ((push / dist)[:, None] * d).sum(0)
        F_ += 0.02 * (0.5 - X)  # gentle pull to the middle
        X += np.clip(F_, -0.02, 0.02)
    lo_ = (X - R_[:, None]).min(0); hi_ = (X + R_[:, None]).max(0)
    span = (hi_ - lo_).max()
    X = (X - lo_) / span + (1 - (hi_ - lo_) / span) / 2
    pos = {i: X[k] for k, i in enumerate(ids_)}
    for k, g in enumerate(groups):
        g["x"], g["y"] = round(float(X[k, 0]), 4), round(float(X[k, 1]), 4)
        g["rad"] = round(float(R_[k] / span), 4)

    # player-level layout for the private player map
    gpos = {g["id"]: (g["x"], g["y"]) for g in groups}
    rad = {g["id"]: 0.9 * g["rad"] for g in groups}
    rng = np.random.default_rng(5)
    xy = np.zeros((N, 2))
    for c_i, cs in enumerate(comms):
        cs = list(cs)
        if c_i in gpos:
            sub = G.subgraph(cs)
            lp = nx.spring_layout(sub, seed=c_i, iterations=60, k=1.2 / np.sqrt(len(cs)))
            arr = np.array([lp[p] for p in cs])
            arr = arr - np.median(arr, 0)
            # keep each player's direction, spread distances evenly over the disc
            rr_ = np.linalg.norm(arr, axis=1)
            rank = pd.Series(rr_).rank(method="first").values / len(rr_)
            arr = arr / np.maximum(rr_, 1e-9)[:, None] * np.sqrt(rank)[:, None]
            cx, cy = gpos[c_i]
            xy[cs] = np.c_[cx + arr[:, 0] * rad[c_i], cy + arr[:, 1] * rad[c_i]]
        else:  # players outside the named groups: a strip under the map
            for p in cs:
                xy[p] = (rng.uniform(0.02, 0.98), rng.uniform(1.05, 1.13))
    for k, p in enumerate(players):
        p["x"], p["y"] = round(float(xy[k, 0]), 4), round(float(xy[k, 1]), 4)

    # ── level mixing ──
    mix = Counter()
    for (a, b2) in W:
        la, lb = players[a]["lvl"], players[b2]["lvl"]
        if "unrated" in (la, lb):
            continue
        mix[tuple(sorted((la, lb), key=CLASSES.index))] += 1
    rated_ties = [(players[a]["r"], players[b2]["r"]) for (a, b2) in W if players[a]["r"] is not None and players[b2]["r"] is not None]
    ra = np.array(rated_ties)
    assort = float(np.corrcoef(np.r_[ra[:, 0], ra[:, 1]], np.r_[ra[:, 1], ra[:, 0]])[0, 1])
    gap = float(np.median(np.abs(ra[:, 0] - ra[:, 1])))
    # null model: shuffle ratings across rated players
    rs = np.array([p["r"] for p in players if p["r"] is not None])
    null_gap = float(np.median(np.abs(rng.choice(rs, 20000) - rng.choice(rs, 20000))))

    # ── venue network facts ──
    vnet = {}
    regular_here = Counter()
    for (a, b2), w in W.items():
        if w >= 2:
            for v in pair_venues[(a, b2)]:
                regular_here[v] += 1
    for v in vmap:
        here = [k for k, p in enumerate(players) if any(vv == v for vv, _ in p["vs"])]
        if not here:
            continue
        dates = fin_ok[fin_ok["si"].map(meta["fid"]).astype(int) == v].groupby("pi")["si"].apply(lambda x: sorted(meta.loc[list(set(x)), "date"]))
        mid = sorted(f["date"])[len(f) // 2]
        early = [d for d in dates if d[0] < mid]
        cms = Counter(int(cm[k]) for k in here if cm[k] in gid)
        vnet[v] = {
            "netPlayers": len(here),
            "partners": float(np.median([players[k]["deg"] for k in here])),
            "reach2": float(np.median([players[k]["r2"] for k in here])),
            "oneTimers": round(100 * np.mean([players[k]["n"] == 1 for k in here]), 1),
            "cameBack": round(100 * np.mean([len(d) >= 2 for d in early]), 1) if len(early) >= 10 else None,
            "regularPairs": int(regular_here[v]),
            "groups": sum(1 for c_, n in cms.items() if n >= 5),
            "groupMix": [[c_, n] for c_, n in cms.most_common(6) if n >= 5],
        }

    city = {
        "players": N, "ties": len(W), "regularPairs": sum(1 for w in W.values() if w >= 2),
        "medianPartners": float(np.median(deg)), "medianReach2": float(np.median(reach2)),
        "giant": round(100 * in_giant.mean(), 1), "giantN": int(in_giant.sum()), "components": int(ncomp),
        "avgSep": round(avg_sep, 2), "steps": steps_hist, "pairs": int(len(dvals)),
        "modularity": round(modularity, 2), "groups": len(groups), "inGroups": sum(g["size"] for g in groups),
        "clustering": round(float(np.mean(list(clus.values()))), 2),
        "events": events, "eventCap": EVENT_CAP, "rosterSessions": int(fin_ok["si"].nunique()),
        "sessions": int(len(f)), "bookings": int(len(fin_ok)),
        "assort": round(assort, 2), "gap": round(gap, 2), "nullGap": round(null_gap, 2),
        "mix": [[a, b2, n] for (a, b2), n in mix.items()],
        "multiVenuePairs": sum(1 for k in pair_venues.values() if len(k) >= 2),
        "degHist": [[lbl, int(((deg >= lo_) & (deg <= hi_)).sum())] for lbl, lo_, hi_ in
                    [("0", 0, 0), ("1–5", 1, 5), ("6–10", 6, 10), ("11–20", 11, 20), ("21–40", 21, 40), ("41–80", 41, 80), ("81+", 81, 10 ** 6)]],
    }
    edges = [[int(a), int(b2), int(w)] for (a, b2), w in W.items()]
    return {"city": city, "players": players, "edges": edges, "groups": groups, "glinks": glinks,
            "series": series, "venues": {int(k): v for k, v in vnet.items()}}
