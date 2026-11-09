"""
court_free.py -- court hire, free and taken, per venue, weekday and hour.

The court log (court_slot_states) says which courts were free in each 30-minute
slot. PlayByPoint blocks a court for a hire and for the venue's own session
alike, so "taken" alone isn't hire. The live function ins_court_hire
(20261015100000) already splits each slot that has started into:

    n_courts    the venue's bookable courts that day
    taken       courts not free
    programmed  courts the venue's own sessions were using (estimate: capacity
                / 4 rounded up, from start to listed end), never more than taken
    hire_taken  taken - programmed
    hireable    n_courts - programmed

fetch() reads it a week at a time, paged (PostgREST returns at most 1,000 rows
per call), retrying like rpc() and keeping the weeks it could read. summarise() turns the slots into one row per venue x weekday x hour,
for "where could a new session go":

    dow       0 = Monday (the same as heat and sessions[].dow)
    weeks     weeks with both of that hour's half-hour slots read
    free2     of those weeks, how many had 2+ courts free for the whole hour
              (the hour's two slots both had 2+ free)
    freeMed   median, over weeks, of the courts free for the whole hour
    courts    the venue's courts (largest seen)
    hire      courts hired / courts hireable, over the window
    prog      courts used by the venue's own sessions / courts, over the window

Court counts, not players: the 5-or-more rule doesn't apply.
"""
from datetime import date, timedelta

COLS = ["fid", "slot_date", "slot_start", "n_courts", "taken", "programmed", "hire_taken", "hireable"]
OUT_COLS = ["v", "dow", "h", "weeks", "free2", "freeMed", "courts", "hire", "prog", "hireH"]
PAGE = 1000


def fetch(client, url, h, frm: date, to: date, chunk_days: int = 7, tries: int = 3, wait: float = 5.0) -> dict:
    """Every started slot from frm to to, from ins_court_hire, a week at a time.

    Each call is retried on a server error or a timeout (statement timeouts
    come back as 5xx, or as a 4xx carrying PostgREST code 57014), like rpc().
    A week that still fails is skipped and the weeks already read are kept:
    {"rows": [...], "ok": every week read, "weeksRead": n, "weeks": n asked}.
    Paging stops only on an empty page, so a server row cap below PAGE can't
    cut a week short."""
    import time
    rows, a, read, asked = [], frm, 0, 0
    while a <= to:
        b = min(to, a + timedelta(days=chunk_days - 1))
        asked += 1
        week, off, good = [], 0, True
        while True:
            r = None
            for attempt in range(tries):
                try:
                    r = client.post(f"{url}/rest/v1/rpc/ins_court_hire", headers=h,
                                    params={"select": ",".join(COLS), "order": "fid,slot_date,slot_start",
                                            "limit": PAGE, "offset": off},
                                    json={"p_from": str(a), "p_to": str(b)})
                except Exception:              # network error or client timeout
                    r = None
                if r is not None and r.status_code < 300:
                    break
                timeout = r is not None and r.status_code < 500 and "57014" in (r.text or "")
                if r is not None and r.status_code < 500 and not timeout:
                    break                      # a real client error: retrying won't help
                time.sleep(wait * (attempt + 1))
            if r is None or r.status_code >= 300:
                print(f"  courthire: week from {a} skipped (HTTP {getattr(r, 'status_code', 'none')})")
                good = False
                break
            got = r.json() or []
            if not got:
                break
            week += got
            off += len(got)
        if good:
            rows += week
            read += 1
        a = b + timedelta(days=1)
    return {"rows": rows, "ok": read == asked, "weeksRead": read, "weeks": asked}


def summarise(df) -> dict:
    """{"cols": OUT_COLS, "rows": [[...], ...]} from the slots in df (COLS)."""
    import pandas as pd
    if df is None or df.empty:
        return {"cols": OUT_COLS, "rows": []}
    d = df.copy()
    for c in ("n_courts", "taken", "programmed", "hire_taken", "hireable"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["fid", "slot_date", "slot_start", "n_courts"])
    d = d[d["n_courts"] > 0]
    if d.empty:
        return {"cols": OUT_COLS, "rows": []}
    d["fid"] = d["fid"].astype(int)
    day = pd.to_datetime(d["slot_date"], errors="coerce")
    d["dow"] = day.dt.dayofweek.astype(int)                    # 0 = Monday, like heat and sessions
    d["week"] = (day - pd.to_timedelta(d["dow"], unit="D")).dt.strftime("%Y-%m-%d")
    d["h"] = d["slot_start"].astype(str).str.slice(0, 2).astype(int)
    d["free"] = (d["n_courts"] - d["taken"].fillna(0)).clip(lower=0)
    # the whole hour: the fewest courts free across its two half-hour slots,
    # per date. An hour with only one slot logged (opening at :30, a missed
    # read) isn't counted as free or taken: we can't say for the whole hour.
    hour = d.groupby(["fid", "dow", "h", "week", "slot_date"], as_index=False).agg(
        free=("free", "min"), courts=("n_courts", "max"), slots=("slot_start", "nunique"))
    hour = hour[hour["slots"] >= 2]
    if hour.empty:
        return {"cols": OUT_COLS, "rows": []}
    wk = hour.groupby(["fid", "dow", "h", "week"], as_index=False).agg(free=("free", "min"), courts=("courts", "max"))
    per = wk.groupby(["fid", "dow", "h"]).agg(
        weeks=("week", "nunique"), free2=("free", lambda s: int((s >= 2).sum())),
        freeMed=("free", "median"), courts=("courts", "max"))
    tot = d.groupby(["fid", "dow", "h"]).agg(ht=("hire_taken", "sum"), hb=("hireable", "sum"),
                                             pg=("programmed", "sum"), nc=("n_courts", "sum"))
    per = per.join(tot)
    rows = []
    for (v, dow, hr), r in per.sort_index().iterrows():
        hire = round(float(r.ht) / float(r.hb), 3) if r.hb and r.hb > 0 else None
        prog = round(float(r.pg) / float(r.nc), 3) if r.nc and r.nc > 0 else None
        # hireH: hireable court-hours in the period (each slot is half an hour)
        hire_h = round(float(r.hb) / 2, 1) if r.hb and r.hb > 0 else 0.0
        rows.append([int(v), int(dow), int(hr), int(r.weeks), int(r.free2),
                     float(r.freeMed), int(r.courts), hire, prog, hire_h])
    return {"cols": OUT_COLS, "rows": rows}


def waitlist_times(sessions, waits):
    """Adds waitlist_first_at (the first reading before the start with anyone
    waiting) and waitlist_first (how many were waiting then) to sessions.

    The fill log writes a new row whenever a waitlist appears, grows, shrinks
    or empties (record_inventory_snapshot, 20261018100000), so the first row
    with waitlist > 0 is when we first saw a queue, to the capture job's pace
    (hourly for sessions 0-1 days out, every 3 h for 2-7, daily beyond). The
    waitlist is only read while a session is full, and only for sessions that
    belong to a programme page."""
    import pandas as pd
    s = sessions.copy()
    s["waitlist_first_at"] = None
    s["waitlist_first"] = None
    if waits is None or waits.empty or s.empty:
        return s
    w = waits.copy()
    w["observed_at"] = pd.to_datetime(w["observed_at"], utc=True, errors="coerce")
    w["waitlist"] = pd.to_numeric(w["waitlist"], errors="coerce")
    w = w[(w["waitlist"] > 0) & w["observed_at"].notna()]
    st = pd.to_datetime(s["starts_at"], utc=True, errors="coerce")
    w = w.merge(pd.DataFrame({"session_key": s["session_key"], "_st": st}), on="session_key")
    w = w[w["observed_at"] < w["_st"]]
    if w.empty:
        return s
    first = w.sort_values("observed_at").drop_duplicates("session_key")[["session_key", "observed_at", "waitlist"]]
    m = s[["session_key"]].merge(first, on="session_key", how="left")
    s["waitlist_first_at"] = [None if pd.isna(x) else x.isoformat() for x in m["observed_at"]]
    s["waitlist_first"] = [None if pd.isna(x) else int(x) for x in m["waitlist"]]
    return s
