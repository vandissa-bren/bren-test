"""
cancellations.py -- the one cancellation rule every Insights figure uses
(venue_report.rosters, timing_preview.build, and through them player counts,
ratings, the network, new/returning, the PM level and Sessions › Timing).

A booking counts as CANCELLED only if a roster pass that ran BEFORE the
session started read the session and the player was no longer on it:

    there is a pass P with  last_seen < P < starts_at,  P <= the session's
    last known read,  and P read this session.

Why not "last_seen < the session's latest last_seen" (the old rule): rosters
are re-read for hours after a session starts (the capture band is by UTC
date), so a player the venue removes after the start (no-show tidy-up,
refund) has last_seen = the last pass before the start and looked like a
cancellation. They were booked at the start, so they count as attended.

How we know a pass P read the session. Every roster_sightings first_seen /
last_seen is the capture run's own run_at (record_roster_sightings stamps
observed_at), so two kinds of evidence exist in the export:
  1. session-local: another sighting of this session stamped at P (a later
     booker, or another player's last read). Exact, but sparse: a player still
     on the roster has only their first and latest read; the passes between
     are overwritten.
  2. global runs: every distinct stamp across all rosters is a run time. Each
     run reads one band of session dates relative to its UTC date
     (capture_roster_sightings.yml: near 0-1 hourly, mid 2-7 every 3 h, far
     8-21 daily). A run's band is recovered from the smallest date offset it
     stamped (a near run never touches day 2+, a mid run never day 0-1), so
     "run P covers this session" is: its band contains the session's offset.
Session-local evidence alone would miss most real cancellations (nobody else
need be stamped between a canceller's last sighting and the start), so both
are used. To keep a pulled or hidden session (venue took it down, hid the
roster, or the roster fetch for it kept failing) from showing its whole
roster as cancelled, P must not be later than the session's last known
ROSTER read (its latest last_seen). The fill log is NOT used as evidence: the
capture job keeps writing spots for a session whose roster it can no longer
read, so a fill-log reading says nothing about the roster. A consequence: if
every player's last sighting is the same pass, nobody is cancelled (the
session vanished, or everyone left at once; we can't tell which).

A pass's stamp is the run's start (run_at is taken before any roster is
read), and a run can take up to its 30-minute timeout to reach a session. So
P must also be at least READ_LAG before the start: otherwise a run stamped
10:09 that reads a 10:30 session at 10:35 would turn a player removed at
10:32 (after the start) into a cancellation. Real cancellations in the last
READ_LAG before the start are therefore not counted as cancellations.

The 6-hour rule stays: if the capture job as a whole didn't run again within
6 h of the player's last sighting, the gap is ours (an outage), and the
departure is reported as `capture_gap`, not a cancellation. It no longer
creates false cancellations (that's what the pass test is for); it keeps the
"hours before the start" of a cancellation honest.

Not knowable from the exports: a run that failed to read one session (a
single 403) while reading others looks like a pass that saw the player gone.
"""
import numpy as np
import pandas as pd

GAP_HOURS = 6
# a run's stamp is its start; it can take up to its timeout (30 min) to reach a session
READ_LAG = pd.Timedelta(minutes=30)
# capture_roster_sightings.yml bands, as (lo, hi) days ahead of the run's UTC date
BANDS = [(0, 1), (2, 7), (8, 21)]


def _band_of(min_offset):
    for lo, hi in BANDS:
        if min_offset <= hi:
            return lo, hi
    return BANDS[-1]


def _naive(x):
    """UTC timestamps as tz-naive (numpy-friendly)."""
    return x.dt.tz_convert("UTC").dt.tz_localize(None)


def capture_runs(r):
    """Every roster capture run seen in the export: its time (to the minute)
    and the band of session dates (days ahead, by UTC date) it read."""
    st = pd.concat([r[["session_date", "first_seen"]].rename(columns={"first_seen": "t"}),
                    r[["session_date", "last_seen"]].rename(columns={"last_seen": "t"})]).dropna()
    if st.empty:
        return pd.DataFrame({"t": pd.Series([], dtype="datetime64[ns, UTC]"), "lo": [], "hi": []})
    st["t"] = st["t"].dt.floor("min")
    d = pd.to_datetime(st["session_date"], errors="coerce")
    st["off"] = (d - _naive(st["t"]).dt.normalize()).dt.days
    runs = st.groupby("t")["off"].min().reset_index()
    runs[["lo", "hi"]] = [(_band_of(o) if o == o else (0, 21)) for o in runs["off"]]
    return runs.sort_values("t").reset_index(drop=True)


def classify(r, s=None):
    """Adds left_roster, capture_gap, cancelled and cancel_hours_before to r.

    r: roster rows with session_key, session_date, first_seen, last_seen and
       starts_at (the session's start, UTC). s: accepted for compatibility and
       ignored (fill-log readings are not roster evidence; see module doc)."""
    r = r.copy()
    if r.empty:
        for c in ("left_roster", "capture_gap", "cancelled"):
            r[c] = pd.Series(dtype=bool)
        r["cancel_hours_before"] = pd.Series(dtype=float)
        return r
    if "session_date" not in r or r["session_date"].isna().all():
        r["session_date"] = r["starts_at"].dt.tz_convert("Australia/Melbourne").dt.strftime("%Y-%m-%d")
    last = r["last_seen"].dt.floor("min")
    start = r["starts_at"]
    # the session's last known roster read (fill log deliberately not used)
    last_read = r.groupby("session_key")["last_seen"].transform("max").dt.floor("min")
    upper = pd.concat([start - READ_LAG, last_read + pd.Timedelta(seconds=1)], axis=1).min(axis=1)  # passes < upper count
    cand = last.notna() & start.notna() & (last < upper)

    hit = pd.Series(False, index=r.index)
    # 1. session-local: another stamp of this session strictly between
    stamps = pd.concat([r[["session_key", "first_seen"]].rename(columns={"first_seen": "t"}),
                        r[["session_key", "last_seen"]].rename(columns={"last_seen": "t"})]).dropna()
    stamps["t"] = stamps["t"].dt.floor("min")
    stamps["t"] = _naive(stamps["t"]).astype("datetime64[ns]")
    by_s = {k: np.sort(v.values) for k, v in stamps.groupby("session_key")["t"]}
    lo_all = _naive(last).astype("datetime64[ns]")
    hi_all = _naive(upper).astype("datetime64[ns]")
    for k, grp in r.loc[cand, ["session_key"]].groupby("session_key"):
        tsn = by_s.get(k)
        if tsn is None:
            continue
        idx = grp.index
        i = np.searchsorted(tsn, lo_all[idx].values, side="right")
        hit.loc[idx] = (i < len(tsn)) & (tsn[np.minimum(i, len(tsn) - 1)] < hi_all[idx].values)
    # 2. global runs whose band covers the session's date
    runs = capture_runs(r)
    sday = pd.to_datetime(r["session_date"], errors="coerce").dt.tz_localize("UTC")
    for lo_d, hi_d in BANDS:
        ts = runs.loc[(runs["lo"] == lo_d) & (runs["hi"] == hi_d), "t"]
        if ts.empty:
            continue
        tsn = np.sort(_naive(ts).astype("datetime64[ns]").values)
        # a run on UTC day D reads dates D+lo .. D+hi, so for this session it
        # must run on UTC days (date - hi) .. (date - lo)
        w0 = pd.concat([last + pd.Timedelta(seconds=1), sday - pd.Timedelta(days=hi_d)], axis=1).max(axis=1)
        w1 = pd.concat([upper, sday - pd.Timedelta(days=lo_d - 1)], axis=1).min(axis=1)
        ok = cand & w0.notna() & w1.notna() & (w0 < w1)
        if not ok.any():
            continue
        a = _naive(w0[ok]).astype("datetime64[ns]").values
        b = _naive(w1[ok]).astype("datetime64[ns]").values
        i = np.searchsorted(tsn, a, side="left")
        hit.loc[ok[ok].index] |= (i < len(tsn)) & (tsn[np.minimum(i, len(tsn) - 1)] < b)

    r["left_roster"] = cand & hit
    # outage check: the next run of the capture job (any band) after the last sighting
    allruns = _naive(runs["t"]).astype("datetime64[ns]").values
    nxt = np.searchsorted(allruns, lo_all.values, side="right")
    nat = np.datetime64("NaT", "ns")
    nr = allruns[np.minimum(nxt, len(allruns) - 1)] if len(allruns) else np.full(len(nxt), nat)
    next_run = pd.Series(np.where(nxt < len(allruns), nr, nat), index=r.index,
                         dtype="datetime64[ns]").dt.tz_localize("UTC")
    gap = (next_run - last) > pd.Timedelta(hours=GAP_HOURS)
    r["capture_gap"] = r["left_roster"] & gap.fillna(True)
    r["cancelled"] = r["left_roster"] & ~r["capture_gap"]
    r["cancel_hours_before"] = np.where(
        r["cancelled"], (r["starts_at"] - r["last_seen"]).dt.total_seconds() / 3600, np.nan)
    return r
