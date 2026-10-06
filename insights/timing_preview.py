"""
timing_preview.py -- timing metrics for one venue, against its rivals and the
PlayByPoint market, from three exports:

  sessions.csv   one row per session in inventory_snapshots (query 1)
  rosters.csv    roster_sightings as stored (query 2)
  catalogue.csv  today's catalogue sessions, for titles and levels (query 3)

Usage: python3 timing_preview.py <dir with the CSVs> <facility_id> > preview.json

Every figure carries its sample size. The data-quality checks from the
catalogue's testing list run first and are reported alongside.
"""
import json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

MEL = "Australia/Melbourne"
EXACT_GAP_MIN = 20          # same rule as player_booking_leads / the feed's LEAD trait
LEAD_BUCKETS = [(0, 1, "Same day"), (1, 2, "1 day"), (2, 4, "2–3 days"),
                (4, 8, "4–7 days"), (8, 10_000, "8+ days")]
LEAD_WINDOW_DAYS = 8        # leads are fully observed up to this many days
REGISTRY = Path(__file__).resolve().parent.parent / "venues.json"
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def fid_of(venue_id):
    m = re.search(r"(\d+)$", str(venue_id or ""))
    return int(m.group(1)) if m else None


def registry():
    data = json.loads(REGISTRY.read_text())
    if isinstance(data, dict):
        data = data.get("venues", data)
    if isinstance(data, dict):
        data = list(data.values())
    return {int(v["id"]): v for v in data if str(v.get("id", "")).isdigit()
            and v.get("platform") == "playbypoint"}


def km(a, b):
    from math import radians, sin, cos, asin, sqrt
    dl, dn = radians(b["lat"] - a["lat"]), radians(b["lng"] - a["lng"])
    h = sin(dl / 2) ** 2 + cos(radians(a["lat"])) * cos(radians(b["lat"])) * sin(dn / 2) ** 2
    return 2 * 6371 * asin(sqrt(h))


def pct(x, n):
    return None if not n else round(100.0 * x / n, 1)


def q(s, p):
    s = pd.Series(s).dropna()
    return None if s.empty else round(float(s.quantile(p)), 2)


def load(d: Path):
    s = pd.read_csv(d / "sessions.csv", na_values=["null"])
    r = pd.read_csv(d / "rosters.csv", na_values=["null"])
    c = pd.read_csv(d / "catalogue.csv", na_values=["null"]) if (d / "catalogue.csv").exists() else pd.DataFrame()
    for col in ("starts_at", "first_obs", "last_obs", "last_obs_before", "first_full_at"):
        if col in s:
            s[col] = pd.to_datetime(s[col], utc=True, errors="coerce", format="mixed")
    for col in ("first_seen", "last_seen"):
        r[col] = pd.to_datetime(r[col], utc=True, errors="coerce", format="mixed")
    s["fid"] = s["venue_id"].map(fid_of)
    return s, r, c


def build(s, r, now):
    """Join rosters to sessions and derive exact leads and cancellations."""
    sess = s.set_index("session_key")
    capture_start = r["first_seen"].min()
    r = r.merge(s[["session_key", "starts_at", "fid", "session_type", "capacity", "first_obs",
                   "first_full_at"]], on="session_key", how="left")
    r["ever_full"] = r["first_full_at"].notna()
    g = r.groupby("session_key")
    r["session_last_pass"] = g["last_seen"].transform("max")
    # We were watching this roster from when the session entered the catalogue,
    # or from when roster capture began, whichever is later. A player first seen
    # after that was booked then (exact, to within the capture interval); one
    # seen at the very start may have booked any time before (censored).
    r["watch_start"] = r["first_obs"].where(r["first_obs"] > capture_start, capture_start)
    r["exact"] = r["first_seen"] > r["watch_start"] + pd.Timedelta(minutes=EXACT_GAP_MIN)
    r["lead_days"] = (r["starts_at"] - r["first_seen"]).dt.total_seconds() / 86400
    # Usable for lead-time shares: the session has started (no late bookers to
    # come) and we watched it for at least LEAD_WINDOW_DAYS, so every booking is
    # either exact or known to be LEAD_WINDOW_DAYS+ ahead.
    r["lead_ok"] = (r["starts_at"] < now) & (
        (r["starts_at"] - r["watch_start"]) >= pd.Timedelta(days=LEAD_WINDOW_DAYS))
    # cancelled: gone while we kept watching, and gone before the start. The
    # capture job must have run again within 6 h of their last sighting --
    # otherwise the gap is ours (an outage), not theirs, and it isn't counted.
    runs = pd.DatetimeIndex(np.sort(pd.concat([r["first_seen"], r["last_seen"]]).dt.floor("min").unique()))
    nxt = runs.searchsorted(r["last_seen"].dt.floor("min"), side="right")
    next_run = pd.Series([runs[i] if i < len(runs) else pd.NaT for i in nxt], index=r.index)
    r["left_roster"] = (r["last_seen"] < r["session_last_pass"]) & (r["last_seen"] < r["starts_at"])
    r["capture_gap"] = r["left_roster"] & ((next_run - r["last_seen"]) > pd.Timedelta(hours=6))
    r["cancelled"] = r["left_roster"] & ~r["capture_gap"]
    r["cancel_hours_before"] = np.where(
        r["cancelled"], (r["starts_at"] - r["last_seen"]).dt.total_seconds() / 3600, np.nan)
    r["finished"] = r["starts_at"] < now
    return r


def lead_profile(rows):
    u = rows[rows["lead_ok"] & (rows["lead_days"] > 0)]
    # censored rows were booked at least LEAD_WINDOW_DAYS ahead
    x = u["lead_days"].where(u["exact"], np.maximum(u["lead_days"], LEAD_WINDOW_DAYS))
    n = len(x)
    buckets = []
    for lo, hi, label in LEAD_BUCKETS:
        k = int(((x >= lo) & (x < hi)).sum())
        buckets.append({"bucket": label, "n": k, "share": pct(k, n)})
    cap = lambda v: None if v is None else (f"{LEAD_WINDOW_DAYS}+" if v >= LEAD_WINDOW_DAYS else v)
    return {"n": n, "sessions": int(u["session_key"].nunique()),
            "median_days": cap(q(x, .5)), "p25_days": cap(q(x, .25)), "p75_days": cap(q(x, .75)),
            "last_minute_share": pct(int((x < 1).sum()), n), "buckets": buckets}


def session_profile(ss, now):
    fin = ss[(ss["starts_at"] < now) & ss["capacity"].gt(0) & ss["spots_at_start"].notna()].copy()
    # a session resized after it was logged can show more spots than its latest
    # capacity; those readings aren't comparable, so the session is left out
    fin = fin[fin["spots_at_start"] <= fin["capacity"]]
    n = len(fin)
    fin["fill"] = 1 - fin["spots_at_start"] / fin["capacity"]
    full = fin[fin["spots_at_start"] == 0]
    # time to sell out, only where we saw it before it was full
    seen_open = full[full["first_full_at"] > full["first_obs"]]
    hrs = (seen_open["starts_at"] - seen_open["first_full_at"]).dt.total_seconds() / 3600
    curve = {}
    for col, label in (("spots_7d", "7 days out"), ("spots_3d", "3 days out"),
                       ("spots_1d", "1 day out"), ("spots_at_start", "At start")):
        f = fin[fin[col].notna() & (fin[col] <= fin["capacity"])]
        curve[label] = {"n": len(f), "fill_pct": None if f.empty else
                        round(100 * float((1 - f[col] / f["capacity"]).mean()), 1)}
    return {"n_finished": n,
            "avg_fill_pct": None if not n else round(100 * float(fin["fill"].mean()), 1),
            "sellout_share": pct(len(full), n),
            "n_sellouts_seen_filling": len(seen_open),
            "sellout_hours_before_median": q(hrs, .5),
            "sellout_hours_before_p25": q(hrs, .25),
            "sellout_hours_before_p75": q(hrs, .75),
            "fill_curve": curve}


def listing_profile(ss, warm_after, cap_days):
    x = ss[ss["first_obs"] > warm_after]
    h = (x["starts_at"] - x["first_obs"]).dt.total_seconds() / 86400
    h = h[h > 0]
    n = len(h)
    return {"n": n, "median_days": q(h, .5), "p25_days": q(h, .25), "p75_days": q(h, .75),
            "at_catalogue_edge_share": pct(int((h >= cap_days - 1).sum()), n)}


def cancel_profile(rows):
    fin = rows[rows["finished"]]
    n = len(fin)
    c = fin[fin["cancelled"]]
    h = c["cancel_hours_before"]
    # rebooked: in a session that sold out at some point, someone else booked
    # after the place was freed (in a session with spare places a later booking
    # isn't a resale)
    resold, freed = 0, 0
    for key, grp in c[c["ever_full"]].groupby("session_key"):
        later = rows[(rows["session_key"] == key)]
        for _, cr in grp.iterrows():
            freed += 1
            if (later["first_seen"] > cr["last_seen"]).any():
                resold += 1
    return {"n_bookings": n, "cancel_share": pct(len(c), n), "n_cancels": len(c),
            "cancel_hours_before_median": q(h, .5),
            "within_24h_share": pct(int((h < 24).sum()), len(c)),
            "freed_in_sold_out_sessions": freed, "rebooked_share": pct(resold, freed)}


def booking_clock(rows):
    x = rows[rows["exact"] & (rows["lead_days"] > 0)]  # any watched booking; only its clock time is used
    loc = x["first_seen"].dt.tz_convert(MEL)
    by_dow = [{"day": DOW[i], "n": int((loc.dt.dayofweek == i).sum())} for i in range(7)]
    bands = [("Before 9am", 0, 9), ("9am–12pm", 9, 12), ("12–5pm", 12, 17), ("5–9pm", 17, 21), ("After 9pm", 21, 24)]
    by_band = [{"band": b, "n": int(((loc.dt.hour >= lo) & (loc.dt.hour < hi)).sum())} for b, lo, hi in bands]
    return {"n": len(x), "by_day": by_dow, "by_time": by_band}


def week_grid(ss, now):
    fin = ss[(ss["starts_at"] < now) & ss["capacity"].gt(0) & ss["spots_at_start"].notna()].copy()
    loc = fin["starts_at"].dt.tz_convert(MEL)
    fin["dow"], fin["hour"] = loc.dt.dayofweek, loc.dt.hour
    fin["fill"] = 1 - fin["spots_at_start"] / fin["capacity"]
    out = []
    for (d, h), grp in fin.groupby(["dow", "hour"]):
        out.append({"day": DOW[int(d)], "hour": int(h), "n": len(grp),
                    "fill_pct": round(100 * float(grp["fill"].mean()), 1),
                    "sellouts": int((grp["spots_at_start"] == 0).sum())})
    return out


def by_type(ss, rows, now):
    out = []
    for t, grp in ss.groupby(ss["session_type"].fillna("Unknown")):
        rr = rows[rows["session_key"].isin(grp["session_key"])]
        lp, sp = lead_profile(rr), session_profile(grp, now)
        out.append({"type": t, "sessions": len(grp), "finished": sp["n_finished"],
                    "avg_fill_pct": sp["avg_fill_pct"], "sellout_share": sp["sellout_share"],
                    "sellout_hours_median": sp["sellout_hours_before_median"],
                    "n_sellouts_seen_filling": sp["n_sellouts_seen_filling"],
                    "lead_n": lp["n"], "lead_median_days": lp["median_days"],
                    "last_minute_share": lp["last_minute_share"]})
    return sorted(out, key=lambda x: -x["sessions"])


def checks(s, r, rows, now, fid):
    out = {}
    mine = s[s["fid"] == fid]
    mine_rows = rows[rows["fid"] == fid]
    out["coverage"] = {
        "sessions": len(mine), "finished": int((mine["starts_at"] < now).sum()),
        "first_session": str(mine["starts_at"].min()), "last_session": str(mine["starts_at"].max()),
        "obs_per_session_median": q(mine["n_obs"], .5),
        "sessions_with_roster": int(mine["session_key"].isin(r["session_key"]).sum()),
        "roster_rows": len(mine_rows), "distinct_players": int(mine_rows["pbp_user_id"].nunique()),
        # PlayByPoint fills whole numbers (1, 2, 3, 5) when a player has no DUPR;
        # a real DUPR rating has 2-3 decimals.
        "dupr_rated_share": pct(int(mine_rows["rating_at_time"].map(
            lambda v: pd.notna(v) and abs(float(v) * 100 - round(float(v) * 100)) < 1e-6
            and round(float(v) * 100) % 50 != 0).sum()), len(mine_rows)),
        "exact_lead_share": pct(int(mine_rows["exact"].sum()), len(mine_rows)),
        "sessions_without_type": int(mine["session_type"].isna().sum()),
    }
    # fill cross-check: roster still on at the start vs capacity - spots
    fin = mine[(mine["starts_at"] < now) & mine["spots_at_start"].notna() & mine["capacity"].gt(0)]
    on = rows[~rows["cancelled"]].groupby("session_key").size()
    cmp = fin.assign(roster=fin["session_key"].map(on).fillna(0),
                     booked=fin["capacity"] - fin["spots_at_start"])
    cmp = cmp[cmp["session_key"].isin(r["session_key"])]
    diff = (cmp["booked"] - cmp["roster"]).abs()
    out["fill_crosscheck"] = {"n": len(cmp), "within_1_share": pct(int((diff <= 1).sum()), len(cmp)),
                              "median_abs_diff": q(diff, .5),
                              "booked_more_than_roster_share": pct(int((cmp["booked"] > cmp["roster"] + 1).sum()), len(cmp))}
    # false cancellations: a "cancel" is only real if the capture job ran again
    # soon after the player's last sighting. Capture runs are recovered from the
    # timestamps the job stamped (every first_seen / last_seen is a run time).
    runs = pd.DatetimeIndex(np.sort(pd.concat([r["first_seen"], r["last_seen"]]).dt.floor("min").unique()))
    gaps = pd.Series(runs).diff().dt.total_seconds() / 3600
    left, sus = int(rows["left_roster"].sum()), int(rows["capture_gap"].sum())
    out["false_cancellations"] = {"left_roster": left, "excluded_capture_gap": sus,
                                  "share": pct(sus, left)}
    out["capture_runs"] = {"runs": len(runs), "first": str(runs.min()), "last": str(runs.max()),
                           "median_gap_hours": q(gaps, .5), "max_gap_hours": q(gaps, 1.0)}
    # venue attribution
    unk = ~r["session_key"].isin(s["session_key"])
    no_venue = rows["fid"].isna()
    out["venue_attribution"] = {"roster_rows": len(r), "no_session_match": int(unk.sum()),
                                "no_venue": int(no_venue.sum()),
                                "unattributed_share": pct(int((unk | no_venue).sum()), len(r))}
    # identity: players seen at more than one venue
    v = rows.dropna(subset=["fid"]).groupby("pbp_user_id")["fid"].nunique()
    out["identity"] = {"players": int(len(v)), "multi_venue_players": int((v > 1).sum()),
                       "multi_venue_share": pct(int((v > 1).sum()), len(v))}
    # week-to-week stability for the venue
    weeks = []
    mrows = mine_rows.assign(week=mine_rows["starts_at"].dt.tz_convert(MEL).dt.strftime("%G-W%V"))
    msess = mine.assign(week=mine["starts_at"].dt.tz_convert(MEL).dt.strftime("%G-W%V"))
    for w, grp in msess.groupby("week"):
        sp = session_profile(grp, now)
        lp = lead_profile(mrows[mrows["week"] == w])
        weeks.append({"week": w, "finished": sp["n_finished"], "avg_fill_pct": sp["avg_fill_pct"],
                      "lead_n": lp["n"], "lead_median_days": lp["median_days"]})
    out["weekly"] = weeks
    return out


def rivals(rows, reg, fid, k=4):
    """Venues ranked by shared play: share of this venue's players' sessions held elsewhere."""
    mine = set(rows.loc[rows["fid"] == fid, "pbp_user_id"])
    theirs = rows[rows["pbp_user_id"].isin(mine) & rows["fid"].notna() & (rows["fid"] != fid)]
    total = len(rows[rows["pbp_user_id"].isin(mine)])
    share = theirs.groupby("fid").agg(sessions=("session_key", "size"), players=("pbp_user_id", "nunique"))
    share["shared_play_pct"] = share["sessions"] / max(total, 1) * 100
    share = share.sort_values("sessions", ascending=False)
    out = []
    for f, row in share.iterrows():
        f = int(f)
        v = reg.get(f, {})
        out.append({"fid": f, "name": v.get("name", str(f)),
                    "km": round(km(reg[fid], v), 1) if fid in reg and v.get("lat") else None,
                    "shared_players": int(row["players"]), "shared_sessions": int(row["sessions"]),
                    "shared_play_pct": round(float(row["shared_play_pct"]), 1)})
    return {"players": len(mine), "their_sessions": total, "venues": out}


def main(d, fid):
    s, r, c = load(Path(d))
    now = max(s["last_obs"].max(), r["last_seen"].max())
    reg = registry()
    rows = build(s, r, now)
    cap_days = float(((s["starts_at"] - s["first_obs"]).dt.total_seconds() / 86400).quantile(.99))
    warm_after = s["first_obs"].min() + pd.Timedelta(days=2)

    def venue(f):
        ss, rr = s[s["fid"] == f], rows[rows["fid"] == f]
        return {"fid": f, "name": reg.get(f, {}).get("name", str(f)),
                "sessions": len(ss), "lead": lead_profile(rr), "fill": session_profile(ss, now),
                "listing": listing_profile(ss, warm_after, cap_days), "cancel": cancel_profile(rr)}

    venues = sorted({int(x) for x in s["fid"].dropna()})
    per_venue = [venue(f) for f in venues]
    rv = rivals(rows, reg, fid)
    named = [v["fid"] for v in rv["venues"][:4]]
    nearest = sorted((km(reg[fid], reg[f]), f) for f in reg if f != fid and reg[f].get("lat")
                     and reg[f].get("status") == "active")[:1]
    for _, f in nearest:
        if f not in named:
            named.append(f)
    out = {
        "as_of": str(now), "facility_id": fid, "venue": reg.get(fid, {}).get("name"),
        "catalogue_horizon_days": round(cap_days, 1),
        "checks": checks(s, r, rows, now, fid),
        "venue_detail": {
            "by_type": by_type(s[s["fid"] == fid], rows, now),
            "booking_clock": booking_clock(rows[rows["fid"] == fid]),
            "week_grid": week_grid(s[s["fid"] == fid], now),
        },
        "market": {
            "lead": lead_profile(rows[rows["fid"].notna()]),
            "fill": session_profile(s, now),
            "listing": listing_profile(s, warm_after, cap_days),
            "cancel": cancel_profile(rows[rows["fid"].notna()]),
        },
        "rivals_by_shared_play": rv,
        "named": named,
        "per_venue": per_venue,
    }
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3 or not sys.argv[2].isdigit():
        sys.exit("usage: python3 timing_preview.py <dir with the CSVs> <facility_id>")
    print(json.dumps(main(sys.argv[1], int(sys.argv[2])), indent=1, default=str))
