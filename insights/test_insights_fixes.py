"""
test_insights_fixes.py -- the cancellation rule, the 21 Sep window cut and the
like-for-like money figures, on small synthetic data.

Run: python -m pytest -q insights/test_insights_fixes.py
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cancellations as CX          # noqa: E402
import build_insights as BI         # noqa: E402
import build_overview as B          # noqa: E402

DAY = "2026-10-20"
START = pd.Timestamp(f"{DAY} 09:00", tz="UTC")


def t(hhmm, day=DAY):
    return pd.Timestamp(f"{day} {hhmm}", tz="UTC")


def background():
    """Another session the same day, read by the hourly runs 05:09-11:09: one
    booker joining at each run, so every run leaves a stamp."""
    rows = []
    for h in range(5, 12):
        rows.append({"pbp_user_id": 900 + h, "session_key": "pbp-bg", "session_date": DAY,
                     "first_seen": t(f"{h:02d}:09"), "last_seen": t("11:09"),
                     "starts_at": t("12:00")})
    return rows


def frame(rows):
    r = pd.DataFrame(rows)
    for c in ("first_seen", "last_seen", "starts_at"):
        r[c] = pd.to_datetime(r[c], utc=True)
    return r


def flags(r):
    out = CX.classify(r)
    return dict(zip(zip(out["session_key"], out["pbp_user_id"]), out["cancelled"])), out


def test_removed_after_start_is_not_a_cancellation():
    # A was on the roster at the last pass before the 09:00 start (08:09) and
    # was gone by the first pass after it: booked at the start, so attended.
    rows = background() + [
        {"pbp_user_id": 1, "session_key": "pbp-s", "session_date": DAY,
         "first_seen": t("05:09"), "last_seen": t("08:09"), "starts_at": START},
        {"pbp_user_id": 2, "session_key": "pbp-s", "session_date": DAY,
         "first_seen": t("05:09"), "last_seen": t("11:09"), "starts_at": START},
    ]
    f, _ = flags(frame(rows))
    assert not f[("pbp-s", 1)]
    assert not f[("pbp-s", 2)]


def test_gone_before_a_later_pre_start_pass_is_a_cancellation():
    # C was last seen at 06:09; the 07:09 and 08:09 runs read the session
    # (it was still being read at 11:09) and C was gone.
    rows = background() + [
        {"pbp_user_id": 3, "session_key": "pbp-s", "session_date": DAY,
         "first_seen": t("05:09"), "last_seen": t("06:09"), "starts_at": START},
        {"pbp_user_id": 2, "session_key": "pbp-s", "session_date": DAY,
         "first_seen": t("05:09"), "last_seen": t("11:09"), "starts_at": START},
    ]
    f, out = flags(frame(rows))
    assert f[("pbp-s", 3)]
    h = out.loc[(out["session_key"] == "pbp-s") & (out["pbp_user_id"] == 3), "cancel_hours_before"].iloc[0]
    assert h == pytest.approx(2 + 51 / 60)


def test_session_read_once_before_start_has_no_cancellations():
    # every sighting from one pass at 06:09; later runs happened (background)
    # but nothing shows this session was read again (pulled / out of band)
    rows = background() + [
        {"pbp_user_id": p, "session_key": "pbp-once", "session_date": DAY,
         "first_seen": t("06:09"), "last_seen": t("06:09"), "starts_at": START}
        for p in (4, 5, 6)
    ]
    f, _ = flags(frame(rows))
    assert not any(f[("pbp-once", p)] for p in (4, 5, 6))


def test_run_outside_the_sessions_band_is_not_evidence():
    # a session 5 days ahead is read by the 3-hourly mid runs, not the hourly
    # near ones. Player 7 was last read at 03:24; player 8 stayed on (seen
    # again only after the start). Only near runs (05:09-11:09) ran in
    # between: they didn't read this session, so 7 isn't known to have left.
    far = "2026-10-25"
    rows = background() + [
        {"pbp_user_id": 7, "session_key": "pbp-far", "session_date": far,
         "first_seen": t("03:24"), "last_seen": t("03:24"), "starts_at": t("09:00", far)},
        {"pbp_user_id": 8, "session_key": "pbp-far", "session_date": far,
         "first_seen": t("03:24"), "last_seen": t("09:40", far), "starts_at": t("09:00", far)}]
    out = CX.classify(frame(rows))
    assert not out.loc[out["session_key"] == "pbp-far", "cancelled"].any()
    # ...whereas a mid run at 06:24 (seen on another far session) did read it
    rows.append({"pbp_user_id": 99, "session_key": "pbp-far2", "session_date": far,
                 "first_seen": t("06:24"), "last_seen": t("06:24"), "starts_at": t("10:00", far)})
    out = CX.classify(frame(rows))
    f = dict(zip(zip(out["session_key"], out["pbp_user_id"]), out["cancelled"]))
    assert f[("pbp-far", 7)] and not f[("pbp-far", 8)]


def test_hidden_roster_is_not_a_mass_cancellation():
    # critic's case: the venue hides the roster (or the fetch keeps failing)
    # after one run; the capture job keeps writing spots to the fill log up to
    # the start, and other sessions the same day are read hourly. Nobody on
    # the hidden session is cancelled, with or without the fill-log reading.
    T = lambda x: pd.Timestamp(x, tz="UTC")
    rows = [dict(pbp_user_id=p, session_key="S", session_date="2026-11-10",
                 first_seen=T("2026-11-08 06:09"), last_seen=T("2026-11-09 10:09"),
                 starts_at=T("2026-11-10 08:00")) for p in range(1, 7)]
    rows += [dict(pbp_user_id=100 + h, session_key="O", session_date="2026-11-10",
                  first_seen=T(f"2026-11-09 {h:02d}:09"), last_seen=T("2026-11-10 09:09"),
                  starts_at=T("2026-11-10 08:00")) for h in range(11, 24)]
    r = frame(rows)
    s = pd.DataFrame([dict(session_key="S", last_obs_before="2026-11-10T07:09:00+00:00"),
                      dict(session_key="O", last_obs_before="2026-11-10T07:09:00+00:00")])
    for out in (CX.classify(r, s), CX.classify(r)):
        assert not out.loc[out["session_key"] == "S", "cancelled"].any()


def test_pass_inside_the_read_lag_is_not_evidence():
    # run stamped 08:39 for a 09:00 start may not reach the session until
    # after the start, so a player last seen at 08:09 and gone at the 08:39
    # pass isn't counted as cancelled; one gone at the 07:09 pass is.
    rows = background() + [
        {"pbp_user_id": 1, "session_key": "pbp-s", "session_date": DAY,
         "first_seen": t("05:09"), "last_seen": t("08:09"), "starts_at": START},
        {"pbp_user_id": 3, "session_key": "pbp-s", "session_date": DAY,
         "first_seen": t("05:09"), "last_seen": t("07:09"), "starts_at": START},
        {"pbp_user_id": 2, "session_key": "pbp-s", "session_date": DAY,
         "first_seen": t("05:09"), "last_seen": t("11:09"), "starts_at": START},
        # a run stamped 08:39 that read this session (another booker)
        {"pbp_user_id": 4, "session_key": "pbp-s", "session_date": DAY,
         "first_seen": t("08:39"), "last_seen": t("11:09"), "starts_at": START},
    ]
    f, _ = flags(frame(rows))
    assert not f[("pbp-s", 1)]
    assert f[("pbp-s", 3)]


def test_venue_report_and_timing_use_the_same_rule():
    import venue_report as R
    import timing_preview as TP
    rows = background() + [
        {"pbp_user_id": 1, "session_key": "pbp-s", "session_date": DAY,
         "first_seen": t("05:09"), "last_seen": t("08:09")},
        {"pbp_user_id": 3, "session_key": "pbp-s", "session_date": DAY,
         "first_seen": t("05:09"), "last_seen": t("06:09")},
        {"pbp_user_id": 2, "session_key": "pbp-s", "session_date": DAY,
         "first_seen": t("05:09"), "last_seen": t("11:09")},
    ]
    r = pd.DataFrame([{k: v for k, v in x.items() if k != "starts_at"} for x in rows]).assign(rating_at_time=3.0)
    s = pd.DataFrame({"session_key": ["pbp-s", "pbp-bg"], "fid": [1.0, 1.0],
                      "starts_at": [START, t("12:00")], "first_obs": [t("01:00")] * 2,
                      "first_full_at": [pd.NaT] * 2, "session_type": ["Social"] * 2, "capacity": [8, 8]})
    for c in ("title", "family", "label", "band_lo", "band_hi", "dow", "hour", "hm"):
        s[c] = np.nan
    m = R.rosters(s, r)
    tl = TP.build(s, r, t("23:00"))
    a = dict(zip(m["pbp_user_id"], m["cancelled"]))
    b = dict(zip(tl["pbp_user_id"], tl["cancelled"]))
    assert a == b
    assert not a[1] and a[3] and not a[2]


# ── windows from 21 Sep ──

def test_window_clamped_to_data_start():
    w = BI.window(90, date(2026, 11, 7))
    assert BI.DATA_START == date(2026, 9, 21)
    assert w["from"] == date(2026, 9, 21) and w["clamped"]
    assert w["covered"] == 48
    assert w["prev"] is None                      # would start before 21 Sep


def test_window_prev_kept_or_omitted():
    w7 = BI.window(7, date(2026, 11, 7))
    assert w7["from"] == date(2026, 11, 1) and not w7["clamped"] and w7["covered"] == 7
    assert w7["prev"] == (date(2026, 10, 25), date(2026, 10, 31))
    w28 = BI.window(28, date(2026, 11, 7))
    assert w28["prev"] is None                    # 13 Sep - 10 Oct reaches back too far
    assert BI.window(28, date(2026, 11, 20))["prev"] == (date(2026, 9, 26), date(2026, 10, 23))


def test_per_week_divides_by_days_covered():
    start, end = pd.Timestamp("2026-09-21", tz="UTC"), pd.Timestamp("2026-11-07", tz="UTC")
    old = B.COVERED_DAYS
    try:
        B.COVERED_DAYS = 48
        assert B.covered_days(start, end) == 48
        assert 96 / (B.covered_days(start, end) / 7) == pytest.approx(14.0)   # 96 sessions → 14 a week
        B.COVERED_DAYS = None
        assert B.covered_days(start, end) == 48       # the sessions' own span
    finally:
        B.COVERED_DAYS = old


# ── money like for like ──

def sessions(fmt, n, cap, price, booked, dur, hire=None):
    return pd.DataFrame({"fmt": [fmt] * n, "capacity": [cap] * n, "price_n": [float(price)] * n,
                         "mprice_n": [np.nan] * n, "booked": [booked] * n, "spots_at_start": [cap - booked] * n,
                         "dur_h": [float(dur)] * n, "price_basis": ["listed"] * n, "hire_rate": [hire] * n})


def test_full_court_hour_arithmetic():
    # doubles: 8 places → 2 courts; $20 × 8 / (2 h × 2 courts) = $40 full, $20 × 6 / 4 = $30 taken
    m = B.money(sessions("round_robin", 5, 8, 20, 6, 2, hire=40.0))
    assert m["fullCourtHour"] == 40.0
    assert m["takenCourtHour"] == 30.0
    assert m["perCourtHour"] == 30.0              # old key kept, old rule (a court per 4)
    assert m["hireRate"] == 40.0
    assert m["emptyValue"] == 5 * 20 * 2


def test_courts_by_format():
    # social: 10 places / 5 a court = 2 courts → $15 × 10 / (2 h × 2) = $37.5
    assert B.money(sessions("social", 5, 10, 15, 10, 2))["fullCourtHour"] == 37.5
    # clinic: 8 places / 8 a court = 1 court → $40 × 8 / 1 h = $320
    assert B.money(sessions("clinic", 5, 8, 40, 8, 1))["fullCourtHour"] == 320.0
    # under 5 timed sessions: no figure
    assert B.money(sessions("social", 4, 10, 15, 10, 2))["fullCourtHour"] is None


def test_hire_rate_for_venue_day_and_slot():
    f = pd.DataFrame({"fid": [1, 1], "starts_at": [pd.Timestamp("2026-10-20 18:00", tz="Australia/Melbourne").tz_convert("UTC"),
                                                   pd.Timestamp("2026-10-27 10:15", tz="Australia/Melbourne").tz_convert("UTC")]})
    shifts = pd.DataFrame({"facility_id": [1, 1], "slot_date": ["2026-10-20", "2026-10-20"],
                           "slot_start": ["18:00:00", "10:00:00"], "shift": ["peak", "offpeak"]})
    rates = pd.DataFrame({"facility_id": [1, 1, 1, 1], "court_id": ["a", "b", "a", "b"],
                          "shift": ["peak", "peak", "offpeak", "offpeak"], "price": [50, 60, 30, 30],
                          "member_price": [None] * 4, "first_seen": ["2026-10-01T00:00:00Z"] * 4})
    h = B.hire_rates(f, rates, shifts)
    assert h.iloc[0] == 55.0                      # that date's shift, median of the courts
    assert h.iloc[1] == 30.0                      # another Tuesday 10:15: the usual shift for Tue 10:00


# ── REST exports don't stop at the server's row cap ──

class _Resp:
    def __init__(self, rows, total):
        self.status_code = 206
        self._rows = rows
        self.headers = {"Content-Range": (f"{rows[0]['id']}-{rows[-1]['id']}/{total}" if rows else f"*/{total}")}
        self.text = ""

    def json(self):
        return self._rows


class _CappedClient:
    """Fake PostgREST: 2,500 rows, never more than 1,000 per response."""
    def __init__(self, n=2500, cap=1000, total_known=True):
        self.data = [{"id": i} for i in range(n)]
        self.cap, self.calls, self.total_known = cap, 0, total_known

    def get(self, url, headers, params):
        self.calls += 1
        lo, hi = map(int, headers["Range"].split("-"))
        rows = self.data[lo:min(hi + 1, lo + self.cap)]
        return _Resp(rows, len(self.data) if self.total_known else "*")


@pytest.mark.parametrize("page", [1000, 10000])
@pytest.mark.parametrize("known", [True, False])
def test_table_reads_past_the_row_cap(page, known):
    c = _CappedClient(total_known=known)
    rows = BI.table(c, "http://x", {}, "court_slot_states", {}, page=page)
    assert [r["id"] for r in rows] == list(range(2500))
