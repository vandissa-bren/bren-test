"""
Groundwork for the new figures: when a waitlist first appeared, and courts free
per venue x weekday x hour (court_free.py, build_insights export/build).

    python -m pytest -q test_groundwork_courts.py
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import court_free as C  # noqa: E402


def test_waitlist_first_seen_before_start_only():
    s = pd.DataFrame({"session_key": ["a", "b", "c"],
                      "starts_at": ["2026-10-20T08:00:00+00:00"] * 3})
    w = pd.DataFrame({"session_key": ["a", "a", "a", "b", "c"],
                      "observed_at": ["2026-10-19T10:00:00+00:00", "2026-10-18T09:00:00+00:00",
                                      "2026-10-20T09:00:00+00:00",   # after the start: ignored
                                      "2026-10-20T08:30:00+00:00",   # only after the start
                                      "2026-10-19T12:00:00+00:00"],
                      "waitlist": [4, 2, 9, 3, 0]})                 # c: 0 isn't a queue
    out = C.waitlist_times(s, w).set_index("session_key")
    assert out.loc["a", "waitlist_first_at"].startswith("2026-10-18T09:00")
    assert out.loc["a", "waitlist_first"] == 2
    assert pd.isna(out.loc["b", "waitlist_first_at"])
    assert pd.isna(out.loc["c", "waitlist_first_at"])


def test_waitlist_times_without_readings():
    s = pd.DataFrame({"session_key": ["a"], "starts_at": ["2026-10-20T08:00:00+00:00"]})
    out = C.waitlist_times(s, None)
    assert out["waitlist_first_at"].isna().all()


def _slots(rows):
    return pd.DataFrame(rows, columns=C.COLS)


def test_hour_needs_both_slots_free():
    # Mon 19:00 at venue 1, two weeks; week 1 both slots 3 free, week 2 one slot 1 free
    rows = []
    for d, frees in (("2026-10-12", (3, 3)), ("2026-10-19", (3, 1))):
        for mm, f in zip(("19:00:00", "19:30:00"), frees):
            rows.append([1, d, mm, 6, 6 - f, 2, 4 - f, 4])
    out = C.summarise(_slots(rows))
    assert out["cols"] == C.OUT_COLS
    (r,) = out["rows"]
    rec = dict(zip(C.OUT_COLS, r))
    assert rec["v"] == 1 and rec["dow"] == 1 and rec["h"] == 19
    assert rec["weeks"] == 2 and rec["free2"] == 1      # week 2's hour has only 1 free throughout
    assert rec["freeMed"] == 2.0 and rec["courts"] == 6
    assert rec["prog"] == round(8 / 24, 3)
    assert rec["hire"] == round(sum(4 - f for f in (3, 3, 3, 1)) / 16, 3)


def test_summarise_empty_and_bad_rows():
    assert C.summarise(None)["rows"] == []
    assert C.summarise(_slots([[1, "2026-10-12", "19:00:00", 0, 0, 0, 0, 0]]))["rows"] == []


def test_build_carries_court_free_and_waitlist_columns():
    import fixture_build as F
    import build_insights as BI
    with tempfile.TemporaryDirectory() as t:
        data, work = Path(t) / "data", Path(t) / "work"
        work.mkdir()
        F.write(data)
        BI.add_waitlist_times(data)
        s = pd.read_csv(data / "sessions.csv")
        assert {"waitlist_first_at", "waitlist_first"} <= set(s.columns)
        assert s["waitlist_first_at"].notna().any()
        d, net = BI.build(data, work)
        cf = d["courtFree"]
        assert cf["cols"] == C.OUT_COLS and cf["rows"]
        out = BI.shareable(d, net)
        ids = {v["id"] for v in out["venues"]}
        assert all(r[0] in ids for r in out["courtFree"]["rows"])
        assert all(r[4] <= r[3] for r in out["courtFree"]["rows"])   # free2 <= weeks


# ── after the groundwork review ──

def test_hour_with_one_slot_logged_is_not_counted():
    rows = [[1, "2026-10-12", "19:00:00", 6, 1, 0, 1, 6], [1, "2026-10-12", "19:30:00", 6, 1, 0, 1, 6],
            [1, "2026-10-19", "19:30:00", 6, 1, 0, 1, 6]]                 # week 2: only the :30 slot
    (r,) = C.summarise(_slots(rows))["rows"]
    rec = dict(zip(C.OUT_COLS, r))
    assert rec["weeks"] == 1 and rec["free2"] == 1


class _Resp:
    def __init__(self, code, body=None, text=""):
        self.status_code, self._body, self.text = code, body, text

    def json(self):
        return self._body


class _Client:
    """Answers ins_court_hire from a script: each call pops the next response for its week."""
    def __init__(self, script):
        self.script, self.calls = script, []

    def post(self, url, headers=None, params=None, json=None):
        self.calls.append((json["p_from"], params["offset"]))
        resp = self.script[json["p_from"]].pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


def _row(d, mm="19:00:00"):
    return {"fid": 1, "slot_date": d, "slot_start": mm, "n_courts": 4, "taken": 1, "programmed": 0,
            "hire_taken": 1, "hireable": 4}


def test_fetch_retries_keeps_good_weeks_and_pages_until_empty():
    from datetime import date
    script = {
        # week 1: a 502, then a short page (a server cap below 1,000), then more, then empty
        "2026-10-05": [_Resp(502), _Resp(200, [_row("2026-10-05")]), _Resp(200, [_row("2026-10-06")]), _Resp(200, [])],
        # week 2: a statement timeout on every try -> skipped, the run carries on
        "2026-10-12": [_Resp(400, text='{"code":"57014"}')] * 3,
        # week 3: a dropped connection, then fine
        "2026-10-19": [ConnectionError("reset"), _Resp(200, [_row("2026-10-19")]), _Resp(200, [])],
    }
    got = C.fetch(_Client(script), "http://x", {}, date(2026, 10, 5), date(2026, 10, 25), wait=0)
    assert [r["slot_date"] for r in got["rows"]] == ["2026-10-05", "2026-10-06", "2026-10-19"]
    assert got["ok"] is False and got["weeksRead"] == 2 and got["weeks"] == 3


def test_export_fetches_court_hire_once_and_skips_extras_for_prev(monkeypatch):
    import build_insights as BI
    from datetime import date
    calls = []

    def fake_fetch(client, url, h, frm, to, **kw):
        calls.append((frm, to))
        return {"rows": [_row("2026-10-05"), _row("2026-10-20")], "ok": True, "weeksRead": 3, "weeks": 3}

    monkeypatch.setattr(C, "fetch", fake_fetch)
    monkeypatch.setattr(BI, "rpc", lambda *a, **k: [])
    monkeypatch.setattr(BI, "table", lambda *a, **k: [])
    cache = {}
    with tempfile.TemporaryDirectory() as t:
        long_, short = Path(t) / "long", Path(t) / "short"
        long_.mkdir(); short.mkdir()
        BI.export(None, "u", {}, date(2026, 10, 5), date(2026, 10, 25), long_, courthire=cache)
        BI.export(None, "u", {}, date(2026, 10, 19), date(2026, 10, 25), short, courthire=cache)
        assert calls == [(date(2026, 10, 5), date(2026, 10, 25))]           # fetched once
        assert list(pd.read_csv(short / "courthire.csv")["slot_date"]) == ["2026-10-20"]
        prev = Path(t) / "prev"
        prev.mkdir()
        BI.export(None, "u", {}, date(2026, 9, 21), date(2026, 10, 4), prev, skip=("courthire", "waitlists"))
        assert not (prev / "courthire.csv").exists() and not (prev / "waitlists.csv").exists()
        assert len(calls) == 1
