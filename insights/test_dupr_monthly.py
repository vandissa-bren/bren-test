"""Tests for dupr_monthly.py and the build's DUPR history merge. No network.

    python -m pytest test_dupr_monthly.py -q
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pandas as pd

import dupr_monthly as dm

sys.path.insert(0, str(Path(__file__).parent / "insights"))


def test_dry_run_picks_round_robin_and_never_signs_in(monkeypatch, capsys):
    calls = []

    async def fake_rpc(client, name, payload):
        calls.append((name, payload))
        return [{"pbp_user_id": 1, "venue": "pbp-597"}, {"pbp_user_id": 2, "venue": "pbp-597"},
                {"pbp_user_id": 3, "venue": "pbp-755"}]

    monkeypatch.setattr(dm.lr, "SUPABASE_KEY", "test")
    monkeypatch.setattr(dm.lr, "_rpc", fake_rpc)
    monkeypatch.setattr(dm, "DRY_RUN", True)
    assert asyncio.run(dm.main()) == 0
    assert calls == [("dupr_monthly_targets", {"p_limit": dm.LIMIT, "p_min_sessions": 4})]
    out = capsys.readouterr().out
    assert "Targets: 3 across 2 venues" in out
    assert "pbp_user_id" not in out


def test_refuses_in_the_nightly_window(monkeypatch):
    monkeypatch.setattr(dm.lr, "SUPABASE_KEY", "test")
    monkeypatch.setattr(dm, "DRY_RUN", False)
    monkeypatch.setattr(dm.S, "FORCE", False)
    monkeypatch.setattr(dm.S, "in_nightly_window", lambda now: True)
    assert asyncio.run(dm.main()) == 1


def test_dupr_log_adds_signed_in_readings_to_the_history():
    import build_insights as BI
    import pm_v2 as V2
    log = pd.DataFrame({"pbp_user_id": [1], "rating": [3.12], "prev_rating": [3.04],
                        "seen_at": ["2026-10-01T00:00:00Z"], "prev_seen_at": ["2026-09-01T00:00:00Z"],
                        "session_key": ["k"], "source": ["roster"]})
    reads = pd.DataFrame({"pbp_user_id": [1, 2], "doubles": [3.31, 3.82],
                          "read_at": ["2026-11-02T03:30:00Z", "2026-11-02T03:31:00Z"]})
    m = pd.DataFrame(columns=["pbp_user_id", "last_seen", "rating_at_time"])
    h = V2.dupr_history(m, BI.dupr_log(log, reads))
    assert h[h.pbp_user_id == 1]["v"].round(2).tolist() == [3.04, 3.12, 3.31]
    assert h[h.pbp_user_id == 2]["v"].round(2).tolist() == [3.82]
    assert BI.dupr_log(log, None) is log
    assert BI.dupr_log(None, reads)["rating"].tolist() == [3.31, 3.82]
