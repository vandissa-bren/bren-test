"""Per-session prices: seasons split or left out, never counted each night (25 Nov 2026)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import venue_report as R  # noqa: E402


def frame(rows):
    R.PROGRAMS = pd.DataFrame(columns=["venue_id", "program_slug", "tiers", "lessons_ahead_max"])
    base = {"fid": 1, "venue_id": "pbp-1", "program_slug": None, "price_tiers": None, "fmt": "social"}
    return R.session_prices(pd.DataFrame([{**base, **r} for r in rows]))


SOCIALS = [{"price": "$20", "title": "Open Play"}] * 6


def test_league_by_its_format_is_never_a_nightly_price():
    s = frame(SOCIALS + [{"price": "$250", "title": "Tuesday Doubles Comp", "fmt": "league"}])
    assert s["price_n"].iloc[-1] != 250
    assert s["price_basis"].iloc[-1] == "season"


def test_league_split_by_its_weeks():
    s = frame(SOCIALS + [{"price": "$250", "title": "Tuesday Doubles 10 weeks", "fmt": "league"}])
    assert s["price_n"].iloc[-1] == 25 and s["price_basis"].iloc[-1] == "season-weeks"


def test_commitment_tier_marks_a_season():
    tiers = [{"price": 125, "player_category": None, "lessons": 1, "lesson_unit": "per_week_per_next_sessions"}]
    s = frame(SOCIALS + [{"price": "$125", "title": "Tuesday Night Doubles", "price_tiers": tiers}])
    assert pd.isna(s["price_n"].iloc[-1]) and s["price_basis"].iloc[-1] == "season"


def test_many_leagues_dont_make_250_typical():
    rows = [{"price": "$20", "title": "Open Play"}] * 3 + [{"price": "$250", "title": f"Comp {i}", "fmt": "league"} for i in range(5)]
    rows.append({"price": "$240", "title": "Thursday Night Doubles"})        # not titled a league, still 12x a social
    s = frame(rows)
    assert pd.isna(s["price_n"].iloc[-1])


def test_ordinary_sessions_keep_their_price():
    s = frame(SOCIALS + [{"price": "$22", "title": "Ladder Night", "fmt": "ladder"}])
    assert list(s["price_n"]) == [20] * 6 + [22]
    assert set(s["price_basis"]) == {"listed"}


def test_a_session_tier_wins():
    tiers = [{"price": 25, "player_category": None, "lesson_unit": "session"}]
    s = frame(SOCIALS + [{"price": "$250", "title": "League night", "fmt": "league", "price_tiers": tiers}])
    assert s["price_n"].iloc[-1] == 25 and s["price_basis"].iloc[-1] == "tier"
