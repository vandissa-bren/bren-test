"""What a programme says about itself (programme_terms.py, 27 Nov 2026)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import programme_terms as PT  # noqa: E402
import venue_report as R  # noqa: E402


def test_rating_range_field():
    assert PT.parse("x", "Clinic", "> 3.0", "")["band"] == (3.0, 8.0)
    assert PT.parse("x", "Clinic", "< 3.6", "")["band"] == (1.0, 3.6)
    assert PT.parse("x", "Clinic", "1.0 / 6.0", "")["band"] is None


def test_level_from_the_description():
    d = "Fast rallies.\nRecommended DUPR 2.75 – 3.50 for entry\nBring water."
    p = PT.parse("Intermediate Socials", "Social Play", None, d)
    assert p["band"] == (2.75, 3.5)
    p = PT.parse("x", "Social", None, "For advanced players (3.5+)\nRotating partners")
    assert p["band"][0] == 3.5


def test_a_line_ruling_a_level_out_is_skipped():
    p = PT.parse("x", "Social", None, "Fast games.\nNot suitable for beginners.\nFor intermediate players.")
    assert p["band"] is not None and p["band"][0] >= 2.5


def test_coaching_weeks_members_format():
    p = PT.parse("x", "League ", None, "League will be a total of 10 weeks.\nMEMBERS ONLY")
    assert p["weeks"] == 10 and p["members"] and p["format"] == "league"
    assert PT.parse("x", "Social", None, "Group coaching session with Pro player Sam")["coached"]
    assert not PT.parse("x", "Social", None, "no pressure, no formal coach")["coached"]
    assert PT.parse("x", "Tournaments", None, "")["format"] == "league"


def test_all_levels():
    p = PT.parse("x", "Open Play", None, "Everyone welcome. All levels.")
    assert p["all_levels"] and p["band"] is None


def test_season_split_by_the_description_weeks():
    R.PROGRAMS = pd.DataFrame(columns=["venue_id", "program_slug", "tiers", "lessons_ahead_max"])
    base = {"fid": 1, "venue_id": "pbp-1", "program_slug": None, "price_tiers": None, "fmt": "social", "prog_weeks": None}
    rows = [{**base, "price": "$20", "title": "Open Play"}] * 6
    rows.append({**base, "price": "$200", "title": "Wednesday Ladder Comp", "fmt": "league", "prog_weeks": 10})
    s = R.session_prices(pd.DataFrame(rows))
    assert s["price_n"].iloc[-1] == 20 and s["price_basis"].iloc[-1] == "season-weeks"
    # description weeks never make an ordinary price a season
    rows = [{**base, "price": "$20", "title": "Youth Skills", "prog_weeks": 6}]
    s = R.session_prices(pd.DataFrame(rows))
    assert s["price_n"].iloc[0] == 20
