"""one_off_competitions (28 Nov 2026): a tournament division that runs once is
not a visit; a weekly league night is."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from build_overview import one_off_competitions


def _rows(title, dates, fid=1009, hm="18:00"):
    return [{"session_key": f"{title}-{d}", "fid": fid, "title": title, "capacity": 16,
             "starts_at": pd.Timestamp(f"{d} {hm}", tz="Australia/Melbourne").tz_convert("UTC")} for d in dates]


def test_a_one_off_division_is_left_out_and_a_weekly_league_stays():
    f = pd.DataFrame(_rows("Men's Singles 3-3.49", ["2026-11-07"])
                     + _rows("Tuesday League", ["2026-11-03", "2026-11-10", "2026-11-17"], hm="19:00")
                     + _rows("Social Play", ["2026-11-04"], hm="10:00"))
    out = one_off_competitions(f)
    assert out == {"Men's Singles 3-3.49-2026-11-07"}
