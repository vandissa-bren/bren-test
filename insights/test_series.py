"""Tests for series.assign_series: the one 'same weekly session' rule."""
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

import series as S

MEL = ZoneInfo("Australia/Melbourne")
MON = date(2026, 9, 28)   # a Monday


def row(week, hhmm, title=None, slug=None, kind="open play", lvl="intermediate", fid=1557, dow=3):
    d = MON + timedelta(days=7 * week + dow)
    h, m = map(int, hhmm.split(":"))
    st = datetime.combine(d, time(h, m), MEL).astimezone(timezone.utc)
    return {"fid": fid, "starts_at": st, "title": title, "program_slug": slug, "session_type": kind, "lvl": lvl}


def ids(rows):
    return list(S.assign_series(pd.DataFrame(rows)))


def test_same_title_same_time_is_one_series():
    out = ids([row(w, "19:00", "Intermediate Open Play") for w in range(4)])
    assert len(set(out)) == 1


def test_retitle_keeps_series():
    out = ids([row(0, "19:00", "Intermediate Open Play"), row(1, "19:00", "Intermediate Open Play"),
               row(2, "19:00", "Thursday Intermediate Open Play"), row(3, "19:00", "Intermediate Open Play (3.0-3.5)")])
    assert len(set(out)) == 1


def test_time_shift_15_minutes_keeps_series():
    out = ids([row(0, "19:00", "Social Round Robin"), row(1, "19:15", "Social Round Robin"),
               row(2, "19:15", "Social Round Robin")])
    assert len(set(out)) == 1


def test_time_shift_over_30_minutes_breaks_without_slug():
    out = ids([row(0, "19:00", "Social Round Robin"), row(1, "20:00", "Social Round Robin")])
    assert len(set(out)) == 2


def test_two_sessions_same_night_stay_apart():
    rows = []
    for w in range(3):
        rows += [row(w, "18:00", "Intermediate Open Play", slug="iop"), row(w, "19:00", "Intermediate Open Play", slug="iop")]
    out = ids(rows)
    assert len(set(out)) == 2
    assert out[0] == out[2] == out[4] and out[1] == out[3] == out[5]


def test_untitled_sessions_do_not_merge_on_time_alone():
    out = ids([row(0, "19:00", None, kind="open play", lvl="beginner"),
               row(1, "19:00", None, kind="clinic", lvl="beginner"),
               row(2, "19:00", None, kind="open play", lvl="advanced")])
    assert len(set(out)) == 3


def test_untitled_same_time_type_level_merge():
    out = ids([row(w, "07:00", None, kind="open play", lvl="beginner") for w in range(3)])
    assert len(set(out)) == 1


def test_untitled_joins_by_slug():
    out = ids([row(0, "07:00", "Early Birds", slug="early-birds"), row(1, "07:30", None, slug="early-birds-october")])
    assert len(set(out)) == 1


def test_season_rollover_new_slug_same_title_and_time():
    out = ids([row(0, "19:00", "Tuesday Ladder", slug="tuesday-ladder-season-3"),
               row(1, "19:00", "Tuesday Ladder", slug="tuesday-ladder-season-3"),
               row(2, "19:00", "Tuesday Ladder", slug="tuesday-ladder-s4-copy-a21bb8")])
    assert len(set(out)) == 1


def test_skipped_week_keeps_series():
    out = ids([row(0, "19:00", "Intermediate Open Play"), row(1, "19:00", "Intermediate Open Play"),
               row(3, "19:00", "Intermediate Open Play")])          # week 2 off (holiday)
    assert len(set(out)) == 1


def test_long_gap_breaks_series():
    out = ids([row(0, "19:00", "Intermediate Open Play"), row(5, "19:00", "Intermediate Open Play")])   # 5 weeks apart
    assert len(set(out)) == 2


def test_different_weekdays_or_venues_never_merge():
    out = ids([row(0, "19:00", "Open Play", slug="op", dow=1), row(1, "19:00", "Open Play", slug="op", dow=3),
               row(2, "19:00", "Open Play", slug="op", dow=3, fid=597)])
    assert len(set(out)) == 3


def test_unrelated_title_same_slot_breaks():
    out = ids([row(0, "19:00", "Intermediate Open Play"), row(1, "19:00", "Ladies Beginner Clinic")])
    assert len(set(out)) == 2


def test_series_label_uses_latest_title():
    df = pd.DataFrame([row(0, "19:00", "Open Play"), row(1, "19:00", "Open Play Intermediate")])
    df["series"] = S.assign_series(df)
    lab = S.series_label(df)
    assert len(lab) == 1 and lab.iloc[0]["title"] == "Open Play Intermediate" and lab.iloc[0]["runs"] == 2
    assert lab.iloc[0]["titles"] == 2


def test_same_words_different_level_breaks():
    out = ids([row(0, "19:00", "Open Play 3.0-3.5", lvl="intermediate"),
               row(1, "19:00", "Open Play 4.0+", lvl="advanced")])
    assert len(set(out)) == 2


# ── after the groundwork review ──

def test_mixed_levels_elsewhere_do_not_split_unlevelled_series():
    """Pandas 3 turned missing levels into NaN, which split every unlevelled series
    once any other session had a level. Reproduces the reviewer's case."""
    rows = [row(w, "19:00", "Intermediate Open Play", lvl=None) for w in range(4)]
    rows += [row(0, "10:00", "Beginner Clinic", lvl="beginner", fid=597)]
    rows += [row(w, "07:00", None, kind="open play", lvl="no level stated", dow=5) for w in range(3)]
    df = pd.DataFrame(rows)
    out = list(S.assign_series(df))
    assert len(set(out[:4])) == 1
    assert len(set(out[5:8])) == 1


def test_missing_slug_does_not_erase_series_slug():
    out = ids([row(0, "19:00", "Wed Ladder", slug="wed-ladder"), row(1, "19:00", "Wed Ladder", slug=None),
               row(2, "19:20", None, slug="wed-ladder")])
    assert len(set(out)) == 1


def test_same_slug_two_levels_same_night_stay_put_whatever_the_row_order():
    rows = []
    for w in range(5):
        rows += [row(w, "19:00", "Open Play", slug="op", lvl="intermediate"),
                 row(w, "19:00", "Open Play", slug="op", lvl="advanced")]
    df = pd.DataFrame(rows)
    for seed in range(5):
        sh = df.sample(frac=1, random_state=seed)
        out = S.assign_series(sh)
        inter = set(out[sh["lvl"] == "intermediate"])
        adv = set(out[sh["lvl"] == "advanced"])
        assert len(inter) == 1 and len(adv) == 1 and inter != adv


def test_drift_over_weeks_splits():
    times = ["17:00", "18:00", "19:00", "20:00", "21:00"]
    out = ids([row(w, t, "Evening Social", slug="evening-social") for w, t in enumerate(times)])
    assert out[0] != out[-1]
    titled = ids([row(w, t, "Evening Social") for w, t in
                  enumerate(["18:00", "18:25", "18:50", "19:15", "19:40"])])
    assert titled[0] != titled[-1]


def test_rotating_friday_events_under_one_slug_do_not_chain():
    titles = ["Advanced Social Play", "Friday Night DUPR Points Tournament Night", "LUZZ LUCKY FRIDAY",
              "Men's Singles >4", "Social Play", "Women's Singles >3.5"]
    out = ids([row(w, "19:00", t, slug="friday-nights", lvl=None, dow=4) for w, t in enumerate(titles)])
    assert len(set(out)) == 6


def test_level_word_on_one_side_blocks():
    out = ids([row(0, "19:00", "Social Play", lvl=None), row(1, "19:00", "Advanced Social Play", lvl=None)])
    assert len(set(out)) == 2
    out = ids([row(0, "19:00", "Social Play: Night Session - All Skill Levels", lvl=None),
               row(1, "19:00", "Social Play: Night Session - Beginners", lvl=None)])
    assert len(set(out)) == 2


def test_competition_sessions_get_no_series():
    out = ids([row(w, "19:00", f"Team Eastern <3 Division - EIPC vs {opp}", lvl=None)
               for w, opp in enumerate(["Pickleball Manningham", "Box Hill", "Ringwood"])]
              + [row(0, "10:00", "Club Champs", kind="Tournament", lvl=None)])
    assert out == [None, None, None, None]
