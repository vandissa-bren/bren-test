"""
programme_terms.py -- what a programme says about itself (27 Nov 2026).

capture_programme_details.py keeps, for every PlayByPoint programme, its
category (the venue's own: "Clinic", "Social Play", "League ", "DUPR Session"),
its rating range (ntrp_str: "> 3.0", "< 3.6", "3.0 / 7.0") and its description.
Titles carry no level for most sessions (1,627 of 1,796 in the live check), so
the nightly build reads the level, the format, the coaching and a season's
length from here when the title doesn't say:

  band       (lo, hi) from the rating range, else the description's first line
             that gives a level ("Recommended DUPR 2.75 - 3.50", "For advanced
             players (3.5+)", "Level: Intermediate (roughly 2.5-4.0)"), read
             with the same parser as titles (catalogue_terms.session_terms)
  all_levels the description says all levels / everyone welcome
  format     from the category ("Clinic" -> clinic, "League" -> league,
             "Tournaments" -> league, "Open Play" -> open_play)
  coached    a coach leads it ("Group coaching session with ...", "our friendly
             coach will guide you"), not "no formal coach"
  weeks      a season's length ("League will be a total of 10 weeks", "Term
             Payments (10 weeks)", "an 8 week process"): only used to split a
             price already known to be a season (venue_report.session_prices)
  members    members only

Only applied where the title says nothing (venue_report.attach_titles).
"""
from __future__ import annotations

import re
from typing import Optional

from catalogue_terms import session_terms, title_format, label_band, _fmt

LEVEL_WORDS = re.compile(r"\b(beginner|intermediate|advanced|novice|improver|all\s+levels?|dupr|level|rating|rated)\b", re.I)
RANGE = re.compile(r"\b[2-6](?:\.\d{1,2})?\s*(?:-|–|—|to)\s*[2-6]\.\d{1,2}\b|\b[2-6]\.\d{1,2}\s*(?:-|–|—|to)\s*[2-6](?:\.\d{1,2})?\b|\b[2-6]\.\d{1,2}\s*\+")
ALL_LEVELS = re.compile(r"\ball\s+levels?\b|\beveryone\s+(?:is\s+)?welcome\b|\bany\s+level\b|\ball\s+abilities\b", re.I)
COACH = re.compile(r"\bcoach(?:es|ed|ing)?\b|\binstructor\b|\bguided\s+by\b|\bled\s+by\b|\bpro\s+player\b", re.I)
NO_COACH = re.compile(r"\bno\s+(?:formal\s+)?coach(?:es|ing)?\b|\bwithout\s+(?:a\s+)?coach\b|\bnot\s+(?:a\s+)?coach(?:ed|ing)\b|\bself[\s-]guided\b", re.I)
WEEKS = re.compile(r"\b(\d{1,2})\s*-?\s*(?:weeks?|wks?)\b", re.I)
# "not suitable for beginners", "no beginners", "isn't for beginners": a line ruling a level out
NEG = re.compile(r"\b(?:not|no|isn['’]?t|unsuitable|excluding|except)\b[^.\n]{0,25}\b(?:beginner|intermediate|advanced)", re.I)
MEMBERS = re.compile(r"\bmembers?\s+only\b", re.I)
NTRP = re.compile(r"^\s*([<>])\s*(\d(?:\.\d+)?)\s*$|^\s*(\d(?:\.\d+)?)\s*/\s*(\d(?:\.\d+)?)\s*$")


def _ntrp_band(s: Optional[str]) -> Optional[tuple]:
    """PlayByPoint's rating range: '> 3.0' -> (3.0, 8), '< 3.6' -> (1.0, 3.6),
    '3.0 / 7.0' -> (3.0, 8); '1.0 / 6.0' (everyone) -> None."""
    m = NTRP.match(s or "")
    if not m:
        return None
    if m.group(1):
        v = float(m.group(2))
        return (v, 8.0) if m.group(1) == ">" else (1.0, v)
    lo, hi = float(m.group(3)), float(m.group(4))
    if lo <= 1.5 and hi >= 5.5:
        return None
    return (lo, 8.0 if hi >= 5.5 else hi)


def _level_line(desc: str) -> Optional[str]:
    """The description's first line that states a level: a rating range first,
    else a line naming levels in its first 160 characters."""
    lines = [x.strip() for x in desc.splitlines() if x.strip()]
    for x in lines:
        if RANGE.search(x) and LEVEL_WORDS.search(x):
            return x[:200]
    for x in lines:
        if RANGE.search(x):
            return x[:200]
    for x in lines:
        if re.search(r"\b(beginner|intermediate|advanced)", x[:160], re.I) and not NEG.search(x[:160]):
            return x[:160]
    return None


def parse(name: Optional[str], category: Optional[str], ntrp: Optional[str], desc: Optional[str]) -> dict:
    desc = desc or ""
    band = _ntrp_band(ntrp)
    label = None
    line = _level_line(desc) if desc else None
    if band:
        label = f"{_fmt(band[0])}+" if band[1] >= 8 else (f"<{_fmt(band[1])}" if band[0] <= 1.0 else f"{_fmt(band[0])}-{_fmt(band[1])}")
    elif line:
        t = session_terms(line)
        if t["band"]:
            band, label = tuple(t["band"]), t["label"]
    all_levels = bool(ALL_LEVELS.search(desc)) and band is None
    cat = (category or "").strip()
    fmt = title_format(cat) if cat else None
    if fmt is None and re.search(r"\btournaments?\b|\bcompetitions?\b", cat, re.I):
        fmt = "league"
    coached = bool(COACH.search(desc)) and not NO_COACH.search(desc)
    weeks = None
    for m in WEEKS.finditer(desc):
        n = int(m.group(1))
        if 2 <= n <= 52:
            weeks = n
            break
    return {"band": band, "label": label, "all_levels": all_levels, "format": fmt, "category": cat or None,
            "coached": coached, "weeks": weeks, "members": bool(MEMBERS.search(desc))}


def table(details) -> dict:
    """{program_slug: parse(...)} from the progdetails export (a DataFrame)."""
    if details is None or len(details) == 0:
        return {}
    out = {}
    for r in details.itertuples(index=False):
        g = lambda k: getattr(r, k, None) if isinstance(getattr(r, k, None), str) else None
        out[r.program_slug] = parse(g("name"), g("category"), g("ntrp_str"), g("description"))
    return out


__all__ = ["parse", "table", "label_band"]
