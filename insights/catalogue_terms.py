"""
catalogue_terms.py — the app's own vocabulary, so the model and the app agree.

Player model §06: type comes from mapPBPType, level from inferSkillLevel. The
model previously used its own divergent mapping (a "Session" catch-all that
swallowed Events, court hire and "New to Pickleball"), so it learned and
rewarded a category the app never shows. These are line-for-line ports of
src/lib/sessionTerms.ts — if the app changes, change these.

Also here: a session's LEVEL BAND (§06: roster first, label as fallback),
PROGRAMME identity (§03) and PLAY FAMILY (F109).

── TITLES FIRST (F116, 2 Oct 2026) ─────────────────────────────────────────
PlayByPoint's `type` is each venue's own category, and its `skill_level` is
filled on about 1 session in 10. The title is where venues actually say what
a session is. Tested against all 207 distinct titles in the 2 Oct catalogue:

  · format: "AMBER INT - ADV Match Play 3.25+" (typed Open Play), every
    round robin, "Set Pairs", "Fixed Partners" and DUPR round robins were all
    read as SOCIAL. They are competitive. "Beginners Social Play" typed Clinic
    was read as LEARNING. The title now decides; the type only when the title
    says nothing.
  · level: "S5 -8pm ... DUPR 3.0 Level and above" read as 5-8 (a season code
    and a time); "Below 3.00", "Intro to", "Lvl 1 Fundamentals", "Come and
    Try", "Low Int", "Upper Beginner", "All Skill Levels" weren't read at all;
    "Daytime Session - Intermediate" with PlayByPoint's "1.0 / 3.5" was read
    as 1.0-3.5. Numbers that aren't ratings are stripped first, more level
    words are known, and a loose skill_level field (a floor or a ceiling
    only) is combined with the title instead of overriding it.
  · "competitive" is a FORMAT word, not a level: "BEGINNERS COMPETITIVE PLAY"
    was read as 2.0-8.0.
The level SCALE is unchanged (beginner 2.0-2.5 ... advanced 3.5+): the app's
filters and the model's level classes are built on it.
"""
from __future__ import annotations
import re
from typing import Optional

DUPR_MIN, DUPR_MAX = 2.0, 8.0
# A top of 6.0 or more is no real ceiling (PBP writes "3.25 / 7.0" for "3.25 and up").
OPEN_TOP = 6.0
NUM = r"(\d(?:\.\d{1,3})?)"


# ── text clean-up: what can never be a level ─────────────────────────────────
def _strip_negated(t: str) -> str:
    t = re.sub(r"\((?:\s*)(?:no|not|excluding|excl\.?|non)\b[^)]*\)", " ", t)
    t = re.sub(r"\b(?:no|not)\s+(?:for\s+)?(?:beginner|intermediate|advanced|elite|competitive)[a-z]*\b", " ", t)
    # aspiration, not level: "INTERMEDIATE SOCIALS - ROAD TO ADVANCED"
    t = re.sub(r"\broad\s+to\s+[a-z]+", " ", t)
    return t

def _strip_non_ratings(t: str) -> str:
    """Numbers that look like ratings but aren't: prices, clock times, ages,
    durations, weeks, season codes, programme levels."""
    t = re.sub(r"\$\s*\d+(?:\.\d+)?", " ", t)                                  # $12, $5
    t = re.sub(r"\b\d{1,2}[:.]\d{2}\s*(?:am|pm)\b|\b\d{1,2}\s*(?:am|pm)\b", " ", t)  # 6:30am, 8pm
    t = re.sub(r"\b\d+(?:\.\d+)?\s*h\b|\b\d+\s*-?\s*weeks?\b|\bweek\s+\d+\b", " ", t)  # 1.5h, 6-week, week 2
    t = re.sub(r"\bs\d+\b|\blvl\s*\d\b|\bpart\s+\d\b", " ", t)                  # S12, Lvl 1, Part 2
    t = re.sub(r"(?<![\d.])\d{2,}\s*\+|(?<![\d.])\d{2,}\s*years?\b", " ", t)    # 50+, 55years (not the 75 of 3.75+)
    return t

def _clean(title: Optional[str]) -> str:
    return _strip_non_ratings(_strip_negated((title or "").lower()))

def _is_rating_like(raw: str, mentions_dupr: bool) -> bool:
    try:
        n = float(raw)
    except ValueError:
        return False
    if n < DUPR_MIN or n > DUPR_MAX:
        return False
    return "." in raw or mentions_dupr


def _fmt(x: float) -> str:
    s = f"{x:.2f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


# ── level from numbers in the title ──────────────────────────────────────────
def _title_numbers(t: str, mentions: bool) -> Optional[tuple]:
    """(lo, hi) — hi DUPR_MAX = open top, lo DUPR_MIN = open bottom. t is _clean()ed."""
    m = re.search(NUM + r"\s*(?:-|–|—|to)\s*" + NUM, t)
    # "3-3.5": a whole number joined to a decimal is a rating too; "9-12" is not
    joined = m is not None and "." in m.group(1) + m.group(2)
    if m and _is_rating_like(m.group(1), mentions or joined) and _is_rating_like(m.group(2), mentions or joined):
        lo, hi = float(m.group(1)), float(m.group(2))
        if lo < hi:
            return lo, hi
    m = re.search(NUM + r"\s*\+|(?:>|over|above)\s*=?\s*" + NUM + r"|" + NUM + r"\s*(?:level\s+)?and\s+(?:above|up|over)", t)
    if m:
        x = next(g for g in m.groups() if g)
        if _is_rating_like(x, mentions):
            return float(x), DUPR_MAX
    m = re.search(r"(?:below|under|<|up\s+to)\s*=?\s*" + NUM, t)
    if m and _is_rating_like(m.group(1), True):
        return DUPR_MIN, float(m.group(1))
    m = re.search(r"dupr\s*:?\s*" + NUM, t)
    if m and _is_rating_like(m.group(1), True):
        return float(m.group(1)), DUPR_MAX
    return None


# ── level from words: phrases first, their words consumed ───────────────────
_PHRASES = (
    (r"\badv(?:anced)?\.?\s*-?\s*beg(?:inner)?s?\b|\bupper\s+beg(?:inner)?s?\b|\bbeg(?:inner)?\s*\+", (2.5, 3.0)),
    (r"\b(?:low(?:er)?|early|emerg(?:ing|ent))[\s–-]*(?:mid[\s–-]*)?int(?:ermediate)?\b", (2.5, 3.0)),
    (r"\b(?:upper|high)\s+int(?:ermediate|er)?\b", (3.0, 3.5)),
    (r"\b(?:lower|early|emerging)\s+adv(?:anced)?\b", (3.5, 4.0)),
    (r"\binbetweener\b", (2.0, 3.5)),
)
_WORDS = (
    (r"\bbeg(?:inner)?s?\b", (2.0, 2.5)),
    (r"\bint(?:ermediate|er)?\b", (2.5, 3.5)),
    (r"\badv(?:anced)?\b|\belite\b", (3.5, DUPR_MAX)),
)
# Words that mean "new to the game". A level only when the title names no
# other: "Intermediate Coaching • Learn & Play" is an intermediate session.
_NEW_PLAYER = (r"\blearn\s*(?:to|&)\s*play\b|\bnew\s+to\b|\bnew\s+players?\b|\bstarter\b|"
               r"\bintro(?:duction)?\s+to\b|\bfirst\s+time\b|\bcome\s+and\s+try\b|\bfundamentals?\b|\bfoundational\b")
_ALL_LEVELS = re.compile(r"\ball[\s-]*(?:skill\s+)?(?:levels?|lvls?)\b|\bopen\s+to\s+all\b")

def _title_bands(t: str) -> list:
    """Every level the title names, as bands. t is _clean()ed."""
    found = []
    for rx, b in _PHRASES:
        if re.search(rx, t):
            found.append(b)
            t = re.sub(rx, " ", t)          # consumed: its words don't count again
    for rx, b in _WORDS:
        if re.search(rx, t):
            found.append(b)
    if not found and re.search(_NEW_PLAYER, t):
        found.append((2.0, 2.5))
    return found

def _union(bands: list) -> Optional[tuple]:
    """Several level words ("Beginner/Intermediate") describe a RANGE. An open-top
    'advanced' beside a closed band ("High Intermediate / Advanced") reaches one
    band above it, not to 8.0."""
    if not bands:
        return None
    closed = [b for b in bands if b[1] < OPEN_TOP]
    lo = min(b[0] for b in bands)
    if len(bands) == 1 or not closed:
        return (lo, max(b[1] for b in bands))
    hi = max(b[1] for b in closed)
    if len(closed) < len(bands):
        hi = max(hi, 4.0)
    return lo, hi


def _field_band(skill_level: Optional[str]) -> Optional[tuple]:
    """PlayByPoint's skill_level as (lo|None, hi|None): '1.0 / 3.5' -> (None, 3.5),
    '3.5 / 7.0' -> (3.5, None), '> 3.0' -> (3.0, None), '2.0 / 3.0' -> (2.0, 3.0)."""
    s = (skill_level or "").strip().lower()
    m = re.match(r"^" + NUM + r"\s*(?:/|-|–|to)\s*" + NUM + r"$", s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (None if lo <= 1.0 else lo, None if hi >= OPEN_TOP else hi)
    m = re.match(r"^(?:>\s*=?\s*)" + NUM + r"$|^" + NUM + r"\s*\+$", s)
    if m:
        return (float(next(g for g in m.groups() if g)), None)
    m = re.match(r"^<\s*=?\s*" + NUM + r"$", s)
    if m:
        return (None, float(m.group(1)))
    return None


# ── inferSkillLevel (sessionTerms.ts) ────────────────────────────────────────
def infer_skill_level(title: Optional[str], skill_level: Optional[str]) -> str:
    """The display label: '2.5-3.5', '3.75+', '<3.0', or the field as PBP wrote it,
    or 'All levels'."""
    field = (skill_level or "").strip()
    has_field = bool(field) and field.lower() != "all levels"
    raw = (title or "").lower()
    mentions = "dupr" in raw
    t = _clean(title)

    fb = _field_band(field) if has_field else None
    # A CLOSED field ("2.0 / 3.0") is the venue being specific: it wins.
    if has_field and (fb is None or (fb[0] is not None and fb[1] is not None)):
        return field

    nums = _title_numbers(t, mentions)
    if nums:
        lo, hi = nums
        if hi >= OPEN_TOP:
            return f"{_fmt(lo)}+"
        if lo <= DUPR_MIN and not re.search(NUM + r"\s*(?:-|–|—|to)\s*" + NUM, t):
            return f"<{_fmt(hi)}"
        return f"{_fmt(lo)}-{_fmt(hi)}"

    words = _union(_title_bands(t))
    if words:
        lo, hi = words
        # A loose field ("> 3.0", "1.0 / 3.5") only cuts the title's band (F115).
        if fb:
            if fb[0] is not None:
                lo = max(lo, fb[0])
            if fb[1] is not None:
                hi = min(hi, fb[1])
            if hi <= lo:
                hi = lo + 0.5
        return f"{_fmt(lo)}+" if hi >= OPEN_TOP else f"{_fmt(lo)}-{_fmt(hi)}"

    if has_field:
        return field
    return "All levels"


# ── a label -> (lo, hi) band ─────────────────────────────────────────────────
def label_band(label: Optional[str]) -> Optional[tuple[float, float]]:
    """'2.5-3.5' -> (2.5,3.5) · '3.5+' / '> 3.5' -> (3.5, 8) · '1.0 / 3.5' -> (1.0,3.5)
    · '<3.0' -> (2.0, 3.0) · 'All levels' -> None."""
    if not label:
        return None
    s = label.strip().lower()
    m = re.match(r"^(\d(?:\.\d+)?)\s*(?:-|–|—|/|to)\s*(\d(?:\.\d+)?)$", s)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
        return (min(lo, hi), max(lo, hi))
    m = re.match(r"^(?:>\s*=?\s*)?(\d(?:\.\d+)?)\s*\+?$", s)
    if m and (s.endswith("+") or s.startswith(">")):
        return (float(m.group(1)), DUPR_MAX)
    m = re.match(r"^<\s*=?\s*(\d(?:\.\d+)?)$", s)
    if m:
        return (DUPR_MIN, float(m.group(1)))
    return None


def _floor_with_title(floor: float, t: str) -> Optional[tuple]:
    """An open-ended rating ("> 3.0", "3.5+", "3.25 / 7.0") says who may JOIN, not who
    the session is for. When the title names the level ("Early Intermediate"), that is
    the target, and the floor only cuts it from below. A higher floor written in the
    title ("DUPR >3.6") counts too. No level words: the floor and one point above, as
    before. F115: "Early Intermediate > 3.0" was read as 3.0-and-up = advanced."""
    for m in re.finditer(r"(?:>|over|above)\s*=?\s*(\d(?:\.\d+)?)"
                         r"|(\d(?:\.\d+)?)\s*(?:\+|(?:level\s+)?and\s+(?:above|up|over))", t):
        x = m.group(1) or m.group(2)
        if _is_rating_like(x, True) and DUPR_MIN <= float(x) <= OPEN_TOP:
            floor = max(floor, float(x))
    named = _union(_title_bands(t))
    if not named or named[1] >= OPEN_TOP:
        # "1.0 / 7.0" is open to everyone: it says nothing about level. A title
        # naming only "advanced" keeps the open top.
        if floor <= DUPR_MIN and not named:
            return None
        return (max(floor, named[0] if named else floor), DUPR_MAX)
    lo = max(floor, named[0])
    hi = named[1]                       # "INT - ADV ... 3.25+" -> 3.25-4.0, not 3.25-3.5
    return (lo, hi) if hi > lo else (lo, lo + 0.5)


# ── §06 session level band: the roster first, the label as fallback ──────────
# roster rating 1.0 is the unrated default and 2.0 is an unconfirmed spike —
# neither is evidence. Fewer than three real ratings -> fall back to the label.
def session_band(title: Optional[str], skill_level: Optional[str],
                 roster_ratings: Optional[list]) -> tuple[Optional[tuple[float, float]], str]:
    real = []
    for r in roster_ratings or []:
        try:
            x = float(r)
        except (TypeError, ValueError):
            continue
        if x in (1.0, 2.0) or x < DUPR_MIN or x > DUPR_MAX:
            continue
        real.append(x)
    if len(real) >= 3:
        return (min(real), max(real)), "roster"          # the range the room spans (§06)
    t = _clean(title)
    label = infer_skill_level(title, skill_level)
    if label == "All levels" and _ALL_LEVELS.search(t):
        return None, "none"                               # stated open to all: no band
    band = label_band(label)
    if band and band[1] >= OPEN_TOP:
        band = _floor_with_title(band[0], t)          # an ENTRY FLOOR, not the room's level (F115)
    return (band, "label") if band else (None, "none")


# ── §03 programme identity ───────────────────────────────────────────────────
# PBP re-slugs a programme when it is re-listed ("...-september", "-copy-a21bb8"),
# so the same Tuesday session would count as a new programme each month.
_MONTHS = ("january|february|march|april|may|june|july|august|september|october|"
           "november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")

def programme_key(slug: Optional[str]) -> Optional[str]:
    if not slug:
        return None
    s = slug.lower()
    s = re.sub(r"-copy(?:-[0-9a-f]+)?$", "", s)
    s = re.sub(rf"(?:^|-)(?:{_MONTHS})(?:-\d{{2,4}})?(?=-|$)", "", s)
    s = re.sub(r"(?:^|-)20\d\d(?=-|$)", "", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or None


# ── FORMAT: what is played, read from the title first ───────────────────────
# Narrowest first; the first hit wins. Coaching and learning sit above match
# play so "Drills & Match Play" stays a coached session.
_FORMATS = (
    ("round_robin", r"\bround[\s-]*robin\b"),
    ("ladder", r"\bladder\b"),
    ("league", r"\bleague\b|\btournament\b|\bchampionship\b|\bcompetition\b|\bthrowdown\b|\bteam\s+battle\b"),
    ("learn_to_play", r"\blearn\s*(?:to|&)\s*play\b|\bintro(?:duction)?\s+to\b|\bfirst\s+time\b|\bnew\s+to\s+pickle|"
                      r"\bcome\s+and\s+try\b|\bstarter\b|\bfundamentals?\b|\bplayer\s+assessment\b"),
    ("coaching", r"\bcoach(?:ing|ed)?\b|\blesson\b"),
    ("clinic", r"\bclinic\b|\bdrill[sz]?\b|\bskill[sz]\b|\bboot\s*camp\b|\bdevelopment\b|\baccelerate\b|"
               r"\bacademy\b|\bprogram(?:me)?\b|\bpathway\b|\bjuniors?\b|\bkids?\b|\byouth\b"),
    ("match_play", r"\bmatch\s*play\b|\bset\s+pairs\b|\bfixed\s+partners?\b|\bpower\s+play\b|\bcompetitive\b"),
    ("open_play", r"\bopen\s+play\b"),
    ("social", r"\bsocials?\b"),
)
_FAMILY = {"round_robin": "competitive", "ladder": "competitive", "league": "competitive",
           "match_play": "competitive", "learn_to_play": "learning", "coaching": "learning",
           "clinic": "learning", "open_play": "social", "social": "social"}
_DUPR_RATED = re.compile(r"\bdupr\s+(?:rated|recorded|session|nights?|doubles|singles|round\s+robin|league|points)\b|"
                         r"\b(?:rated|recorded)\s+session\b|\bsuper\s+dupr\b|\(dupr\)|\bdupr\s+rated\b")

def title_format(text: Optional[str]) -> Optional[str]:
    t = _strip_negated((text or "").lower())
    for name, rx in _FORMATS:
        if re.search(rx, t):
            return name
    return None

def dupr_rated(text: Optional[str]) -> bool:
    return bool(_DUPR_RATED.search((text or "").lower()))


# ── mapPBPType (sessionTerms.ts) — the app's display category ───────────────
_APP_CATEGORY = {"round_robin": "Round Robin", "ladder": "Ladder", "league": "League",
                 "learn_to_play": "Clinic", "coaching": "Coaching", "clinic": "Clinic",
                 "match_play": "Match Play", "open_play": "Open Play", "social": "Social Event"}

def _map_type_field(raw: Optional[str]) -> str:
    t = (raw or "").lower()
    if "round robin" in t or "roundrobin" in t:            return "Round Robin"
    if "ladder" in t:                                       return "Ladder"
    if "open play" in t or "openplay" in t:                 return "Open Play"
    if "coaching" in t or "lesson" in t:                    return "Coaching"
    if ("clinic" in t or "skill" in t
            or "academy" in t or "junior" in t):            return "Clinic"
    if ("league" in t or "tournament" in t
            or "championship" in t or "competition" in t):  return "League"
    if "social" in t or "socials" in t or "event" in t:     return "Social Event"
    return "Open Play"

def map_pbp_type(raw: Optional[str], title: Optional[str] = None) -> str:
    """The title decides when it names a format; the venue's category otherwise."""
    f = title_format(title)
    return _APP_CATEGORY[f] if f else _map_type_field(raw)


# ── PLAY FAMILY: what kind of evening it is (F109, F116) ─────────────────────
# PBP's type field is labelled per venue: the same social session is "Open Play" at
# one venue and "Social Play"/"Event" at another, and a DUPR tournament is typed
# "Event". For PREFERENCE, the distinction a player actually makes is social vs
# competitive vs learning. The TITLE decides when it names a format; the type is
# read only when it doesn't. Results sent to DUPR make a social format competitive.
_LEARNING = ("clinic", "lesson", "coaching", "learn to play", "academy", "drill", "junior", "first time")
_COMPETITIVE = ("tournament", "league", "competition", "championship", "ladder", "team battle", "round robin")

def play_family(raw_type: Optional[str], text: Optional[str] = None) -> Optional[str]:
    both = f"{text or ''} {raw_type or ''}".lower().replace("-", " ")
    if not both.strip() or "court booking" in both:
        return None
    f = title_format(text)
    if f:
        fam = _FAMILY[f]
    else:
        ty = (raw_type or "").lower().replace("-", " ")
        if any(w in ty for w in _LEARNING):
            fam = "learning"
        elif any(w in ty for w in _COMPETITIVE):
            fam = "competitive"
        else:
            fam = "social"
    if fam == "social" and (dupr_rated(text) or "dupr session" in (raw_type or "").lower()):
        fam = "competitive"
    return fam


# ── everything a title says, for reports (venue insights) ───────────────────
_AUDIENCE = (
    ("women", r"\bwomen'?s\b|\bladies\b|\bgirls\b"),
    ("men", r"\bmen'?s\b"),
    ("mixed", r"\bmixed\b"),
    ("seniors", r"\b(?:4[05]|5[05]|60)\s*\+|\b5[05]\s*years?\b|\bseniors?\b"),
    ("juniors", r"\bkids?\b|\bjuniors?\b|\byouth\b|\bschool\s+holiday\b"),
    ("school", r"\bschool\s+clinic\b"),
    ("lgbtq", r"\bqueer\b"),
    ("parents", r"\bmumm?a\b|\bmums?\b"),
)

def session_terms(title: Optional[str], raw_type: Optional[str] = None,
                  skill_level: Optional[str] = None) -> dict:
    low = (title or "").lower()
    band, _ = session_band(title, skill_level, [])
    label = infer_skill_level(title, skill_level)
    f = title_format(title)
    return {
        "label": label,
        "band": band,
        "all_levels": label == "All levels" and bool(_ALL_LEVELS.search(_clean(title))),
        "format": f or {"Round Robin": "round_robin", "Ladder": "ladder", "League": "league",
                        "Coaching": "coaching", "Clinic": "clinic", "Open Play": "open_play",
                        "Social Event": "social"}[_map_type_field(raw_type)],
        "format_from": "title" if f else "type",
        "family": play_family(raw_type, title),
        "category": map_pbp_type(raw_type, title),
        "dupr_rated": dupr_rated(title),
        "singles": True if re.search(r"\bsingles?\b", low) else (False if re.search(r"\bdoubles\b", low) else None),
        "audience": [n for n, rx in _AUDIENCE if re.search(rx, low)],
    }
