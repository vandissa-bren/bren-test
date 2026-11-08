"""
series.py -- one rule for "the same weekly session", used everywhere the
build needs it (network.py's session series, Quality's fixtures and titles
that vary, and the metrics built on series: another court, same faces each
week, sessions that share regulars).

Two sessions belong to the same weekly series when they run at the same venue
on the same weekday, in different weeks no more than MAX_GAP_WEEKS apart, and
one of these holds:

  * the same programme: their PlayByPoint slugs agree once re-listing noise
    ("-september", "-copy-a21bb8", years) is stripped (catalogue_terms
    .programme_key), and they start within SLUG_MINUTES of each other. A slug
    can cover several sessions a week, so it never joins two sessions on
    different weekdays or at very different times;
  * both are titled, start within TIME_MINUTES of each other, and their titles
    share at least TITLE_SIM of their words (Jaccard on word sets, case and
    punctuation ignored; season markers, weekday names and bare numbers such
    as rating bands stripped), against any of the series' last 3 titles. A retitle keeps the series;
    a new season under a new slug with the same title and time keeps it too;
  * either is untitled, and they start at exactly the same time with the same
    session type and level. Untitled sessions never merge on time alone.

What a session states about who it's for must agree on both sides, in every
branch: its level (the title's level word, else the stated level; "all
levels" and none count as none), any gender words (men's, women's, ladies,
mixed) and singles or doubles. So "Advanced Social Play" and "Social Play",
or "Men's Singles" and "Women's Singles", are never one series, even under
one slug. The cost: a session retitled from "Intermediate Open Play" to plain
"Open Play" with no stated level starts a new series.

No drift: time is compared with the series' median start, and the title with
its core words (words in at least half its titled runs) or its first title,
never only the latest run, so a series can't creep from 17:00 to 21:00 or
morph into a different session a word at a time. Under one slug, two titled
sessions also need some title overlap (SLUG_SIM), and a slug match beyond
TIME_MINUTES (up to SLUG_MINUTES) needs the titles to agree too.

Competition sessions (titles with "vs", "division", "final", "round 3",
"play-offs", or a league / tournament / competition type) get no series: a
team fixture with rotating opponents isn't a weekly session.

Matching is one to one within a week, best score first (slug, then title,
then time), with a fixed tie-break, so two sessions on the same night are
always two series and the result doesn't depend on row order. Rating bands in
titles ("3.0-3.5", ">4") nudge ties toward the same band.

A series id is "<fid>-<dow>-<n>", numbered in order of each series' first
session. It is stable within a build, not across builds whose window starts
on a different day.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from catalogue_terms import programme_key

MEL = "Australia/Melbourne"
MAX_GAP_WEEKS = 3        # a series survives up to 3 weeks off (holidays); 4+ starts a new one
TIME_MINUTES = 30        # titled sessions: start within ±30 min
SLUG_MINUTES = 60        # same programme: start within ±60 min
TITLE_SIM = 0.6          # share of title words in common (Jaccard)
SLUG_SIM = 0.34          # same programme, both titled: at least this much in common

_MONTHS = ("january|february|march|april|may|june|july|august|september|october|november|december|"
           "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")
_STOP = {"the", "and", "a", "at", "of", "with", "for", "&", "pm", "am", "s",
         "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
         "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun", "night", "nights"}


def title_words(t) -> frozenset:
    """A title's words, without season markers, dates, week counts or punctuation."""
    if not isinstance(t, str) or not t.strip():
        return frozenset()
    x = t.lower()
    x = re.sub(r"copy[-\s]*[0-9a-f]{6}", " ", x)
    x = re.sub(r"\b(season|term|series|round|week|wk)\s*#?\d+\b", " ", x)
    x = re.sub(r"\b[st]\d{1,3}\b|\bq[1-4]\b", " ", x)
    x = re.sub(r"\b\d+\s*(weeks?|wks?)\b", " ", x)
    x = re.sub(r"\b(19|20)\d\d\b", " ", x)
    x = re.sub(rf"\b({_MONTHS})\b", " ", x)
    x = re.sub(r"\b\d{1,2}(st|nd|rd|th)\b", " ", x)
    x = re.sub(r"[^a-z0-9+.]+", " ", x)
    # weekday names repeat what the series already fixes; bare numbers are
    # rating bands, times and dates, which venues edit freely
    return frozenset(w for w in x.split() if w not in _STOP and not re.fullmatch(r"[0-9.+]+", w))


def similarity(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


_LEVEL_WORDS = [("beginner", r"beginners?|novice|intro(?:duction)?|learn(?:\s+to\s+play)?|foundation(?:al)?|starter"),
                ("intermediate", r"intermediate|improvers?"),
                ("advanced", r"advanced|elite|expert|high\s+level")]
_GENDER = [("men", r"men'?s|mens|male"), ("women", r"women'?s|womens|ladies|female"), ("mixed", r"mixed")]
_FORMAT = [("singles", r"singles"), ("doubles", r"doubles")]
_COMP_TITLE = re.compile(r"\bvs\.?\b|\bdivision\b|\b(?:grand\s+|semi[-\s]?)?finals?\b|\bround\s*\d+\b|\bplay[-\s]?offs?\b", re.I)
_COMP_TYPE = re.compile(r"league|tournament|competition|championship", re.I)
_BANDS = re.compile(r"[<>]\s*\d(?:\.\d{1,2})?|\d\.\d{1,2}\+?")


def _tags(text, table):
    if not isinstance(text, str):
        return frozenset()
    t = text.lower()
    return frozenset(name for name, pat in table if re.search(rf"\b(?:{pat})\b", t))


def _level_class(v):
    """beginner / intermediate / advanced from a level field or a title, else None."""
    tags = _tags(v, _LEVEL_WORDS)
    return min(tags) if len(tags) == 1 else ("|".join(sorted(tags)) if tags else None)


def is_competition(title, kind) -> bool:
    return bool((isinstance(title, str) and _COMP_TITLE.search(title))
                or (isinstance(kind, str) and _COMP_TYPE.search(kind)))


@dataclass
class _Series:
    sid: str
    week: int
    minutes: list                    # every run's start minute
    first_words: frozenset           # the first titled run's words
    titled: list                     # every titled run's words
    slug: Optional[str]
    kind: Optional[str]
    level: Optional[str]
    tags: frozenset
    bands: frozenset
    members: list = field(default_factory=list)

    @property
    def minute(self) -> float:
        return float(np.median(self.minutes))

    @property
    def mode_minute(self) -> int:
        return max(set(self.minutes), key=lambda m: (self.minutes.count(m), -m))

    @property
    def core(self) -> frozenset:
        if not self.titled:
            return frozenset()
        n = len(self.titled)
        cnt = {}
        for w in self.titled:
            for x in w:
                cnt[x] = cnt.get(x, 0) + 1
        core = frozenset(x for x, c in cnt.items() if 2 * c >= n)
        return core or self.first_words


def _score(s: _Series, r: dict) -> Optional[float]:
    if r["level"] != s.level or r["tags"] != s.tags:
        return None                                  # different level, gender or singles/doubles
    dm = abs(r["minute"] - s.minute)
    words = r["words"]
    sim = max(similarity(words, s.core), similarity(words, s.first_words)) if words and s.titled else None
    band = 0.0
    if r["bands"] and s.bands:
        band = 0.2 if r["bands"] == s.bands else -0.2
    if r["slug"] and s.slug and r["slug"] == s.slug:
        agree = sim is not None and sim >= TITLE_SIM
        if sim is not None and sim < SLUG_SIM:
            return None                              # one programme, different sessions (a rotating night)
        if dm <= TIME_MINUTES or (agree and dm <= SLUG_MINUTES):
            return 3.0 + (sim or 0) - dm / 120 + band
    if sim is not None:
        if dm <= TIME_MINUTES and sim >= TITLE_SIM:
            return 2.0 + sim - dm / 120 + band
        return None
    # one side untitled: its usual start exactly, same type and level
    if r["minute"] == s.mode_minute and r["kind"] is not None and r["kind"] == s.kind:
        return 1.0
    return None


def _clean(v):
    return v.strip() if isinstance(v, str) and v.strip() else None


_NO_LEVEL = {"no level stated", "all levels", "all skill levels", "mixed", "open", "unknown", "unrated"}


def _level(v):
    """Kept for callers: a stated level, lower-cased, or None."""
    v = _clean(v)
    return None if v is None or v.lower() in _NO_LEVEL else v.lower()


_CACHE: dict = {}


def assign_series(df: pd.DataFrame, *, fid="fid", starts_at="starts_at", title="title",
                  slug="program_slug", kind="session_type", level="lvl") -> pd.Series:
    """A series id for every row of df (index kept). Rows without a venue or a
    start time, and competition sessions, get None. Columns other than fid and
    starts_at are optional. The result is cached on the columns' contents, so
    the build's several callers in one window pay for it once."""
    if df.empty:
        return pd.Series([None] * len(df), index=df.index, dtype=object)
    cols = [c for c in (fid, starts_at, title, slug, kind, level) if c in df]
    try:
        key = (tuple(cols), len(df), int(pd.util.hash_pandas_object(df[cols].astype(str), index=True).sum()))
    except Exception:
        key = None
    if key is not None and key in _CACHE:
        return _CACHE[key].copy()
    out = _assign(df, fid, starts_at, title, slug, kind, level)
    if key is not None:
        if len(_CACHE) >= 4:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = out.copy()
    return out


def _assign(df, fid, starts_at, title, slug, kind, level) -> pd.Series:
    out = pd.Series([None] * len(df), index=df.index, dtype=object)
    st = pd.to_datetime(df[starts_at], utc=True, errors="coerce").dt.tz_convert(MEL)
    col = lambda c: (df[c].tolist() if c in df else [None] * len(df))   # plain Python values, never NaN-typed
    fids, titles, slugs, kinds, levels = col(fid), col(title), col(slug), col(kind), col(level)
    recs = []
    for pos, (idx, t) in enumerate(zip(df.index, st)):
        f = fids[pos]
        if f is None or (isinstance(f, float) and np.isnan(f)) or pd.isna(t):
            continue
        ti, ki = _clean(titles[pos]), _clean(kinds[pos])
        if is_competition(ti, ki):
            continue
        lv = _level_class(ti) or _level_class(_level(levels[pos])) or _level(levels[pos])
        day = t.date()
        recs.append({
            "idx": idx, "pos": pos, "fid": f, "t": t, "dow": t.dayofweek,
            "week": (day.toordinal() - day.weekday()) // 7, "minute": t.hour * 60 + t.minute,
            "title": ti, "words": title_words(ti), "slug": programme_key(_clean(slugs[pos])),
            "kind": ki, "level": lv, "tags": _tags(ti, _GENDER) | _tags(ti, _FORMAT),
            "bands": frozenset(b.replace(" ", "") for b in _BANDS.findall(ti or "")),
        })
    groups: dict = {}
    for r in recs:
        groups.setdefault((r["fid"], r["dow"]), []).append(r)
    for (f, dow) in sorted(groups, key=lambda k: (str(k[0]), k[1])):
        g = sorted(groups[(f, dow)], key=lambda r: (r["t"], r["title"] or "", r["slug"] or "", r["pos"]))
        active: list[_Series] = []
        n = 0
        weeks: dict = {}
        for r in g:
            weeks.setdefault(r["week"], []).append(r)
        for wk in sorted(weeks):
            gw = weeks[wk]
            live = [s for s in active if 0 < wk - s.week <= MAX_GAP_WEEKS]
            pairs = []
            for i, r in enumerate(gw):
                for k, s in enumerate(live):
                    sc = _score(s, r)
                    if sc is not None:
                        pairs.append((-sc, abs(r["minute"] - s.minute), i, k))
            pairs.sort()
            taken_rows, taken_series = set(), set()
            for _, _, i, k in pairs:
                if i in taken_rows or k in taken_series:
                    continue
                taken_rows.add(i); taken_series.add(k)
                s, r = live[k], gw[i]
                s.week = wk
                s.minutes.append(r["minute"])
                s.members.append(r["idx"])
                if r["words"]:
                    s.titled.append(r["words"])
                    if not s.first_words:
                        s.first_words = r["words"]
                if r["slug"]:
                    s.slug = r["slug"]
                if r["kind"]:
                    s.kind = r["kind"]
                if r["bands"]:
                    s.bands = r["bands"]
                out.at[r["idx"]] = s.sid
            for i, r in sorted(enumerate(gw), key=lambda p: (p[1]["minute"], p[1]["title"] or "", p[1]["pos"])):
                if i in taken_rows:
                    continue
                n += 1
                s = _Series(f"{int(float(f))}-{int(dow)}-{n}", wk, [r["minute"]], r["words"],
                            [r["words"]] if r["words"] else [], r["slug"], r["kind"], r["level"],
                            r["tags"], r["bands"], [r["idx"]])
                active.append(s)
                out.at[r["idx"]] = s.sid
            active = [s for s in active if wk - s.week <= MAX_GAP_WEEKS]
    return out


def series_label(df: pd.DataFrame, sid_col="series", title="title", label="label",
                 starts_at="starts_at") -> pd.DataFrame:
    """One row per series: its latest title (else label), weekday, most common
    start time, and number of runs."""
    st = pd.to_datetime(df[starts_at], utc=True, errors="coerce").dt.tz_convert(MEL)
    x = pd.DataFrame({"sid": df[sid_col], "t": st,
                      "title": df[title] if title in df else None,
                      "label": df[label] if label in df else None}).dropna(subset=["sid"])
    x["hm"] = x["t"].dt.strftime("%H:%M")
    x["dow"] = x["t"].dt.dayofweek
    rows = []
    for sid, g in x.sort_values("t").groupby("sid", sort=False):
        titled = g["title"].map(_clean).dropna()
        labelled = g["label"].map(_clean).dropna()
        name = titled.iloc[-1] if len(titled) else (labelled.iloc[-1] if len(labelled) else "Untitled session")
        rows.append({"sid": sid, "title": name, "dow": int(g["dow"].iloc[-1]),
                     "hm": g["hm"].mode().iloc[0], "runs": int(len(g)),
                     "titles": int(titled.nunique())})
    return pd.DataFrame(rows, columns=["sid", "title", "dow", "hm", "runs", "titles"])
