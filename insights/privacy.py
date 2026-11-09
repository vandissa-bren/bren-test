"""
privacy.py -- the 5-or-more rule for everything the build publishes.

Players are only ever shown as counts of 5 or more. Every builder tries to hold
small groups back itself, but a few counts slipped through (a group's level
mix, a weekly session's players, the "partners" histogram, the period before's
players per venue). So the published build passes through one last filter here:

  COUNTS  every field that counts players (or things that stand for players:
          rating changes, players in a level bin). A value of 1-4 becomes None.
          0 stays 0: it means "none", and a zero next to a hidden number gives
          nothing away that the hidden number didn't.
  RATES   every percentage whose numerator is players. If the players behind
          it (base x rate) come to 1-4, or the base itself is under 5, the rate
          becomes None: "1 of 20 came back" is one person.

  PARTS   (partitions()) every total published next to its breakdown or a
          subset of it: subtracting the visible parts must never leave 1-4
          ("40 players, 36 of them new" says 4 returned). The smallest visible
          part is hidden until no remainder is 1-4.

Paths: dot-separated keys; "*" is every key of a dict; "[]" after a key is
every item of a list; "[2]" is item 2. A path that isn't in the build is
skipped, so the list can name optional parts.

Counts of BOOKINGS (sessions booked, cancellations, waitlist places) are not
players and are left alone; neither are sessions, courts or money.
"""
from __future__ import annotations

import re

MIN_GROUP = 5

COUNTS = [
    # city and player summaries
    "city.players",
    "players.total", "players.freq.*", "players.venues.*", "players.level.*",
    # venues
    "venues[].players", "venues[].homeHere", "venues[].rated",
    # levels
    "levelFit[].players", "levelFit[].verified", "levelClass[].players", "levelClass[].verified",
    "levelMix[].players", "levelSupply[].players.*",
    # coaches, travel, market
    "coaches[].players", "coaches[].regulars", "coaches[].after[0]",
    "travel[].players",
    "market.*.rivals[].players", "market.*.travellers.players", "market.*.travellers.travellers",
    "market.*.travellers.classes.*", "market.*.travellers.homes[][1]", "market.*.travellers.draws[][1]",
    "market.*.away.cells[][3]",
    # new and returning
    "newret.city.players", "newret.city.new", "newret.city.returning",
    "newret.weeks[][1]", "newret.weeks[][2]", "newret.venues.*[]", "newret.firsts.*",
    "newret.back[0]", "newret.back[1]", "newret.backBy.*[0]", "newret.backBy.*[1]",
    # weekly sessions and sessions that share regulars (the player-level
    # network itself is no longer published)
    "net.series[].players", "net.series[].regulars",
    "net.shared.city[].shared", "net.shared.venues.*[].shared",
    # venues that share players
    "shares.rows[][2]", "shares.base.*", "shares.players",
    # scorecard: new players, overall and per week
    "scorecard.city.newHere", "scorecard.city.series.newHere[]",
    "scorecard.venues.*.newHere", "scorecard.venues.*.series.newHere[]",
    # PM level
    "pm.players", "pm.kinds.*", "pm.mix.*[]", "pm.tests.players", "pm.tests.high.n", "pm.tests.low.n",
    "pm.selfCal.players", "pm.regulars[0]", "pm.venues.*[0]",
    # ratings
    "ratings.city.dupr", "ratings.city.players", "ratings.city.measured", "ratings.city.changes",
    "ratings.city.up", "ratings.city.down",
    "ratings.weeks[][1]", "ratings.weeks[][2]", "ratings.weeks[][3]", "ratings.weeks[][4]",
    "ratings.after.*[0]", "ratings.room[][1]", "ratings.progression[0]",
    "ratings.venues.*.*[0]", "ratings.programmes[].players", "ratings.programmes[].after[0]",
    # the period before
    "prev.players", "prev.venuePlayers.*",
]

# (rate path, base path), both relative to the same parent object. The rate is
# a percentage of the base (players).
RATES = [
    ("venues[]", "loyal", "players"),
    ("venues[]", "fromAfar", "fromAfarN"),
    ("venues[]", "cameBack", "cameBackN"),
    ("ratings.after.*", 2, 0), ("ratings.after.*", 3, 0),
    ("ratings.venues.*.all", 2, 0), ("ratings.venues.*.all", 3, 0),
    ("ratings.progression", 2, 0), ("ratings.progression", 3, 0),
    ("pm.regulars", 1, 0),
    ("pm.venues.*", 4, 0),
]
# bases the builders add only for this check; removed before publishing
INTERNAL = [("venues[]", "fromAfarN"), ("venues[]", "cameBackN")]

_TOK = re.compile(r"([^.\[\]]+)|\[(\d*)\]")


def _parse(path: str):
    """'net.series[].also[][2]' -> ['net', 'series', [], 'also', [], 2]"""
    out = []
    for name, idx in _TOK.findall(path):
        if name:
            out.append(name)
        else:
            out.append(int(idx) if idx != "" else [])
    return out


def _parents(obj, segs):
    """Every (container, key) the path ends on."""
    if not segs:
        return
    head, rest = segs[0], segs[1:]
    if head == []:
        items = list(enumerate(obj)) if isinstance(obj, list) else []
    elif head == "*":
        items = list(obj.items()) if isinstance(obj, dict) else []
    elif isinstance(head, int):
        items = [(head, obj[head])] if isinstance(obj, list) and -len(obj) <= head < len(obj) else []
    else:
        items = [(head, obj[head])] if isinstance(obj, dict) and head in obj else []
    for k, v in items:
        if not rest:
            yield obj, k
        else:
            yield from _parents(v, rest)


def _objects(obj, segs):
    """Every object the path leads to (for RATES' parent paths)."""
    if not segs:
        yield obj
        return
    for parent, k in _parents(obj, segs):
        yield parent[k]


def small(n) -> bool:
    return isinstance(n, (int, float)) and not isinstance(n, bool) and n == n and 1 <= n < MIN_GROUP


def _get(o, k):
    if isinstance(o, dict):
        return o.get(k)
    if isinstance(o, list) and isinstance(k, int) and -len(o) <= k < len(o):
        return o[k]
    return None


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x


def _ref_ok(ref) -> bool:
    o, k = ref
    if isinstance(o, dict):
        return k in o
    return isinstance(o, list) and isinstance(k, int) and -len(o) <= k < len(o)


def partitions(d: dict):
    """Every (total, parts) pair the build publishes where the parts are a
    breakdown, or a subset, of the total, as (container, key) references.
    Subtracting the visible parts from the total must never leave 1-4: that
    remainder is a group of players too. Used by enforce() and by the tests."""
    G = []

    def add(total, parts):
        parts = [p for p in parts if _ref_ok(p)]
        if _ref_ok(total) and parts:
            G.append((total, parts))

    pl = d.get("players")
    if isinstance(pl, dict):
        for sub in ("freq", "venues", "level"):
            if isinstance(pl.get(sub), dict):
                add((pl, "total"), [(pl[sub], k) for k in pl[sub]])
    nr = d.get("newret")
    if isinstance(nr, dict):
        c = nr.get("city")
        if isinstance(c, dict):
            add((c, "players"), [(c, "new"), (c, "returning")])
        for w in nr.get("weeks") or []:
            if isinstance(w, list) and len(w) >= 3:
                add((w, 1), [(w, 2)])
        for v in (nr.get("venues") or {}).values():
            if isinstance(v, list) and len(v) >= 3:
                add((v, 0), [(v, 1)])
                add((v, 1), [(v, 2)])
        for b in [nr.get("back")] + list((nr.get("backBy") or {}).values()):
            if isinstance(b, list) and len(b) >= 2:
                add((b, 0), [(b, 1)])
    sc = d.get("scorecard")
    if isinstance(sc, dict):
        cards = [sc.get("city")] + list((sc.get("venues") or {}).values())
        for card in cards:
            if isinstance(card, dict) and isinstance((card.get("series") or {}).get("newHere"), list):
                wk = card["series"]["newHere"]
                add((card, "newHere"), [(wk, i) for i in range(len(wk))])
    base = (d.get("shares") or {}).get("base") or {}
    for v in d.get("venues") or []:
        if not isinstance(v, dict):
            continue
        for sub in ("homeHere", "rated"):
            add((v, "players"), [(v, sub)])
        if str(v.get("id")) in base:
            add((v, "players"), [(base, str(v.get("id")))])
    for fid, mk in (d.get("market") or {}).items():
        tr = (mk or {}).get("travellers")
        if isinstance(tr, dict):
            add((tr, "players"), [(tr, "travellers")])
            if isinstance(tr.get("classes"), dict):
                add((tr, "travellers"), [(tr["classes"], k) for k in tr["classes"]])
    for s in ((d.get("net") or {}).get("series") or []):
        if isinstance(s, dict):
            add((s, "players"), [(s, "regulars")])
    for c in d.get("coaches") or []:
        if isinstance(c, dict):
            add((c, "players"), [(c, "regulars")])
    for r in (d.get("levelFit") or []) + (d.get("levelClass") or []):
        if isinstance(r, dict):
            add((r, "players"), [(r, "verified")])
    return G


def remainder(total_ref, parts) -> float | None:
    t = _get(*total_ref)
    if not _num(t):
        return None
    return t - sum(_get(*p) for p in parts if _num(_get(*p)))


def _complements(d: dict) -> None:
    """Hide parts until no total minus its visible parts comes to 1-4: the
    smallest visible non-zero part goes each time (it gives the least away)."""
    for _ in range(20):
        changed = False
        for total, parts in partitions(d):
            rem = remainder(total, parts)
            if rem is None or not small(rem):
                continue
            vis = [p for p in parts if _num(_get(*p)) and _get(*p) > 0]
            if not vis:
                continue
            o, k = min(vis, key=lambda p: _get(*p))
            o[k] = None
            changed = True
        if not changed:
            return


def _shares_bases(d: dict) -> None:
    """A venue whose 'players with 2+ sessions' base is hidden: its rows'
    share and compared-with-chance would give the base back (shared / share),
    so they go too."""
    sh = d.get("shares") or {}
    base = sh.get("base") or {}
    hidden = {int(k) for k, n in base.items() if n is None}
    for r in sh.get("rows") or []:
        if isinstance(r, list) and len(r) >= 5 and r[0] in hidden:
            r[3] = r[4] = None


def enforce(d: dict) -> dict:
    """Apply RATES then COUNTS, then hide whatever subtraction would give away
    (partitions), to the build in place; returns it."""
    for parent_path, rate_k, base_k in RATES:
        for o in _objects(d, _parse(parent_path)):
            rate, base = _get(o, rate_k), _get(o, base_k)
            if not isinstance(rate, (int, float)) or isinstance(rate, bool):
                continue
            if not isinstance(base, (int, float)) or small(base) or base == 0:
                o[rate_k] = None
                continue
            n = round(base * rate / 100)
            # the players behind it, and the players not behind it: "95% of 20
            # came back" says 1 didn't
            if small(n) or small(round(base) - n):
                o[rate_k] = None
    for path in COUNTS:
        for parent, k in list(_parents(d, _parse(path))):
            if small(parent[k]):
                parent[k] = None
    _complements(d)
    _shares_bases(d)
    for parent_path, k in INTERNAL:
        for o in _objects(d, _parse(parent_path)):
            if isinstance(o, dict):
                o.pop(k, None)
    return d
