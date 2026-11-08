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
    "venues[].players", "venues[].homeHere", "venues[].netPlayers", "venues[].rated",
    "venues[].groupMix[][1]",
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
    # co-play network
    "net.city.players", "net.city.giantN", "net.city.inGroups", "net.city.degHist[][1]",
    "net.city.multiVenuePairs", "net.city.steps.*",     # pairs of players
    "net.groups[].size", "net.groups[].lvl.*",
    "net.series[].players", "net.series[].regulars", "net.series[].also[][2]",
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
    ("venues[]", "repeat", "players"),
    ("venues[]", "loyal", "players"),
    ("venues[]", "oneTimers", "netPlayers"),
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


def enforce(d: dict) -> dict:
    """Apply RATES then COUNTS to the build in place; returns it."""
    for parent_path, rate_k, base_k in RATES:
        for o in _objects(d, _parse(parent_path)):
            rate, base = _get(o, rate_k), _get(o, base_k)
            if not isinstance(rate, (int, float)) or isinstance(rate, bool):
                continue
            if not isinstance(base, (int, float)) or small(base) or base == 0:
                o[rate_k] = None
                continue
            if small(round(base * rate / 100)):
                o[rate_k] = None
    for path in COUNTS:
        for parent, k in list(_parents(d, _parse(path))):
            if small(parent[k]):
                parent[k] = None
    for parent_path, k in INTERNAL:
        for o in _objects(d, _parse(parent_path)):
            if isinstance(o, dict):
                o.pop(k, None)
    return d
