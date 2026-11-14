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
          part is hidden until no remainder is 1-4. Three kinds:
            subset  the parts are some of the total (quiet regulars of all
                    regulars): total minus visible parts must not be 1-4;
            exact   the parts add up to the total (quiet = booked elsewhere +
                    not seen): as subset, and also no single hidden number may
                    be worked back out (one hidden among total and parts is
                    total minus the rest, or the sum of the rest);
            city    a city figure next to the venue figures it covers (players
                    can be in more than one venue, so it's not exact): city
                    minus the visible venues must not be 1-4, and when it is,
                    the CITY figure is hidden, not a venue's.
          Repeated until nothing changes.

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
    # PM level v2 (pm_v2.py holds these at source and checks subtraction; listed as a second guard)
    "pm.v2.players", "pm.v2.kinds.*", "pm.v2.sure", "pm.v2.regulars", "pm.v2.regularsSure", "pm.v2.coaches",
    "pm.v2.mix.*[]", "pm.v2.vsDupr.players", "pm.v2.vsDupr.above", "pm.v2.vsDupr.below", "pm.v2.vsDupr.within",
    "pm.v2.rooms[].players", "pm.v2.groups[].pairs",
    "pm.v2.tests.forward.players", "pm.v2.tests.forward.movers", "pm.v2.tests.forward.called",
    "pm.v2.tests.forward.above.players", "pm.v2.tests.forward.below.players", "pm.v2.tests.forward.near.players",
    "pm.v2.tests.hidden.players", "pm.v2.tests.hidden.fresh.players",
    # ratings
    "ratings.city.dupr", "ratings.city.players", "ratings.city.measured", "ratings.city.changes",
    "ratings.city.up", "ratings.city.down",
    "ratings.weeks[][1]", "ratings.weeks[][2]", "ratings.weeks[][3]", "ratings.weeks[][4]",
    "ratings.after.*[0]", "ratings.room[][1]", "ratings.progression[0]",
    "ratings.venues.*.*[0]", "ratings.programmes[].players", "ratings.programmes[].after[0]",
    # the period before
    "prev.players", "prev.venuePlayers.*",
    # the seven figures (figures.py)
    "figures.quiet.city.regulars", "figures.quiet.city.quiet",
    "figures.quiet.venues.*.regulars", "figures.quiet.venues.*.quiet",
    "figures.quiet.venues.*.elsewhere", "figures.quiet.venues.*.notSeen", "figures.quiet.venues.*.to[][1]",
    "figures.retention.city[].n", "figures.retention.venues.*[].n",
    "figures.sameFaces.venues.*[].players",
    "figures.where.venues.*[].players",
    "figures.leaning.venues.*.players", "figures.leaning.venues.*.top",
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
    # new players coming back: wK is a % of the cohort's n
    *[(p, f"w{k}", "n") for p in ("figures.retention.city[]", "figures.retention.venues.*[]") for k in (1, 2, 4, 8)],
    # same faces: % of a run's players (its median size) who were at the run before
    ("figures.sameFaces.venues.*[]", "pct", "players"),
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
    remainder is a group of players too. Used by the tests; enforce() uses
    partitions_ex(), which also says what kind each pair is."""
    return [(t, p) for t, p, _ in partitions_ex(d)]


def partitions_ex(d: dict):
    """partitions() with each pair's kind: 'subset', 'exact' or 'city'."""
    G = []

    def add(total, parts, kind="subset"):
        parts = [p for p in parts if _ref_ok(p)]
        if _ref_ok(total) and parts:
            G.append((total, parts, kind))

    pl = d.get("players")
    if isinstance(pl, dict):
        for sub in ("freq", "venues", "level"):
            if isinstance(pl.get(sub), dict):
                # freq and venues bins cover every player; level only the rated
                add((pl, "total"), [(pl[sub], k) for k in pl[sub]], "exact" if sub != "level" else "subset")
    nr = d.get("newret")
    if isinstance(nr, dict):
        c = nr.get("city")
        if isinstance(c, dict):
            add((c, "players"), [(c, "new"), (c, "returning")], "exact")
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
                add((card, "newHere"), [(wk, i) for i in range(len(wk))], "exact")
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
                add((tr, "travellers"), [(tr["classes"], k) for k in tr["classes"]], "exact")
    for s in ((d.get("net") or {}).get("series") or []):
        if isinstance(s, dict):
            add((s, "players"), [(s, "regulars")])
    for c in d.get("coaches") or []:
        if isinstance(c, dict):
            add((c, "players"), [(c, "regulars")])
    for r in (d.get("levelFit") or []) + (d.get("levelClass") or []):
        if isinstance(r, dict):
            add((r, "players"), [(r, "verified")])
    # regulars who've gone quiet: quiet = elsewhere + not seen; quiet is a
    # subset of regulars; each "now books at X" is a subset of elsewhere
    q = ((d.get("figures") or {}).get("quiet") or {})
    if isinstance(q.get("city"), dict):
        add((q["city"], "regulars"), [(q["city"], "quiet")])
    qv = [vv for vv in (q.get("venues") or {}).values() if isinstance(vv, dict)]
    for vv in qv:
        add((vv, "quiet"), [(vv, "elsewhere"), (vv, "notSeen")], "exact")
        add((vv, "regulars"), [(vv, "quiet")])
        for t in vv.get("to") or []:
            if isinstance(t, list) and len(t) >= 2:
                add((vv, "elsewhere"), [(t, 1)])
    # the city's quiet regulars (and regulars) against the venues listed:
    # "15 city-wide, 12 at the only venue shown" says 3 at the hidden ones
    if isinstance(q.get("city"), dict) and qv:
        for k in ("quiet", "regulars"):
            add((q["city"], k), [(vv, k) for vv in qv], "city")
    return G


def remainder(total_ref, parts) -> float | None:
    t = _get(*total_ref)
    if not _num(t):
        return None
    return t - sum(_get(*p) for p in parts if _num(_get(*p)))


def _hide_smallest(parts) -> bool:
    vis = [p for p in parts if _num(_get(*p)) and _get(*p) > 0]
    if not vis:
        return False
    o, k = min(vis, key=lambda p: _get(*p))
    o[k] = None
    return True


def _exact_totals(d: dict) -> set:
    """The 'exact' breakdowns that really add up in this build (checked before
    anything is hidden): only those can be worked back out. One that doesn't
    add up (a 6-week series under a 90-day total) is treated as a subset."""
    out = set()
    for total, parts, kind in partitions_ex(d):
        if kind != "exact":
            continue
        vals = [_get(*r) for r in [total] + parts]
        if all(_num(v) for v in vals) and abs(vals[0] - sum(vals[1:])) < 1e-9:
            out.add((id(total[0]), total[1]))
    return out


def _complements(d: dict, exact: set | None = None) -> None:
    """Hide numbers until no total minus its visible parts comes to 1-4 and
    no hidden number in an exact breakdown can be worked back out; repeat to a
    fixpoint (hiding one number can expose another pair)."""
    for _ in range(100):
        changed = False
        for total, parts, kind in partitions_ex(d):
            t = _get(*total)
            if kind == "exact" and (exact is None or (id(total[0]), total[1]) in exact):
                # exactly one unknown among the total and its parts can be
                # worked out from the others: hide one more
                unknown = [r for r in [total] + parts if not _num(_get(*r))]
                if len(unknown) == 1 and _hide_smallest(parts if unknown[0] is total else
                                                        [p for p in parts if p is not unknown[0]] or parts):
                    changed = True
                    continue
            rem = remainder(total, parts)
            if rem is None or not small(rem):
                continue
            if kind == "city":
                if _num(t):
                    total[0][total[1]] = None             # hide the city figure, not a venue's
                    changed = True
                continue
            if _hide_smallest(parts):
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


def _quiet_split(d: dict) -> None:
    """Regulars who've gone quiet: the split is shown whole or not at all
    (half a split next to the total gives the other half away), and when
    "booked elsewhere" is hidden, the venues they now book at go too."""
    q = ((d.get("figures") or {}).get("quiet") or {})
    for vv in (q.get("venues") or {}).values():
        if not isinstance(vv, dict):
            continue
        if vv.get("elsewhere") is None or vv.get("notSeen") is None:
            vv["elsewhere"] = vv["notSeen"] = None
        if vv.get("elsewhere") is None and vv.get("to"):
            vv["to"] = []


def enforce(d: dict) -> dict:
    """Apply RATES then COUNTS, then hide whatever subtraction would give away
    (partitions), to the build in place; returns it."""
    exact = _exact_totals(d)
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
    for _ in range(3):                    # the follow-on rules can hide more; check again
        _complements(d, exact)
        _shares_bases(d)
        _quiet_split(d)
    for parent_path, k in INTERNAL:
        for o in _objects(d, _parse(parent_path)):
            if isinstance(o, dict):
                o.pop(k, None)
    return d
