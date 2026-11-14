"""
dupr_sample.py -- one-off DUPR sample at the venues that hide DUPR (8 Nov 2026)

Run by hand: Actions > "DUPR sample" (workflow_dispatch only).

13 venues show DUPR-style ratings on <=4% of their public roster players
(20-58% elsewhere), though every roster read asks for rating_provider=dupr.
This reads PlayByPoint's signed-in per-player ratings for up to DUPR_SAMPLE_LIMIT
(default 200) players seen ONLY at those venues, to measure how many have a DUPR
we can't see today.

1. dupr_sample_targets(p_limit) (migration 20270205100000) picks them, server
   side, round-robin by home venue, most recent first.
2. Each is read with league_rosters.read_ratings, the nightly job's own call,
   about one request a second, signed in the same way (PBP_COOKIES_JSON).
3. Readings are saved with the nightly job's pbp_ratings_ingest (unchanged),
   after dupr_sample_register adds the pbp_people row that RPC needs.

Protects the account: stops at the first 401/403/429 (and makes no further
PlayByPoint request, retries included), or after 3 failed reads in a row,
saving what it read and exiting non-zero.

Prints aggregate counts only: never names, PlayByPoint ids or DUPR ids
(errors are shown as type / HTTP status only: this repo is public, and a
Postgres key error would otherwise print the id).

Re-checks the 14:45-17:30 UTC window before every read, so a long run that
starts just before it stops cleanly (saving what it read) instead of running
into the nightly scrape.

Separate from the nightly league ratings read: its own run, no shared counter.
The nightly job's 400-a-night cap is a per-run constant, and its 20-hour rule is
per person (pbp_people.ratings_read_at), and only for people on a live league or
tournament roster, which these players are not.

    DUPR_SAMPLE_LIMIT=200 python dupr_sample.py
    DUPR_SAMPLE_DRY_RUN=1 python dupr_sample.py     (pick and count only, no PlayByPoint calls)
"""
from __future__ import annotations

import asyncio
import os
import sys
from collections import Counter, OrderedDict
from datetime import datetime, timezone

import httpx

import league_rosters as lr

console = lr.console

LIMIT = int(os.environ.get("DUPR_SAMPLE_LIMIT") or 200)
MAX_LIMIT = 500
SLEEP = 1.0                  # seconds between PlayByPoint reads
STOP_AFTER_ERRORS = 3        # failed reads in a row
STOP_STATUSES = (401, 403, 429)
CHUNK = 50                   # players per save
DRY_RUN = os.environ.get("DUPR_SAMPLE_DRY_RUN", "").lower() in ("1", "true", "yes")
# The nightly scrape starts 15:00 UTC (cron in daily_scrape.yml) and reads
# league ratings at its end; keep off the account while it runs.
NIGHTLY_WINDOW_UTC = (14 * 60 + 45, 17 * 60 + 30)
FORCE = os.environ.get("DUPR_SAMPLE_FORCE", "").lower() in ("1", "true", "yes")


class AccountGuard(Exception):
    """PlayByPoint refused us (401/403/429): stop and touch nothing more."""


class GuardedClient:
    """Wraps PlayByPointAPI's HTTP session. After the first 401/403/429 every
    further GET fails at once, without a request, so PlayByPointAPI's own
    retries can't hit the account again."""

    def __init__(self, inner):
        self._inner = inner
        self.tripped: int | None = None

    async def get(self, *a, **kw):
        if self.tripped is not None:
            raise AccountGuard(f"stopped after HTTP {self.tripped}")
        resp = await self._inner.get(*a, **kw)
        if getattr(resp, "status_code", 0) in STOP_STATUSES:
            self.tripped = resp.status_code
        return resp

    def __getattr__(self, name):
        return getattr(self._inner, name)


def round_robin(targets: list[dict], limit: int) -> list[dict]:
    """[{pbp_user_id, venue}] in order -> one per venue in turn, keeping each
    venue's own order and the order venues first appear; at most `limit`,
    no id twice. dupr_sample_targets already returns this order; applied again
    so the script never depends on it."""
    by_venue: OrderedDict[str, list[int]] = OrderedDict()
    seen: set[int] = set()
    for t in targets or []:
        if not isinstance(t, dict):
            continue
        try:
            uid = int(t.get("pbp_user_id"))
        except (TypeError, ValueError):
            continue
        if uid in seen:
            continue
        seen.add(uid)
        by_venue.setdefault(str(t.get("venue") or "unknown"), []).append(uid)
    out, i = [], 0
    while len(out) < limit and any(i < len(v) for v in by_venue.values()):
        for venue, ids in by_venue.items():
            if i < len(ids) and len(out) < limit:
                out.append({"pbp_user_id": ids[i], "venue": venue})
        i += 1
    return out


def _num(v) -> float | None:
    try:
        f = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return f if 1 <= f <= 8.5 else None


def _truthy(v) -> bool:
    return v is True or str(v).strip().lower() == "true"


def summarise(entries: list[dict] | None) -> dict:
    """What one reply says about DUPR (no values kept)."""
    d = next((e for e in entries or [] if str(e.get("provider") or "").strip().lower() == "dupr"), None)
    return {
        "dupr": d is not None,
        "doubles": d is not None and _num(d.get("double")) is not None,
        "singles": d is not None and _num(d.get("single")) is not None,
        "id": d is not None and bool(str(d.get("external_user_id") or "").strip()),
        "doubles_verified": d is not None and _truthy(d.get("double_verified")),
    }


def _why(e: Exception) -> str:
    """An error as type (+ HTTP status when there is one): never its text,
    which can carry ids (Postgres 'Key (pbp_user_id)=(...)')."""
    import re
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status is None:                       # lr._rpc: RuntimeError("<name>: HTTP 409 <body>")
        m = re.search(r"\bHTTP (\d{3})\b", str(e))
        status = m.group(1) if m else None
    return f"{type(e).__name__}" + (f" (HTTP {status})" if status else "")


async def _save(client, rows) -> int:
    ids = [r["pbp_user_id"] for r in rows if r["ratings"] is not None]
    if not ids:
        return 0
    try:
        await lr._rpc(client, "dupr_sample_register", {"p_ids": ids})
    except Exception as e:
        console.print(f"  [yellow]{len(ids)} players not registered: {_why(e)}[/yellow]")
        return 0
    # pbp_ratings_ingest called directly (not lr._save_ratings, which prints the error text)
    try:
        return (await lr._rpc(client, "pbp_ratings_ingest", {"p_rows": rows})).get("new_readings", 0)
    except Exception as e:
        console.print(f"  [yellow]{len(rows)} players not saved: {_why(e)}[/yellow]")
        return 0


async def run_sample(client, api, targets: list[dict], sleep: float = SLEEP,
                     clock=lambda: datetime.now(timezone.utc), force: bool | None = None) -> dict:
    """Reads and saves. Returns the counts and why it stopped (None = finished)."""
    force = FORCE if force is None else force
    guard = GuardedClient(api._client)
    api._client = guard
    totals = Counter()
    per_venue: dict[str, Counter] = {}
    rows, in_a_row, stopped = [], 0, None
    for n, t in enumerate(targets):
        if n:
            await asyncio.sleep(sleep)
        if not force and in_nightly_window(clock()):
            stopped = "reached the nightly scrape window (14:45-17:30 UTC)"
            break
        entries = await lr.read_ratings(api, t["pbp_user_id"])
        v = per_venue.setdefault(t["venue"], Counter())
        totals["tried"] += 1
        if guard.tripped is not None:
            stopped = f"PlayByPoint answered HTTP {guard.tripped}"
            break
        if entries is None:
            totals["failed"] += 1
            v["failed"] += 1
            in_a_row += 1
            if in_a_row >= STOP_AFTER_ERRORS:
                stopped = f"{STOP_AFTER_ERRORS} failed reads in a row"
                break
            continue
        in_a_row = 0
        s = summarise(entries)
        for k in ("read",) + tuple(k for k, on in s.items() if on):
            totals[k] += 1
            v[k] += 1
        rows.append({"pbp_user_id": t["pbp_user_id"], "ratings": entries})
        if len(rows) >= CHUNK:
            totals["new_readings"] += await _save(client, rows)
            rows = []
    if rows:
        totals["new_readings"] += await _save(client, rows)
    return {"totals": totals, "per_venue": per_venue, "stopped": stopped}


def report(res: dict, picked: int, title: str = "DUPR sample") -> None:
    t = res["totals"]
    pct = lambda a, b: f"{100 * a / b:.0f}%" if b else "-"
    console.print(f"{title}: {picked} picked · {t['tried']} tried · {t['read']} read · {t['failed']} failed")
    console.print(f"  with a DUPR entry {t['dupr']} ({pct(t['dupr'], t['read'])}) · DUPR doubles {t['doubles']}"
                  f" ({pct(t['doubles'], t['read'])}) · DUPR singles {t['singles']} · DUPR id {t['id']}"
                  f" ({pct(t['id'], t['read'])}) · doubles verified {t['doubles_verified']}")
    console.print(f"  new readings saved: {t['new_readings']}")
    console.print("  venue        read  dupr  doubles  id  failed")
    for venue, c in sorted(res["per_venue"].items()):
        console.print(f"  {venue:<11} {c['read']:>5} {c['dupr']:>5} {c['doubles']:>8} {c['id']:>3} {c['failed']:>7}")
    if res["stopped"]:
        console.print(f"[red]Stopped early: {res['stopped']}. Nothing more was sent to PlayByPoint.[/red]")


def in_nightly_window(now: datetime) -> bool:
    m = now.hour * 60 + now.minute
    return NIGHTLY_WINDOW_UTC[0] <= m < NIGHTLY_WINDOW_UTC[1]


async def main() -> int:
    if not lr.SUPABASE_KEY:
        console.print("[red]No SUPABASE_KEY[/red]")
        return 1
    if not DRY_RUN and not FORCE and in_nightly_window(datetime.now(timezone.utc)):
        console.print("[red]The nightly scrape runs 15:00 UTC and reads league ratings at its end; "
                      "run this outside 14:45-17:30 UTC (or set DUPR_SAMPLE_FORCE=1).[/red]")
        return 1
    limit = max(1, min(LIMIT, MAX_LIMIT))
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            raw = await lr._rpc(client, "dupr_sample_targets", {"p_limit": limit})
        except Exception as e:
            console.print(f"[red]Could not pick targets: {_why(e)}[/red]")
            return 1
        targets = round_robin(raw, limit)
        by_venue = Counter(t["venue"] for t in targets)
        console.print(f"Targets: {len(targets)} across {len(by_venue)} venues: "
                      + ", ".join(f"{v} {n}" for v, n in sorted(by_venue.items())))
        if DRY_RUN or not targets:
            return 0

        from push_to_supabase import PBP_SLUG_MAP, PROXY_URL, _load_cookies  # same sign-in as the nightly job
        from extract_thejar import PlayByPointAPI
        cookies, user_id = _load_cookies()
        if not cookies:
            console.print("[red]No PBP_COOKIES_JSON[/red]")
            return 1
        async with PlayByPointAPI(cookies=cookies, club_slug=next(iter(PBP_SLUG_MAP.values()), ""),
                                  proxy=PROXY_URL) as api:
            api._user_id = user_id
            res = await run_sample(client, api, targets)
        report(res, len(targets))
        return 2 if res["stopped"] else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
