"""
venue_registry.py -- the single answer to "what venues do we support?"

THE LIST LIVES IN ONE PLACE: the frontend's public/venues.json, published at
https://picklematch.com.au/venues.json. Adding a venue is one entry there and
a publish; the scrapers, the API server and the email importer all read it
through this module. Nothing in this repo keeps its own list any more.

Read from the site, cached for VENUE_REFRESH_SECONDS (10 minutes), so a new
venue reaches the long-running API server without a redeploy.

NEVER A HARD DEPENDENCY ON THE SITE. Venue configuration is needed to build a
booking request, so an unreachable site must not stop booking. If the fetch
fails -- the site is down, or answers with its HTML shell because the file
is not deployed yet -- the last good list is kept, and before any has loaded
the committed venues.json in this repo (a copy of the site file) is used.
Refresh that copy when you add a venue, so the fallback stays current:
    curl -s https://picklematch.com.au/venues.json -o venues.json

VENUE_SOURCE_URL overrides the address; set it to "" to use only the
committed copy (the tests do).

The lookup layer, not the caller, enforces the distinctions:

    resolve(1664)   -> Venue(status='active', booking_enabled=True)
    resolve(1826)   -> Venue(status='delisted')   historical rows still work
    resolve('pickleplay') -> Unresolved('pickleplay')
    resolve(99999)  -> Unresolved('99999')

`Unresolved` is a RESULT, never a registry row. Historical data contains
values we cannot identify, and the fix is to return an explicit unresolved
state -- not to invent configuration so old rows fit.

Callers must never fall back to another venue. Two such fallbacks exist in
the frontend today (SessionsPage `?? VENUES[0]`, which silently renders an
unknown venue as The Jar) and one in the booking server (`courts[0]`). Every
function here either returns a Venue for the id asked for, or says it
cannot.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional, Union

SNAPSHOT_PATH = os.environ.get(
    "VENUE_SNAPSHOT_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "venues.json"),
)
SOURCE_URL = os.environ.get("VENUE_SOURCE_URL", "https://picklematch.com.au/venues.json")
REFRESH_SECONDS = int(os.environ.get("VENUE_REFRESH_SECONDS", "600"))
FETCH_TIMEOUT = 4
# After a failed fetch, wait this long before trying the site again.
RETRY_SECONDS = 60

VALID_STATUS = {"active", "delisted"}
VALID_FETCHER = {"court_blocks", "sportswell", None}
VALID_PLATFORM = {"playbypoint"}


@dataclass(frozen=True)
class Venue:
    facility_id: int
    platform: str
    slug: str
    name: str
    status: str
    fetcher: Optional[str]
    booking_enabled: bool

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def is_bookable(self) -> bool:
        """Active AND explicitly booking-enabled. Both, always."""
        return self.status == "active" and self.booking_enabled


@dataclass(frozen=True)
class Unresolved:
    """
    A venue reference we cannot identify. Carries the original value so the
    UI can say which reference failed rather than rendering a wrong venue.
    """
    venue_id: str

    @property
    def is_active(self) -> bool:
        return False

    @property
    def is_bookable(self) -> bool:
        return False


Resolution = Union[Venue, Unresolved]


class RegistryError(RuntimeError):
    pass


_cache: Optional[dict] = None
_cache_source = ""          # "site" or "snapshot"
_next_check = 0.0
_lock = threading.Lock()


def _parse(raw: dict) -> dict:
    """
    {facility_id: Venue} for the Play By Point entries of a venues file.

    Reads the site format (facilityId, bookingEnabled) and, so an old copy
    still works, the earlier snapshot format (facility_id, booking_enabled).
    Map-only venues have no facility and are skipped. Anything malformed
    raises, so a bad edit on the site is rejected whole rather than half-used.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get("venues"), list):
        raise RegistryError("venues file has no venue list")
    venues = {}
    for row in raw["venues"]:
        fid = row.get("facilityId", row.get("facility_id"))
        if fid is None:
            continue
        if row.get("platform") not in VALID_PLATFORM:
            continue
        v = Venue(
            facility_id=int(fid),
            platform=row["platform"],
            slug=row["slug"],
            name=row["name"],
            status=row["status"],
            fetcher=row.get("fetcher"),
            booking_enabled=bool(row.get("bookingEnabled", row.get("booking_enabled", False))),
        )
        if v.status not in VALID_STATUS:
            raise RegistryError(f"facility {v.facility_id}: status {v.status!r} is not one of {sorted(VALID_STATUS)}")
        if v.fetcher not in VALID_FETCHER:
            raise RegistryError(f"facility {v.facility_id}: fetcher {v.fetcher!r} is not one of {sorted(f for f in VALID_FETCHER if f)} or null")
        if not v.slug:
            raise RegistryError(f"facility {v.facility_id}: no slug")
        if v.facility_id in venues:
            raise RegistryError(f"facility {v.facility_id} is listed twice")
        venues[v.facility_id] = v
    if not venues:
        raise RegistryError("venues file contains no Play By Point venues")
    return venues


def _fetch_site() -> dict:
    req = urllib.request.Request(SOURCE_URL, headers={
        "Accept": "application/json", "User-Agent": "picklematch-venue-registry"})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
        # The site answers unknown paths with its HTML shell and a 200, so
        # "not deployed yet" arrives as a parse error here, not a 404.
        return _parse(json.loads(resp.read().decode("utf-8")))


def _read_snapshot() -> dict:
    try:
        with open(SNAPSHOT_PATH) as fh:
            return _parse(json.load(fh))
    except FileNotFoundError as e:
        raise RegistryError(f"venue snapshot missing at {SNAPSHOT_PATH}") from e


def _load() -> dict:
    global _cache, _cache_source, _next_check
    now = time.monotonic()
    if _cache is not None and now < _next_check:
        return _cache
    with _lock:
        if _cache is not None and time.monotonic() < _next_check:
            return _cache
        if SOURCE_URL:
            try:
                _cache, _cache_source = _fetch_site(), "site"
                _next_check = time.monotonic() + REFRESH_SECONDS
                return _cache
            except Exception as e:  # network, HTML shell, or a malformed edit
                print(f"  !! venue_registry: could not use {SOURCE_URL} ({e}); "
                      f"{'keeping the last good list' if _cache else 'using the committed venues.json'}")
                _next_check = time.monotonic() + RETRY_SECONDS
                if _cache is not None:
                    return _cache
        _cache, _cache_source = _read_snapshot(), "snapshot"
        if not SOURCE_URL:
            _next_check = float("inf")
        return _cache


def source() -> str:
    """Where the current list came from: 'site' or 'snapshot'."""
    _load()
    return _cache_source


def resolve(venue_id) -> Resolution:
    """
    Resolve any venue reference -- an int, or the text `venue_id` stored on
    favorites / created_sessions / booked_sessions.

    Never raises for an unknown id and never substitutes another venue.
    """
    if venue_id is None:
        return Unresolved("")
    raw = str(venue_id).strip()
    if not raw or not raw.isdigit():
        return Unresolved(raw)
    return _load().get(int(raw)) or Unresolved(raw)


def get_venue(facility_id: int) -> Venue:
    """A registered venue of any status. Raises if there is none."""
    v = _load().get(int(facility_id))
    if v is None:
        raise RegistryError(f"facility {facility_id} is not in the registry")
    return v


def get_bookable_venue(facility_id: int) -> Venue:
    """
    The only correct entry point for pricing and booking.

    Fails closed. A delisted venue, or an active one whose booking is not
    enabled, raises here rather than proceeding with a slug that would send
    the request somewhere plausible but wrong.
    """
    v = get_venue(facility_id)
    if v.status == "delisted":
        raise RegistryError(f"facility {facility_id} ({v.name}) is delisted")
    if not v.booking_enabled:
        raise RegistryError(
            f"facility {facility_id} ({v.name}) is not booking-enabled")
    return v


def active_venues() -> list[Venue]:
    return [v for v in sorted(_load().values(), key=lambda x: x.facility_id)
            if v.status == "active"]


def bookable_venues() -> list[Venue]:
    return [v for v in active_venues() if v.booking_enabled]


def venues_for_fetcher(fetcher: str) -> list[Venue]:
    """Which venues a given scraper is responsible for."""
    return [v for v in active_venues() if v.fetcher == fetcher]


def slug_for(facility_id: int) -> str:
    """
    Replaces `SLUG_MAP.get(facility_id, "nplpickleball")`.

    That default is why five venues absent from SLUG_MAP sent their price and
    booking requests carrying another facility's Referer. There is no default
    here on purpose.
    """
    return get_venue(facility_id).slug


def all_facility_ids() -> set[int]:
    return set(_load().keys())


def validate_facility_keys(name: str, facility_ids, *, require_bookable=False):
    """
    Check a facility-keyed constant that lives OUTSIDE the registry -- e.g.
    booking_server's FALLBACK_PLANS, FALLBACK_CLINIC_IDS,
    INDIVIDUAL_PRICE_FACILITIES. Those are pricing capability, not venue
    identity, so they stay where they are; but an entry for a facility that
    is not registered, or is delisted, is a defect.

    Returns a list of human-readable problems; empty means clean.
    """
    problems = []
    for fid in facility_ids:
        try:
            v = get_venue(int(fid))
        except RegistryError:
            problems.append(f"{name}: facility {fid} is not in the registry")
            continue
        if v.status == "delisted":
            problems.append(f"{name}: facility {fid} ({v.name}) is delisted")
        elif require_bookable and not v.booking_enabled:
            problems.append(
                f"{name}: facility {fid} ({v.name}) is not booking-enabled")
    return problems
