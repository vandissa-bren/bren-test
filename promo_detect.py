"""
promo_detect.py — is this venue announcement offering something? (F134)

Used by fetch_announcements_gmail.py as each email is stored, and by
backfill_promo_candidates.py for announcements already in the table. Either way
the result is only a CANDIDATE in `promos`: a person confirms it in the SQL
editor (select * from promo_review;), because announcements rarely say when an
offer ends and some hits are events rather than offers.

    detect(subject, body) -> dict | None
    store_candidate(supabase_url, key, announcement_id, facility_id, venue, subject, body, sent_at) -> str
"""
from __future__ import annotations
import re

# An OFFER: something cheaper, free or extra. Prize pools and tournaments alone are not.
OFFER = re.compile(
    r"(\d{1,3}\s?%\s?off"
    r"|half[- ]price"
    r"|\$\s?\d+(?:\.\d{2})?\s?off"
    r"|\$\s?\d+(?:\.\d{2})?\s+(?:first|intro|trial)\b[^.\n]{0,30}"
    r"|free (?:session|sessions|court|courts|court time|game|games|entry|paddle|paddles|clinic|coaching"
    r"|trial|hour|open play|play|learn to play|assessment|player assessment)"
    r"|buy \w+ get \w+"
    r"|bring a (?:friend|mate)"
    r"|early[- ]bird"
    r"|intro(?:ductory)? offer"
    r"|first (?:session|visit|game) (?:free|for \$\s?\d+)"
    r"|special offer"
    r"|membership sale|sale is on"
    r"|loyalty rewards?"
    r"|discount(?:ed)?)",
    re.I,
)
# Codes are written in capitals: "use code MPC20", "grab BOOSTER", "code: SPRING".
CODE = re.compile(r"\b(?:code|coupon|grab|use|enter|apply)\b[:\s\-]+[\"“'‘]?([A-Z][A-Z0-9]{3,19})\b")
QUOTED = re.compile(r"[\"“'‘]([A-Z][A-Z0-9]{3,19})[\"”'’]")
NOT_CODES = {"DUPR", "FREE", "TODAY", "NOW", "THIS", "THE", "YOUR", "BOOK", "BOOKING", "PLAY", "OPEN",
             "SOCIAL", "SPOTS", "NOTE", "HERE", "LINK", "APP", "PBP", "PLAYBYPOINT", "CLICK", "NEW",
             "SALE", "WEEK", "WEEKEND", "COURT", "COURTS", "LIVE", "MPC", "NPL", "MLPA", "RSVP"}
ENDS = re.compile(r"((?:until|ends?|ending|valid (?:until|till|to|through)|expires?|offer closes|"
                  r"last (?:day|week|chance)|before|by)\s[^.\n!]{0,40})", re.I)


def _kind(t: str) -> str:
    if re.search(r"membership", t) and re.search(r"%\s?off|sale|discount|\$\s?\d+\s?off", t):
        return "membership"
    if re.search(r"early[- ]bird", t):
        return "early_bird"
    if re.search(r"bring a (friend|mate)", t):
        return "bring_a_friend"
    if re.search(r"%\s?off|half[- ]price|\$\s?\d+(\.\d{2})?\s?off|discount", t):
        return "discount"
    if re.search(r"\bfree\b", t):
        return "free"
    return "other"


def _sentence_with(text: str, span: tuple[int, int]) -> str:
    a = max(text.rfind(".", 0, span[0]), text.rfind("\n", 0, span[0]), text.rfind("!", 0, span[0])) + 1
    ends = [i for i in (text.find(".", span[1]), text.find("\n", span[1]), text.find("!", span[1])) if i != -1]
    b = min(ends) + 1 if ends else len(text)
    return re.sub(r"\s+", " ", text[a:b]).strip()[:240]


def detect(subject: str, body: str) -> dict | None:
    text = f"{subject or ''}\n{body or ''}"
    hits = list(OFFER.finditer(text))
    if not hits:
        return None
    signals = []
    for m in hits:
        s = re.sub(r"\s+", " ", m.group(1).lower()).strip()
        if s not in signals:
            signals.append(s)
    codes = [c for c in dict.fromkeys(CODE.findall(text) + QUOTED.findall(text)) if c not in NOT_CODES]
    end = ENDS.search(text)
    lower = text.lower()
    return {
        "kind": _kind(lower),
        "signals": signals[:8],
        "code": codes[0] if codes else None,
        "summary": _sentence_with(text, hits[0].span()),
        "ends_hint": re.sub(r"\s+", " ", end.group(1)).strip()[:80] if end else None,
    }


def store_candidate(supabase_url: str, service_key: str, announcement_id: str, facility_id: int,
                    venue: str, subject: str, body: str, sent_at: str, client=None) -> str:
    """Add a candidate for this announcement. Idempotent: one per announcement id."""
    found = detect(subject, body)
    if not found:
        return "no offer"
    import httpx
    row = {"facility_id": facility_id, "venue_name": venue, "title": (subject or "Offer")[:120],
           "summary": found["summary"], "kind": found["kind"], "code": found["code"],
           "signals": found["signals"], "ends_hint": found["ends_hint"],
           "status": "candidate", "source": "announcement", "announcement_id": announcement_id,
           "starts_at": sent_at}
    hdr = {"apikey": service_key, "Authorization": f"Bearer {service_key}",
           "Content-Type": "application/json", "Prefer": "resolution=ignore-duplicates,return=minimal"}
    post = (client or httpx).post
    r = post(f"{supabase_url}/rest/v1/promos?on_conflict=announcement_id", headers=hdr, json=row, timeout=20)
    if r.status_code in (200, 201, 204):
        return f"candidate ({found['kind']}: {', '.join(found['signals'][:3])})"
    return f"candidate NOT stored (HTTP {r.status_code}: {r.text[:120]})"
