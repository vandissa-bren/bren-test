"""
promo_detect.py — is this venue announcement offering something? (F134)

Used by fetch_announcements_gmail.py as each email is stored, and by
backfill_promo_candidates.py for announcements already in the table. Either way
the result is only a CANDIDATE in `promos`: a person confirms it in the SQL
editor (select * from promo_review;), because announcements rarely say when an
offer ends and some hits are events rather than offers.

    detect(subject, body) -> dict | None
    store_candidate(supabase_url, key, announcement_id, facility_id, venue, subject, body, sent_at) -> str

F142: a venue mentions the same offer in email after email (Melbourne Pickle Club's
20% off memberships arrived three times in September). A follow-up ATTACHES to the
offer already being tracked — recorded in its seen_announcements — instead of adding
another row. Same venue, and either the same code, the same headline discount, or a
title that reads the same.
"""
from __future__ import annotations
import difflib, re

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


def _same_offer(found: dict, subject: str, row: dict) -> str | None:
    """Why this announcement is the offer `row` already tracks, or None.

    A shared CODE is never enough on its own. Melbourne Pickle Club ran its
    birthday open play and its 20% off memberships under one MPCTURNS1
    campaign: two offers, two banners, two rows. The announcements must also
    agree on WHAT is offered — the same kind, the same headline discount, or a
    title that reads the same."""
    # The headline signal: "20% off" twice at one venue is one offer, not two.
    theirs = {s for s in (row.get("signals") or []) if "%" in s or "$" in s}
    mine = {s for s in found["signals"] if "%" in s or "$" in s}
    shared = theirs & mine if (theirs and mine) else set()

    a, b = _norm(subject), _norm(row.get("title") or "")
    alike = bool(a and b and difflib.SequenceMatcher(None, a, b).ratio() >= 0.6)
    same_kind = bool(found.get("kind")) and found["kind"] == row.get("kind")

    code = (found.get("code") or "").upper()
    if code and (row.get("code") or "").upper() == code:
        if shared:
            return f"same code {code}, same offer ({', '.join(sorted(shared))})"
        if same_kind or alike:
            return f"same code {code}"
        return None            # one campaign code over two different offers
    if shared:
        return f"same offer ({', '.join(sorted(shared))})"
    if alike:
        return "the same announcement, sent again"
    return None


def _norm(t: str) -> str:
    """Titles for comparison: no emoji, punctuation or filler."""
    t = re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower())
    return re.sub(r"\b(the|a|an|is|are|for|at|on|in|our|your|new|now|this|week|and)\b", " ", t).strip()


def store_candidate(supabase_url: str, service_key: str, announcement_id: str, facility_id: int,
                    venue: str, subject: str, body: str, sent_at: str, client=None) -> str:
    """Add a candidate for this announcement, unless the offer is already tracked.
    Idempotent: one row per announcement id, one row per offer."""
    found = detect(subject, body)
    if not found:
        return "no offer"
    import httpx
    hdr = {"apikey": service_key, "Authorization": f"Bearer {service_key}",
           "Content-Type": "application/json"}
    get = (client or httpx).get
    post = (client or httpx).post
    patch = (client or httpx).patch

    # Is this the same offer as one already on the shortlist for this venue?
    try:
        r = get(f"{supabase_url}/rest/v1/promos", headers=hdr, timeout=20, params={
            "select": "id,title,code,signals,seen_announcements,status",
            "facility_id": f"eq.{facility_id}",
            "status": "in.(candidate,live)"})
        existing = r.json() if r.status_code == 200 else []
    except Exception:
        existing = []
    for row in existing:
        why = _same_offer(found, subject, row)
        if not why:
            continue
        seen = list(row.get("seen_announcements") or [])
        if announcement_id in seen:
            return f"already tracked ({why})"
        seen.append(announcement_id)
        try:
            patch(f"{supabase_url}/rest/v1/promos", headers={**hdr, "Prefer": "return=minimal"},
                  params={"id": f"eq.{row['id']}"}, json={"seen_announcements": seen}, timeout=20)
        except Exception:
            pass
        return f"attached to the offer already tracked ({why})"

    row = {"facility_id": facility_id, "venue_name": venue, "title": (subject or "Offer")[:120],
           "summary": found["summary"], "kind": found["kind"], "code": found["code"],
           "signals": found["signals"], "ends_hint": found["ends_hint"],
           "status": "candidate", "source": "announcement", "announcement_id": announcement_id,
           "seen_announcements": [announcement_id], "starts_at": sent_at}
    r = post(f"{supabase_url}/rest/v1/promos?on_conflict=announcement_id",
             headers={**hdr, "Prefer": "resolution=ignore-duplicates,return=minimal"},
             json=row, timeout=20)
    if r.status_code in (200, 201, 204):
        return f"candidate ({found['kind']}: {', '.join(found['signals'][:3])})"
    return f"candidate NOT stored (HTTP {r.status_code}: {r.text[:120]})"
