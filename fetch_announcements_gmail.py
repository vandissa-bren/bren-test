"""
fetch_announcements_gmail.py — pull venue announcements out of the
announcements mailbox and store them in Supabase.

WHAT CHANGED, AND WHY

1. IT NO LONGER ASKS ONLY FOR PLAYBYPOINT MAIL.
   The old query was `from:@playbypoint.com is:unread`. A survey of the
   mailbox found 82 PlayByPoint messages, all of them already processed, and
   13 unread ones the fetcher could not see at all — because they come from
   the venues' own domains:

       info@easternindoor.com.au          Eastern Indoor Pickleball Club
       southmelbourne@thejar.club         The Jar | South Melbourne
       admin@therallypickleball.com.au    The Rally Pickleball

   Those are three of the venues reported as "going missing". They were never
   missed by the matcher; they were never fetched. The mailbox is dedicated to
   announcements, so this now reads everything in it and decides by SENDER,
   not by domain.

2. UNREAD IS NO LONGER THE QUEUE.
   The old version used unread state as durable storage: anything a human
   opened was never ingested and never recorded. The ledger is now the
   `announcements` table itself — the row id is derived from the Gmail message
   id, so "have we seen this" is a database question, not an inbox one.
   Read/unread is now only a signal to YOU: read means handled, unread means
   it still needs a venue.

3. NOTHING IS GUESSED FROM THE SUBJECT LINE.
   The old fallback scanned subjects for fragments, so "dink" matched Dink &
   Drive and any venue writing about "dinking" was filed under it. A
   misattributed announcement is worse than a missing one. Sender address
   only: exact address, then domain, then PlayByPoint slug. No match means the
   message stays unread and is reported, so the map can be extended from
   evidence.

4. THE VENUE NAMES COME FROM THE REGISTRY.
   Two hand-maintained dicts used to duplicate venues.json and had already
   drifted — facility 885 exists in the table under two different spellings.
   Names now come from venue_registry, so they cannot drift again.

5. IT NO LONGER WRITES TO promo_codes.
   That table has never been read by anything in the app, and it was filled by
   a regex that treats any quoted capitalised word as a discount code. Codes
   come from the admin panel.

Run:  python3 fetch_announcements_gmail.py --dry-run     # reads only
      python3 fetch_announcements_gmail.py
      WINDOW_DAYS=120 python3 fetch_announcements_gmail.py   # one-off catch-up
"""
import argparse
import base64
import json
import os
import re
import sys
from datetime import datetime, timezone
from email import message_from_bytes
from email.header import decode_header

import httpx
from dotenv import load_dotenv

import venue_registry

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://stwohmddmdwttasbyblt.supabase.co")
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
GMAIL_TOKEN_PATH = os.environ.get("GMAIL_TOKEN_PATH", "/app/.gmail_token.json")
UNMATCHED_LOG_PATH = os.environ.get("UNMATCHED_LOG_PATH", "/app/unmatched_announcements.log")
GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me"

# How far back to look. Steady state only needs a few days — cron runs every
# fifteen minutes — but the window is what makes a backfill possible at all.
# Fourteen days is roughly a thousand times the fifteen-minute cron interval,
# so it tolerates a long outage while keeping the per-run re-read small. The
# legacy rows this re-reads age out of the window on their own.
WINDOW_DAYS = int(os.environ.get("WINDOW_DAYS", "14"))

# ── who sent it ───────────────────────────────────────────────────────────────
# Three lookups, most specific first. Every entry here was observed in the
# mailbox; none of it is inferred.

# A full address. Needed where one domain serves more than one venue.
SENDERS_EXACT = {
    "southmelbourne@thejar.club": 597,
}

# A whole domain, where the venue owns it outright.
SENDERS_DOMAIN = {
    "easternindoor.com.au": 1009,
    "therallypickleball.com.au": 1664,
    # thejar.club is deliberately NOT here. Two venues share it — The Jar |
    # South Melbourne (597) and The Jar HQ | Maidstone (1883) — so the local
    # part decides. An unrecognised address at that domain must surface as
    # unmatched rather than default to one of them; defaulting is how Maidstone
    # announcements would silently become South Melbourne's.
}

# PlayByPoint senders whose local part is not the venue's booking slug.
# Everything else is derived from the registry below.
PBP_ALIASES = {
    "realdill": 1461,           # books as therealdill
    "melbournepickle": 1383,    # books as MelbournePickleClub
    "leveluppickleball": 755,   # books as leveluppickleballknoxcity
    "pascoevale": 885,          # books as sportswellpickleballpalace
}

# Mail that arrives here and is not a venue announcement. Ignored means
# marked read and not reported -- unlike an unmatched venue, there is nothing
# here for anyone to fix, so it must not sit in the queue crying wolf.
IGNORE_SENDERS = {
    "no-reply@accounts.google.com",
    "no-reply@mailing.playbypoint.com",
}
IGNORE_DOMAINS = {
    # The National Pickleball League. Shares a name with The Jar | South
    # Melbourne's PlayByPoint slug (nplpickleball) but is a separate entity,
    # so its league news is not any venue's announcement.
    "nplpickleball.com.au",
}

SKIP_SUBJECTS = (
    "welcome to",
    "you are now subscribed",
    "thanks for subscribing",
    "subscription confirmed",
)


def _norm_slug(s: str) -> str:
    return s.lower().replace("-", "").replace("_", "").replace(" ", "")


def pbp_localparts() -> dict:
    """
    PlayByPoint sender slug -> facility id, built from the registry so it can
    never drift from venues.json, plus the handful of senders whose local part
    differs from their booking slug.
    """
    m = {_norm_slug(v.slug): v.facility_id for v in venue_registry.active_venues()}
    m.update(PBP_ALIASES)
    return m


def venue_name(facility_id: int) -> str:
    try:
        return venue_registry.get_venue(facility_id).name
    except Exception:
        return f"facility {facility_id}"


ADDR_RE = re.compile(r"[\w.+-]+@[\w.-]+")


def sender_address(raw_from: str) -> str:
    """`Pickle Haus <picklehaus@mailing.playbypoint.com>` -> the address."""
    m = ADDR_RE.search(raw_from or "")
    return m.group(0).lower() if m else ""


def match_venue(addr: str, localparts: dict):
    """
    Returns (facility_id, how) or (None, reason). Sender only — never the
    subject, never the body.
    """
    if not addr:
        return None, "no sender address"
    if addr in IGNORE_SENDERS:
        return None, "ignored sender"
    if addr.rpartition("@")[2] in IGNORE_DOMAINS:
        return None, "ignored sender"
    if addr in SENDERS_EXACT:
        return SENDERS_EXACT[addr], "exact address"
    local, _, domain = addr.partition("@")
    if domain in SENDERS_DOMAIN:
        return SENDERS_DOMAIN[domain], "domain"
    if domain == "playbypoint.com" or domain.endswith(".playbypoint.com"):
        fid = localparts.get(_norm_slug(local))
        if fid:
            return fid, "playbypoint slug"
        return None, f"unknown playbypoint slug '{local}'"
    return None, f"unknown sender domain '{domain}'"


# ── gmail ─────────────────────────────────────────────────────────────────────
def access_token() -> str:
    with open(GMAIL_TOKEN_PATH) as f:
        t = json.load(f)
    r = httpx.post(t["token_uri"], timeout=30, data={
        "client_id": t["client_id"],
        "client_secret": t["client_secret"],
        "refresh_token": t["refresh_token"],
        "grant_type": "refresh_token",
    })
    r.raise_for_status()
    return r.json()["access_token"]


def list_messages(tok: str, query: str) -> list:
    """Every page, not the first fifty. The old cap silently truncated any
    backlog bigger than one batch."""
    out, page = [], None
    while True:
        params = {"q": query, "maxResults": 100}
        if page:
            params["pageToken"] = page
        r = httpx.get(f"{GMAIL_API}/messages", params=params, timeout=30,
                      headers={"Authorization": f"Bearer {tok}"})
        r.raise_for_status()
        body = r.json()
        out.extend(m["id"] for m in body.get("messages", []))
        page = body.get("nextPageToken")
        if not page:
            return out


def get_message(tok: str, msg_id: str):
    r = httpx.get(f"{GMAIL_API}/messages/{msg_id}", params={"format": "raw"},
                  timeout=30, headers={"Authorization": f"Bearer {tok}"})
    r.raise_for_status()
    data = r.json()
    raw = base64.urlsafe_b64decode(data["raw"] + "==")
    return message_from_bytes(raw), data.get("internalDate")


def mark_as_read(tok: str, msg_id: str) -> None:
    httpx.post(f"{GMAIL_API}/messages/{msg_id}/modify", timeout=30,
               headers={"Authorization": f"Bearer {tok}",
                        "Content-Type": "application/json"},
               json={"removeLabelIds": ["UNREAD"]})


# ── the message itself ────────────────────────────────────────────────────────
def decode_subject(s: str) -> str:
    """
    Decodes encoded words AND unfolds the header. A long subject is wrapped
    across lines by the sending server, and the old version kept the newline —
    which is why one stored title reads "20% Off Courts\\n Inside".
    """
    out = ""
    for part, enc in decode_header(s or ""):
        out += part.decode(enc or "utf-8", errors="replace") if isinstance(part, bytes) else part
    return re.sub(r"\s+", " ", out).strip()


def strip_html(html: str) -> str:
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<(br|p|div|tr|li|h[1-6])[^>]*>", "\n", html, flags=re.I)
    html = re.sub(r"<[^>]+>", "", html)
    for a, b in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                 ("&#39;", "'"), ("&quot;", '"')):
        html = html.replace(a, b)
    html = re.sub(r"\n\s*\n+", "\n\n", html)
    return re.sub(r"[ \t]+", " ", html).strip()


def extract_bodies(msg):
    """Returns (text, html). The old version stored the stripped text in
    body_html as well, so that column never held what its name said."""
    text, html = "", ""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            payload = part.get_payload(decode=True)
            if not payload:
                continue
            if ct == "text/plain" and not text:
                text = payload.decode("utf-8", errors="replace").strip()
            elif ct == "text/html" and not html:
                html = payload.decode("utf-8", errors="replace")
    else:
        payload = msg.get_payload(decode=True) or b""
        if msg.get_content_type() == "text/html":
            html = payload.decode("utf-8", errors="replace")
        else:
            text = payload.decode("utf-8", errors="replace").strip()

    if not text and html:
        text = strip_html(html)
    text = re.sub(r"You are receiving this email.*$", "", text, flags=re.S | re.I).strip()
    text = re.sub(r"To unsubscribe.*$", "", text, flags=re.S | re.I).strip()
    return text[:2000], html[:5000]


def stable_announcement_id(msg_id: str) -> int:
    """
    Derived from the Gmail message id, so the same email always produces the
    same number. The old version used abs(hash(subject)) % 10**9, and Python
    randomises str hashing per process — a retried email became a second row.
    Kept under 10**9 to stay inside a Postgres integer.
    """
    return int(msg_id, 16) % (10 ** 9) if re.fullmatch(r"[0-9a-f]+", msg_id) \
        else sum(ord(c) * (31 ** i) for i, c in enumerate(msg_id)) % (10 ** 9)


# ── supabase ──────────────────────────────────────────────────────────────────
def headers(extra: dict = None) -> dict:
    h = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
         "Content-Type": "application/json"}
    if extra:
        h.update(extra)
    return h


def title_key(facility_id, title, when) -> str:
    """
    Facility, normalised title AND THE DAY. The day is not optional: Pascoe
    Vale sends "Thursday 8-10pm - Intermediate Open Play" every single week,
    so a key without a date would treat next week's as a duplicate of this
    week's and drop it, for ever. This key only has to recognise rows the old
    code wrote under a random id; it must not become a recurring-subject
    filter.
    """
    t = re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()
    return f"{facility_id}|{t}|{(when or '')[:10]}"


def existing_rows():
    """
    The ledger. Returns (ids already stored, {facility|title} already stored).

    The second one matters for the changeover: rows written by the old code are
    keyed `gmail-<random hash>`, so the new id scheme cannot recognise them.
    Matching on facility and normalised title as well means the first run over
    a wide window re-reads old mail without duplicating it.
    """
    r = httpx.get(f"{SUPABASE_URL}/rest/v1/announcements", timeout=60,
                  params={"select": "id,facility_id,title,fetched_at",
                          "limit": "100000"},
                  headers=headers())
    if r.status_code != 200:
        sys.exit(f"Cannot read the announcements ledger: HTTP {r.status_code}")
    ids, titles = set(), set()
    for row in r.json():
        ids.add(row.get("id"))
        titles.add(title_key(row.get("facility_id"), row.get("title"),
                             row.get("fetched_at")))
    return ids, titles


def store(record: dict):
    r = httpx.post(f"{SUPABASE_URL}/rest/v1/announcements", json=record, timeout=30,
                   headers=headers({"Prefer": "resolution=merge-duplicates"}))
    ok = r.status_code in (200, 201, 204)
    return ok, (None if ok else f"HTTP {r.status_code} {r.text[:160]}")


def log_unmatched(reason, addr, subject, sent_at):
    """
    One line per problem, and it says WHICH problem. The old log mixed "no
    venue matched" with "the database write failed" in the same file with no
    way to tell them apart, which is why three Pickle Haus and MPC entries
    looked like matching failures when they were write failures.
    """
    try:
        with open(UNMATCHED_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat()}\t{reason}\t"
                    f"{addr}\t{sent_at}\t{subject[:160]}\n")
    except Exception as e:
        print(f"    (could not write the log: {e})")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="read everything, write nothing, mark nothing read")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode = "DRY RUN — nothing will be written" if args.dry_run else "live"
    print(f"[{stamp}] Reading the announcements mailbox ({mode}, {WINDOW_DAYS}d window)")

    localparts = pbp_localparts()
    tok = access_token()
    known_ids, known_titles = existing_rows()
    print(f"Ledger: {len(known_ids)} announcements already stored")

    # Everything in the window. NOT `is:unread` — see the note at the top.
    msg_ids = list_messages(tok, f"newer_than:{WINDOW_DAYS}d")
    print(f"Mailbox: {len(msg_ids)} messages in the window")

    stored = skipped = unmatched = failed = ignored = 0

    for msg_id in msg_ids:
        row_id = f"gmail-{msg_id}"
        if row_id in known_ids:
            skipped += 1
            continue
        try:
            msg, internal = get_message(tok, msg_id)
        except Exception as e:
            print(f"  ! could not read message {msg_id}: {e}")
            failed += 1
            continue

        addr = sender_address(msg.get("From", ""))
        subject = decode_subject(msg.get("Subject", "")) or "(no subject)"
        sent_at = (datetime.fromtimestamp(int(internal) / 1000, tz=timezone.utc).isoformat()
                   if internal else datetime.now(timezone.utc).isoformat())

        if any(s in subject.lower() for s in SKIP_SUBJECTS):
            ignored += 1
            if not args.dry_run:
                mark_as_read(tok, msg_id)
            continue

        facility_id, how = match_venue(addr, localparts)
        if facility_id is None:
            if how == "ignored sender":
                ignored += 1
                if not args.dry_run:
                    mark_as_read(tok, msg_id)
                continue
            # LEFT UNREAD ON PURPOSE. An unmatched announcement stays visible
            # in the inbox until its sender is added above.
            unmatched += 1
            print(f"  ? {addr:<42} {how}")
            print(f"    {subject[:90]}")
            if not args.dry_run:
                log_unmatched("no_venue_match", addr, subject, sent_at)
            continue

        name = venue_name(facility_id)
        if title_key(facility_id, subject, sent_at) in known_titles:
            skipped += 1
            continue

        text, html = extract_bodies(msg)
        record = {
            "id": row_id,
            "announcement_id": stable_announcement_id(msg_id),
            "facility_id": facility_id,
            "facility_name": name,
            "title": subject,
            "body_text": text,
            "body_html": html or text,
            "date_str": f"{name} - {sent_at[:10]}",
            "url": "",
            "fetched_at": sent_at,     # when the venue sent it, not when we looked
        }

        if args.dry_run:
            print(f"  + would store [{how}] {name}: {subject[:70]}")
            stored += 1
            known_titles.add(title_key(facility_id, subject, sent_at))
            continue

        ok, err = store(record)
        if ok:
            stored += 1
            known_ids.add(row_id)
            known_titles.add(title_key(facility_id, subject, sent_at))
            mark_as_read(tok, msg_id)
            print(f"  + {name}: {subject[:70]}")
        else:
            failed += 1
            print(f"  x write failed, left unread: {name}: {subject[:60]} — {err}")
            log_unmatched("store_failed", addr, subject, sent_at)

    print(f"\nStored {stored} · already had {skipped} · unmatched {unmatched} "
          f"· failed {failed} · ignored {ignored}")
    if unmatched:
        print("Unmatched senders stay UNREAD in the mailbox. Add them to "
              "SENDERS_EXACT or SENDERS_DOMAIN and run again.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
