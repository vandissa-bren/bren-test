"""
charge_league_weeks.py — charge each weekly league player's saved card for
the coming week.

The database decides who owes what and when (league_charges_due, in
picklematch-app 20270106100000_host_league_weekly.sql); this moves the money,
as settle_refunds.py does for refunds. Nothing here decides an amount.

    league_charges_due()                 the weeks to charge now (the day
                                         before each week), made once each,
                                         with the card saved for each
    league_charge_result(id, ok, …)      how it went. Not through: the week
                                         is owed, the player is told to pay
                                         it in the app, the host sees "Owes"

Each charge is an off-session PaymentIntent on the host's connected account,
to the Customer and card Stripe saved at entry. The idempotency key is the
charge's id, so a run that dies half way and runs again can't charge twice.

TWO OUTCOMES:

    paid        the card went through
    owed        declined, expired, or the bank wants the player to confirm
                (authentication_required): the player pays it in the app

Run hourly on the API server (the droplet that runs picklematch.service):

    cd /app && /app/venv/bin/python3 charge_league_weeks.py --dry-run
    cd /app && /app/venv/bin/python3 charge_league_weeks.py

    crontab -e:
    7 * * * * cd /app && /app/venv/bin/python3 charge_league_weeks.py >> /app/charge_league_weeks.log 2>&1
"""
import os
import sys
from datetime import datetime, timezone

import httpx
import stripe

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")

DRY = "--dry-run" in sys.argv

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}


def log(msg):
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}", flush=True)


def rpc(client, name, payload=None):
    r = client.post(f"{SUPABASE_URL}/rest/v1/rpc/{name}", headers=HEADERS, json=payload or {})
    r.raise_for_status()
    return r.json() if r.text else None


def main():
    if not (SUPABASE_URL and SUPABASE_KEY and STRIPE_SECRET_KEY):
        log("ERROR missing SUPABASE_URL, SUPABASE_SERVICE_KEY or STRIPE_SECRET_KEY")
        return 1
    stripe.api_key = STRIPE_SECRET_KEY

    with httpx.Client(timeout=30.0) as client:
        due = rpc(client, "league_charges_due") or []
        if not due:
            log("nothing to charge")
            return 0
        log(f"{len(due)} week(s) to charge")

        # The host's connected account: every charge lives on it, as the card does.
        host_ids = sorted({d["host_id"] for d in due if d.get("host_id")})
        accounts = {}
        if host_ids:
            in_list = ",".join(f'"{h}"' for h in host_ids)
            rp = client.get(f"{SUPABASE_URL}/rest/v1/profiles?id=in.({in_list})&select=id,stripe_account_id", headers=HEADERS)
            rp.raise_for_status()
            accounts = {p["id"]: p.get("stripe_account_id") for p in rp.json()}

        paid = owed = skipped = 0
        for d in due:
            cid = d["charge_id"]
            amount = int(d.get("amount_cents") or 0)
            label = f"{cid} ${amount / 100:.2f} ({d.get('description')})"
            account_id = accounts.get(d.get("host_id"))
            if amount <= 0 or not d.get("customer_id") or not d.get("payment_method_id"):
                log(f"SKIP {label} — no amount or no saved card")
                skipped += 1
                continue
            if not account_id:
                # left due (attempts 0): it's charged once the host's Stripe is back
                log(f"SKIP {label} — host has no Stripe account")
                skipped += 1
                continue
            if DRY:
                log(f"WOULD CHARGE {label} on {account_id}")
                continue
            try:
                intent = stripe.PaymentIntent.create(
                    amount=amount, currency="aud",
                    customer=d["customer_id"], payment_method=d["payment_method_id"],
                    off_session=True, confirm=True,
                    description=(d.get("description") or "PickleMatch league")[:200],
                    metadata={"league_week_charge_id": cid, "player_user_id": d.get("user_id") or "", "host_user_id": d.get("host_id") or ""},
                    stripe_account=account_id,
                    idempotency_key=f"leagueweek-{cid}",
                )
            except stripe.error.CardError as e:
                # declined, or the bank wants the player to confirm it
                pi = getattr(e, "payment_intent", None) or (getattr(e, "error", None) and getattr(e.error, "payment_intent", None))
                pi_id = pi.get("id") if isinstance(pi, dict) else getattr(pi, "id", None)
                rpc(client, "league_charge_result", {"p_charge": cid, "p_ok": False, "p_intent": pi_id,
                                                     "p_error": (getattr(e, "code", "") or str(e))[:300]})
                log(f"OWED {label} — {getattr(e, 'code', '') or e}")
                owed += 1
                continue
            except stripe.error.StripeError as e:
                # Stripe itself failed (network, rate limit, the account): not the
                # player's card. Left due; the next run tries again with the same key.
                log(f"RETRY LATER {label} — {str(e)[:200]}")
                skipped += 1
                continue

            if intent.status == "succeeded":
                rpc(client, "league_charge_result", {"p_charge": cid, "p_ok": True, "p_intent": intent.id})
                log(f"PAID {label} {intent.id}")
                paid += 1
            else:
                rpc(client, "league_charge_result", {"p_charge": cid, "p_ok": False, "p_intent": intent.id,
                                                     "p_error": intent.status})
                log(f"OWED {label} — {intent.status}")
                owed += 1

        log(f"done: {paid} paid, {owed} owed, {skipped} left for later")
    return 0


if __name__ == "__main__":
    sys.exit(main())
