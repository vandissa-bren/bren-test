"""
settle_refunds.py — pay what the ledger says is owed.

The database decides what is owed and why; this moves the money. Nothing here
makes a policy decision — if you find yourself computing an amount or deciding
whether a refund applies, something is in the wrong place.

    refunds_due()                       what is owed, to which intent, on
                                        whose connected account
    settle_refund(id, ok, failure)      mark it done, and tell the player
    refund_needs_funds(id)              tell the host their balance is short

THREE OUTCOMES, and they are not the same:

    refunded    money returned, player notified, row closed

    needs funds the connected account is empty. Left `due` so the next run
                tries again — this resolves on its own when the host takes
                another booking — and the host is told ONCE rather than hourly

    failed      disputed, already refunded, no such intent, account
                restricted. Terminal: retrying either loops forever or
                double-refunds the moment the cause clears. Needs a person.

FIRST, PAYMENTS THE APP NEVER CONFIRMED (5 Nov 2026). An entry, week or sub
payment is recorded as taken (succeeded_at) only when the app calls
/payment_confirm. If Stripe took the money and that call never came (the app
closed, the network dropped), nothing recorded it, so nothing refunded it.
Each run asks Stripe about every such payment from the last three days that's
over 20 minutes old, and stamps the ones it took. The database's sweeps then
settle them: used, or refunded within the hour as the payment sheet says.

    cd /app && /app/venv/bin/python3 settle_refunds.py --dry-run
    cd /app && /app/venv/bin/python3 settle_refunds.py
"""
import os
import sys
from datetime import datetime, timedelta, timezone

import httpx
import stripe

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
# No hardcoded fallback. A service key baked into a file is that key published
# the moment the file reaches a repo — which is exactly how one leaked here.
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
    r = client.post(f"{SUPABASE_URL}/rest/v1/rpc/{name}",
                    headers=HEADERS, json=payload or {})
    r.raise_for_status()
    return r.json() if r.text else None


# the kinds of payment the app confirms itself (payment_intents_issued.session_id)
CONFIRMED_KINDS = ("tournament:", "league:", "leagueweek:", "leaguesub:")


def host_accounts(client, host_ids):
    if not host_ids:
        return {}
    in_list = ",".join(f'"{h}"' for h in sorted(host_ids))
    rp = client.get(f"{SUPABASE_URL}/rest/v1/profiles?id=in.({in_list})&select=id,stripe_account_id", headers=HEADERS)
    rp.raise_for_status()
    return {p["id"]: p.get("stripe_account_id") for p in rp.json()}


def reconcile(client):
    """Stamp payments Stripe took that the app never confirmed."""
    now = datetime.now(timezone.utc)
    since = (now - timedelta(days=3)).isoformat()
    until = (now - timedelta(minutes=20)).isoformat()
    kinds = ",".join(f"session_id.like.{k}*" for k in CONFIRMED_KINDS)
    r = client.get(
        f"{SUPABASE_URL}/rest/v1/payment_intents_issued",
        params={"succeeded_at": "is.null", "and": f"(created_at.gte.{since},created_at.lte.{until})",
                "or": f"({kinds})", "select": "intent_id,host_id,amount_cents"},
        headers=HEADERS,
    )
    r.raise_for_status()
    rows = [x for x in r.json() if str(x.get("intent_id", "")).startswith("pi_")]
    if not rows:
        return
    accounts = host_accounts(client, {x["host_id"] for x in rows if x.get("host_id")})
    stamped = 0
    for x in rows:
        intent_id, account_id = x["intent_id"], accounts.get(x.get("host_id"))
        try:
            intent = (stripe.PaymentIntent.retrieve(intent_id, stripe_account=account_id) if account_id
                      else stripe.PaymentIntent.retrieve(intent_id))
        except stripe.error.StripeError as e:
            log(f"RECONCILE SKIP {intent_id} — {str(e)[:200]}")
            continue
        if intent.status != "succeeded" or int(intent.amount_received or 0) < int(x.get("amount_cents") or 0):
            continue
        if DRY:
            log(f"WOULD STAMP {intent_id} (taken, never confirmed)")
            stamped += 1
            continue
        w = client.patch(
            f"{SUPABASE_URL}/rest/v1/payment_intents_issued?intent_id=eq.{intent_id}&succeeded_at=is.null",
            headers={**HEADERS, "Prefer": "return=minimal"},
            json={"succeeded_at": now.isoformat()},
        )
        w.raise_for_status()
        log(f"STAMPED {intent_id} (taken, never confirmed)")
        stamped += 1
    if stamped:
        log(f"reconciled {stamped} payment(s)")


def main():
    if not (SUPABASE_URL and SUPABASE_KEY and STRIPE_SECRET_KEY):
        log("ERROR missing SUPABASE_URL, SUPABASE_SERVICE_KEY or STRIPE_SECRET_KEY")
        return 1
    stripe.api_key = STRIPE_SECRET_KEY

    with httpx.Client(timeout=30.0) as client:
        try:
            reconcile(client)
        except Exception as e:   # never stops the refunds below
            log(f"RECONCILE ERROR {str(e)[:300]}")
        due = rpc(client, "refunds_due") or []
        if not due:
            log("nothing owed")
            return 0
        log(f"{len(due)} refund(s) owed")

        # Host Stripe accounts. A direct charge lives on the connected account,
        # so the refund must be issued against it — not the platform. The
        # legacy manual refund endpoint got this wrong and never worked.
        host_ids = sorted({d["host_id"] for d in due if d.get("host_id")})
        accounts = {}
        if host_ids:
            in_list = ",".join(f'"{h}"' for h in host_ids)
            rp = client.get(
                f"{SUPABASE_URL}/rest/v1/profiles?id=in.({in_list})"
                "&select=id,stripe_account_id",
                headers=HEADERS,
            )
            rp.raise_for_status()
            accounts = {p["id"]: p.get("stripe_account_id") for p in rp.json()}

        settled = short = failed = 0

        for d in due:
            rid = d["refund_id"]
            intent = d["payment_intent_id"]
            amount = float(d["refund_amount"] or 0)
            label = f"{intent} ${amount:.2f} ({d.get('reason')})"

            if amount <= 0:
                # `refunds_zero_is_not_due` should make this unreachable.
                log(f"SKIP {label} — nothing to refund")
                continue

            account_id = accounts.get(d.get("host_id"))
            if not account_id:
                log(f"SKIP {label} — host has no Stripe account")
                continue

            if DRY:
                log(f"WOULD REFUND {label} on {account_id}")
                settled += 1
                continue

            try:
                # DOLLARS IN THE LEDGER, CENTS AT STRIPE. Getting this wrong by
                # a factor of 100 is the likeliest bug here, in either
                # direction, and neither direction is recoverable quietly.
                # ONE REFUND PER LEDGER ROW, EVER. Each refund carries its
                # ledger row's id. If a run died after Stripe refunded but
                # before settle_refund recorded it, this run finds that refund
                # and records it instead of refunding again (a part refund
                # would otherwise go through twice). Not an idempotency key:
                # Stripe keeps a key's error for a day, so a refund that once
                # met an empty balance couldn't be retried when it fills.
                done_before = any(
                    (r.get("metadata") or {}).get("ledger_id") == str(rid)
                    for r in stripe.Refund.list(payment_intent=intent, limit=100,
                                                stripe_account=account_id).auto_paging_iter()
                )
                if not done_before:
                    stripe.Refund.create(
                        payment_intent=intent,
                        amount=int(round(amount * 100)),
                        stripe_account=account_id,
                        metadata={"ledger_id": str(rid)},
                    )
            except stripe.error.StripeError as e:
                code = getattr(e, "code", "") or ""
                detail = str(e)[:400]

                if code == "balance_insufficient":
                    # Not terminal: the host takes another booking and the
                    # money is there. Left `due`, host told once.
                    rpc(client, "refund_needs_funds", {"p_refund_id": rid})
                    log(f"NEEDS FUNDS {label} — left due, will retry"
                        + ("" if d.get("host_notified") else " (host notified)"))
                    short += 1
                    continue

                rpc(client, "settle_refund",
                    {"p_refund_id": rid, "p_ok": False, "p_failure": detail})
                log(f"FAILED {label} — {detail}")
                failed += 1
                continue

            rpc(client, "settle_refund", {"p_refund_id": rid, "p_ok": True})
            log(f"REFUNDED {label}")
            settled += 1

        log(f"done — {settled} refunded, {short} awaiting funds, {failed} failed"
            + (" (dry run)" if DRY else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
