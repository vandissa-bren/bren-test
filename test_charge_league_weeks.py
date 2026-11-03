"""charge_league_weeks.py: each outcome reports back once, with the right words."""
import types
from unittest import mock

import stripe
import charge_league_weeks as job

DUE = [
    {"charge_id": "c-paid", "amount_cents": 1200, "host_id": "h1", "user_id": "u1", "customer_id": "cus_1", "payment_method_id": "pm_1", "description": "L · week 1"},
    {"charge_id": "c-card", "amount_cents": 1200, "host_id": "h1", "user_id": "u2", "customer_id": "cus_2", "payment_method_id": "pm_2", "description": "L · week 1"},
    {"charge_id": "c-down", "amount_cents": 1200, "host_id": "h1", "user_id": "u3", "customer_id": "cus_3", "payment_method_id": "pm_3", "description": "L · week 1"},
]


def run():
    calls = []

    def fake_rpc(client, name, payload=None):
        calls.append((name, payload))
        return DUE if name == "league_charges_due" else None

    def fake_create(**kw):
        assert kw["off_session"] and kw["confirm"] and kw["stripe_account"] == "acct_h1"
        assert kw["idempotency_key"] == "leagueweek-" + kw["metadata"]["league_week_charge_id"]
        cid = kw["metadata"]["league_week_charge_id"]
        if cid == "c-card":
            e = stripe.error.CardError("Your card needs authentication.", None, "authentication_required")
            e.payment_intent = {"id": "pi_card"}
            raise e
        if cid == "c-down":
            raise stripe.error.APIConnectionError("network")
        return types.SimpleNamespace(status="succeeded", id="pi_ok")

    profiles = mock.MagicMock()
    profiles.json.return_value = [{"id": "h1", "stripe_account_id": "acct_h1"}]
    client = mock.MagicMock()
    client.__enter__.return_value = client
    client.get.return_value = profiles
    with mock.patch.object(job, "SUPABASE_URL", "http://x"), mock.patch.object(job, "SUPABASE_KEY", "k"), \
         mock.patch.object(job, "STRIPE_SECRET_KEY", "sk_test_x"), mock.patch.object(job, "rpc", fake_rpc), \
         mock.patch.object(job.httpx, "Client", return_value=client), mock.patch.object(stripe.PaymentIntent, "create", side_effect=fake_create):
        assert job.main() == 0
    return [c for c in calls if c[0] == "league_charge_result"]


def test_outcomes():
    results = {p["p_charge"]: p for _, p in run()}
    assert results["c-paid"]["p_ok"] is True and results["c-paid"]["p_intent"] == "pi_ok"
    assert results["c-card"]["p_ok"] is False and results["c-card"]["p_intent"] == "pi_card"
    assert results["c-card"]["p_error"] == "authentication_required"
    # Stripe down is not the player's card: nothing reported, left due for the next run
    assert "c-down" not in results
