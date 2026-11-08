"""
test_member_prices.py -- the parsing in fetch_member_prices.py.

Run: python -m pytest -q test_member_prices.py
"""
from datetime import date, datetime

import fetch_member_prices as fmp


def test_split_key_matches_record_court_rates():
    # record_court_rates: court_id '^([^_]+)_', shift '^[^_]+_(.+)$'
    assert fmp.split_key("15020_primetime") == ("15020", "primetime")
    assert fmp.split_key("15020_primetime_weekend") == ("15020", "primetime_weekend")
    assert fmp.split_key("15020") is None


def test_melbourne_midnight():
    ts = fmp.melbourne_midnight_unix(date(2026, 11, 10))
    assert datetime.fromtimestamp(ts, fmp.MEL).strftime("%Y-%m-%d %H:%M") == "2026-11-10 00:00"


def test_member_venues():
    ms = [{"facility_id": 1557, "is_valid": True}, {"facility_id": 999, "is_valid": False}]
    assert fmp.member_venues(ms) == {1557: None}
    assert fmp.member_venues('[{"facility_id": "42", "is_valid": true}]') == {42: None}
    assert fmp.member_venues(None) == fmp.FALLBACK_VENUES
    assert fmp.member_venues([]) == fmp.FALLBACK_VENUES


def test_price_windows_uses_real_blocks_and_weekend_days():
    prices = {"1_day": 40, "1_primetime_weekend": 50, "2_lowtime": 30, "x_day": 1}
    by_date = {
        "2026-11-08": [{"court_id": "1", "shift": "day", "start": "12:00"}],   # before first day
        "2026-11-10": [{"court_id": "1", "shift": "day", "start": "13:30"}],
        "2026-11-14": [{"court_id": "1", "shift": "primetime_weekend", "start": "18:00"}],
    }
    w = fmp.price_windows(prices, by_date, date(2026, 11, 9))       # a Monday
    assert w["1_day"] == (date(2026, 11, 10), 13 * 3600 + 1800)
    assert w["1_primetime_weekend"] == (date(2026, 11, 14), 18 * 3600)
    assert w["2_lowtime"] == (date(2026, 11, 9), 8 * 3600)           # guessed, weekday
    assert "x_day" not in w
    # no block: a weekend key falls on the next Saturday
    w2 = fmp.price_windows({"3_day_weekend": 1}, {}, date(2026, 11, 9))
    assert w2["3_day_weekend"][0] == date(2026, 11, 14)


def test_parse_member_price():
    data = {"total": {"original_reservation_fare": "32.5"},
            "prices_per_user": [{"id": 7, "price": {"affiliation": {"name": "Member"},
                                                     "shift_prices": [{"price": 32.5}]}}]}
    assert fmp.parse_member_price(data, 7) == (32.5, {"name": "Member"})
    unpriced = {"total": {"original_reservation_fare": 0.0},
                "prices_per_user": [{"id": 7, "price": {"shift_prices": []}}]}
    assert fmp.parse_member_price(unpriced, 7) == (None, None)
    assert fmp.parse_member_price(None, 7) == (None, None)


def test_refusal():
    assert "signed out" in fmp.refusal(401, "", {})
    assert "signed out" in fmp.refusal(403, "null", {})
    assert "Cloudflare" in fmp.refusal(403, "<title>Just a moment</title>", {})
    assert fmp.refusal(200, "{}", {}) is None


# ── review fixes ──

import asyncio


class _R:
    def __init__(self, status, data=None, text=""):
        self.status_code, self._d, self.text, self.headers = status, data, text, {}

    def json(self):
        if self._d is None:
            raise ValueError("not json")
        return self._d


class _PBP:
    def __init__(self, aff):
        self.aff = aff

    async def get(self, url, params=None, headers=None, timeout=None):
        return _R(200, {"prices_per_user": [{"id": 7, "price": {"affiliation": self.aff}}],
                        "total": {"original_reservation_fare": 40}})


def _run(aff, monkeypatch):
    async def nosleep(_):
        return None
    monkeypatch.setattr(fmp.asyncio, "sleep", nosleep)
    row = {"court_prices": {"1_day": 40, "2_day": 40}, "by_date": {}}
    return asyncio.run(fmp.price_venue(_PBP(aff), 1557, "x", row, 7))


def test_no_affiliation_anywhere_is_not_saved(monkeypatch):
    prices, err = _run(None, monkeypatch)
    assert err and "casual" in err


def test_member_affiliation_is_saved(monkeypatch):
    prices, err = _run({"name": "Member"}, monkeypatch)
    assert err is None and prices == {"1_day": 40.0, "2_day": 40.0}


def test_has_member_affiliation():
    assert not fmp.has_member_affiliation([None, None, ""])
    assert fmp.has_member_affiliation([None, {"id": 3}])


def test_saved_prices_replace_old_and_drop_gone_courts():
    prices = {"1_day": 30.0, "9_day": 20.0}           # 9 no longer in court_prices
    assert fmp.member_prices_to_save(prices, {"1_day": 40, "2_day": 40}) == {"1_day": 30.0}


def test_logs_never_show_user_id_or_bodies(monkeypatch, capsys):
    class Leaky(_PBP):
        async def get(self, *a, **k):
            return _R(500, None, text="SECRET-BODY user 7")
    async def nosleep(_):
        return None
    monkeypatch.setattr(fmp.asyncio, "sleep", nosleep)
    asyncio.run(fmp.price_venue(Leaky(None), 1557, "x", {"court_prices": {"1_day": 40}, "by_date": {}}, 7))
    assert "SECRET-BODY" not in capsys.readouterr().out
