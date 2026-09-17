"""Fetch court blocks for geo-restricted venues and push to Supabase.

These venues (SportsWell, Raya, Pickle4Real) are geo-restricted by PBP and
can only be reliably fetched from an Australian IP -- this DO server, not
GitHub Actions runners. Hence a separate script from fetch_court_blocks.py.

C7, 17 Sep 2026 -- runs on picklematch-geo-syd1 (170.64.183.118) as the
unprivileged user `pmjobs`. What changed, and what did not:

  * UNCHANGED: how blocks and prices are built (fetch_blocks_and_prices).
    These venues book by the hour, so one block per available hour.
  * SETTINGS come from a private JSON file (GEO_SECRETS, default
    /home/pmjobs/secrets.json: supabase_url, supabase_secret_key,
    pbp_email, pbp_password), read when the job runs -- not at import.
    SUPABASE_URL / SUPABASE_SECRET_KEY in the environment override it.
  * THE KEY is a new-style sb_secret_ key, sent on the `apikey` header only;
    Supabase rejects those as Authorization: Bearer. A legacy JWT key still
    gets both headers.
  * THE SESSION lives in GEO_SESSION (default /home/pmjobs/pbp_session.json).
    It is checked once per run; if PlayByPoint refuses it, the job logs in
    again the same browserless way the app's connect step does, and saves
    the new session. The account number comes from whoami(), which logs
    nothing, and is kept with the session for price lookups.
  * WINDOWS: DAYS_START..DAYS_AHEAD-1 (defaults 0..13). DRY_RUN=1 fetches
    everything and writes nothing.
  * SAVING goes through merge_availability (C4b): only this job's court days,
    court prices and their timestamps, and fetch_status; past days pruned.
  * FAILURES (C5): a day that errors is not saved, so its stored courts stay;
    fetch_status is ok / ok_empty / partial / failed. The run exits non-zero
    if every venue failed or any save was rejected.
"""
import asyncio, json, os, re, sys, time, httpx
import venue_registry
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

from extract_thejar import PlayByPointAPI

SECRETS_PATH = os.environ.get("GEO_SECRETS", "/home/pmjobs/secrets.json")
SESSION_PATH = os.environ.get("GEO_SESSION", "/home/pmjobs/pbp_session.json")
PBP_BASE = os.environ.get("PBP_BASE_OVERRIDE") or "https://app.playbypoint.com"
DAYS_START = int(os.environ.get("DAYS_START", "0"))
DAYS_AHEAD = int(os.environ.get("DAYS_AHEAD", "14"))
DRY_RUN = os.environ.get("DRY_RUN", "") == "1"

# How old a cached price can get before we refetch it. Matches
# fetch_court_blocks.py's PRICE_REFRESH_HOURS behaviour.
PRICE_REFRESH_HOURS = int(os.environ.get("PRICE_REFRESH_HOURS", "168"))

# Superseded by venue_registry (fetcher='sportswell'). No consumers remain;
# deleted once the frontend migrates. Do not add to it.
GEO_RESTRICTED = {885: "sportswellpickleballpalace", 1770: "rayapickleballclub", 1783: "PICKLE4REAL"}


def sec_to_hhmm(s): return f"{s//3600:02d}:{(s%3600)//60:02d}"


def get_shift(sec, target_date=None):
    """
    DEPRECATED -- last-resort fallback only, if PBP ever omits the real shift
    field for a slot. PBP's own shift label (captured in fetch_blocks_and_prices)
    is used everywhere else now, same as fetch_court_blocks.py.
    """
    hour = sec // 3600
    if hour >= 17: shift = "primetime"
    elif hour >= 12: shift = "day"
    else: shift = "lowtime"
    if target_date and target_date.weekday() >= 5:
        shift = f"{shift}_weekend"
    return shift


async def fetch_blocks_and_prices(api, fid, target, existing_prices, existing_fetched_at, errors=None):
    """
    SportsWell-style venues use session-style available_hours (one block per
    hour) -- we deliberately build one block per available hour slot rather
    than merging consecutive hours, since these venues book hourly not in
    30-min increments (a different, still-open issue for the main pipeline).
    """
    # C7: when `errors` is a list, any failure that makes the day's blocks
    # incomplete is appended to it, and the caller does not save that day.
    # Price errors are not among them -- a missing price is shown as
    # unpriced, which is honest; a missing court is not.
    blocks = []
    new_prices = dict(existing_prices)
    new_fetched_at = dict(existing_fetched_at)
    now = datetime.utcnow()

    try:
        h = await api.available_hours(fid, target, surface="pickleball")
        slots = (h or {}).get("available_hours", [])

        # Capture PBP's real shift label per slot, tagged with weekday/weekend
        # so a shared label (e.g. "primetime") on both a weekday and a
        # weekend with genuinely different prices doesn't collide.
        sec_shift_map = {}
        valid = []
        for s in slots:
            if not (s.get("available") and isinstance(s.get("seconds_from_midnight"), (int, float))):
                continue
            sec = int(s["seconds_from_midnight"])
            valid.append(sec)
            real_shift = s.get("shift")
            if real_shift:
                if target.weekday() >= 5:
                    real_shift = f"{real_shift}_weekend"
                sec_shift_map[sec] = real_shift

        def shift_for(sec):
            return sec_shift_map.get(sec) or get_shift(sec, target)

        # Sample one representative court per distinct real shift present today.
        sample_courts = {}  # {shift: court_id}
        distinct_shifts = set(shift_for(s) for s in valid)
        for shift in distinct_shifts:
            shift_sec = next((s for s in valid if shift_for(s) == shift), None)
            if shift_sec is None:
                continue
            try:
                courts = await api.available_courts(fid, target, shift_sec, shift_sec + 1800, surface="pickleball")
                if courts:
                    sample_courts[shift] = courts[0].get("id")
                await asyncio.sleep(0.2)
            except Exception:
                pass

        # Fetch missing or stale prices using sample courts.
        for shift, court_id in sample_courts.items():
            if not court_id:
                continue
            cache_key = f"{court_id}_{shift}"
            needs_fetch = cache_key not in new_prices
            if not needs_fetch:
                last = new_fetched_at.get(cache_key)
                if last:
                    try:
                        age_hours = (now - datetime.fromisoformat(last)).total_seconds() / 3600
                        needs_fetch = age_hours >= PRICE_REFRESH_HOURS
                    except Exception:
                        needs_fetch = True
                else:
                    needs_fetch = True  # pre-existing entry from before this feature
            if not needs_fetch:
                continue
            shift_sec = next((s for s in valid if shift_for(s) == shift), None)
            if shift_sec is None:
                continue
            try:
                price_data = await api.court_price(int(court_id), target, shift_sec, shift_sec + 3600, user_id=api._user_id)
                fare = (price_data or {}).get("total", {}).get("original_reservation_fare")
                new_prices[cache_key] = round(float(fare), 2) if fare is not None else None
                new_fetched_at[cache_key] = now.isoformat()
                print(f"    Price {shift}: ${new_prices[cache_key]}")
                await asyncio.sleep(0.3)
            except Exception as e:
                print(f"    Price error {shift}: {e}")

        # Build price lookup by shift (use first court_id per shift as representative).
        shift_price = {}
        for shift, court_id in sample_courts.items():
            cache_key = f"{court_id}_{shift}"
            if cache_key in new_prices:
                shift_price[shift] = new_prices[cache_key]

        # Build one block per court per available hour.
        for sec in valid:
            shift = shift_for(sec)
            try:
                courts = await api.available_courts(fid, target, sec, sec + 1800, surface="pickleball")
            except Exception as e:
                courts = []
                if errors is not None:
                    errors.append(f"courts at {sec_to_hhmm(sec)}: {e}")
            if not courts:
                courts = [{"id": None, "name": "Court"}]
            for court in courts:
                cname = court.get("name") or "Court"
                cid = court.get("id")
                blocks.append({
                    "court": cname,
                    "court_id": str(cid) if cid else None,
                    "start": sec_to_hhmm(sec),
                    "end": sec_to_hhmm(sec + 3600),
                    "duration_min": 60,
                    "price": shift_price.get(shift),
                    "shift": shift,
                    "pricingTier": shift.replace("_weekend", ""),
                })
            await asyncio.sleep(0.2)

    except Exception as e:
        print(f"  Error: {e}")
        if errors is not None:
            errors.append(str(e))

    return blocks, new_prices, new_fetched_at


def load_settings() -> dict:
    s = {}
    if os.path.exists(SECRETS_PATH):
        with open(SECRETS_PATH) as f:
            s = json.load(f)
    s["supabase_url"] = os.environ.get("SUPABASE_URL") or s.get("supabase_url") \
        or "https://stwohmddmdwttasbyblt.supabase.co"
    key = (os.environ.get("SUPABASE_SECRET_KEY") or s.get("supabase_secret_key")
           or os.environ.get("SUPABASE_SERVICE_KEY"))
    if not key:
        raise SystemExit(f"No Supabase key: expected {SECRETS_PATH} or SUPABASE_SECRET_KEY")
    s["supabase_secret_key"] = key
    return s


def supabase_headers(key: str) -> dict:
    h = {"apikey": key, "Content-Type": "application/json"}
    if key.startswith("eyJ"):            # legacy JWT service key only
        h["Authorization"] = f"Bearer {key}"
    return h


def _write_private(path: str, obj: dict) -> None:
    fd = os.open(path + ".new", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)
    os.replace(path + ".new", path)


def load_session():
    try:
        with open(SESSION_PATH) as f:
            s = json.load(f)
        return s if (s.get("cookies") or {}).get("_paybycourt_session") else None
    except Exception:
        return None


async def login(settings: dict):
    """Browserless PlayByPoint login, as the app's connect step does it."""
    from curl_cffi.requests import AsyncSession
    email, password = settings.get("pbp_email"), settings.get("pbp_password")
    if not (email and password):
        print("  LOGIN NOT POSSIBLE: no PlayByPoint email/password in the settings file")
        return None
    try:
        async with AsyncSession(impersonate="chrome124") as s:
            r = await s.get(f"{PBP_BASE}/users/sign_in")
            m = re.search(r'<meta name="csrf-token" content="([^"]+)"', r.text or "")
            if not m:
                print(f"  LOGIN FAILED: sign-in page HTTP {r.status_code}, no token")
                return None
            r2 = await s.post(
                f"{PBP_BASE}/users/sign_in",
                json={"user": {"email": email, "password": password, "remember_me": "1"}},
                headers={"Accept": "application/json", "Content-Type": "application/json",
                         "X-CSRF-Token": m.group(1), "X-Requested-With": "XMLHttpRequest",
                         "Referer": f"{PBP_BASE}/users/sign_in"},
            )
            try:
                ok = bool(r2.json().get("success"))
            except Exception:
                ok = False
            jar = {k: v for k, v in s.cookies.items()}
    except Exception as e:
        print(f"  LOGIN FAILED: {type(e).__name__}: {str(e)[:100]}")
        return None
    if not (ok and jar.get("_paybycourt_session")):
        print(f"  LOGIN FAILED: HTTP {r2.status_code}, not accepted")
        return None
    session = {"cookies": jar, "obtained_at": int(time.time())}
    _write_private(SESSION_PATH, session)
    print("  logged in again; new session saved")
    return session


def _api(cookies, slug):
    kw = {"app_base_url": PBP_BASE} if os.environ.get("PBP_BASE_OVERRIDE") else {}
    return PlayByPointAPI(cookies=cookies, club_slug=slug, **kw)


async def session_accepted(cookies, venue) -> bool:
    async with _api(cookies, venue.slug) as api:
        try:
            await api.court_types(venue.facility_id, kind=None)
            return True
        except PermissionError:
            return False
        except Exception as e:
            print(f"  session check could not complete: {type(e).__name__}: {str(e)[:80]}")
            return False


async def ensure_session(settings, venue):
    session = load_session()
    if session and await session_accepted(session["cookies"], venue):
        return session
    print("  saved session missing or refused -- logging in")
    session = await login(settings)
    if session and not await session_accepted(session["cookies"], venue):
        print("  NEW SESSION ALSO REFUSED")
        return None
    return session


async def main():
    settings = load_settings()
    headers = supabase_headers(settings["supabase_secret_key"])
    base = settings["supabase_url"].rstrip("/")
    today = datetime.now(ZoneInfo('Australia/Melbourne')).date()
    dates = [today + timedelta(days=i) for i in range(DAYS_START, DAYS_AHEAD)]
    venues = list(venue_registry.venues_for_fetcher("sportswell"))
    print(f"Geo court job: days {DAYS_START}-{DAYS_AHEAD - 1}, {len(venues)} venues"
          + ("  [DRY RUN - nothing will be saved]" if DRY_RUN else ""))

    results = {v.facility_id: {"name": v.name, "ok": False, "error": None, "dates_ok": 0,
                               "failed_dates": [], "by_date": {}} for v in venues}
    save_failed = []

    session = await ensure_session(settings, venues[0]) if venues else None
    if session and not session.get("user_id"):
        async with _api(session["cookies"], venues[0].slug) as api:
            uid = await api.whoami()
        if uid:
            session["user_id"] = uid
            _write_private(SESSION_PATH, session)
        else:
            print("  note: account number not found; prices may come back empty")

    async with httpx.AsyncClient(timeout=60.0) as client:
        async def merge(row_id, name, p_set, p_by_date=None, prune_before=None):
            if DRY_RUN:
                days = ", ".join(f"{d}:{len(b)}" for d, b in sorted((p_by_date or {}).items()))
                print(f"  DRY RUN {name}: would save status={p_set.get('fetch_status', {}).get('state')} days [{days}]")
                return {"result": "dry-run"}
            body = {"p_id": row_id, "p_set": p_set}
            if p_by_date is not None:
                body["p_by_date"] = p_by_date
            if prune_before is not None:
                body["p_prune_before"] = prune_before
            try:
                r = await client.post(f"{base}/rest/v1/rpc/merge_availability", headers=headers, json=body)
            except Exception as e:
                save_failed.append((row_id, f"{type(e).__name__}: {e}"))
                print(f"  SAVE FAILED {name}: {type(e).__name__}: {e}")
                return None
            if r.status_code != 200:
                save_failed.append((row_id, f"HTTP {r.status_code}"))
                print(f"  SAVE FAILED {name}: HTTP {r.status_code} {r.text[:200]}")
                return None
            out = r.json()
            if isinstance(out, dict) and out.get("result") == "missing":
                print(f"  NOT SAVED {name}: row {row_id} does not exist")
                return None
            return out

        for v in venues:
            fid, slug, res = v.facility_id, v.slug, results[v.facility_id]
            row_id = f"pbp-{fid}"
            status = "no response"
            try:
                resp = await client.get(f"{base}/rest/v1/availability_cache",
                                        params={"id": f"eq.{row_id}", "select": "id,data"}, headers=headers)
                status = f"HTTP {resp.status_code}"
                rows = resp.json() if resp.status_code == 200 else []
            except Exception as e:
                rows, res["error"] = [], f"read failed: {e}"
            if not rows:
                res["error"] = res["error"] or f"no stored row {row_id} ({status})"
                print(f"  SKIPPED {v.name}: {res['error']}")
                continue
            data = rows[0].get("data") or {}
            existing_prices = data.get("court_prices", {}) or {}
            existing_fetched_at = data.get("court_prices_fetched_at", {}) or {}
            res["prices"], res["fetched_at"] = dict(existing_prices), dict(existing_fetched_at)

            if not session:
                res["error"] = "no PlayByPoint session (login failed)"
            else:
                try:
                    async with _api(session["cookies"], slug) as api:
                        api._user_id = session.get("user_id")
                        for target in dates:
                            date_errors = []
                            blocks, res["prices"], res["fetched_at"] = await fetch_blocks_and_prices(
                                api, fid, target, res["prices"], res["fetched_at"], errors=date_errors)
                            ds = target.isoformat()
                            if date_errors:
                                res["failed_dates"].append(ds)
                                res["error"] = res["error"] or date_errors[0][:300]
                                print(f"  {slug} {ds}: FAILED ({len(date_errors)} error(s)) -- stored day kept")
                            else:
                                res["by_date"][ds] = blocks
                                res["dates_ok"] += 1
                                print(f"  {slug} {ds}: {len(blocks)} blocks")
                            await asyncio.sleep(1)
                    res["ok"] = res["dates_ok"] > 0
                except Exception as e:
                    res["error"] = f"{type(e).__name__}: {e}"
                    print(f"  {v.name} FAILED: {res['error']}")

            now_iso = datetime.now().astimezone().isoformat()
            if not res["ok"]:
                await merge(row_id, v.name, {"fetch_status": {
                    "state": "failed", "at": now_iso, "error": res["error"],
                    "dates_ok": res["dates_ok"], "failed_dates": res["failed_dates"]}})
                print(f"  NOT SAVED {v.name}: fetch failed, existing courts left intact")
                continue
            today_str = date.today().isoformat()
            stored = {d: b for d, b in (data.get("by_date") or {}).items() if str(d) >= today_str}
            stored.update(res["by_date"])
            total = sum(len(b) for b in stored.values() if isinstance(b, list))
            state = "partial" if res["failed_dates"] else ("ok" if total else "ok_empty")
            out = await merge(
                row_id, v.name,
                {"court_prices": res["prices"], "court_prices_fetched_at": res["fetched_at"],
                 "fetch_status": {"state": state, "at": now_iso, "error": res["error"],
                                  "dates_ok": res["dates_ok"], "failed_dates": res["failed_dates"],
                                  "blocks": total}},
                p_by_date=res["by_date"], prune_before=today_str)
            if out is not None and not DRY_RUN:
                print(f"Saved {slug}: {total} total blocks, {len(res['prices'])} prices cached"
                      + (f"  [PARTIAL -- kept stored days {', '.join(res['failed_dates'])}]" if res["failed_dates"] else ""))

    failed = [f for f, r in results.items() if not r["ok"]]
    partial = [f for f, r in results.items() if r["ok"] and r["failed_dates"]]
    print("=" * 60)
    print(f"GEO SUMMARY  attempted={len(results)}  ok={len(results) - len(failed) - len(partial)}  "
          f"partial={len(partial)}  failed={len(failed)}  save_failed={len(save_failed)}"
          + ("  DRY RUN" if DRY_RUN else ""))
    for f in failed:
        print(f"  FAILED {f} {results[f]['name']} -- {results[f]['error']}")
    for f in partial:
        print(f"  PARTIAL {f} {results[f]['name']} -- kept {', '.join(results[f]['failed_dates'])}")
    if save_failed or (results and len(failed) == len(results)):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
