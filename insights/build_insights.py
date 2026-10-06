"""
build_insights.py -- rebuilds the full Insights dataset behind /admin/insights.

Runs once a day (see .github/workflows/build_insights.yml). For each window
(last 7, 28 and 90 days, ending yesterday in Melbourne) it:

  1. exports sessions, rosters and the catalogue from Supabase through three
     service-only functions (insights_export_sessions / _rosters / _catalogue),
  2. runs the same analysis as the Melbourne overview prototype
     (build_overview.py -> venue_report, report_more, timing_preview, network),
  3. keeps the aggregate view only -- venues named, players only as counts,
     no player rows or ties, groups / shared-player links of 5 or more --
  4. stores it in insights_builds (one row per window). Only admins can read it,
     through admin_insights_full().

Environment: SUPABASE_URL, SUPABASE_SERVICE_KEY (or SUPABASE_KEY).
Optional: INSIGHTS_WINDOWS="7,28,90".

Local run against CSV exports, no Supabase:
    python insights/build_insights.py --csv path/to/csvdir --out out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MEL = ZoneInfo("Australia/Melbourne")
MIN_GROUP = 5
COLS = {
    "sessions": ["session_key", "session_date", "start_time", "capacity", "price", "status", "starts_at",
                 "venue_id", "session_type", "first_obs", "last_obs", "n_obs", "n_obs_before", "last_obs_before",
                 "spots_at_start", "first_full_at", "spots_7d", "spots_3d", "spots_1d", "n_prices",
                 "n_start_times", "venue_name", "end_time"],
    "rosters": ["pbp_user_id", "session_key", "lesson_id", "session_date", "first_seen", "last_seen",
                "rating_at_time"],
    "catalogue": ["venue_id", "venue_name", "lesson_id", "title", "type", "category", "skill_level",
                  "program_slug", "date", "start", "end_time", "capacity", "spots_left", "price"],
}


def sb():
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        sys.exit("ERROR: SUPABASE_URL and SUPABASE_SERVICE_KEY are needed")
    h = {"apikey": key, "Content-Type": "application/json"}
    if key.startswith("eyJ"):
        h["Authorization"] = f"Bearer {key}"
    return url, h


def rpc(client, url, h, name, args):
    for attempt in range(3):
        r = client.post(f"{url}/rest/v1/rpc/{name}", headers=h, json=args)
        if r.status_code < 500:
            break
        time.sleep(5 * (attempt + 1))
    if r.status_code >= 300:
        raise RuntimeError(f"{name}: HTTP {r.status_code} {r.text[:300]}")
    return r.json() or []


def export(client, url, h, frm, to, d: Path, chunk_days: int = 14):
    """The three exports for sessions dated frm..to, fetched a fortnight at a time
    so no single call runs long."""
    import pandas as pd
    counts = {}
    for kind, cols in COLS.items():
        rows, a = [], frm
        while a <= to:
            b = min(to, a + timedelta(days=chunk_days - 1))
            rows += rpc(client, url, h, f"insights_export_{kind}", {"p_from": str(a), "p_to": str(b)})
            a = b + timedelta(days=1)
        df = pd.DataFrame(rows, columns=cols)
        key = {"sessions": ["session_key"], "rosters": ["pbp_user_id", "session_key"], "catalogue": ["lesson_id"]}[kind]
        df = df.drop_duplicates(key)
        df.to_csv(d / f"{kind}.csv", index=False)
        counts[kind] = len(df)
    return counts


def build(data_dir: Path, work_dir: Path):
    """Run the prototype's analysis over the CSVs in data_dir."""
    import importlib
    os.environ["INSIGHTS_DATA"] = str(data_dir)
    os.environ["INSIGHTS_WORK"] = str(work_dir)
    import venue_report as R
    R.DATA, R.WORK = data_dir, work_dir
    import report_more as M
    import build_overview as B
    importlib.reload(M)
    importlib.reload(B)
    M.main()                      # market-wide only; writes report_more.json, which build_overview reads
    return B.main()


def shareable(d: dict, net: dict) -> dict:
    """The aggregate view: tracked venues only; no player rows or ties; anything
    that counts a group of players needs 5 or more of them."""
    d = dict(d)
    d["venues"] = [v for v in d["venues"] if v.get("tracked")]
    ids = {v["id"] for v in d["venues"]}
    d["sessions"] = [s for s in d["sessions"] if s["v"] in ids]
    d["links"] = [l for l in d["links"] if l["a"] in ids and l["b"] in ids and l["n"] >= MIN_GROUP]
    net = dict(net)
    net.pop("players", None)
    net.pop("edges", None)
    for s in net.get("series", []):
        s["also"] = [a for a in s.get("also", []) if a[2] >= MIN_GROUP]
    net["groups"] = [g for g in net.get("groups", []) if g["size"] >= MIN_GROUP]
    net["glinks"] = [l for l in net.get("glinks", []) if l["n"] >= MIN_GROUP]
    d["net"] = net
    return d


def clean(o):
    """NaN/inf -> None so the JSON is valid for Postgres."""
    if isinstance(o, float):
        return None if o != o or o in (float("inf"), float("-inf")) else o
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="build from CSV exports in this folder instead of Supabase")
    ap.add_argument("--out", help="write the result here instead of storing it")
    a = ap.parse_args()

    if a.csv:
        with tempfile.TemporaryDirectory() as w:
            d, net = build(Path(a.csv), Path(w))
        out = clean(dict(shareable(d, net), window={"days": None, "builtAt": datetime.now(timezone.utc).isoformat()}))
        Path(a.out or "insights.json").write_text(json.dumps(out, separators=(",", ":"), default=str))
        print(f"{len(out['venues'])} venues, {len(out['sessions'])} sessions -> {a.out or 'insights.json'}")
        return 0

    import httpx
    url, h = sb()
    windows = [int(x) for x in os.environ.get("INSIGHTS_WINDOWS", "7,28,90").split(",") if x.strip()]
    to = datetime.now(MEL).date() - timedelta(days=1)
    ok = 0
    with httpx.Client(timeout=300) as client:
        for days in windows:
            frm = to - timedelta(days=days - 1)
            t0 = time.time()
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    data_dir, work_dir = Path(tmp) / "data", Path(tmp) / "work"
                    data_dir.mkdir()
                    work_dir.mkdir()
                    n = export(client, url, h, frm, to, data_dir)
                    if not n["sessions"]:
                        print(f"{days}d: no sessions between {frm} and {to}; skipped")
                        continue
                    d, net = build(data_dir, work_dir)
                out = clean(dict(shareable(d, net), window={
                    "days": days, "from": str(frm), "to": str(to),
                    "builtAt": datetime.now(timezone.utc).isoformat(), "exported": n}))
                body = json.dumps({"days": days, "built_at": out["window"]["builtAt"], "data": out},
                                  separators=(",", ":"), default=str)
                r = client.post(f"{url}/rest/v1/insights_builds?on_conflict=days",
                                headers={**h, "Prefer": "resolution=merge-duplicates,return=minimal"},
                                content=body)
                if r.status_code >= 300:
                    raise RuntimeError(f"store: HTTP {r.status_code} {r.text[:300]}")
                ok += 1
                print(f"{days}d ({frm} to {to}): {n} -> {len(out['venues'])} venues, "
                      f"{len(out['sessions'])} sessions, {len(body) // 1024} KB in {time.time() - t0:.0f}s")
            except Exception as e:
                print(f"{days}d: FAILED {type(e).__name__}: {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
