"""
fixture_build.py -- a small made-up export for testing the build end to end
(test_privacy.py). Six venues, five weeks of weekly sessions, a few hundred
players with stable habits, so every part of the build has something to count,
including plenty of groups of 1-4 that must never leave the build.

    python fixture_build.py OUT_DIR     # writes the CSVs the build reads
"""
import json
import random
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

MEL = ZoneInfo("Australia/Melbourne")
VENUES = ["597", "1557", "1355", "1783", "755", "1919"]
TINY = "1883"          # a venue only 3 players use: every count about it is under 5
TYPES = [("Intermediate Open Play", "open play", "intermediate", 16, 22.0, 2.0),
         ("Beginner Social", "social", "beginner", 12, 18.0, 1.5),
         ("Advanced Drills", "clinic", "advanced", 8, 35.0, 1.5),
         ("Round Robin", "round robin", None, 16, 25.0, 2.0)]


def write(out: Path, start=date(2026, 9, 28), weeks=5, players=420, seed=7):
    rnd = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    pool = list(range(100000, 100000 + players))
    # each player: a home venue, a usual series or two, a rating (some DUPR-like, some defaults)
    habit = {}
    for p in pool:
        home = rnd.choice(VENUES)
        k = rnd.random()
        rating = (round(rnd.uniform(2.5, 4.6), 3) if k < 0.3 else (1.0 if k < 0.7 else None))
        habit[p] = {"home": home, "rating": rating, "keen": rnd.random(),
                    "other": rnd.choice(VENUES) if rnd.random() < 0.3 else None}
    sess, rost, cat = [], [], []
    lesson = 500000
    for p in pool[:3]:
        habit[p]["other"] = TINY
    for vi, v in enumerate(VENUES + [TINY]):
        for ti, (title, typ, lvl, cap, price, dur) in enumerate(TYPES[: 2 + vi % 3]):
            dow, hour = (vi + ti) % 7, [7, 10, 18, 19][ti]
            for w in range(weeks):
                d = start + timedelta(days=7 * w + dow)
                lesson += 1
                key = f"pbp-{lesson}"
                st = datetime.combine(d, time(hour), MEL).astimezone(timezone.utc)
                first_obs = st - timedelta(days=10)
                regulars = [p for p in pool if habit[p]["home"] == v or habit[p]["other"] == v]
                keen = [p for p in regulars if rnd.random() < 0.15 + 0.5 * habit[p]["keen"]]
                booked = keen[: rnd.randint(max(2, cap // 3), cap)]
                full = len(booked) >= cap
                spots = cap - len(booked)
                sess.append({
                    "session_key": key, "session_date": d.isoformat(), "start_time": f"{hour:02d}:00:00",
                    "capacity": cap, "price": f"${price:.0f}", "status": "full" if full else "open",
                    "starts_at": st.isoformat(), "venue_id": f"pbp-{v}", "session_type": typ,
                    "first_obs": first_obs.isoformat(), "last_obs": (st + timedelta(hours=1)).isoformat(),
                    "n_obs": 30, "n_obs_before": 25, "last_obs_before": (st - timedelta(minutes=20)).isoformat(),
                    "spots_at_start": spots,
                    "first_full_at": (st - timedelta(hours=60)).isoformat() if full else None,
                    "spots_7d": min(cap, spots + 6), "spots_3d": min(cap, spots + 3), "spots_1d": spots,
                    "n_prices": 1, "n_start_times": 1, "venue_name": f"Venue {v}",
                    "end_time": f"{int(hour + dur):02d}:{30 if dur % 1 else 0:02d}:00", "price_tiers": None,
                    "program_slug": f"prog-{v}-{ti}", "waitlist_max": rnd.choice([0, 0, 1, 3, 6]) if full else 0,
                    "coaches": json.dumps(["Coach A"]) if typ == "clinic" else None})
                cat.append({"venue_id": f"pbp-{v}", "venue_name": f"Venue {v}", "lesson_id": lesson,
                            "title": title, "type": typ, "category": typ, "skill_level": lvl,
                            "program_slug": f"prog-{v}-{ti}", "date": d.isoformat(), "start": f"{hour:02d}:00",
                            "end_time": None, "capacity": cap, "spots_left": spots, "price": price})
                for p in booked:
                    seen0 = st - timedelta(hours=rnd.randint(2, 200))
                    last = st - timedelta(minutes=40) if rnd.random() > 0.05 else seen0
                    r = habit[p]["rating"]
                    rost.append({"pbp_user_id": p, "session_key": key, "lesson_id": lesson,
                                 "session_date": d.isoformat(), "first_seen": seen0.isoformat(),
                                 "last_seen": last.isoformat(),
                                 "rating_at_time": None if r is None else (f"{r:.3f}" if r > 2 else "1.0")})
    s, r, c = pd.DataFrame(sess), pd.DataFrame(rost), pd.DataFrame(cat)
    # waitlists: fill-log readings with people waiting, for full sessions with a queue
    wl = []
    for x in sess:
        if x["waitlist_max"]:
            st = datetime.fromisoformat(x["starts_at"])
            for hrs, n in ((50, 1), (30, x["waitlist_max"])):
                wl.append({"session_key": x["session_key"], "observed_at": (st - timedelta(hours=hrs)).isoformat(),
                           "waitlist": n})
    pd.DataFrame(wl, columns=["session_key", "observed_at", "waitlist"]).to_csv(out / "waitlists.csv", index=False)
    # courthire: each venue's courts, 6am-10pm, every day; its sessions take
    # courts at their hour, and hire fills evenings more than mornings
    ch = []
    days = sorted({x["session_date"] for x in sess})
    prog = {}
    for x in sess:
        k = (int(x["venue_id"][4:]), x["session_date"], int(x["start_time"][:2]))
        prog[k] = prog.get(k, 0) + -(-int(x["capacity"]) // 4)
    for v in VENUES + [TINY]:
        n = 6 + int(v) % 4
        for dd in days:
            for hr in range(6, 22):
                for mm in (0, 30):
                    pg = min(n, prog.get((int(v), dd, hr), 0))
                    hire = min(n - pg, rnd.randint(0, n) if hr >= 17 else rnd.randint(0, 2))
                    ch.append({"fid": int(v), "slot_date": dd, "slot_start": f"{hr:02d}:{mm:02d}:00",
                               "n_courts": n, "taken": pg + hire, "programmed": pg, "hire_taken": hire,
                               "hireable": n - pg})
    pd.DataFrame(ch).to_csv(out / "courthire.csv", index=False)
    s.to_csv(out / "sessions.csv", index=False)
    r.to_csv(out / "rosters.csv", index=False)
    c.to_csv(out / "catalogue.csv", index=False)
    # firsts: each player's first session at each venue
    r2 = r.merge(s[["session_key", "venue_id", "session_date"]], on="session_key", suffixes=("", "_s"))
    r2["fid"] = r2["venue_id"].str.extract(r"(\d+)$")[0].astype(int)
    f = r2.sort_values("session_date").drop_duplicates(["pbp_user_id", "fid"])
    f[["pbp_user_id", "fid", "session_date", "session_key"]].rename(
        columns={"session_date": "first_date", "session_key": "first_key"}).to_csv(out / "firsts.csv", index=False)
    return out


if __name__ == "__main__":
    print(write(Path(sys.argv[1])))
