"""
coach_names_probe.py -- does PlayByPoint hold coaches' full names anywhere we can read?

PlayByPoint's session data names each coach as an initial and surname
(`teacher_names`: "S. Rivera"; all 36 coaches on 10 Oct). Its "Book a Pro"
pages (/pros/<first-last>) suggest it knows the full names. This looks, read
only and signed out, in the places the nightly scrape can reach:

  1. the public clinic list for a venue (every field on a clinic)
  2. a few program pages' embedded data (every field on a session, and
     anything under a teacher / coach / pro / instructor / staff key)
  3. a handful of likely "pros" addresses for the venue

IT NEVER PRINTS A NAME. Field names are printed as they are; every value is
printed as its SHAPE, letters masked: "Aaaa Aaaaaa" is a full name,
"A. Aaaaaa" an initial and surname, "999" a number. So the log is safe to
paste back.

    python coach_names_probe.py                  (venues 1379 and 885)
    PROBE_FACILITIES=597,1355 python coach_names_probe.py

Exit code 0 always: it's a look, not a check.
"""
from __future__ import annotations

import html as _html
import json
import os
import re
import sys

from curl_cffi import requests as cr

APP = "https://app.playbypoint.com"
FACILITIES = [int(x) for x in os.environ.get("PROBE_FACILITIES", "1379,885").split(",") if x.strip()]
PROGRAMS_PER_VENUE = int(os.environ.get("PROBE_PROGRAMS", "4"))
COACHY = re.compile(r"teach|coach|pro\b|pros\b|instructor|staff|trainer|host", re.I)
INTERESTING = re.compile(r"teach|coach|pro\b|pros\b|instructor|staff|trainer|host|owner|first_?name|last_?name|full_?name|display_?name|^name$", re.I)


def mask(v, n: int = 40) -> str:
    """A value's shape, never the value: letters -> A/a, digits -> 9."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        s = re.sub(r"[A-Z]", "A", re.sub(r"[a-z]", "a", re.sub(r"[0-9]", "9", v)))
        s = re.sub(r"a{3,}", "aaa…", s)
        return f'"{s[:n]}"' + ("…" if len(s) > n else "")
    if isinstance(v, list):
        return f"list[{len(v)}]"
    if isinstance(v, dict):
        return f"object{{{len(v)}}}"
    return type(v).__name__


def looks_full(v) -> bool:
    return isinstance(v, str) and bool(re.fullmatch(r"[A-Za-z'’-]{2,}(\s+[A-Za-z'’-]{2,})+", v.strip()))


def looks_initial(v) -> bool:
    return isinstance(v, str) and bool(re.fullmatch(r"[A-Za-z]\.?\s*[A-Za-z'’-]{2,}", v.strip()))


found_full: set[str] = set()     # paths where a full-name-shaped value sat under an interesting key


def walk(o, path: str, depth: int, out: list, seen_paths: set):
    """Print interesting paths with value shapes; recurse a little."""
    if depth > 6:
        return
    if isinstance(o, dict):
        for k, v in o.items():
            p = f"{path}.{k}"
            gp = re.sub(r"\[\d+\]", "[]", p)
            if INTERESTING.search(str(k)) and gp not in seen_paths:
                seen_paths.add(gp)
                extra = ""
                if isinstance(v, list) and v:
                    extra = f"  first item: {mask(v[0])}"
                    if isinstance(v[0], dict):
                        extra += "  keys: " + ", ".join(sorted(v[0].keys())[:30])
                elif isinstance(v, dict):
                    extra = "  keys: " + ", ".join(sorted(v.keys())[:30])
                out.append(f"    {gp:<60} {mask(v)}{extra}")
            # only under a coach-like field: a programme's own `name` doesn't count
            if INTERESTING.search(str(k)) and COACHY.search(gp):
                vals = v if isinstance(v, list) else [v]
                for x in vals:
                    if looks_full(x):
                        found_full.add(gp)
                    if isinstance(x, dict):
                        for kk, vv in x.items():
                            if looks_full(vv):
                                found_full.add(f"{gp}[].{kk}")
            walk(v, p, depth + 1, out, seen_paths)
    elif isinstance(o, list):
        for i, x in enumerate(o[:3]):
            walk(x, f"{path}[{i}]", depth + 1, out, seen_paths)


def react_props(text: str) -> list[tuple[str, dict]]:
    """Every data-react-props blob on a page, with its component name."""
    out = []
    for m in re.finditer(r'data-react-class="([^"]+)"\s+data-react-props="([^"]*)"', text or ""):
        try:
            out.append((m.group(1), json.loads(_html.unescape(m.group(2)))))
        except Exception:
            pass
    for m in re.finditer(r'data-react-props="([^"]*)"\s+data-react-class="([^"]+)"', text or ""):
        try:
            out.append((m.group(2), json.loads(_html.unescape(m.group(1)))))
        except Exception:
            pass
    return out


def session():
    s = cr.Session(impersonate="chrome", timeout=30, verify=os.environ.get("PROBE_CA_BUNDLE") or True)
    s.headers.update({"Accept": "application/json, text/plain, */*", "X-Requested-With": "XMLHttpRequest",
                      "Origin": APP, "Referer": f"{APP}/home"})
    return s


def get(s, url, accept_html=False):
    try:
        h = {"Accept": "text/html"} if accept_html else {}
        r = s.get(url, headers=h, allow_redirects=False)
        return r
    except Exception as e:
        print(f"    ERROR {type(e).__name__}: {str(e)[:100]}")
        return None


def describe(r) -> str:
    if r is None:
        return "no answer"
    t = r.text or ""
    k = ("CLOUDFLARE" if (r.headers.get("cf-mitigated") or "Just a moment" in t) else
         "json" if t.lstrip().startswith(("{", "[")) else
         "page with data" if "data-react-props" in t else
         "empty" if t.strip() in ("", "null") else "html")
    loc = r.headers.get("location")
    return f"{r.status_code} {k}{f' -> {loc[:60]}' if loc else ''}"


def main() -> int:
    s = session()
    for fid in FACILITIES:
        print(f"\n════ venue {fid} ════")
        # 1. the clinic list
        r = get(s, f"{APP}/api/public/clinics?search=&facility_id={fid}&per_page=50")
        print(f"  clinic list: {describe(r)}")
        stubs = []
        try:
            d = r.json() if r is not None else None
            stubs = (d or {}).get("clinics") or [] if isinstance(d, dict) else (d or [])
        except Exception:
            pass
        if stubs:
            print(f"  {len(stubs)} clinics; fields on a clinic: {', '.join(sorted(stubs[0].keys()))}")
            out, seen = [], set()
            walk({"clinic": stubs[0]}, "", 0, out, seen)
            print("\n".join(out) or "    (nothing teacher-like on a clinic)")

        # 2. program pages
        slugs = [(x.get("url") or "").split("/programs/")[-1] for x in stubs if "/programs/" in (x.get("url") or "")]
        print(f"\n  program pages (first {PROGRAMS_PER_VENUE}):")
        lesson_keys: set[str] = set()
        out, seen = [], set()
        for slug in slugs[:PROGRAMS_PER_VENUE]:
            rp = get(s, f"{APP}/programs/{slug}", accept_html=True)
            print(f"   a program page: {describe(rp)}")
            for comp, props in react_props(rp.text if rp is not None else ""):
                lessons = props.get("sessions") or props.get("clinic_lessons") or []
                if isinstance(lessons, list) and lessons and isinstance(lessons[0], dict):
                    lesson_keys.update(lessons[0].keys())
                walk({comp: props}, "", 0, out, seen)
        if lesson_keys:
            print(f"  fields on a session: {', '.join(sorted(lesson_keys))}")
        print("  teacher-like fields on program pages:")
        print("\n".join(out) or "    (none)")

        # 3. likely "pros" addresses
        print("\n  pros addresses:")
        for label, url, as_html in [
            ("pros page", f"{APP}/pros?facility_id={fid}", True),
            ("api/public/pros", f"{APP}/api/public/pros?facility_id={fid}", False),
            ("api/pros", f"{APP}/api/pros?facility_id={fid}", False),
            ("api/facilities/{id}/pros", f"{APP}/api/facilities/{fid}/pros", False),
            ("api/public/facilities/{id}/pros", f"{APP}/api/public/facilities/{fid}/pros", False),
            ("api/facilities/{id}/teachers", f"{APP}/api/facilities/{fid}/teachers", False),
            ("api/public/teachers", f"{APP}/api/public/teachers?facility_id={fid}", False),
            ("api/facilities/{id}", f"{APP}/api/facilities/{fid}", False),
        ]:
            rr = get(s, url, accept_html=as_html)
            print(f"   {label:<34} {describe(rr)}")
            if rr is None or rr.status_code != 200:
                continue
            out2, seen2 = [], set()
            t = rr.text or ""
            if t.lstrip().startswith(("{", "[")):
                try:
                    walk({"json": rr.json()}, "", 0, out2, seen2)
                except Exception:
                    pass
            for comp, props in react_props(t):
                walk({comp: props}, "", 0, out2, seen2)
            pro_links = len(set(re.findall(r'/pros/[a-z0-9-]+', t)))
            if pro_links:
                print(f"      links to {pro_links} pro pages")
            if out2:
                print("\n".join(out2[:40]))

    print("\nVERDICT")
    if found_full:
        print("  A full-name-shaped value sits under these teacher-like fields:")
        for p in sorted(found_full):
            print(f"    {p}")
        print("  If one of these is the coach, the scrape can read full names from it.")
    else:
        print("  No full-name-shaped value under any teacher-like field in what's readable signed out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
