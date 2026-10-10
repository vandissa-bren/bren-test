"""
coach_names_probe.py -- does PlayByPoint hold coaches' full names anywhere we can read?

PlayByPoint's session data names each coach as an initial and surname
(`teacher_names`: "S. Rivera"; all 36 coaches on 10 Oct). Its "Book a Pro"
pages (/pros/<first-last>) suggest it knows the full names. This looks, read
only, in the places the nightly scrape can reach:

  1. a few program pages' embedded data (every field on a session, and
     anything under a teacher / coach / pro / instructor / staff key)
  2. the venue's pros page, which loads its list from an address in
     PlayByPoint's JavaScript: those addresses (addresses only), each tried
     signed out, and signed in with the scrape's login when PBP_COOKIES_JSON
     is set (never printed)

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
        gpath = re.sub(r"\[\d+\]", "[]", path)
        fn, ln = o.get("first_name"), o.get("last_name")
        if COACHY.search(gpath) and isinstance(fn, str) and isinstance(ln, str) and len(fn.strip()) >= 2 and len(ln.strip()) >= 2:
            found_full.add(f"{gpath}.first_name + last_name")
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
    """Every data-react-props blob on a page, with its component name, whatever
    order and other attributes the tag has (the props are HTML-escaped, so a
    tag never contains a raw '>')."""
    out = []
    for tag in re.findall(r"<[^>]*data-react-class=[^>]*>", text or ""):
        c = re.search(r'data-react-class=["\']([^"\']+)["\']', tag)
        p = re.search(r'data-react-props="([^"]*)"', tag) or re.search(r"data-react-props='([^']*)'", tag)
        if not (c and p):
            continue
        try:
            out.append((c.group(1), json.loads(_html.unescape(p.group(1)))))
        except Exception:
            pass
    return out


def api_paths_in_scripts(s, page_html: str) -> list[str]:
    """Addresses the page's own JavaScript calls that mention teachers or pros.
    Addresses only (they hold no personal data)."""
    found: set[str] = set()
    for src in re.findall(r'<script[^>]+src="([^"]+)"', page_html or "")[:12]:
        url = src if src.startswith("http") else f"{APP}{src}"
        r = get(s, url)
        if r is None or r.status_code != 200:
            continue
        for m in re.findall(r'["\'`](/api/[^"\'`\s]{0,120})["\'`]', r.text or ""):
            if re.search(r"teach|pro\b|pros|coach|instructor", m, re.I):
                found.add(m)
    return sorted(found)


def session(cookies: dict | None = None):
    s = cr.Session(impersonate="chrome", timeout=30, verify=os.environ.get("PROBE_CA_BUNDLE") or True)
    s.headers.update({"Accept": "application/json, text/plain, */*", "X-Requested-With": "XMLHttpRequest",
                      "Origin": APP, "Referer": f"{APP}/home"})
    for k, v in (cookies or {}).items():
        s.cookies.set(k, v, domain=".playbypoint.com")
    return s


def load_cookies() -> dict:
    """The scrape's PlayByPoint login (PBP_COOKIES_JSON), if supplied. Never printed."""
    try:
        return (json.loads(os.environ.get("PBP_COOKIES_JSON") or "{}") or {}).get("cookies") or {}
    except Exception:
        return {}


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


def lesson_lists(o, depth=0):
    """Lists of session-like objects anywhere in a page's props."""
    if depth > 5:
        return
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(v, list) and v and isinstance(v[0], dict) and ("lesson_date" in v[0] or "teacher_names" in v[0] or "hour_start" in v[0]):
                yield k, v
            else:
                yield from lesson_lists(v, depth + 1)
    elif isinstance(o, list):
        for x in o[:3]:
            yield from lesson_lists(x, depth + 1)


def show(r, label: str):
    print(f"   {label:<44} {describe(r)}")
    if r is None or r.status_code != 200:
        return
    out2, seen2 = [], set()
    t = r.text or ""
    if t.lstrip().startswith(("{", "[")):
        try:
            data = r.json()
            first = data[0] if isinstance(data, list) and data else data
            if isinstance(first, dict):
                print(f"      fields: {', '.join(sorted(first.keys())[:40])}")
            # an answer from a coach-list address is coach data throughout
            walk({"teachers" if COACHY.search(label) else "json": data}, "", 0, out2, seen2)
            if isinstance(data, list):
                print(f"      {len(data)} items")
        except Exception:
            pass
    for comp, props in react_props(t):
        walk({comp: props}, "", 0, out2, seen2)
    if out2:
        print("\n".join(out2[:40]))


def main() -> int:
    s = session()
    cookies = load_cookies()
    auth = session(cookies) if cookies else None
    print(f"Signed-in tries: {'yes (PBP_COOKIES_JSON supplied)' if auth else 'no (no PBP_COOKIES_JSON)'}")
    for fid in FACILITIES:
        print(f"\n════ venue {fid} ════")
        r = get(s, f"{APP}/api/public/clinics?search=&facility_id={fid}&per_page=50")
        print(f"  clinic list: {describe(r)}")
        stubs = []
        try:
            d = r.json() if r is not None else None
            stubs = (d or {}).get("clinics") or [] if isinstance(d, dict) else (d or [])
        except Exception:
            pass

        # program pages: the sessions' own fields, wherever they sit
        slugs = [(x.get("url") or "").split("/programs/")[-1] for x in stubs if "/programs/" in (x.get("url") or "")]
        print(f"\n  program pages (first {PROGRAMS_PER_VENUE}):")
        lesson_keys: set[str] = set()
        comps: set[str] = set()
        out, seen = [], set()
        named = 0
        for slug in slugs[:PROGRAMS_PER_VENUE]:
            rp = get(s, f"{APP}/programs/{slug}", accept_html=True)
            for comp, props in react_props(rp.text if rp is not None else ""):
                comps.add(comp)
                for key, lessons in lesson_lists(props):
                    lesson_keys.update(lessons[0].keys())
                    named += sum(1 for x in lessons if x.get("teacher_names"))
                walk({comp: props}, "", 0, out, seen)
        print(f"  components: {', '.join(sorted(comps)) or '(none found)'}")
        if lesson_keys:
            print(f"  fields on a session: {', '.join(sorted(lesson_keys))}")
            print(f"  sessions naming a coach: {named}")
        print("  teacher-like fields on program pages:")
        print("\n".join(out) or "    (none)")

        # the pros page, and the addresses its JavaScript calls
        print("\n  pros list:")
        rp = get(s, f"{APP}/pros?facility_id={fid}", accept_html=True)
        show(rp, "pros page")
        paths = api_paths_in_scripts(s, rp.text if rp is not None else "")
        print(f"   addresses in the page's scripts mentioning teachers/pros: {len(paths)}")
        for pth in paths[:25]:
            print(f"      {pth}")
        tries = {p.replace("${", "{").split("?")[0] for p in paths}
        tries |= {"/api/facilities/{id}/teachers", "/api/teachers", "/api/public/facilities/{id}/teachers"}
        for pth in sorted(tries)[:20]:
            url = APP + re.sub(r"\{[^}]*\}|:\w+", str(fid), pth)
            url += ("&" if "?" in url else "?") + f"facility_id={fid}"
            show(get(s, url), f"signed out {pth}")
            if auth:
                show(get(auth, url), f"signed in  {pth}")

    print("\nVERDICT")
    if found_full:
        print("  A full-name-shaped value sits under these coach fields:")
        for p in sorted(found_full):
            print(f"    {p}")
        print("  If one of these is the coach, the scrape can read full names from it.")
    else:
        print("  No full-name-shaped value under any coach field in what was readable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
