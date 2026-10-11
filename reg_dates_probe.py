"""
reg_dates_probe.py -- where does PlayByPoint say a tournament's entries close?

Read only. Uses the nightly scrape's own login and readers (PBP_COOKIES_JSON,
PlayByPointAPI, program pages) for a few tournament and league programmes at
each venue, and reports:
  1. every field on the programme list entry, the programme page and its
     first session whose NAME suggests a date or entry window (registration,
     deadline, close, cutoff, open, until, start, end, sign-up, enrol, book),
     with its value when it's a date or a number, otherwise the value's shape;
  2. sentences on the programme page that mention registration, entries or a
     deadline, with letters kept but any email or phone removed.
No player data is read (no rosters). Exit 0 always.

    python reg_dates_probe.py                      (venues 1009,1355,1379,1383,885)
    PROBE_FACILITIES=1009 PROBE_PER_VENUE=6 python reg_dates_probe.py
"""
from __future__ import annotations

import asyncio, html as _html, json, os, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import push_to_supabase as P
from extract_thejar import PlayByPointAPI, _extract_react_props_from_html

FACILITIES = [int(x) for x in os.environ.get("PROBE_FACILITIES", "1009,1355,1379,1383,885").split(",") if x.strip()]
PER_VENUE = int(os.environ.get("PROBE_PER_VENUE", "4"))
KEY = re.compile(r"regist|deadline|close|closing|cutoff|cut_off|open|until|start|end|sign|enrol|enroll|book|expire|last_day|available|sale|window|due", re.I)
DATEISH = re.compile(r"^\d{4}-\d{2}-\d{2}([T ][\d:.]+)?(Z|[+-]\d{2}:?\d{2})?$")
TEXT = re.compile(r"[^.!?\n]*(registration|registrations|register by|entries close|entry close|entries open|deadline|sign[- ]?ups? close|closes on|close on|cut[- ]?off)[^.!?\n]*[.!?]?", re.I)


def shape(v, n=40):
    if v is None: return "null"
    if isinstance(v, bool): return str(v).lower()
    if isinstance(v, (int, float)): return repr(v)
    if isinstance(v, str):
        if DATEISH.match(v.strip()): return repr(v.strip())
        s = re.sub(r"[A-Z]", "A", re.sub(r"[a-z]", "a", re.sub(r"\d", "9", v)))
        return '"' + re.sub(r"a{3,}", "aaa…", s)[:n] + '"'
    if isinstance(v, list): return f"list[{len(v)}]"
    if isinstance(v, dict): return f"object{{{len(v)}}}"
    return type(v).__name__


def walk(o, path, out, depth=0):
    if depth > 4: return
    if isinstance(o, dict):
        for k, v in o.items():
            p = f"{path}.{k}"
            if KEY.search(str(k)) and not isinstance(v, (dict, list)):
                out.append(f"      {re.sub(r'\[\d+\]', '[]', p):<58} {shape(v)}")
            if isinstance(v, (dict, list)):
                walk(v, p, out, depth + 1)
    elif isinstance(o, list):
        for i, x in enumerate(o[:1]):
            walk(x, f"{path}[{i}]", out, depth + 1)


def all_components(text: str) -> list[tuple[str, dict]]:
    """Every data-react-props blob on the page, with its component name, in
    any attribute order: the sessions component the scrape reads is one of
    several, and an entry deadline may sit in another."""
    out = []
    for tag in re.findall(r"<[^>]*data-react-class=[^>]*>", text or ""):
        c = re.search(r'data-react-class=["\']([^"\']+)["\']', tag)
        p = re.search(r'data-react-props="([^"]*)"', tag) or re.search(r"data-react-props='([^']*)'", tag)
        if c and p:
            try:
                out.append((c.group(1), json.loads(_html.unescape(p.group(1)))))
            except Exception:
                pass
    return out


def page_text(html: str) -> str:
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html or "", flags=re.S | re.I)
    t = _html.unescape(re.sub(r"<[^>]+>", " ", t))
    t = re.sub(r"\S+@\S+", "[email]", t)
    t = re.sub(r"\+?\d[\d\s()-]{7,}\d", "[phone]", t)
    return re.sub(r"\s+", " ", t)


async def main() -> int:
    cookies, user_id = P._load_cookies()
    print(f"Signed in: {'yes' if cookies else 'NO (no PBP_COOKIES_JSON)'}")
    found_keys: dict[str, int] = {}
    for fid in FACILITIES:
        slug = P.PBP_SLUG_MAP.get(fid)
        if not slug:
            print(f"\n== venue {fid}: not in the registry"); continue
        print(f"\n════ venue {fid} ════")
        async with PlayByPointAPI(cookies=cookies, club_slug=slug, proxy=P.PROXY_URL) as api:
            api._user_id = user_id
            try:
                stubs, _, _ = await P.fetch_clinic_stubs(api, fid)
            except Exception as e:
                print(f"  programme list unreadable: {type(e).__name__}"); continue
            comp = [s for s in stubs if P.is_competitive(s)][:PER_VENUE]
            print(f"  {len(stubs)} programmes, {len([s for s in stubs if P.is_competitive(s)])} tournament/league-like; reading {len(comp)}")
            for s in comp:
                url = s.get("url") or ""
                pslug = url.split("/programs/")[-1] if "/programs/" in url else ""
                print(f"\n  ── programme {shape(s.get('name'), 30)} (category {shape(s.get('category'))})")
                out: list[str] = []
                walk({"list_entry": s}, "", out)
                try:
                    html = await api.program_detail_html(pslug) if pslug else ""
                except Exception as e:
                    html = ""; print(f"    page unreadable: {type(e).__name__}")
                comps = all_components(html)
                print(f"    components on the page: {', '.join(sorted({c for c, _ in comps})) or '(none)'}")
                for cname, cprops in comps:
                    if cname.startswith("ClinicStepper"):
                        continue          # the sessions component: read below
                    walk({cname: cprops}, "", out)
                props = _extract_react_props_from_html(html) if html else None
                if isinstance(props, dict):
                    lessons = props.get("sessions") or props.get("clinic_lessons") or []
                    top = {k: v for k, v in props.items() if k not in ("sessions", "clinic_lessons")}
                    walk({"page": top}, "", out)
                    if lessons: walk({"first_session": lessons[0]}, "", out)
                    print(f"    page fields: {', '.join(sorted(top.keys()))[:600]}")
                    if lessons and isinstance(lessons[0], dict):
                        print(f"    session fields: {', '.join(sorted(lessons[0].keys()))[:600]}")
                for line in out:
                    k = line.split()[0]; found_keys[k] = found_keys.get(k, 0) + 1
                print("\n".join(out) or "      (no date-like fields)")
                said = [m.group(0).strip() for m in TEXT.finditer(page_text(html))][:4]
                for t in said:
                    print(f"    says: {t[:220]}")
                await asyncio.sleep(0.5)
    print("\nSUMMARY: date-like fields seen, how often")
    for k, n in sorted(found_keys.items(), key=lambda x: -x[1])[:40]:
        print(f"  {n:>3}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
