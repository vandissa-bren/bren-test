"""
test_venue_source.py -- venue_registry reads the one venue list from the site
and never lets the site's state stop the backend.

Run: python3 test_venue_source.py

Serves venue files from a local HTTP server and checks each case:
  site good           -> used, source 'site'
  site edited         -> picked up on the next refresh, no restart
  bad edit on site    -> rejected whole; the last good list is kept
  site gives HTML     -> (file not deployed yet) committed copy is used
  site unreachable    -> committed copy is used
"""
import http.server, importlib, json, os, sys, threading, time

failures = []
def check(label, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}" + ("" if cond else f"  {detail}"))
    if not cond: failures.append(label)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = json.load(open(os.path.join(HERE, "venues.json")))
served = {"body": json.dumps(BASE).encode(), "type": "application/json"}

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", served["type"])
        self.end_headers()
        self.wfile.write(served["body"])
    def log_message(self, *a): pass

srv = http.server.HTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_port}/venues.json"

def fresh(source_url, refresh="600"):
    os.environ["VENUE_SOURCE_URL"] = source_url
    os.environ["VENUE_REFRESH_SECONDS"] = refresh
    import venue_registry
    return importlib.reload(venue_registry)

def with_venue(extra):
    doc = json.loads(json.dumps(BASE)); doc["venues"].append(extra); return doc

NEW = {"id": "9999", "name": "Test Courts", "facilityId": 9999, "slug": "testcourts",
       "platform": "playbypoint", "status": "active", "fetcher": "court_blocks",
       "bookingEnabled": True, "city": "Nowhere", "address": "1 Test St", "lat": -37.8, "lng": 145.0}

print("\n-- site good --")
reg = fresh(url)
n = len(reg.active_venues())
check("reads the site", reg.source() == "site", reg.source())
check("same venues as the committed copy", n == 22, n)
check("map-only venues are not backend venues", isinstance(reg.resolve("map-2"), reg.Unresolved))

print("\n-- a venue added on the site arrives on the next refresh --")
reg = fresh(url, refresh="0")
reg.active_venues()
served["body"] = json.dumps(with_venue(NEW)).encode()
time.sleep(0.01)
v = reg.resolve(9999)
check("new venue resolves without a restart", isinstance(v, reg.Venue) and v.slug == "testcourts", v)
check("and the scrapers would fetch it", 9999 in {x.facility_id for x in reg.venues_for_fetcher("court_blocks")})

print("\n-- a bad edit on the site is rejected whole --")
bad = with_venue(dict(NEW, status="actve"))
served["body"] = json.dumps(bad).encode()
reg._next_check = 0
v = reg.resolve(9999)
check("last good list kept (venue still there, typo not applied)", isinstance(v, reg.Venue) and v.status == "active", v)
check("still reports the site as its source", reg.source() == "site")

print("\n-- the site answers with its HTML shell (file not deployed yet) --")
served["body"], served["type"] = b"<!doctype html><html></html>", "text/html"
reg = fresh(url)
check("falls back to the committed copy", reg.source() == "snapshot", reg.source())
check("with the full list", len(reg.active_venues()) == 22)

print("\n-- the site is unreachable --")
srv.shutdown()
reg = fresh("http://127.0.0.1:9/venues.json")
check("falls back to the committed copy", reg.source() == "snapshot", reg.source())
check("booking lookups still work", reg.get_bookable_venue(597).slug == "nplpickleball")

print("\n-- VENUE_SOURCE_URL='' uses only the committed copy --")
reg = fresh("")
check("no network, committed copy", reg.source() == "snapshot")

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}"); sys.exit(1)
print("all checks passed")
