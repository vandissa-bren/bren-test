"""
auth_guard.py — one guard in front of every booking-server endpoint that acts for a
player (book, cancel, bookings, balance, coupons, court prices, sync, member prices).

A request that names a player — {user_id} in the path, user_id in the query or in the
JSON body — is allowed only when it carries EITHER
  · that player's own sign-in token (Authorization: Bearer <Supabase access token>),
    verified by Supabase itself (GET /auth/v1/user), never just decoded, or
  · the server key (X-Internal-Key = SUPABASE_SERVICE_KEY), for server-to-server calls
    (the daily refresh job, the API server's price lookups).

AUTH_GUARD_MODE (in /app/.env):
  log      (default) change nothing; log every request that WOULD be refused
  enforce  refuse them with 401
  off      the guard does nothing

Wire-up in booking_server.py (two lines, immediately BEFORE the CORS line, so the CORS
headers still wrap a 401):
    from auth_guard import AuthGuard
    app.add_middleware(AuthGuard)

Logs never contain a token or a key; a player id is shortened to 8 characters.
"""
from __future__ import annotations
import hmac, json, logging, os, re, time
from urllib.parse import parse_qs

import httpx

log = logging.getLogger("uvicorn")
UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
EXEMPT = {"/health"}
CACHE_TTL = 300            # a verified token is trusted for 5 minutes
_cache: dict[str, tuple[str, float]] = {}


def _mode() -> str:
    m = os.environ.get("AUTH_GUARD_MODE", "log").strip().lower()
    return m if m in ("log", "enforce", "off") else "log"


async def verify_token(token: str) -> str | None:
    """The Supabase user id this access token belongs to, or None. Asks Supabase Auth,
    so a forged or expired token fails (the old check only decoded it)."""
    now = time.time()
    hit = _cache.get(token)
    if hit and hit[1] > now:
        return hit[0]
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        return None
    try:
        async with httpx.AsyncClient(timeout=6.0) as c:
            r = await c.get(f"{url}/auth/v1/user", headers={"apikey": key, "Authorization": f"Bearer {token}"})
        uid = (r.json() or {}).get("id") if r.status_code == 200 else None
    except Exception:
        return None
    if uid:
        if len(_cache) > 2000:
            _cache.clear()
        _cache[token] = (uid, now + CACHE_TTL)
    return uid


def _named_players(path: str, query: bytes, body: bytes, content_type: str) -> set[str]:
    ids = {m.group(0).lower() for m in UUID.finditer(path)}
    for v in parse_qs(query.decode("latin-1")).get("user_id", []):
        ids.add(v.lower())
    if body and "json" in content_type:
        try:
            data = json.loads(body)
            if isinstance(data, dict) and isinstance(data.get("user_id"), str):
                ids.add(data["user_id"].lower())
        except ValueError:
            pass
    return ids


class AuthGuard:
    """Pure ASGI middleware (reads the body once and hands it on unchanged)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        mode = _mode()
        if (scope.get("type") != "http" or mode == "off" or scope.get("method") == "OPTIONS"
                or not scope.get("path", "").startswith("/api/") or scope.get("path") in EXEMPT):
            return await self.app(scope, receive, send)

        # read the whole body, then replay it to the app
        chunks, more = [], True
        while more:
            msg = await receive()
            if msg["type"] != "http.request":
                return await self.app(scope, receive, send)
            chunks.append(msg.get("body", b""))
            more = msg.get("more_body", False)
        body = b"".join(chunks)
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        players = _named_players(scope["path"], scope.get("query_string", b""), body, headers.get("content-type", ""))
        if not players:
            return await self.app(scope, replay, send)

        why = await self._refusal(headers, players)
        if why is None:
            return await self.app(scope, replay, send)

        who = ",".join(sorted(p[:8] for p in players))
        where = UUID.sub(lambda m: m.group(0)[:8] + "…", scope["path"])[:60]
        if mode == "enforce":
            log.warning(f"AUTH-GUARD refused {scope['method']} {where} player={who}: {why}")
            payload = json.dumps({"detail": "Not signed in as this player."}).encode()
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-type", b"application/json"),
                                    (b"content-length", str(len(payload)).encode())]})
            await send({"type": "http.response.body", "body": payload})
            return
        log.warning(f"AUTH-GUARD would-refuse {scope['method']} {where} player={who}: {why}")
        return await self.app(scope, replay, send)

    @staticmethod
    async def _refusal(headers: dict, players: set) -> str | None:
        """None when allowed; otherwise the reason (no secrets in it)."""
        key = os.environ.get("SUPABASE_SERVICE_KEY", "")
        given = headers.get("x-internal-key", "")
        if key and given and hmac.compare_digest(given.encode(), key.encode()):
            return None                                        # server-to-server
        auth = headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            return "no sign-in token"
        uid = await verify_token(auth.split(" ", 1)[1].strip())
        if not uid:
            return "token not accepted by Supabase"
        if players != {uid.lower()}:
            return "token belongs to a different player"
        return None
