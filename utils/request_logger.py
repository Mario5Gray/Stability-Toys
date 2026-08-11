# request_logger.py
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Optional, Set, Dict, Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from utils.env import env_bool, env_int, env_list

# NOTE: the former local `_env_bool` moved to utils.env.env_bool, semantics
# preserved VERBATIM — everything except 0/false/False/no/No is true, empty and
# 'off' and 'FALSE' included. Those oddities are deliberate; normalising them is
# STABL-cfyshjre. The reads below now go through utils.env so a quoted value
# behaves the same under `docker compose env_file` (strips quotes) and
# `docker run --env-file` (does not) — env.dev is loaded BOTH ways, which is how
# LOG_HEADER_ALLOWLIST came to have a literal quote on its first entry under
# runner.sh (STABL-voqsoicx).


@dataclass
class RequestLoggerConfig:
    enabled: bool = field(default_factory=lambda: env_bool("LOG_REQUESTS", True))
    log_headers: bool = field(default_factory=lambda: env_bool("LOG_REQUEST_HEADERS", True))
    log_body: bool = field(default_factory=lambda: env_bool("LOG_REQUEST_BODY", True))

    # bytes of body to log (json/text). multipart becomes summary only.
    body_max: int = field(default_factory=lambda: env_int("LOG_BODY_MAX", 8192))

    # only log these headers (lowercase). keep minimal by default.
    header_allowlist: Set[str] = field(
        default_factory=lambda: {
            h.lower()
            for h in env_list(
                "LOG_HEADER_ALLOWLIST",
                "content-type,content-length,x-forwarded-for,x-real-ip,user-agent,host",
            )
        }
    )

    # redact sensitive headers even if allowlisted
    redact_headers: Set[str] = field(default_factory=lambda: {"authorization", "cookie"})

    # optional: only log these paths (comma-separated prefix match), e.g. "/generate,/superres"
    path_prefix_allowlist: Optional[Set[str]] = field(
        default_factory=lambda: set(env_list("LOG_PATH_PREFIXES")) or None
    )

    # optional: skip these paths (comma-separated prefix match), e.g. "/health,/docs"
    path_prefix_denylist: Set[str] = field(
        default_factory=lambda: set(env_list("LOG_PATH_DENYLIST", "/docs,/openapi.json"))
    )


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    """
    Logs incoming requests (method/path/query), selected headers, and a safe body summary.
    Safe for FastAPI because it reads body once and re-injects it into the request stream.
    """

    def __init__(self, app, config: Optional[RequestLoggerConfig] = None):
        super().__init__(app)
        self.cfg = config or RequestLoggerConfig()

    def _want_log(self, request: Request) -> bool:
        if not self.cfg.enabled:
            return False

        path = request.url.path

        for p in self.cfg.path_prefix_denylist:
            if p and path.startswith(p):
                return False

        if self.cfg.path_prefix_allowlist is None:
            return True

        for p in self.cfg.path_prefix_allowlist:
            if p and path.startswith(p):
                return True

        return False

    def _redact_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for k, v in headers.items():
            lk = k.lower()
            if lk not in self.cfg.header_allowlist:
                continue
            if lk in self.cfg.redact_headers:
                out[k] = "***redacted***"
            else:
                out[k] = v
        return out

    def _summarize_body(self, content_type: str, body: bytes) -> str:
        if not body:
            return ""

        # multipart: do not dump binary; just size
        if "multipart/form-data" in content_type:
            return f"<multipart body: {len(body)} bytes>"

        snippet = body[: self.cfg.body_max]
        truncated = "…(truncated)" if len(body) > self.cfg.body_max else ""

        # JSON: parse and compact if possible
        if "application/json" in content_type:
            try:
                obj = json.loads(snippet.decode("utf-8", errors="replace"))
                return json.dumps(obj, ensure_ascii=False) + truncated
            except Exception:
                return snippet.decode("utf-8", errors="replace") + truncated

        # text-ish default
        return snippet.decode("utf-8", errors="replace") + truncated

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if not self._want_log(request):
            return await call_next(request)

        t0 = time.time()

        # Read body once, then re-inject it so downstream can read it again.
        body = await request.body()

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        request = Request(request.scope, receive)

        ct = (request.headers.get("content-type") or "").lower()

        payload: Dict[str, Any] = {
            "remote": request.client.host if request.client else None,
            "method": request.method,
            "path": request.url.path,
            "query": str(request.url.query) if request.url.query else "",
        }

        if self.cfg.log_headers:
            payload["headers"] = self._redact_headers(dict(request.headers))

        if self.cfg.log_body:
            payload["body"] = self._summarize_body(ct, body)

        print("[REQ]", json.dumps(payload, ensure_ascii=False))

        resp = await call_next(request)

        dt_ms = int((time.time() - t0) * 1000)
        print(f"[RESP] {request.method} {request.url.path} -> {resp.status_code} ({dt_ms}ms)")

        return resp


class RequestLogger:
    """
    Convenience installer so your server file stays clean:
      RequestLogger.install(app)
    """

    @staticmethod
    def install(app, config: Optional[RequestLoggerConfig] = None) -> None:
        app.add_middleware(RequestLoggerMiddleware, config=config)