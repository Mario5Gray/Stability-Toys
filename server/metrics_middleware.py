"""HTTP request metrics as a pure ASGI middleware (STABL-xmsrxvto).

Deliberately NOT BaseHTTPMiddleware: that wraps every request in an anyio task
group, interacts badly with streaming responses and background tasks, and would
still need an explicit WebSocket exclusion. A plain ASGI callable is cheaper and
its skip condition is visible.

Spec: docs/superpowers/specs/2026-08-03-server-observability-seams-design.md §6
"""
import time

from server.metrics import get_metrics, record

UNMATCHED = "__unmatched__"


def route_label(scope) -> str:
    """The matched route TEMPLATE, never the raw path.

    `request.url.path` would make `/api/models/{name}` one series per model and
    `/v1/storage/{key}` one series per stored object. Starlette populates
    `scope["route"]` during routing, so this must be read AFTER the downstream
    app has run. A request that matched nothing has no route, and an unmatched
    path is precisely the unbounded set a scanner probes.
    """
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path if path else UNMATCHED


class MetricsMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        # WebSocket and lifespan scopes have no status and no duration worth
        # charging to an HTTP histogram. The gate is re-read per request rather
        # than cached so reset_metrics() in tests takes effect.
        if scope["type"] != "http" or not get_metrics().enabled:
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        # 500 unless the app says otherwise: an unhandled exception never sends
        # http.response.start through us, and a request that raised still
        # happened — dropping it would hide exactly the traffic being looked for.
        status = {"code": 500}

        async def _send(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, _send)
        finally:
            elapsed = time.perf_counter() - start
            method = scope.get("method", "UNKNOWN")
            route = route_label(scope)
            code = str(status["code"])
            record(lambda met: (
                met.http_requests_total.labels(
                    method=method, route=route, status=code).inc(),
                met.http_request_duration_seconds.labels(
                    method=method, route=route).observe(elapsed),
            ))
