"""HTTP request metrics and entry span as a pure ASGI middleware.

Deliberately NOT BaseHTTPMiddleware: that wraps every request in an anyio task
group, interacts badly with streaming responses and background tasks, and would
still need an explicit WebSocket exclusion. A plain ASGI callable is cheaper and
its skip condition is visible.

Carries BOTH observability pillars for HTTP (STABL-xmsrxvto metrics,
STABL-qnlaclof tracing). One middleware rather than two: they need the same
`scope["route"]`-after-the-app read, and a second pass over every request to
recompute it would be pure cost. The class keeps its name because it is
registered by it; what it does is described here.

Specs: docs/superpowers/specs/2026-08-03-server-observability-seams-design.md §6
       docs/superpowers/specs/2026-08-12-tracing-span-map-and-boundary-fixes.md §3.1
"""
import time

from server.metrics import get_metrics, record
from server.tracing import get_tracer, get_tracing

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
        # charging to an HTTP histogram. The gates are re-read per request rather
        # than cached so reset_metrics()/reset_tracing() in tests take effect, and
        # they are checked SEPARATELY: tracing must not silently require
        # METRICS_ENABLED, which is a coupling nobody would think to look for.
        if scope["type"] != "http" or not (get_metrics().enabled or get_tracing().enabled):
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

        method = scope.get("method", "UNKNOWN")
        # Named by method alone at creation and renamed once the route is known.
        # The template cannot be read before the app has run, and a span name
        # built from the raw path would be the cardinality trap that
        # route_label() exists to avoid — one operation per model id.
        with get_tracer(__name__).start_as_current_span(f"HTTP {method}") as span:
            try:
                await self.app(scope, receive, _send)
            finally:
                elapsed = time.perf_counter() - start
                route = route_label(scope)
                code = str(status["code"])
                record(lambda met: (
                    met.http_requests_total.labels(
                        method=method, route=route, status=code).inc(),
                    met.http_request_duration_seconds.labels(
                        method=method, route=route).observe(elapsed),
                ))
                span.set_attribute("http.request.method", method)
                span.set_attribute("http.route", route)
                span.set_attribute("http.response.status_code", status["code"])
                span.update_name(f"{method} {route}")
