# HTTP and WebSocket Prometheus Metrics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, or execute it directly. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Do NOT use superpowers:subagent-driven-development.** `AGENTS.md` forbids sub-agent driven development in this repo.

**FP:** STABL-xmsrxvto (child of STABL-oxbwjwvu, depends on STABL-asawxgvp — merged as `dd633e6`)
**Spec:** `docs/superpowers/specs/2026-08-03-server-observability-seams-design.md` §6 (approved)
**Contract:** `docs/observability-contract.md` — extended by this issue

**Goal:** Answer request-rate, latency, error-rate and WebSocket-activity questions from Prometheus alone.

**Architecture:** Five new families on the existing `server/metrics.py` facade. HTTP is one **pure ASGI middleware**; WebSocket instrumentation splits between `ws_hub.py` (connections and every outbound message) and the `ws_routes.py` message loop (inbound). No new infrastructure, no new dependency.

**Tech Stack:** Python 3, FastAPI/Starlette, `prometheus_client` (already pinned), pytest.

## Global Constraints

- **Labels stay bounded.** `method`, `route`, `status`, `type`, `direction` only. No `job_id`, no `pid`, no `hostname`, no `client_id`, no raw paths.
- **`route` is the matched route TEMPLATE**, never `request.url.path`. Verified: `scope["route"].path` gives `/api/models/{name}`; an unmatched request has `scope["route"] is None` and is labelled `__unmatched__`.
- **A client-supplied string must never become a label value.** WS message `type` maps through the `HANDLERS` registry; anything unrecognised collapses to a fixed value.
- **`METRICS_ENABLED` still gates everything** — the facade's no-op objects make every new call site inert by default.
- **Instrumentation must never break a request or drop a WS connection.** New call sites go through one guard.
- **`server/metrics.py` remains the only module importing `prometheus_client`.**
- Python env: `conda activate stability-toys`, then `python` (not `python3`).

---

## Findings that shaped this plan

**`scope["route"]` behaves as the spec assumed — verified, not presumed:**

```text
/api/models/SDXL        -> /api/models/{name}   200
/api/models/HunyuanDiT  -> /api/models/{name}   200
/nope/does/not/exist    -> None                 404
```

Two model names collapse to one series, and the `__unmatched__` fallback is load-bearing
rather than defensive — an unmatched path is exactly the unbounded set a scanner probes.

**Outbound WS messages do not all pass through `websocket_endpoint`.** `hub.broadcast()`
is called from `_status_broadcaster` every 5s, entirely outside the endpoint's message
loop. Instrumenting only the endpoint would make the single most frequent outbound
message invisible. So outbound is instrumented in **`ws_hub.py`**, which both paths share.

**`connections_active` is SET from `hub.client_count`, not incremented.** The hub already
owns the authoritative count; an inc/dec pair drifts the first time a disconnect path is
missed, and `broadcast()` has its own dead-client reaping path that calls `disconnect`.
Setting from the source cannot drift.

**Pure ASGI middleware, not `BaseHTTPMiddleware`.** The latter wraps every request in an
anyio task group, interacts badly with background tasks and streaming, and would need its
own WebSocket exclusion anyway. A plain ASGI callable is cheaper and its skip condition
(`scope["type"] != "http"`) is explicit. The app has no streaming responses today, but the
middleware should not be the thing that prevents adding one.

---

## File Structure

| File | Responsibility |
|---|---|
| `server/metrics.py` (modify) | five new families + one shared `record()` guard |
| `server/metrics_middleware.py` (create) | the HTTP ASGI middleware |
| `server/ws_hub.py` (modify) | connections gauge, sessions counter, outbound messages |
| `server/ws_routes.py` (modify) | inbound messages at the dispatch point |
| `server/lcm_sr_server.py` (modify) | add the middleware |
| `docs/observability-contract.md` (modify) | the five new entries |
| `tests/test_metrics_http.py` (create) | middleware behaviour |
| `tests/test_metrics_ws.py` (create) | hub + message-loop behaviour |
| `tests/test_metrics.py` (modify) | family list + contract test already cover the new families |

---

## Task 1: Families and the shared guard

**Files:**
- Modify: `server/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `Metrics`, `get_metrics()` from STABL-asawxgvp.
- Produces: `Metrics.http_requests_total`, `.http_request_duration_seconds`,
  `.ws_connections_active`, `.ws_sessions_total`, `.ws_messages_total`;
  module-level `record(fn) -> None`; `HTTP_DURATION_BUCKETS`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_metrics.py` — extend the existing `_ALL_FAMILIES` list:

```python
_ALL_FAMILIES = [
    "queue_depth", "jobs_in_flight", "job_queue_wait_seconds",
    "job_execution_seconds", "job_terminal_total", "wait_expired_total",
    "mode_load_seconds", "mode_switch_total", "demand_reload_total",
    "unload_total", "worker_recovery_total", "mode_active", "resolution_epoch",
    "device_total_bytes", "device_free_bytes", "device_used_bytes",
    "device_unattributed_bytes", "consumer_reserved_bytes",
    "consumer_allocated_bytes", "device_snapshot_stale",
    # STABL-xmsrxvto
    "http_requests_total", "http_request_duration_seconds",
    "ws_connections_active", "ws_sessions_total", "ws_messages_total",
]
```

and append these tests:

```python
def test_http_duration_buckets_span_health_and_generate(monkeypatch):
    """/health answers in single-digit ms; /generate runs for minutes. One
    histogram covers both, so the buckets must too — prometheus defaults stop at
    10s and would bin every generation into +Inf."""
    monkeypatch.setenv("METRICS_ENABLED", "1")
    assert min(m.HTTP_DURATION_BUCKETS) <= 0.01
    assert max(m.HTTP_DURATION_BUCKETS) >= 600
    met = m.get_metrics()
    met.http_request_duration_seconds.labels(method="POST", route="/generate").observe(240.0)
    body, _ = met.render()
    assert b"st_http_request_duration_seconds_bucket" in body


def test_new_families_carry_no_client_controlled_labels(monkeypatch):
    monkeypatch.setenv("METRICS_ENABLED", "1")
    met = m.get_metrics()
    assert set(met.http_requests_total._labelnames) == {"method", "route", "status"}
    assert set(met.http_request_duration_seconds._labelnames) == {"method", "route"}
    assert set(met.ws_messages_total._labelnames) == {"type", "direction"}
    assert met.ws_connections_active._labelnames == ()
    assert met.ws_sessions_total._labelnames == ()


def test_record_runs_the_side_effect(monkeypatch):
    monkeypatch.setenv("METRICS_ENABLED", "1")
    m.record(lambda met: met.ws_sessions_total.inc())
    body, _ = m.get_metrics().render()
    assert any(ln.startswith("st_ws_sessions_total ") and ln.endswith("1.0")
               for ln in body.decode().splitlines())


def test_record_swallows_anything():
    """Instrumentation must never break a request or drop a WS connection."""
    m.record(lambda met: 1 / 0)                 # returns normally
    m.record(lambda met: met.nope.labels(x=1))  # returns normally
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate stability-toys && python -m pytest tests/test_metrics.py -q`
Expected: FAIL — `AttributeError` on the five new names, and
`AttributeError: module 'server.metrics' has no attribute 'record'`.

- [ ] **Step 3: Implement**

In `server/metrics.py`, add the bucket constant beside the existing ones:

```python
# /health answers in single-digit ms, /generate runs for minutes — one histogram
# spans both. prometheus_client's DEFAULT_BUCKETS stop at 10s.
HTTP_DURATION_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600,
)
```

In `Metrics._declare`, after the DeviceMemory block:

```python
        # --- HTTP (STABL-xmsrxvto) ---
        self.http_requests_total = C(
            "st_http_requests_total", "HTTP requests by method, route and status",
            ["method", "route", "status"], registry=r)
        self.http_request_duration_seconds = H(
            "st_http_request_duration_seconds", "HTTP request duration",
            ["method", "route"], buckets=HTTP_DURATION_BUCKETS, registry=r)

        # --- WebSocket (STABL-xmsrxvto) ---
        self.ws_connections_active = G(
            "st_ws_connections_active", "Currently connected WebSocket clients",
            registry=r)
        self.ws_sessions_total = C(
            "st_ws_sessions_total", "WebSocket sessions accepted", registry=r)
        self.ws_messages_total = C(
            "st_ws_messages_total", "WebSocket messages by type and direction",
            ["type", "direction"], registry=r)
```

Add the five names to the `_declare_noop` tuple:

```python
            "http_requests_total", "http_request_duration_seconds",
            "ws_connections_active", "ws_sessions_total", "ws_messages_total",
```

And add the shared guard at module level, after `reset_metrics()`:

```python
def record(fn) -> None:
    """Run a metrics side effect that must never reach the caller.

    The facade's no-op objects already make a DISABLED call site inert; this
    guards the ENABLED one, where a label-cardinality error or a typo'd attribute
    would otherwise propagate into a request handler or drop a WebSocket
    connection. Mirrors Governor._metric, which cannot be reused here because
    backends/ must not be imported from server/ request paths.
    """
    try:
        fn(get_metrics())
    except Exception:
        logger.debug("[Metrics] side effect failed", exc_info=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: all pass **except** `test_every_family_is_documented_in_the_contract`, which
fails until Task 4 — the contract test is bidirectional and the five new families have no
doc entries yet. That is the intended RED for Task 4; do not weaken the test.

- [ ] **Step 5: Commit**

```bash
git add server/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): declare HTTP + WebSocket families and the shared record() guard (STABL-xmsrxvto)

Explicit HTTP duration buckets: /health answers in ms and /generate runs for
minutes, and prometheus defaults stop at 10s. record() guards ENABLED call sites
so instrumentation cannot break a request or drop a WS connection.

test_every_family_is_documented_in_the_contract fails until Task 4 by design —
the contract test is bidirectional.

next: Task 2 HTTP middleware"
```

---

## Task 2: The HTTP middleware

**Files:**
- Create: `server/metrics_middleware.py`
- Modify: `server/lcm_sr_server.py` (`add_middleware`, near `:1016`)
- Test: `tests/test_metrics_http.py`

**Interfaces:**
- Consumes: `get_metrics()`, `record()` from Task 1.
- Produces: `MetricsMiddleware(app)` — a pure ASGI callable; `route_label(scope) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics_http.py
"""HTTP request metrics (STABL-xmsrxvto)."""
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from server import metrics as m
from server.metrics_middleware import MetricsMiddleware, route_label


@pytest.fixture(autouse=True)
def _metrics_on(monkeypatch):
    monkeypatch.setenv("METRICS_ENABLED", "1")
    m.reset_metrics()
    yield
    m.reset_metrics()


def _app():
    app = FastAPI()
    app.add_middleware(MetricsMiddleware)

    @app.get("/api/models/{name}")
    def one(name: str):
        return {"name": name}

    @app.get("/boom")
    def boom():
        raise HTTPException(503, "nope")

    @app.get("/crash")
    def crash():
        raise RuntimeError("unhandled")

    return app


def _lines(prefix):
    body, _ = m.get_metrics().render()
    return [ln for ln in body.decode().splitlines()
            if ln.startswith(prefix) and not ln.startswith("#")]


def _value(prefix, contains=None):
    for ln in _lines(prefix):
        if contains is None or contains in ln:
            return float(ln.rsplit(" ", 1)[1])
    return None


def test_path_params_collapse_to_one_series():
    """THE cardinality trap: request.url.path would make one series per model."""
    with TestClient(_app()) as c:
        c.get("/api/models/SDXL")
        c.get("/api/models/HunyuanDiT")

    assert _value("st_http_requests_total", 'route="/api/models/{name}"') == 2.0
    assert not _lines('st_http_requests_total{method="GET",route="/api/models/SDXL"')


def test_unmatched_paths_share_one_label():
    """An unmatched path is exactly the unbounded set a scanner probes."""
    with TestClient(_app()) as c:
        c.get("/nope/one")
        c.get("/nope/two")

    assert _value("st_http_requests_total", 'route="__unmatched__"') == 2.0


def test_status_is_labelled():
    with TestClient(_app()) as c:
        c.get("/api/models/x")
        c.get("/boom")

    assert _value("st_http_requests_total", 'status="200"') == 1.0
    assert _value("st_http_requests_total", 'status="503"') == 1.0


def test_duration_is_observed():
    with TestClient(_app()) as c:
        c.get("/api/models/x")

    assert _value("st_http_request_duration_seconds_count",
                  'route="/api/models/{name}"') == 1.0


def test_an_unhandled_exception_is_counted_as_500():
    """A request that raises still happened. Counting only successful responses
    hides exactly the traffic an operator is looking for."""
    with TestClient(_app(), raise_server_exceptions=False) as c:
        c.get("/crash")

    assert _value("st_http_requests_total", 'status="500"') == 1.0
    assert _value("st_http_request_duration_seconds_count", 'route="/crash"') == 1.0


def test_websocket_scope_is_not_counted_as_http():
    """A WS upgrade has no status; counting it as a request would invent one."""
    app = FastAPI()
    app.add_middleware(MetricsMiddleware)

    @app.websocket("/ws")
    async def ws(websocket):
        await websocket.accept()
        await websocket.close()

    with TestClient(app) as c:
        with c.websocket_connect("/ws"):
            pass

    assert not _lines("st_http_requests_total")


def test_disabled_middleware_records_nothing(monkeypatch):
    monkeypatch.delenv("METRICS_ENABLED", raising=False)
    m.reset_metrics()
    with TestClient(_app()) as c:
        assert c.get("/api/models/x").status_code == 200
    body, _ = m.get_metrics().render()
    assert body == b""


def test_a_broken_metrics_backend_does_not_break_the_request(monkeypatch):
    class _Explodes:
        def labels(self, **kw):
            raise RuntimeError("boom")

    monkeypatch.setattr(m.get_metrics(), "http_requests_total", _Explodes())
    with TestClient(_app()) as c:
        assert c.get("/api/models/x").status_code == 200


def test_route_label_helper():
    assert route_label({"route": type("R", (), {"path": "/x/{y}"})()}) == "/x/{y}"
    assert route_label({"route": None}) == "__unmatched__"
    assert route_label({}) == "__unmatched__"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics_http.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.metrics_middleware'`.

- [ ] **Step 3: Implement the middleware**

```python
# server/metrics_middleware.py
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
    `/v1/storage/{key}` one series per object. Starlette populates
    `scope["route"]` during routing; a request that matched nothing has none, and
    an unmatched path is precisely the unbounded set a scanner probes.
    """
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path if path else UNMATCHED


class MetricsMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        # WebSocket and lifespan scopes have no status and no duration worth
        # charging to an HTTP histogram.
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
            # Read AFTER the call — scope["route"] is populated during routing.
            route = route_label(scope)
            record(lambda met: (
                met.http_requests_total.labels(
                    method=method, route=route, status=str(status["code"])).inc(),
                met.http_request_duration_seconds.labels(
                    method=method, route=route).observe(elapsed),
            ))
```

- [ ] **Step 4: Wire it into the app**

In `server/lcm_sr_server.py`, add to the imports beside the other metrics imports:

```python
from server.metrics_middleware import MetricsMiddleware
```

and add it next to the CORS middleware (around `:1016`):

```python
# STABL-xmsrxvto: added AFTER CORSMiddleware, which makes it the OUTER of the two
# (Starlette applies middleware in reverse registration order), so a CORS preflight
# rejection is still counted as the request it was.
app.add_middleware(MetricsMiddleware)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics_http.py -q`
Expected: ALL PASS.

- [ ] **Step 6: Verify against the real app**

```bash
conda activate stability-toys
METRICS_ENABLED=1 CONTROLNET_REGISTRY_VALIDATION=off BACKEND=cpu python -c "
from fastapi.testclient import TestClient
from server.lcm_sr_server import app
with TestClient(app) as c:
    c.get('/health'); c.get('/health'); c.get('/definitely-not-a-route')
    body = c.get('/metrics').text
for l in body.splitlines():
    if l.startswith('st_http_requests_total'): print(l)
"
```

Expected: a `/health` series with value 2 and an `__unmatched__` series with value 1.

**Corrected during execution — there is NO `/metrics` series on the first scrape.** The
counter increments in the middleware's `finally`, which runs after the response body has
already been rendered, so scrape N reports N−1 scrapes. Verified:

```text
scrape 1: (no /metrics series)
scrape 2: st_http_requests_total{method="GET",route="/metrics",status="200"} 1.0
```

`/metrics` does count itself, one scrape behind. Task 4's contract entry must say that
rather than "the scrape counts itself", which would have an operator reading the first
scrape after a restart as a lost request.

- [ ] **Step 7: Commit**

```bash
git add server/metrics_middleware.py server/lcm_sr_server.py tests/test_metrics_http.py
git commit -m "feat(metrics): HTTP request counters and latency histogram (STABL-xmsrxvto)

Labels on the matched route TEMPLATE from scope['route'], never request.url.path
— verified that two model names collapse to /api/models/{name} and that an
unmatched path yields route=None, so __unmatched__ is load-bearing rather than
defensive.

Pure ASGI middleware rather than BaseHTTPMiddleware: no anyio task group per
request, no interaction with streaming or background tasks, and the WebSocket
skip is explicit. An unhandled exception is counted as 500 rather than dropped.

next: Task 3 WebSocket metrics"
```

---

## Task 3: WebSocket metrics

**Files:**
- Modify: `server/ws_hub.py` (`connect`, `disconnect`, `send`, `broadcast`)
- Modify: `server/ws_routes.py` (the message loop, `:850-870`)
- Test: `tests/test_metrics_ws.py`

**Interfaces:**
- Consumes: `record()`, `get_metrics()` from Task 1.
- Produces: no new public symbols — instrumentation only. `ws_routes` gains a
  module-level `_inbound_type(raw_type) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics_ws.py
"""WebSocket metrics (STABL-xmsrxvto)."""
import pytest

from server import metrics as m
from server.ws_hub import WSHub
from server.ws_routes import _inbound_type, _INVALID_JSON, HANDLERS


@pytest.fixture(autouse=True)
def _metrics_on(monkeypatch):
    monkeypatch.setenv("METRICS_ENABLED", "1")
    m.reset_metrics()
    yield
    m.reset_metrics()


def _lines(prefix):
    body, _ = m.get_metrics().render()
    return [ln for ln in body.decode().splitlines()
            if ln.startswith(prefix) and not ln.startswith("#")]


def _value(prefix, contains=None):
    for ln in _lines(prefix):
        if contains is None or contains in ln:
            return float(ln.rsplit(" ", 1)[1])
    return None


class _FakeWS:
    def __init__(self, fail=False):
        self.sent = []
        self.fail = fail

    async def send_json(self, msg):
        if self.fail:
            raise RuntimeError("client gone")
        self.sent.append(msg)


# --- connections ---

async def test_connect_counts_a_session_and_sets_the_gauge():
    hub = WSHub()
    await hub.connect(_FakeWS(), "a")
    await hub.connect(_FakeWS(), "b")

    assert _value("st_ws_sessions_total") == 2.0
    assert _value("st_ws_connections_active") == 2.0


async def test_disconnect_lowers_the_gauge_but_not_the_session_counter():
    hub = WSHub()
    await hub.connect(_FakeWS(), "a")
    await hub.connect(_FakeWS(), "b")
    await hub.disconnect("a")

    assert _value("st_ws_connections_active") == 1.0
    assert _value("st_ws_sessions_total") == 2.0   # counters never go down


async def test_the_gauge_is_set_from_the_hub_not_incremented():
    """A double disconnect is a no-op on the hub, so an inc/dec pair would drift
    negative. The gauge is set from client_count, which cannot."""
    hub = WSHub()
    await hub.connect(_FakeWS(), "a")
    await hub.disconnect("a")
    await hub.disconnect("a")          # idempotent on the hub

    assert _value("st_ws_connections_active") == 0.0


async def test_a_dead_client_reaped_by_broadcast_updates_the_gauge():
    """broadcast() reaps dead clients via disconnect(); the gauge must follow."""
    hub = WSHub()
    await hub.connect(_FakeWS(), "ok")
    await hub.connect(_FakeWS(fail=True), "dead")
    await hub.broadcast({"type": "system:status"})

    assert _value("st_ws_connections_active") == 1.0


# --- messages ---

async def test_send_counts_one_outbound_message_of_its_type():
    hub = WSHub()
    await hub.connect(_FakeWS(), "a")
    await hub.send("a", {"type": "job:progress"})

    assert _value("st_ws_messages_total",
                  'direction="out",type="job:progress"') == 1.0


async def test_broadcast_counts_one_message_per_recipient():
    """The status broadcaster fans out every 5s; per-recipient is the number that
    reflects actual socket writes."""
    hub = WSHub()
    await hub.connect(_FakeWS(), "a")
    await hub.connect(_FakeWS(), "b")
    await hub.broadcast({"type": "system:status"})

    assert _value("st_ws_messages_total",
                  'direction="out",type="system:status"') == 2.0


async def test_send_to_an_unknown_client_counts_nothing():
    hub = WSHub()
    await hub.send("nobody", {"type": "job:progress"})
    assert not _lines("st_ws_messages_total")


# --- inbound type mapping ---

def test_known_types_pass_through():
    for t in HANDLERS:
        assert _inbound_type(t) == t


def test_an_unknown_type_collapses_to_a_fixed_label():
    """The client controls this string. A 1 MB type field must not become a
    label value, and neither must a million distinct ones."""
    assert _inbound_type("job:not-a-real-handler") == "unknown"
    assert _inbound_type("A" * 4096) == "unknown"
    assert _inbound_type(None) == "unknown"
    assert _inbound_type(12345) == "unknown"


def test_malformed_json_has_its_own_label():
    """Distinct operationally from a wrong type: one is a broken client, the
    other is a client asking for something that does not exist."""
    assert _inbound_type(_INVALID_JSON) == "invalid_json"
```

No asyncio marker is needed: `pytest.ini:44` sets `asyncio_mode = auto` (and `:28` passes
`--asyncio-mode=auto`), so bare `async def test_*` functions run as-is.
`pytest-asyncio>=0.21.0` is already in `requirements-test.txt`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics_ws.py -q`
Expected: FAIL — `ImportError: cannot import name '_inbound_type'`.

- [ ] **Step 3: Instrument the hub**

In `server/ws_hub.py`, add the import:

```python
from server.metrics import record
```

and instrument the four methods:

```python
    async def connect(self, ws: WebSocket, client_id: str) -> None:
        async with self._lock:
            self._clients[client_id] = ws
            count = len(self._clients)
        logger.info("WS client connected: %s (%d total)", client_id, count)
        record(lambda met: (
            met.ws_sessions_total.inc(),
            met.ws_connections_active.set(count),
        ))

    async def disconnect(self, client_id: str) -> None:
        async with self._lock:
            self._clients.pop(client_id, None)
            count = len(self._clients)
        logger.info("WS client disconnected: %s (%d total)", client_id, count)
        # SET, never dec: disconnect is idempotent on the hub (a double
        # disconnect pops nothing) and broadcast() reaps dead clients through
        # this same path, so an inc/dec pair would drift negative.
        record(lambda met: met.ws_connections_active.set(count))
```

In `send`, count only a message that actually reached a socket:

```python
        try:
            await ws.send_json(msg)
        except Exception as e:
            logger.warning(...)
            await self.disconnect(client_id)
        else:
            _count_out(msg)
```

In `broadcast`, count per successful recipient:

```python
        for cid, ws in snapshot:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(cid)
            else:
                _count_out(msg)
```

with the helper at module level:

```python
def _count_out(msg: dict) -> None:
    """Outbound messages are SERVER-generated, so `type` is bounded by our own
    vocabulary. Instrumented here rather than in websocket_endpoint because
    _status_broadcaster calls broadcast() every 5s entirely outside that loop —
    the single most frequent outbound message would otherwise be invisible."""
    msg_type = msg.get("type", "unknown") if isinstance(msg, dict) else "unknown"
    record(lambda met: met.ws_messages_total.labels(
        type=msg_type, direction="out").inc())
```

- [ ] **Step 4: Instrument inbound in the message loop**

In `server/ws_routes.py`, add near `HANDLERS`:

```python
_INVALID_JSON = object()


def _inbound_type(raw_type) -> str:
    """Map a client-supplied `type` onto a BOUNDED label value.

    The client controls this string entirely, so it must never reach a label
    unchecked — one long or one million distinct values would be equally fatal.
    Malformed JSON gets its own value because it means something different from
    an unrecognised type: a broken client versus a client asking for something
    that does not exist.
    """
    if raw_type is _INVALID_JSON:
        return "invalid_json"
    return raw_type if raw_type in HANDLERS else "unknown"
```

**`raw_type in HANDLERS` is NOT safe for arbitrary input** — the client controls the value,
and an unhashable one raises `TypeError` on the dict lookup. Verified: `{"type": {"a": 1}}`
and `{"type": ["x"]}` both raise. So:

```python
    try:
        return raw_type if raw_type in HANDLERS else "unknown"
    except TypeError:          # unhashable type field, e.g. {"type": {"a": 1}}
        return "unknown"
```

> **Pre-existing bug found here, NOT fixed by this issue.** The very next line,
> `HANDLERS.get(msg_type)` (`ws_routes.py:858`), has the same hazard *today*:
> `dict.get(unhashable)` raises `TypeError`, which propagates to the loop's outer
> `except Exception` and then `finally: await hub.disconnect(client_id)` — **a client can
> drop its own WebSocket connection by sending `{"type": {}}`**. It is one line to fix
> (`isinstance(msg_type, str)` guard, or wrap the lookup) but it is a behaviour change
> outside this issue's scope. Filed separately; do not silently fold it in, and note that
> the `_inbound_type` guard above must not be mistaken for having fixed it — the
> instrumentation sits *before* that line and would merely stop being the first thing
> to raise.

and count at the three inbound outcomes in the loop:

```python
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                _count_in(_INVALID_JSON)
                await hub.send(client_id, _error("Invalid JSON"))
                continue

            msg_type = msg.get("type")
            _count_in(msg_type)
            handler = HANDLERS.get(msg_type)
```

with:

```python
def _count_in(raw_type) -> None:
    label = _inbound_type(raw_type)
    record(lambda met: met.ws_messages_total.labels(
        type=label, direction="in").inc())
```

and `from server.metrics import record` added to the imports.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics_ws.py -q`
Expected: ALL PASS.

- [ ] **Step 6: Run the WS regression suite**

Run: `python -m pytest tests/test_ws_hub.py tests/test_ws_routes.py tests/test_ws_build_generate_request.py -q`
Expected: no new failures. `test_ws_hub.py` is the one most likely to notice — it
constructs `WSHub` directly, which now emits metrics on connect/disconnect.

- [ ] **Step 7: Commit**

```bash
git add server/ws_hub.py server/ws_routes.py tests/test_metrics_ws.py
git commit -m "feat(metrics): WebSocket connection, session and message metrics (STABL-xmsrxvto)

Outbound is instrumented in ws_hub, not websocket_endpoint: _status_broadcaster
calls broadcast() every 5s entirely outside that loop, so the most frequent
outbound message would otherwise be invisible.

connections_active is SET from the hub's own count, never inc/dec — disconnect is
idempotent and broadcast() reaps dead clients through it, so a paired counter
would drift negative.

Inbound type maps through the HANDLERS registry; the client controls that string
and it must never reach a label unchecked. Malformed JSON gets its own value
because a broken client is not the same event as an unrecognised type.

next: Task 4 contract doc"
```

---

## Task 4: Extend the contract

**Files:**
- Modify: `docs/observability-contract.md`
- Test: `tests/test_metrics.py` (the existing bidirectional test now passes)

**Interfaces:**
- Consumes: the family names from Task 1.
- Produces: the doc entries `../continuous` reads.

- [ ] **Step 1: Confirm the contract test is currently RED**

Run: `python -m pytest tests/test_metrics.py::test_every_family_is_documented_in_the_contract -q`
Expected: FAIL listing the five undocumented families. This has been red since Task 1 —
it is the task's own RED, not a regression.

- [ ] **Step 2: Add the entries**

Append to `docs/observability-contract.md`, after the DeviceMemory section:

```markdown
## HTTP and WebSocket families

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `st_http_requests_total` | counter | `method`, `route`, `status` | HTTP requests |
| `st_http_request_duration_seconds` | histogram | `method`, `route` | request duration |
| `st_ws_connections_active` | gauge | — | currently connected WS clients |
| `st_ws_sessions_total` | counter | — | WS sessions accepted |
| `st_ws_messages_total` | counter | `type`, `direction` | WS messages, `in` / `out` |

**`route` is the matched route TEMPLATE, never the raw path.** `/api/models/SDXL`
and `/api/models/HunyuanDiT` are one series, `/api/models/{name}`. A request that
matched no route is `route="__unmatched__"` — do not expect literal paths there,
and do not add them: that is the unbounded set a scanner probes.

**`/metrics` counts itself.** Each scrape appears as
`st_http_requests_total{route="/metrics"}`. That is intentional — excluding it
would hide real load — so expect a baseline request rate equal to the scrape
interval.

**An unhandled server exception is counted as `status="500"`.** The request
happened; dropping it would hide exactly the traffic being investigated.

**WebSocket upgrades are not HTTP requests here.** They have no status, so they
appear only in the `st_ws_*` families.

**`st_ws_messages_total{direction="in"}` has a bounded `type`.** Values are the
server's own handler names plus two fixed fallbacks: `unknown` (a type the server
has no handler for) and `invalid_json` (a payload that did not parse). The client
controls that field, so its raw value never reaches a label.

**`st_ws_messages_total{direction="out"}` counts per RECIPIENT**, not per
broadcast call: `_status_broadcaster` fans one message out to every client every
5s, and the per-recipient number is the one that reflects actual socket writes.
```

- [ ] **Step 3: Run the contract test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: ALL PASS, including both directions of the contract check.

- [ ] **Step 4: Commit**

```bash
git add docs/observability-contract.md
git commit -m "docs(metrics): extend the contract with HTTP and WS families (STABL-xmsrxvto)

Five new entries plus the behaviours that mislead a reader who assumes otherwise:
route is a template and __unmatched__ is deliberate, /metrics counts itself, an
unhandled exception is a 500, WS upgrades are not HTTP requests, inbound type is
bounded by the handler registry, and outbound counts per recipient rather than
per broadcast call.

next: closeout"
```

---

## Closeout

- [ ] **Run the full suite and record the numbers**

```bash
conda activate stability-toys && python -m pytest tests/ -q 2>&1 | tail -3
```

Baseline before this issue: **1265 passed, 9 skipped, 1 xfailed**.

- [ ] **Check drift**

```bash
drift refs server/lcm_sr_server.py && drift check
```

Editing `lcm_sr_server.py`, `ws_routes.py` and `ws_hub.py` will stale their bound docs.
**Read each binding's prose before relinking** — relink only where the prose is still
accurate; update prose first where it is not. Baseline: 24 stale, none attributable to
the metrics work.

- [ ] **Update FP**

```bash
fp issue assign STABL-xmsrxvto --rev <sha>   # per commit, in order — see below
fp comment STABL-xmsrxvto "<what landed / decisions / next step>"
```

**Assign each commit as you make it, in chronological order.** `fp issue diff` derives its
baseline from the first-listed revision, so back-filling earlier commits after the tip
inverts the range and the diff silently reports no changes.

- [ ] **Report ready for review.** Do not self-advance state or call `fin`.

---

## Deferred (NOT in this issue)

- Per-endpoint SLO or alert rules — `../continuous` owns those.
- Request/response SIZE histograms — not asked for, and `content-length` is absent on
  streamed or chunked responses, so the metric would be quietly incomplete.
- `client_id` or per-client WS series — unbounded, same class as `job_id`.
- Structured logging (`STABL-bpsfmoke`) and tracing (`STABL-qnlaclof`).
