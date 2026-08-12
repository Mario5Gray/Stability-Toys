"""HTTP + WebSocket entry spans (STABL-qnlaclof step 3).

Where a trace BEGINS. Two seams, and the WebSocket one is anchored deliberately:
the message loop has two exits that `continue` before any handler runs — malformed
JSON and an unrecognised type — so a span opened later would miss exactly the
protocol traffic worth tracing. The span map records that as a blocker found at
review; these tests are what stop it coming back.

Spec: docs/superpowers/specs/2026-08-12-tracing-span-map-and-boundary-fixes.md §3.1
"""
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from server import metrics as m
from server import ws_routes
from server.metrics_middleware import MetricsMiddleware
from server.ws_routes import ws_router


class _RecordedSpan:
    def __init__(self, name):
        self.name = name
        self.attributes = {}
        self.exceptions = []
        self.exit_exc_type = None
        self.ended = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exit_exc_type = exc_type
        self.ended = True
        return False

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def update_name(self, name):
        self.name = name

    def add_event(self, *args, **kwargs):
        pass

    def record_exception(self, exc):
        self.exceptions.append(exc)

    def set_status(self, *args, **kwargs):
        pass

    def is_recording(self):
        return True


class _RecordingTracer:
    def __init__(self):
        self.spans = []

    def start_as_current_span(self, name, *args, **kwargs):
        span = _RecordedSpan(name)
        self.spans.append(span)
        return span


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _http_app():
    app = FastAPI()
    app.add_middleware(MetricsMiddleware)

    @app.get("/api/models/{name}")
    def one(name: str):
        return {"name": name}

    @app.get("/boom")
    def boom():
        raise HTTPException(503, "nope")

    # For the scope-exclusion test. Its span, if any, would come from
    # ws_routes.get_tracer, which this fixture does NOT patch — so anything the
    # recorder sees here came from the HTTP middleware.
    app.include_router(ws_router)
    app.state.use_mode_system = False
    app.state.service = None
    app.state.sr_service = None
    app.state.storage = None
    return app


class _EnabledTracing:
    """Stands in for an enabled facade.

    Tracing cannot be enabled for real here: the OTel SDK is not a dependency
    yet (step 6), so `TRACING_ENABLED=1` correctly degrades to disabled. The
    middleware's fast path reads `get_tracing().enabled`, so a test that only
    patched the tracer would exercise the skip branch and see zero spans —
    which is what happened, and is why this exists rather than a bare tracer
    patch.
    """

    enabled = True


@pytest.fixture
def http(monkeypatch):
    """Tracing on, metrics OFF — deliberately. Every HTTP test in this file
    therefore also demonstrates the two pillars gate independently."""
    monkeypatch.delenv("METRICS_ENABLED", raising=False)
    m.reset_metrics()
    tracer = _RecordingTracer()
    monkeypatch.setattr("server.metrics_middleware.get_tracer", lambda name: tracer)
    monkeypatch.setattr("server.metrics_middleware.get_tracing", _EnabledTracing)
    try:
        yield tracer, TestClient(_http_app())
    finally:
        m.reset_metrics()


def test_an_http_request_is_named_by_the_route_TEMPLATE_not_the_path(http):
    """The cardinality trap that already bit the metrics label, now for span
    names: `/api/models/sdxl` and `/api/models/lcm` must be ONE operation, not a
    new trace name per model."""
    tracer, client = http

    client.get("/api/models/sdxl")
    client.get("/api/models/lcm")

    assert [s.name for s in tracer.spans] == ["GET /api/models/{name}", "GET /api/models/{name}"]
    assert tracer.spans[0].attributes["http.route"] == "/api/models/{name}"


def test_the_status_code_lands_on_the_span(http):
    tracer, client = http

    client.get("/boom")

    span = tracer.spans[0]
    assert span.attributes["http.response.status_code"] == 503
    assert span.ended


def test_an_unmatched_path_collapses_rather_than_naming_itself(http):
    """An unmatched path is the unbounded set a scanner probes. It must still be
    traced — that traffic is interesting — but under one name."""
    tracer, client = http

    client.get("/nope/nothing/here")

    assert tracer.spans[0].attributes["http.route"] == "__unmatched__"


def test_the_fast_path_ORs_the_two_pillars_rather_than_ANDing_them(http):
    """The middleware short-circuits when there is nothing to record. That
    condition must be an OR: an AND would make tracing silently require
    METRICS_ENABLED — a coupling nobody would think to look for, and one that
    presents as "tracing is broken" rather than as a config problem.

    The `http` fixture runs with metrics OFF, so a span here is the proof.
    """
    tracer, client = http
    assert m.get_metrics().enabled is False

    client.get("/api/models/sdxl")

    assert len(tracer.spans) == 1


def test_a_websocket_scope_is_left_alone_by_the_http_middleware(http):
    """WS has no status and no duration worth charging to an HTTP histogram, and
    its own span is opened per MESSAGE in ws_routes. A span here would wrap the
    whole connection and be useless."""
    tracer, client = http

    with client.websocket_connect("/v1/ws"):
        pass

    assert tracer.spans == []


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@pytest.fixture
def ws(monkeypatch):
    """The same minimal app tests/test_ws_log_context.py drives.

    The `app.state` attributes are not decoration: `_build_status()` reads them
    on connect, and without them the initial `system:status` never arrives — the
    test then blocks on `receive_json()` rather than failing.
    """
    tracer = _RecordingTracer()
    monkeypatch.setattr("server.ws_routes.get_tracer", lambda name: tracer)
    app = FastAPI()
    app.include_router(ws_router)
    app.state.use_mode_system = False
    app.state.service = None
    app.state.sr_service = None
    app.state.storage = None
    return tracer, TestClient(app)


def _messages(tracer):
    return [s for s in tracer.spans if s.name == "ws.message"]


def test_a_handled_message_opens_one_span_carrying_its_type(ws):
    tracer, client = ws

    with client.websocket_connect("/v1/ws") as sock:
        sock.receive_json()                     # system:status on connect
        sock.send_json({"type": "ping", "id": "c1"})
        sock.receive_json()

    spans = _messages(tracer)
    assert len(spans) == 1
    assert spans[0].attributes["messaging.type"] == "ping"


def test_MALFORMED_JSON_is_still_traced(ws):
    """The blocker the span map review caught. `json.loads` raising exits the
    loop body with `continue` before any handler is looked up, so a span anchored
    on the handler call would leave a broken client completely invisible."""
    tracer, client = ws

    with client.websocket_connect("/v1/ws") as sock:
        sock.receive_json()
        sock.send_text("{not json")
        sock.receive_json()                     # the error reply

    spans = _messages(tracer)
    assert len(spans) == 1, "a malformed message produced no span"
    assert spans[0].attributes["messaging.type"] == "invalid_json"


def test_an_UNKNOWN_type_is_still_traced_and_bounded(ws):
    """The second pre-handler exit. `unknown` rather than the client's string:
    the type is client-controlled, and an unbounded span name is the same
    cardinality failure as an unbounded label."""
    tracer, client = ws

    with client.websocket_connect("/v1/ws") as sock:
        sock.receive_json()
        sock.send_json({"type": "definitely:not:a:handler", "id": "c1"})
        sock.receive_json()

    spans = _messages(tracer)
    assert len(spans) == 1
    assert spans[0].attributes["messaging.type"] == "unknown"


def test_an_UNHASHABLE_type_does_not_break_the_span(ws):
    """STABL-gzfzzsdq: `HANDLERS.get(msg_type)` hashes a client-controlled value,
    so `{"type": {}}` raises TypeError and drops the connection. That bug is not
    fixed here — but the span must not be the thing that raises first, and the
    dropped connection should be visible as an errored span rather than silence.
    """
    tracer, client = ws

    # Deliberately no receive after the send. The connection is dropped WITHOUT a
    # reply, so `receive_json()` blocks forever rather than raising — the same
    # shape that makes this bug hard to see in production. Assert after the
    # context manager has torn the socket down.
    with client.websocket_connect("/v1/ws") as sock:
        sock.receive_json()
        sock.send_json({"type": {}, "id": "c1"})

    spans = _messages(tracer)
    assert len(spans) == 1
    assert spans[0].attributes["messaging.type"] == "unknown"
    assert spans[0].ended, "the span outlived the connection it was tracing"
    assert spans[0].exit_exc_type is TypeError, (
        "the dropped connection should surface as an errored span; "
        "STABL-gzfzzsdq is otherwise silent"
    )


def test_each_message_on_one_connection_gets_its_OWN_span(ws):
    """One span per message, not one per connection — the connection is long
    lived and a single span over it would be useless."""
    tracer, client = ws

    with client.websocket_connect("/v1/ws") as sock:
        sock.receive_json()
        sock.send_json({"type": "ping", "id": "c1"})
        sock.receive_json()
        sock.send_json({"type": "ping", "id": "c2"})
        sock.receive_json()

    assert len(_messages(tracer)) == 2
