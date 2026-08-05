"""HTTP request metrics (STABL-xmsrxvto).

Plan: docs/superpowers/plans/2026-08-05-http-ws-metrics.md Task 2.

The load-bearing test is test_path_params_collapse_to_one_series: labelling on
request.url.path would make /api/models/{name} one series per model name and
/v1/storage/{key} one series per stored object.
"""
import pytest
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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


# --- cardinality ---

def test_path_params_collapse_to_one_series():
    """THE cardinality trap: request.url.path would make one series per model."""
    with TestClient(_app()) as c:
        c.get("/api/models/SDXL")
        c.get("/api/models/HunyuanDiT")

    assert _value("st_http_requests_total", 'route="/api/models/{name}"') == 2.0
    assert not [ln for ln in _lines("st_http_requests_total") if "SDXL" in ln]


def test_unmatched_paths_share_one_label():
    """An unmatched path is exactly the unbounded set a scanner probes."""
    with TestClient(_app()) as c:
        c.get("/nope/one")
        c.get("/nope/two")

    assert _value("st_http_requests_total", 'route="__unmatched__"') == 2.0


# --- labels ---

def test_status_is_labelled():
    with TestClient(_app()) as c:
        c.get("/api/models/x")
        c.get("/boom")

    assert _value("st_http_requests_total", 'status="200"') == 1.0
    assert _value("st_http_requests_total", 'status="503"') == 1.0


def test_method_is_labelled():
    with TestClient(_app()) as c:
        c.get("/api/models/x")
        c.post("/api/models/x")          # 405, but still a POST that happened

    assert _value("st_http_requests_total", 'method="GET"') == 1.0
    assert _value("st_http_requests_total", 'method="POST"') == 1.0


def test_duration_is_observed():
    with TestClient(_app()) as c:
        c.get("/api/models/x")

    assert _value("st_http_request_duration_seconds_count",
                  'route="/api/models/{name}"') == 1.0


# --- failure paths ---

def test_an_unhandled_exception_is_counted_as_500():
    """A request that raises still happened. Counting only successful responses
    hides exactly the traffic an operator is looking for."""
    with TestClient(_app(), raise_server_exceptions=False) as c:
        c.get("/crash")

    assert _value("st_http_requests_total", 'status="500"') == 1.0
    assert _value("st_http_request_duration_seconds_count", 'route="/crash"') == 1.0


def test_a_broken_metrics_backend_does_not_break_the_request(monkeypatch):
    class _Explodes:
        def labels(self, **kw):
            raise RuntimeError("boom")

    monkeypatch.setattr(m.get_metrics(), "http_requests_total", _Explodes())
    with TestClient(_app()) as c:
        assert c.get("/api/models/x").status_code == 200


# --- scope handling ---

def test_websocket_scope_is_not_counted_as_http():
    """A WS upgrade has no status; counting it as a request would invent one."""
    reached = []

    app = FastAPI()
    app.add_middleware(MetricsMiddleware)

    # The `: WebSocket` annotation is required — without it FastAPI treats
    # `websocket` as a query parameter and closes the connection with 1008
    # before the handler ever runs.
    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        reached.append(websocket.scope["type"])
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass

    with TestClient(app) as c:
        with c.websocket_connect("/ws") as sock:
            sock.send_text("hello")

    # the connection really happened and really was a websocket scope, so the
    # empty metric below means "skipped", not "never ran"
    assert reached == ["websocket"]
    assert not _lines("st_http_requests_total")


def test_lifespan_scope_passes_through():
    """The middleware must not swallow or mis-handle the lifespan scope — an app
    that cannot start is a louder failure than a missing metric."""
    started = []

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        started.append(True)
        yield

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(MetricsMiddleware)

    with TestClient(app):
        pass

    assert started == [True]


# --- the gate ---

def test_disabled_middleware_records_nothing(monkeypatch):
    monkeypatch.delenv("METRICS_ENABLED", raising=False)
    m.reset_metrics()
    with TestClient(_app()) as c:
        assert c.get("/api/models/x").status_code == 200
    body, _ = m.get_metrics().render()
    assert body == b""


# --- the helper ---

def test_route_label_helper():
    assert route_label({"route": type("R", (), {"path": "/x/{y}"})()}) == "/x/{y}"
    assert route_label({"route": None}) == "__unmatched__"
    assert route_label({}) == "__unmatched__"
