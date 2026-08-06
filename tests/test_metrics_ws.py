"""WebSocket metrics (STABL-xmsrxvto).

Plan: docs/superpowers/plans/2026-08-05-http-ws-metrics.md Task 3.

Outbound is instrumented in ws_hub, not websocket_endpoint: _status_broadcaster
calls hub.broadcast() every 5s entirely outside that loop, so hooking only the
endpoint would make the most frequent outbound message invisible.
"""
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


async def test_a_dead_client_reaped_by_send_updates_the_gauge():
    hub = WSHub()
    await hub.connect(_FakeWS(fail=True), "dead")
    await hub.send("dead", {"type": "job:progress"})

    assert _value("st_ws_connections_active") == 0.0


# --- outbound messages ---

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


async def test_a_failed_send_is_not_counted_as_delivered():
    """The metric means 'written to a socket', not 'attempted'."""
    hub = WSHub()
    await hub.connect(_FakeWS(fail=True), "dead")
    await hub.send("dead", {"type": "job:progress"})

    assert not _lines("st_ws_messages_total")


async def test_broadcast_counts_only_the_recipients_that_took_it():
    hub = WSHub()
    await hub.connect(_FakeWS(), "ok")
    await hub.connect(_FakeWS(fail=True), "dead")
    await hub.broadcast({"type": "system:status"})

    assert _value("st_ws_messages_total",
                  'direction="out",type="system:status"') == 1.0


async def test_send_to_an_unknown_client_counts_nothing():
    hub = WSHub()
    await hub.send("nobody", {"type": "job:progress"})

    assert not _lines("st_ws_messages_total")


# --- inbound type mapping ---

def test_known_types_pass_through():
    assert HANDLERS, "no handlers registered — the test would be vacuous"
    for t in HANDLERS:
        assert _inbound_type(t) == t


def test_an_unknown_type_collapses_to_a_fixed_label():
    """The client controls this string. A 4 KB type field must not become a label
    value, and neither must a million distinct ones."""
    assert _inbound_type("job:not-a-real-handler") == "unknown"
    assert _inbound_type("A" * 4096) == "unknown"
    assert _inbound_type(None) == "unknown"
    assert _inbound_type(12345) == "unknown"


def test_an_unhashable_type_does_not_raise():
    """`raw in HANDLERS` hashes its argument, and the client controls the value.
    Note this guard does NOT fix STABL-gzfzzsdq — HANDLERS.get() one line later
    has the same hazard and still drops the connection."""
    assert _inbound_type({"a": 1}) == "unknown"
    assert _inbound_type(["x"]) == "unknown"


def test_malformed_json_has_its_own_label():
    """Distinct operationally from a wrong type: one is a broken client, the
    other is a client asking for something that does not exist."""
    assert _inbound_type(_INVALID_JSON) == "invalid_json"


def test_inbound_labels_are_bounded_by_the_handler_registry():
    """The full label set is the handler names plus exactly two fallbacks."""
    allowed = set(HANDLERS) | {"unknown", "invalid_json"}
    for probe in ["ping", "job:submit", "nope", "", None, 0, {"a": 1}, _INVALID_JSON]:
        assert _inbound_type(probe) in allowed


# --- the gate ---

async def test_disabled_hub_records_nothing(monkeypatch):
    monkeypatch.delenv("METRICS_ENABLED", raising=False)
    m.reset_metrics()

    hub = WSHub()
    await hub.connect(_FakeWS(), "a")
    await hub.send("a", {"type": "job:progress"})

    body, _ = m.get_metrics().render()
    assert body == b""


async def test_a_broken_metrics_backend_does_not_break_the_hub(monkeypatch):
    class _Explodes:
        def inc(self, *a, **kw):
            raise RuntimeError("boom")

        def set(self, *a, **kw):
            raise RuntimeError("boom")

        def labels(self, **kw):
            raise RuntimeError("boom")

    met = m.get_metrics()
    monkeypatch.setattr(met, "ws_sessions_total", _Explodes())
    monkeypatch.setattr(met, "ws_messages_total", _Explodes())

    hub = WSHub()
    await hub.connect(_FakeWS(), "a")          # must not raise
    await hub.send("a", {"type": "job:progress"})
    assert hub.client_count == 1               # connection survived
