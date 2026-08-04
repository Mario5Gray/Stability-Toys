"""The /metrics scrape endpoint (STABL-asawxgvp).

Plan: docs/superpowers/plans/2026-08-03-prometheus-substrate.md Task 6.

The load-bearing test here is test_metrics_resolves_with_a_ui_static_mount_present:
app.mount("/", StaticFiles(...)) matches EVERY path and Starlette routes in
registration order, so a late-registered /metrics is unreachable — and the failure
is invisible on a dev box, where the UI dist is absent and the mount is skipped.
"""
import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from server import metrics as m
from server.metrics_routes import router as metrics_router, build_runtime_stats_fn


@pytest.fixture(autouse=True)
def _fresh():
    m.reset_metrics()
    yield
    m.reset_metrics()


def _app():
    app = FastAPI()
    app.include_router(metrics_router)
    return app


# --- the endpoint ---

def test_metrics_endpoint_renders(monkeypatch):
    monkeypatch.setenv("METRICS_ENABLED", "1")
    m.get_metrics().job_terminal_total.labels(mode="SDXL", outcome="ok").inc()

    with TestClient(_app()) as client:
        resp = client.get("/metrics")

    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "st_governor_job_terminal_total" in resp.text


def test_metrics_endpoint_404s_when_disabled(monkeypatch):
    monkeypatch.delenv("METRICS_ENABLED", raising=False)
    with TestClient(_app()) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 404


def test_metrics_is_not_cached(monkeypatch):
    """A cached scrape would flat-line every gauge for the cache lifetime."""
    monkeypatch.setenv("METRICS_ENABLED", "1")
    with TestClient(_app()) as client:
        resp = client.get("/metrics")
    assert resp.headers.get("cache-control") == "no-store"


# --- route ordering: THE trap ---

def test_metrics_resolves_with_a_ui_static_mount_present(tmp_path, monkeypatch):
    """app.mount('/', StaticFiles(...)) matches everything, Starlette matches
    routes in registration order, and the UI dist is absent on a dev box — so a
    late-registered /metrics passes locally and 404s only in the deployed image.
    """
    monkeypatch.setenv("METRICS_ENABLED", "1")
    (tmp_path / "index.html").write_text("<html>ui</html>")

    app = FastAPI()
    app.include_router(metrics_router)                 # BEFORE the mount
    app.mount("/", StaticFiles(directory=str(tmp_path), html=True), name="ui")

    with TestClient(app) as client:
        resp = client.get("/metrics")

    assert resp.status_code == 200, "/metrics was shadowed by the static mount"
    assert "st_" in resp.text


def test_a_static_mount_registered_first_really_does_shadow(tmp_path, monkeypatch):
    """Proves the trap is real rather than folklore — if this ever stops
    shadowing, the ordering test above has quietly stopped testing anything."""
    monkeypatch.setenv("METRICS_ENABLED", "1")
    (tmp_path / "index.html").write_text("<html>ui</html>")

    app = FastAPI()
    app.mount("/", StaticFiles(directory=str(tmp_path), html=True), name="ui")
    app.include_router(metrics_router)                 # AFTER the mount — too late

    with TestClient(app) as client:
        resp = client.get("/metrics")

    assert "st_governor" not in resp.text, (
        "the static mount no longer shadows later routes; the ordering guarantee "
        "this suite protects has changed"
    )


def test_registration_order_in_the_real_app():
    """Pin the ordering in lcm_sr_server itself, not just in a synthetic app."""
    from server import lcm_sr_server

    paths = [getattr(r, "path", None) for r in lcm_sr_server.app.router.routes]
    assert "/metrics" in paths, "/metrics is not mounted on the app"
    if "/" in paths:
        assert paths.index("/metrics") < paths.index("/"), (
            "/metrics must be registered before the catch-all static mount"
        )


# --- the scrape path must not fan out ---

def test_scrape_does_not_fan_out(monkeypatch):
    """The render path must touch gauges only — never DeviceMemory.snapshot().

    snapshot() round-trips every registered consumer, and under subprocess
    isolation the worker consumer's read is a request/reply over the child's
    control pipe.
    """
    monkeypatch.setenv("METRICS_ENABLED", "1")
    from backends import device_memory

    def _explode(*a, **kw):
        raise AssertionError("/metrics MUST NOT fan out to consumers")

    monkeypatch.setattr(device_memory._ConsumerRegistry, "snapshot", _explode)

    with TestClient(_app()) as client:
        assert client.get("/metrics").status_code == 200


# --- runtime stats adapter ---

class _GovernorLike:
    """Deliberately NOT a Mock.

    `build_runtime_stats_fn` does `getattr(pool, "_governor", pool)`, and Mock
    auto-creates every attribute — so a Mock "bare governor" grows a `_governor`
    child Mock, the adapter follows it into nonsense, and the resulting None makes
    a negative test pass for entirely the wrong reason. A real Governor has no
    `_governor` attribute, so the double must not either.
    """

    def __init__(self, queue_size=0, records=None, wedged=False):
        import threading
        self._queue_size = queue_size
        self._job_lock = threading.RLock()
        self._job_records = records or {}
        self._wedged = wedged

    def get_queue_size(self):
        if self._wedged:
            raise RuntimeError("wedged")
        return self._queue_size


class _PoolLike:
    """WorkerPool holds its Governor on _governor (backends/worker_pool.py:71)."""

    def __init__(self, governor):
        self._governor = governor


def _record(executing_since):
    from types import SimpleNamespace
    return SimpleNamespace(executing_since=executing_since)


def test_runtime_stats_counts_only_executing_jobs():
    """jobs_in_flight is jobs PAST the admission barrier, not queue length —
    executing_since is stamped after the demand reload and the epoch barrier."""
    pool = _PoolLike(_GovernorLike(
        queue_size=4,
        records={"a": _record(123.0), "b": _record(None)},
    ))

    stats = build_runtime_stats_fn(lambda: pool)()

    assert stats == {"queue_depth": 4, "jobs_in_flight": 1}


def test_runtime_stats_accepts_a_bare_governor():
    """The adapter is handed a pool in production but must not care."""
    stats = build_runtime_stats_fn(lambda: _GovernorLike(queue_size=2))()

    assert stats == {"queue_depth": 2, "jobs_in_flight": 0}


def test_runtime_stats_returns_none_before_the_pool_exists():
    """The sampler starts with the app; the runtime may not be up yet."""
    assert build_runtime_stats_fn(lambda: None)() is None


def test_runtime_stats_returns_none_when_the_getter_raises():
    def _boom():
        raise RuntimeError("app.state has no worker_pool")

    assert build_runtime_stats_fn(_boom)() is None


def test_runtime_stats_returns_none_on_a_wedged_governor():
    """Degrade to 'no queue gauges' rather than killing the sampler pass — that
    pass also carries the device metrics."""
    wedged = _GovernorLike(wedged=True)

    assert build_runtime_stats_fn(lambda: wedged)() is None
    # and the healthy version of the same double DOES produce stats, so the None
    # above is caused by the wedge and not by the double being unusable
    assert build_runtime_stats_fn(lambda: _GovernorLike(queue_size=1))() is not None
