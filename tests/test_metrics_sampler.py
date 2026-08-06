"""Background metrics sampler (STABL-asawxgvp).

Plan: docs/superpowers/plans/2026-08-03-prometheus-substrate.md Task 5.

The sampler is the ONLY thing permitted to fan out to DeviceMemory consumers.
Under subprocess isolation a consumer's pool_stats() is a request/reply over the
child's control pipe, so putting snapshot() on the /metrics render path would tie
pipe traffic to scrape cadence and put a blocking round-trip inside an ASGI
handler — the event-loop starvation shape that already makes /status time out
during a job.
"""
import time

import pytest

from server import metrics as m
from server.metrics_sampler import MetricsSampler, DEFAULT_INTERVAL_S
from backends.device_memory import ConsumerMemory, DeviceMemorySnapshot, MemoryTopology

GIB = 1024 ** 3


@pytest.fixture(autouse=True)
def _metrics_on(monkeypatch):
    monkeypatch.setenv("METRICS_ENABLED", "1")
    m.reset_metrics()
    yield
    m.reset_metrics()


def _snap(stale=False):
    """24 GiB card, 20 GiB free -> 4 GiB used. Consumers hold 3 GiB reserved,
    so 1 GiB is unattributed (the CUDA context and friends)."""
    return DeviceMemorySnapshot(
        device_uuid="GPU-fake",
        topology=MemoryTopology.DISCRETE,
        total_bytes=24 * GIB,
        free_bytes=20 * GIB,
        consumers=(
            ConsumerMemory(label="server", pid=1,
                           allocated_bytes=1 * GIB, reserved_bytes=2 * GIB,
                           stale=stale),
            ConsumerMemory(label="worker", pid=154,
                           allocated_bytes=1 * GIB, reserved_bytes=1 * GIB,
                           stale=stale),
        ),
    )


def _lines(prefix: str) -> list[str]:
    body, _ = m.get_metrics().render()
    return [ln for ln in body.decode().splitlines()
            if ln.startswith(prefix) and not ln.startswith("#")]


def _value(prefix: str, contains: str | None = None) -> float | None:
    for ln in _lines(prefix):
        if contains is None or contains in ln:
            return float(ln.rsplit(" ", 1)[1])
    return None


# --- device gauges ---

def test_sample_once_writes_device_gauges():
    MetricsSampler(snapshot_fn=_snap).sample_once()

    assert _value("st_device_total_bytes") == float(24 * GIB)
    assert _value("st_device_free_bytes") == float(20 * GIB)
    assert _value("st_device_used_bytes") == float(4 * GIB)
    # used 4 GiB - reserved 3 GiB = 1 GiB unexplained
    assert _value("st_device_unattributed_bytes") == float(1 * GIB)


def test_device_gauges_are_labelled_by_uuid():
    MetricsSampler(snapshot_fn=_snap).sample_once()
    assert all('device_uuid="GPU-fake"' in ln
               for ln in _lines("st_device_total_bytes"))


def test_per_consumer_gauges_key_on_label_never_pid():
    """pid looks bounded and is not: the subprocess handle mints a new one on
    every kill+respawn, so a pid label leaks a dead series per recovery."""
    MetricsSampler(snapshot_fn=_snap).sample_once()

    assert _value("st_consumer_reserved_bytes", 'consumer="server"') == float(2 * GIB)
    assert _value("st_consumer_reserved_bytes", 'consumer="worker"') == float(1 * GIB)
    assert _value("st_consumer_allocated_bytes", 'consumer="worker"') == float(1 * GIB)
    assert not any("pid" in ln for ln in _lines("st_consumer_reserved_bytes"))


def test_stale_snapshot_sets_the_stale_gauge():
    """stale=True means a consumer fan-out timed out — i.e. the child is not
    answering. An early wedged-worker signal with no other surface."""
    MetricsSampler(snapshot_fn=lambda: _snap(stale=True)).sample_once()
    assert _value("st_device_snapshot_stale") == 1.0


def test_fresh_snapshot_clears_the_stale_gauge():
    sampler = MetricsSampler(snapshot_fn=lambda: _snap(stale=True))
    sampler.sample_once()
    assert _value("st_device_snapshot_stale") == 1.0

    sampler._snapshot_fn = _snap          # consumer starts answering again
    sampler.sample_once()
    assert _value("st_device_snapshot_stale") == 0.0


# --- runtime stats ---

def test_runtime_stats_feed_queue_gauges():
    MetricsSampler(
        snapshot_fn=_snap,
        runtime_stats_fn=lambda: {"queue_depth": 3, "jobs_in_flight": 1},
    ).sample_once()

    assert _value("st_governor_queue_depth") == 3.0
    assert _value("st_governor_jobs_in_flight") == 1.0


def test_runtime_stats_none_leaves_gauges_alone():
    """The runtime may not be up yet. A None reading must not zero the gauges."""
    sampler = MetricsSampler(
        snapshot_fn=_snap,
        runtime_stats_fn=lambda: {"queue_depth": 7, "jobs_in_flight": 2},
    )
    sampler.sample_once()

    sampler._runtime_stats_fn = lambda: None
    sampler.sample_once()

    assert _value("st_governor_queue_depth") == 7.0


def test_a_raising_runtime_stats_still_writes_device_gauges():
    """The two readers are independent; one failing must not blank the other."""
    def _boom():
        raise RuntimeError("governor not ready")

    MetricsSampler(snapshot_fn=_snap, runtime_stats_fn=_boom).sample_once()
    assert _value("st_device_total_bytes") == float(24 * GIB)


def test_a_raising_snapshot_still_writes_runtime_gauges():
    def _boom():
        raise RuntimeError("NVML exploded")

    MetricsSampler(
        snapshot_fn=_boom,
        runtime_stats_fn=lambda: {"queue_depth": 5, "jobs_in_flight": 0},
    ).sample_once()
    assert _value("st_governor_queue_depth") == 5.0


# --- thread lifecycle ---

def test_a_raising_snapshot_does_not_kill_the_thread():
    """A sampler that dies silently is worse than no sampler."""
    calls = []

    def _boom():
        calls.append(1)
        raise RuntimeError("NVML exploded")

    s = MetricsSampler(snapshot_fn=_boom, interval_s=0.01)
    s.start()
    try:
        deadline = time.time() + 2.0
        while len(calls) < 3 and time.time() < deadline:
            time.sleep(0.01)
    finally:
        s.stop()
    assert len(calls) >= 3, "thread stopped after the first exception"


def test_stop_joins_the_thread():
    s = MetricsSampler(snapshot_fn=_snap, interval_s=0.01)
    s.start()
    s.stop()
    assert s._thread is None


def test_stop_is_idempotent():
    s = MetricsSampler(snapshot_fn=_snap, interval_s=0.01)
    s.start()
    s.stop()
    s.stop()          # must not raise on an already-stopped sampler


def test_double_start_does_not_spawn_a_second_thread():
    s = MetricsSampler(snapshot_fn=_snap, interval_s=0.01)
    s.start()
    first = s._thread
    s.start()
    try:
        assert s._thread is first
    finally:
        s.stop()


def test_stop_before_start_is_safe():
    MetricsSampler(snapshot_fn=_snap).stop()      # must not raise


def test_the_thread_is_a_daemon():
    """A non-daemon sampler would hold the process open at shutdown."""
    s = MetricsSampler(snapshot_fn=_snap, interval_s=0.01)
    s.start()
    try:
        assert s._thread is not None and s._thread.daemon
    finally:
        s.stop()


# --- the gate ---

def test_disabled_sampler_never_calls_snapshot(monkeypatch):
    """No gate means the sampler round-trips the child control pipe every 15s on
    a server that is not exporting anything."""
    monkeypatch.delenv("METRICS_ENABLED", raising=False)
    m.reset_metrics()

    def _must_not_run():
        raise AssertionError("disabled sampler MUST NOT fan out")

    s = MetricsSampler(snapshot_fn=_must_not_run, interval_s=0.01)
    s.start()
    time.sleep(0.1)
    s.stop()
    assert s._thread is None


def test_disabled_sample_once_is_inert(monkeypatch):
    """sample_once is reachable directly, not only through the thread."""
    monkeypatch.delenv("METRICS_ENABLED", raising=False)
    m.reset_metrics()
    calls = []

    MetricsSampler(snapshot_fn=lambda: calls.append(1) or _snap()).sample_once()
    assert calls == []


# --- interval ---

def test_default_interval_is_15(monkeypatch):
    monkeypatch.delenv("METRICS_SAMPLE_INTERVAL_S", raising=False)
    assert MetricsSampler(snapshot_fn=_snap).interval_s == 15.0
    assert DEFAULT_INTERVAL_S == 15.0


def test_interval_from_env(monkeypatch):
    monkeypatch.setenv("METRICS_SAMPLE_INTERVAL_S", "42")
    assert MetricsSampler(snapshot_fn=_snap).interval_s == 42.0


def test_explicit_interval_beats_env(monkeypatch):
    monkeypatch.setenv("METRICS_SAMPLE_INTERVAL_S", "42")
    assert MetricsSampler(snapshot_fn=_snap, interval_s=1.5).interval_s == 1.5


def test_a_garbage_interval_falls_back_to_the_default(monkeypatch):
    """A typo in deployment config must not stop the server from starting."""
    monkeypatch.setenv("METRICS_SAMPLE_INTERVAL_S", "not-a-number")
    assert MetricsSampler(snapshot_fn=_snap).interval_s == DEFAULT_INTERVAL_S


@pytest.mark.parametrize("raw", ["0", "0.0", "-1", "-0.5"])
def test_a_non_positive_env_interval_falls_back_to_the_default(monkeypatch, raw):
    """Event.wait(0) and wait(<0) return IMMEDIATELY, so a non-positive interval
    turns the sampler into a tight loop hammering snapshot_fn — and snapshot_fn is
    the child control-pipe round-trip. Worse than a crash, because it looks like
    the server is merely busy."""
    monkeypatch.setenv("METRICS_SAMPLE_INTERVAL_S", raw)
    assert MetricsSampler(snapshot_fn=_snap).interval_s == DEFAULT_INTERVAL_S


@pytest.mark.parametrize("bad", [0, 0.0, -1, -0.5])
def test_a_non_positive_explicit_interval_falls_back_to_the_default(bad):
    """The explicit argument bypasses the env parser, so it needs the same guard."""
    assert MetricsSampler(snapshot_fn=_snap, interval_s=bad).interval_s == DEFAULT_INTERVAL_S


@pytest.mark.parametrize("good", [0.01, 0.5, 1, 900])
def test_small_positive_intervals_are_honoured(good):
    """The guard rejects non-positive, NOT small — the test suite itself runs the
    sampler at 0.01s and deployments may legitimately want sub-second sampling."""
    assert MetricsSampler(snapshot_fn=_snap, interval_s=good).interval_s == float(good)


def test_a_non_positive_interval_does_not_spin(monkeypatch):
    """The consequence, asserted directly rather than inferred from the value."""
    monkeypatch.setenv("METRICS_SAMPLE_INTERVAL_S", "0")
    calls = []

    def _count():
        calls.append(1)
        return _snap()

    s = MetricsSampler(snapshot_fn=_count)
    s.start()
    try:
        time.sleep(0.2)
    finally:
        s.stop()
    # at 15s fallback: one sample. at 0s: thousands.
    assert len(calls) <= 2, f"sampler span {len(calls)} times in 0.2s — tight loop"


def test_the_structural_mirror_matches_the_real_dataclasses():
    """server/metrics_sampler.py declares _SnapshotLike/_ConsumerLike instead of
    importing from backends/. A field rename on the real dataclass would otherwise
    surface as sample_once swallowing an AttributeError into a debug log — device
    metrics would just silently stop, which is the worst possible failure for an
    observability component."""
    from server.metrics_sampler import _ConsumerLike, _SnapshotLike

    def _declares(proto, attr):
        # fields land in __annotations__; used_bytes/unattributed_bytes are
        # declared as properties on the Protocol, so check both.
        return attr in getattr(proto, "__annotations__", {}) or hasattr(proto, attr)

    snap = _snap()
    for attr in ("device_uuid", "total_bytes", "free_bytes", "consumers",
                 "used_bytes", "unattributed_bytes"):
        assert hasattr(snap, attr), f"DeviceMemorySnapshot lost {attr}"
        assert _declares(_SnapshotLike, attr), f"_SnapshotLike does not mirror {attr}"

    consumer = snap.consumers[0]
    for attr in ("label", "allocated_bytes", "reserved_bytes", "stale"):
        assert hasattr(consumer, attr), f"ConsumerMemory lost {attr}"
        assert _declares(_ConsumerLike, attr), f"_ConsumerLike does not mirror {attr}"

    # the check has teeth: a name neither side declares must be rejected
    assert not _declares(_SnapshotLike, "not_a_real_field")

    # pid exists on the real dataclass and is deliberately absent from the mirror
    assert hasattr(consumer, "pid")
    assert "pid" not in _ConsumerLike.__annotations__


def test_module_imports_nothing_from_backends():
    """Readers arrive as callables so the sampler stays backends-free."""
    import importlib.util

    spec = importlib.util.find_spec("server.metrics_sampler")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as fh:
        head = [ln for ln in fh if ln.startswith(("import ", "from "))]
    assert not any("backends" in ln for ln in head), (
        "server/metrics_sampler.py must not import from backends/"
    )


# ============================================================================
# STABL-cxbwwgly: OS resource gauges
# ============================================================================

from server.resource_probe import ResourceCounts, probe_resources


def _counts(sems=2, segments=1, fds=42):
    return ResourceCounts(leaked_semaphores=sems, shm_segments=segments, open_fds=fds)


def test_sample_once_writes_resource_gauges():
    MetricsSampler(snapshot_fn=_snap, resource_probe_fn=_counts).sample_once()

    assert _value("st_process_leaked_semaphores") == 2.0
    assert _value("st_process_shm_segments") == 1.0
    assert _value("st_process_open_fds") == 42.0


def test_resource_gauges_fire_without_a_runtime_stats_reader():
    """sample_once returns early when no runtime stats reader is injected. If the
    resource block sits behind that return it silently never fires — and the
    production sampler is the only caller that always passes one, so this would
    look fine in the app and be dead in every test."""
    s = MetricsSampler(snapshot_fn=_snap, resource_probe_fn=_counts)
    assert s._runtime_stats_fn is None
    s.sample_once()

    assert _value("st_process_open_fds") == 42.0


def test_an_unavailable_source_leaves_its_series_ABSENT():
    """Absent, never zero — a 0 here reads as 'no leak' on a host that simply
    cannot see /dev/shm, which is the exact mistake a host-side check made during
    the STABL-nstyyrhh investigation."""
    probe = lambda: ResourceCounts(
        leaked_semaphores=None, shm_segments=None, open_fds=17)
    MetricsSampler(snapshot_fn=_snap, resource_probe_fn=probe).sample_once()

    assert not _lines("st_process_leaked_semaphores")
    assert not _lines("st_process_shm_segments")
    assert _value("st_process_open_fds") == 17.0     # per-source, not all-or-nothing


def test_a_zero_reading_IS_rendered(tmp_path):
    """Readable-and-empty is a real measurement and must render, or the absent
    case would be indistinguishable from a genuinely clean host."""
    probe = lambda: ResourceCounts(leaked_semaphores=0, shm_segments=0, open_fds=5)
    MetricsSampler(snapshot_fn=_snap, resource_probe_fn=probe).sample_once()

    assert _value("st_process_leaked_semaphores") == 0.0
    assert _value("st_process_shm_segments") == 0.0


def test_a_raising_probe_still_writes_device_gauges():
    """The three readers are independent; one failing must not blank the others."""
    def _boom():
        raise RuntimeError("probe exploded")

    MetricsSampler(snapshot_fn=_snap, resource_probe_fn=_boom).sample_once()
    assert _value("st_device_total_bytes") == float(24 * GIB)


def test_a_raising_snapshot_still_writes_resource_gauges():
    def _boom():
        raise RuntimeError("NVML exploded")

    MetricsSampler(snapshot_fn=_boom, resource_probe_fn=_counts).sample_once()
    assert _value("st_process_open_fds") == 42.0


def test_the_default_probe_is_the_real_one():
    """Wiring check: an injectable with no default would sample nothing in
    production the day someone forgets to pass it."""
    s = MetricsSampler(snapshot_fn=_snap)
    assert s._resource_probe_fn is probe_resources


def test_disabled_sampler_never_probes(monkeypatch):
    monkeypatch.delenv("METRICS_ENABLED", raising=False)
    m.reset_metrics()

    def _must_not_run():
        raise AssertionError("disabled sampler MUST NOT probe")

    MetricsSampler(snapshot_fn=_snap, resource_probe_fn=_must_not_run).sample_once()
