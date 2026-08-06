"""Repo-local Prometheus facade (STABL-asawxgvp).

Plan: docs/superpowers/plans/2026-08-03-prometheus-substrate.md Tasks 1-2.
Spec: docs/superpowers/specs/2026-08-03-server-observability-seams-design.md
"""
import importlib.util

import pytest

from server import metrics as m


@pytest.fixture(autouse=True)
def _fresh():
    m.reset_metrics()
    yield
    m.reset_metrics()


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("METRICS_ENABLED", raising=False)
    assert m.get_metrics().enabled is False


def test_disabled_facade_accepts_every_call_unchanged(monkeypatch):
    monkeypatch.delenv("METRICS_ENABLED", raising=False)
    met = m.get_metrics()
    # Instrumentation code is unconditional: these must all be no-ops, not errors.
    met.job_terminal_total.labels(mode="SDXL", outcome="ok").inc()
    met.job_execution_seconds.labels(mode="SDXL").observe(1.5)
    met.device_free_bytes.labels(device_uuid="GPU-x").set(123)


def test_enabled_registers_and_renders(monkeypatch):
    monkeypatch.setenv("METRICS_ENABLED", "1")
    met = m.get_metrics()
    assert met.enabled is True
    met.job_terminal_total.labels(mode="SDXL", outcome="ok").inc()
    body, content_type = met.render()
    assert b"st_governor_job_terminal_total" in body
    assert "text/plain" in content_type


def test_disabled_render_is_empty(monkeypatch):
    monkeypatch.delenv("METRICS_ENABLED", raising=False)
    body, content_type = m.get_metrics().render()
    assert body == b""
    assert "text/plain" in content_type


def test_missing_prometheus_client_degrades_instead_of_raising(monkeypatch):
    monkeypatch.setenv("METRICS_ENABLED", "1")
    monkeypatch.setattr(m, "_import_prometheus", lambda: (_ for _ in ()).throw(ImportError()))
    met = m.get_metrics()
    assert met.enabled is False          # degraded, not crashed
    met.job_terminal_total.labels(mode="x", outcome="ok").inc()


def test_warns_when_multiple_web_workers(monkeypatch, caplog):
    monkeypatch.setenv("METRICS_ENABLED", "1")
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    with caplog.at_level("WARNING"):
        m.get_metrics()
    assert any("WEB_CONCURRENCY" in r.message for r in caplog.records)


def test_module_does_not_import_prometheus_at_module_scope():
    spec = importlib.util.find_spec("server.metrics")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as fh:
        head = [ln for ln in fh if ln.startswith(("import ", "from "))]
    assert not any("prometheus_client" in ln for ln in head), (
        "prometheus_client must be imported lazily so a missing dep degrades"
    )


# --- Task 2: family declarations ---

_ALL_FAMILIES = [
    "queue_depth", "jobs_in_flight", "job_queue_wait_seconds",
    "job_execution_seconds", "job_terminal_total", "wait_expired_total",
    "mode_load_seconds", "mode_switch_total", "demand_reload_total",
    "unload_total", "worker_recovery_total", "mode_active", "resolution_epoch",
    "device_total_bytes", "device_free_bytes", "device_used_bytes",
    "device_unattributed_bytes", "consumer_reserved_bytes",
    "consumer_allocated_bytes", "device_snapshot_stale",
    "http_requests_total", "http_request_duration_seconds",
    "ws_connections_active", "ws_sessions_total", "ws_messages_total",
    # STABL-cxbwwgly
    "process_leaked_semaphores", "process_shm_segments", "process_open_fds",
]


@pytest.mark.parametrize("name", _ALL_FAMILIES)
def test_every_family_exists_in_both_modes(name, monkeypatch):
    monkeypatch.delenv("METRICS_ENABLED", raising=False)
    assert hasattr(m.get_metrics(), name), f"disabled facade missing {name}"
    m.reset_metrics()
    monkeypatch.setenv("METRICS_ENABLED", "1")
    assert hasattr(m.get_metrics(), name), f"enabled facade missing {name}"


def test_no_forbidden_labels(monkeypatch):
    """job_id, pid and hostname are excluded for three DIFFERENT reasons (spec §5):
    job_id is unbounded; pid looks bounded and is not, because the subprocess
    handle mints a new one on every kill+respawn; hostname duplicates the
    scrape target's `instance` label and goes stale on a container move."""
    monkeypatch.setenv("METRICS_ENABLED", "1")
    met = m.get_metrics()
    for name in _ALL_FAMILIES:
        family = getattr(met, name)
        labels = set(getattr(family, "_labelnames", ()))
        assert "job_id" not in labels, f"{name} labels on job_id"
        assert "pid" not in labels, f"{name} labels on pid"
        assert "hostname" not in labels, f"{name} labels on hostname"


def test_histogram_buckets_cover_generation_timescales(monkeypatch):
    """prometheus_client's DEFAULT_BUCKETS top out at 10s. A generation runs for
    tens of seconds and a mode load for minutes, so defaults would put every real
    observation in +Inf."""
    monkeypatch.setenv("METRICS_ENABLED", "1")
    met = m.get_metrics()
    assert max(m.EXECUTION_BUCKETS) >= 600
    assert max(m.QUEUE_WAIT_BUCKETS) >= 900   # matches ADMISSION_TIMEOUT_S default
    assert max(m.MODE_LOAD_BUCKETS) >= 300
    met.job_execution_seconds.labels(mode="SDXL").observe(240.0)
    body, _ = met.render()
    assert b"st_governor_job_execution_seconds_bucket" in body


def test_http_duration_buckets_span_health_and_generate(monkeypatch):
    """/health answers in single-digit ms; /generate runs for minutes. One
    histogram covers both, so the buckets must too - prometheus defaults stop at
    10s and would bin every generation into +Inf."""
    monkeypatch.setenv("METRICS_ENABLED", "1")
    assert min(m.HTTP_DURATION_BUCKETS) <= 0.01
    assert max(m.HTTP_DURATION_BUCKETS) >= 600
    met = m.get_metrics()
    met.http_request_duration_seconds.labels(
        method="POST", route="/generate"
    ).observe(240.0)
    body, _ = met.render()
    assert b"st_http_request_duration_seconds_bucket" in body


def test_all_families_carry_the_st_namespace(monkeypatch):
    monkeypatch.setenv("METRICS_ENABLED", "1")
    met = m.get_metrics()
    # touch one series per family so every one renders
    met.queue_depth.set(0)
    met.jobs_in_flight.set(0)
    met.resolution_epoch.set(0)
    met.job_queue_wait_seconds.labels(mode="m").observe(1)
    met.job_execution_seconds.labels(mode="m").observe(1)
    met.mode_load_seconds.labels(mode="m").observe(1)
    met.job_terminal_total.labels(mode="m", outcome="ok").inc()
    met.wait_expired_total.labels(budget="execution").inc()
    met.mode_switch_total.labels(mode="m").inc()
    met.demand_reload_total.labels(mode="m").inc()
    met.unload_total.labels(mode="m", reason="explicit").inc()
    met.worker_recovery_total.labels(reason="oom").inc()
    met.mode_active.labels(mode="m").set(1)
    for g in ("device_total_bytes", "device_free_bytes", "device_used_bytes",
              "device_unattributed_bytes", "device_snapshot_stale"):
        getattr(met, g).labels(device_uuid="GPU-x").set(1)
    met.consumer_reserved_bytes.labels(device_uuid="GPU-x", consumer="worker").set(1)
    met.consumer_allocated_bytes.labels(device_uuid="GPU-x", consumer="worker").set(1)
    met.http_requests_total.labels(method="GET", route="/health", status="200").inc()
    met.http_request_duration_seconds.labels(method="GET", route="/health").observe(0.01)
    met.ws_connections_active.set(1)
    met.ws_sessions_total.inc()
    # "in"/"out" is the vocabulary ws_hub and ws_routes actually emit (Task 3)
    met.ws_messages_total.labels(type="ping", direction="in").inc()

    body, _ = met.render()
    emitted = {
        ln.split("{")[0].split(" ")[0]
        for ln in body.decode().splitlines()
        if ln and not ln.startswith("#")
    }
    assert emitted, "nothing rendered"
    assert all(n.startswith("st_") for n in emitted), (
        f"non-namespaced series: {sorted(n for n in emitted if not n.startswith('st_'))}"
    )


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
    assert any(
        line.startswith("st_ws_sessions_total ") and line.endswith("1.0")
        for line in body.decode().splitlines()
    )


def test_record_swallows_anything():
    """Instrumentation must never break a request or drop a WS connection."""
    m.record(lambda met: 1 / 0)
    m.record(lambda met: met.nope.labels(x=1))


def _registry_family_names(met) -> set[str]:
    """Names as they appear in scraped output.

    Sourced from the REGISTRY, not from a rendered body: a labelled family with no
    observations yet renders no sample lines at all, so a body-derived set would
    silently shrink to the three unlabelled gauges and the contract test would
    pass while checking almost nothing.

    prometheus_client strips `_total` from a Counter's family name, so both
    spellings are offered and the caller accepts either.

    Histogram CHILDREN are included for the same reason (STABL-cxbwwgly): `_count`
    is a genuinely emitted series and the normal way to query a histogram, so a
    contract that cannot name it cannot explain how to use its own histograms.
    `_created` stays out — it is a client-library artifact, not part of the
    contract.
    """
    names = set()
    for family in met._registry.collect():
        names.add(family.name)
        if family.type == "counter":
            names.add(f"{family.name}_total")
        elif family.type == "histogram":
            for suffix in ("_bucket", "_count", "_sum"):
                names.add(f"{family.name}{suffix}")
    return names


def test_every_family_is_documented_in_the_contract(monkeypatch):
    """The contract doc is the cross-repo interface. A family that ships without
    an entry is invisible to whoever writes the dashboards in ../continuous."""
    import pathlib
    import re

    monkeypatch.setenv("METRICS_ENABLED", "1")
    met = m.get_metrics()
    doc = pathlib.Path("docs/observability-contract.md").read_text()

    undocumented = []
    for family in met._registry.collect():
        spellings = {family.name}
        if family.type == "counter":
            spellings.add(f"{family.name}_total")
        elif family.type == "histogram":
            spellings.update(f"{family.name}{s}" for s in ("_bucket", "_count", "_sum"))
        if not any(s in doc for s in spellings):
            undocumented.append(family.name)
    assert not undocumented, f"undocumented metric families: {sorted(undocumented)}"

    # ...and the other direction: a doc entry for a metric that no longer exists
    # sends ../continuous chasing a series that will never appear.
    known = _registry_family_names(met)
    documented = set(re.findall(r"\bst_[a-z0-9_]+", doc))
    stale = sorted(n for n in documented if n not in known)
    assert not stale, f"contract documents metrics that do not exist: {stale}"


def test_outcome_enum_excludes_timeout(monkeypatch):
    """Ratified amendment (spec §5, 2026-08-03): a timeout is a waiter-side budget
    breach counted by wait_expired_total; the job that follows it reaches the
    dispatch loop as a cancel. Counting it as a terminal outcome would
    double-count the same job."""
    monkeypatch.setenv("METRICS_ENABLED", "1")
    met = m.get_metrics()
    for outcome in ("ok", "cancelled", "oom", "error"):
        met.job_terminal_total.labels(mode="m", outcome=outcome).inc()
    body, _ = met.render()
    assert b'outcome="timeout"' not in body
    # wait expiry is its own family, keyed on the budget that blew — not an outcome
    assert set(getattr(met.wait_expired_total, "_labelnames", ())) == {"budget"}
    assert set(getattr(met.job_terminal_total, "_labelnames", ())) == {"mode", "outcome"}


def test_the_contract_may_reference_histogram_children(monkeypatch):
    """_count is how you query a histogram. The contract documents a leak ratio
    built on st_governor_mode_load_seconds_count (STABL-cxbwwgly); if the
    known-set does not accept histogram children, that documentation cannot
    exist and the tempting fix is to delete the query instead."""
    monkeypatch.setenv("METRICS_ENABLED", "1")
    known = _registry_family_names(m.get_metrics())
    assert "st_governor_mode_load_seconds" in known
    assert "st_governor_mode_load_seconds_count" in known
    assert "st_governor_mode_load_seconds_sum" in known
    assert "st_governor_mode_load_seconds_bucket" in known
    # _created stays OUT: a client-library artifact, not part of the contract
    assert "st_governor_mode_load_seconds_created" not in known
    # and the widening must not have made the check vacuous
    assert "st_governor_mode_load_seconds_nonsense" not in known
