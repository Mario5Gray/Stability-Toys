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


def _registry_family_names(met) -> set[str]:
    """Names as they appear in scraped output.

    Sourced from the REGISTRY, not from a rendered body: a labelled family with no
    observations yet renders no sample lines at all, so a body-derived set would
    silently shrink to the three unlabelled gauges and the contract test would
    pass while checking almost nothing.

    prometheus_client strips `_total` from a Counter's family name, so both
    spellings are offered and the caller accepts either.
    """
    names = set()
    for family in met._registry.collect():
        names.add(family.name)
        if family.type == "counter":
            names.add(f"{family.name}_total")
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
