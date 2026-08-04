# Prometheus Substrate + Baseline Governor/DeviceMemory Metrics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, or execute it directly. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Do NOT use superpowers:subagent-driven-development.** `AGENTS.md` forbids sub-agent driven development in this repo, and repo instructions override the skill's default recommendation.

**FP:** STABL-asawxgvp (child of STABL-oxbwjwvu)
**Spec:** `docs/superpowers/specs/2026-08-03-server-observability-seams-design.md` (approved)

**Goal:** Give the server a scrapable `/metrics` endpoint backed by a repo-local facade, with baseline Governor and DeviceMemory series, inert by default.

**Architecture:** One module — `server/metrics.py` — owns the `prometheus_client` import, the enable gate, and every metric family declaration. Runtime code calls `get_metrics()` and touches metric objects unconditionally, because when the gate is off the facade hands back no-op objects. A background `MetricsSampler` thread owns all `DeviceMemory.snapshot()` fan-out on a fixed interval; the `/metrics` render path touches only in-memory gauges.

**Tech Stack:** Python 3, FastAPI/Starlette, `prometheus_client`, pytest. No new infrastructure.

## Global Constraints

- **`METRICS_ENABLED` default `off`.** Nothing changes in default repo behaviour until it is set.
- **`METRICS_SAMPLE_INTERVAL_S` default `15`**, env-overridable.
- **Labels are `device_uuid`, `mode`, plus `outcome` / `reason` / `consumer` / `budget` where declared.** `job_id`, `pid` and `hostname` appear in NO label set, ever — `job_id` and `hostname` belong in structured logs (`STABL-bpsfmoke`), host identity arrives as Prometheus's `instance` label from the scrape target, and `pid` looks bounded but is not (respawn mints a new one).
- **`server/metrics.py` is the only module that may import `prometheus_client`.**
- **`server/metrics.py` imports nothing from `backends/`.** Injection is by callable. (`backends/governor.py` already imports `server.mode_config` at module top, so `backends → server` is the established direction; the reverse is not.)
- **Nothing added may be able to kill the dispatch loop.** `STABL-hdzggeir` landed on this exact failure mode. Every metrics call site in `governor.py` must be inside a helper that cannot raise.
- **Metric namespace prefix `st_`.**
- Python env: `conda activate stability-toys`, then `python` (not `python3`).

---

## Design decision made during planning — read before Task 3

The spec (§5) says to observe job durations "on every terminal branch". Reading
`_dispatch_loop` shows that is five scattered sites (early-cancel `continue` at `:895`,
in-proc post-execute cancel, in-proc success, subprocess terminal, and
`_deliver_job_failure`) — five chances to miss one, and five new statements inside the
loop's `try`.

**Every one of them funnels through `_finalize_job_record(job_id)`.** That is the single
choke point, and by the time it is called the job's future is resolved on every path. So
the plan instruments `_finalize_job_record` once and derives the outcome from the future.

**Consequence for the outcome enum, which diverges from the spec and needs Sigma's eye:**
`timeout` is not a dispatch terminal. It is a *waiter-side* event — `wait_for_result`
raises `TimeoutError` at `_expire` (`:716`), and what reaches the dispatch loop afterwards
is a *cancel*. Counting it as a terminal outcome would either double-count (once as
`timeout`, once as `cancelled`) or lie about which one happened.

- `st_governor_job_terminal_total{mode,outcome}` — `outcome` ∈ `ok` / `cancelled` / `oom` / `error`
- `st_governor_wait_expired_total{budget}` — `budget` ∈ `admission` / `execution`, emitted at `_expire`

The operator question "how often do jobs time out?" is answered by the second series, and
"what happened to the job afterwards" by the first. Flag this to Sigma at review; it is a
strict improvement in fidelity, not a scope change.

---

## File Structure

| File | Responsibility |
|---|---|
| `server/metrics.py` (create) | `prometheus_client` import, gate, no-op objects, all family declarations, `get_metrics()` / `reset_metrics()` singleton, `render()` |
| `server/metrics_sampler.py` (create) | `MetricsSampler` thread: owns all `snapshot()` fan-out, writes device + queue gauges |
| `server/metrics_routes.py` (create) | `/metrics` route |
| `backends/governor.py` (modify) | `JobRecord.enqueued_at`; terminal observation at `_finalize_job_record`; lifecycle counters |
| `server/lcm_sr_server.py` (modify) | mount `/metrics` **before** the static mount; start/stop sampler in lifespan |
| `requirements.txt` (modify) | `prometheus_client` |
| `docs/observability-contract.md` (create) | the cross-repo metric-name contract |

Families live in `server/metrics.py` with the gate rather than in a separate module: they
change together on every metric added, and the skill's own guidance is that files which
change together live together. The sampler and route are separate because they have
genuinely different lifecycles (a thread vs. a route) and different test setups.

---

## Task 1: The facade and its gate

**Files:**
- Create: `server/metrics.py`
- Modify: `requirements.txt`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `get_metrics() -> Metrics`, `reset_metrics() -> None`, `Metrics.enabled: bool`, `Metrics.render() -> tuple[bytes, str]`. The singleton pattern deliberately mirrors `get_device_memory()` / `reset_device_memory()` in `backends/device_memory.py`.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
import importlib
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
    src = importlib.util.find_spec("server.metrics").origin
    with open(src) as fh:
        head = [ln for ln in fh if ln.startswith(("import ", "from "))]
    assert not any("prometheus_client" in ln for ln in head), (
        "prometheus_client must be imported lazily so a missing dep degrades"
    )
```

- [x] **Step 2: Run tests to verify they fail**

Run: `conda activate stability-toys && python -m pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.metrics'`

- [x] **Step 3: Add the dependency**

Append to `requirements.txt`:

```text
prometheus_client==0.21.1
```

Pure Python, no transitive dependencies — unlike the Compel situation that forced
`--no-deps` in `requirements-conditioning.txt`, this needs no special install handling.

Install into the dev env: `conda activate stability-toys && python -m pip install prometheus_client==0.21.1`

- [x] **Step 4: Write the facade**

```python
# server/metrics.py
"""Repo-local Prometheus facade (STABL-asawxgvp).

The ONLY module permitted to import prometheus_client. Runtime code calls
get_metrics() and touches metric objects unconditionally: when the gate is off
the facade returns no-op objects, so instrumentation sites carry no branches.

Imports nothing from backends/ — the sampler injects readers by callable.
"""
import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

QUEUE_WAIT_BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 900)
EXECUTION_BUCKETS = (0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600)
MODE_LOAD_BUCKETS = (1, 2, 5, 10, 20, 30, 60, 120, 300)

_PLAIN_TEXT = "text/plain; version=0.0.4; charset=utf-8"


def _import_prometheus():
    """Seam: patched in tests to prove the ImportError degrade path."""
    import prometheus_client
    return prometheus_client


class _NoopMetric:
    """Accepts every metric call and does nothing. Returns self from labels()
    so chained calls work identically to the real object."""

    def labels(self, *args, **kwargs):
        return self

    def inc(self, *args, **kwargs):
        pass

    def observe(self, *args, **kwargs):
        pass

    def set(self, *args, **kwargs):
        pass


class Metrics:
    """Holds every declared family. Attribute names are stable API for call sites."""

    # NOTE ON `hostname`: deliberately not a label on any family (spec §5, amended
    # 2026-08-03). Prometheus attaches host identity as the `instance` label from
    # the scrape target; a second host label on every series would duplicate it and
    # go stale after a container move. hostname remains a structured-LOG field
    # (STABL-bpsfmoke). Recorded in docs/observability-contract.md so ../continuous
    # does not go looking for a label that is absent on purpose.

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._registry = None
        if enabled:
            try:
                prom = _import_prometheus()
            except ImportError:
                logger.warning(
                    "METRICS_ENABLED is set but prometheus_client is not installed; "
                    "metrics degrade to no-op"
                )
                self.enabled = False
            else:
                self._registry = prom.CollectorRegistry()
                self._declare(prom)
                return
        self._declare_noop()

    # --- family declarations (Task 2 fills these in) ---

    def _declare(self, prom):
        raise NotImplementedError  # Task 2

    def _declare_noop(self):
        raise NotImplementedError  # Task 2

    def render(self) -> tuple[bytes, str]:
        if not self.enabled or self._registry is None:
            return b"", _PLAIN_TEXT
        prom = _import_prometheus()
        return prom.generate_latest(self._registry), _PLAIN_TEXT


_metrics: Optional[Metrics] = None
_metrics_lock = threading.Lock()


def _enabled_from_env() -> bool:
    return os.getenv("METRICS_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def get_metrics() -> Metrics:
    """Singleton accessor (get_device_memory pattern).

    Deliberately called at each site rather than cached on the caller: the sites
    are cold (once per job, once per load, once per sample interval), and caching
    a facade on a long-lived object is how patch targets drift out from under
    tests (see the STABL-anxqlxkm torch-binding class of failure).
    """
    global _metrics
    with _metrics_lock:
        if _metrics is None:
            enabled = _enabled_from_env()
            if enabled:
                try:
                    concurrency = int(os.getenv("WEB_CONCURRENCY", "1"))
                except ValueError:
                    concurrency = 1
                if concurrency > 1:
                    logger.warning(
                        "WEB_CONCURRENCY=%s with METRICS_ENABLED: prometheus_client's "
                        "registry is process-local, so /metrics reports whichever worker "
                        "answered the scrape and counters will appear to go backwards. "
                        "Run a single uvicorn worker or take multiprocess mode.",
                        concurrency,
                    )
            _metrics = Metrics(enabled)
        return _metrics


def reset_metrics() -> None:
    """Test seam (reset_device_memory pattern)."""
    global _metrics
    with _metrics_lock:
        _metrics = None
```

- [x] **Step 5: Run tests — expect all but one to fail**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: **1 passed, 6 failed** — only
`test_module_does_not_import_prometheus_at_module_scope` passes, because it reads the
source file and never constructs a `Metrics`. Every other test calls `get_metrics()`,
which constructs `Metrics` and hits `NotImplementedError` in `_declare` / `_declare_noop`.

**Task 1 has no independently green state**, and that is by design: the facade's gate is
only observable once there is at least one family to observe it through. Task 2 turns the
file green. Do not try to "fix" the failures here.

(Corrected during execution — this step previously claimed three tests would pass.)

- [x] **Step 6: Commit**

```bash
git add server/metrics.py tests/test_metrics.py requirements.txt
git commit -m "feat(metrics): repo-local Prometheus facade + enable gate (STABL-asawxgvp)

Gate lives inside the facade so instrumentation sites stay unconditional.
METRICS_ENABLED defaults off; missing prometheus_client degrades to no-op
rather than raising. Singleton mirrors get_device_memory/reset_device_memory.

next: Task 2 declare the metric families"
```

---

## Task 2: Declare the metric families

**Files:**
- Modify: `server/metrics.py` (`_declare` / `_declare_noop`)
- Test: `tests/test_metrics.py` (extend)

**Interfaces:**
- Consumes: `Metrics` from Task 1.
- Produces: these attribute names, relied on by every later task —
  `queue_depth`, `jobs_in_flight`, `job_queue_wait_seconds`, `job_execution_seconds`,
  `job_terminal_total`, `wait_expired_total`, `mode_load_seconds`, `mode_switch_total`,
  `demand_reload_total`, `unload_total`, `worker_recovery_total`, `mode_active`,
  `resolution_epoch`, `device_total_bytes`, `device_free_bytes`, `device_used_bytes`,
  `device_unattributed_bytes`, `consumer_reserved_bytes`, `consumer_allocated_bytes`,
  `device_snapshot_stale`.

- [x] **Step 1: Write the failing tests**

Append to `tests/test_metrics.py`:

```python
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
    """job_id and pid are unbounded label values (spec §5). pid looks bounded and
    is not: the subprocess handle mints a new pid on every kill+respawn."""
    monkeypatch.setenv("METRICS_ENABLED", "1")
    met = m.get_metrics()
    for name in _ALL_FAMILIES:
        family = getattr(met, name)
        labels = set(getattr(family, "_labelnames", ()))
        assert "job_id" not in labels, f"{name} labels on job_id"
        assert "pid" not in labels, f"{name} labels on pid"


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
    assert b'st_governor_job_execution_seconds_bucket' in body
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL with `NotImplementedError` from `_declare` / `_declare_noop`.

- [x] **Step 3: Implement the declarations**

Replace the two placeholder methods in `server/metrics.py`:

```python
    def _declare(self, prom):
        C, G, H = prom.Counter, prom.Gauge, prom.Histogram
        r = self._registry

        # --- Governor: queue pressure ---
        self.queue_depth = G(
            "st_governor_queue_depth", "Jobs currently queued", registry=r)
        self.jobs_in_flight = G(
            "st_governor_jobs_in_flight", "Jobs currently executing", registry=r)
        self.job_queue_wait_seconds = H(
            "st_governor_job_queue_wait_seconds",
            "Seconds between enqueue and execution start",
            ["mode"], buckets=QUEUE_WAIT_BUCKETS, registry=r)

        # --- Governor: execution cost ---
        self.job_execution_seconds = H(
            "st_governor_job_execution_seconds",
            "Seconds spent executing a job",
            ["mode"], buckets=EXECUTION_BUCKETS, registry=r)
        self.mode_load_seconds = H(
            "st_governor_mode_load_seconds",
            "Seconds to load a mode's worker",
            ["mode"], buckets=MODE_LOAD_BUCKETS, registry=r)

        # --- Governor: failure paths ---
        self.job_terminal_total = C(
            "st_governor_job_terminal_total",
            "Jobs by terminal outcome (ok|cancelled|oom|error)",
            ["mode", "outcome"], registry=r)
        self.wait_expired_total = C(
            "st_governor_wait_expired_total",
            "Waiters that exceeded their budget (admission|execution)",
            ["budget"], registry=r)
        self.worker_recovery_total = C(
            "st_governor_worker_recovery_total",
            "Worker kill+respawn recoveries by cause (oom|dead)",
            ["reason"], registry=r)

        # --- Governor: churn ---
        self.mode_switch_total = C(
            "st_governor_mode_switch_total",
            "Mode switches, labelled by TARGET mode",
            ["mode"], registry=r)
        self.demand_reload_total = C(
            "st_governor_demand_reload_total",
            "Reloads from a retained snapshot after idle eviction",
            ["mode"], registry=r)
        self.unload_total = C(
            "st_governor_unload_total", "Model unloads by reason",
            ["mode", "reason"], registry=r)

        # --- Governor: authority ---
        self.mode_active = G(
            "st_governor_mode_active",
            "1 for the loaded mode, 0 for every other configured mode",
            ["mode"], registry=r)
        self.resolution_epoch = G(
            "st_governor_resolution_epoch", "Current resolution epoch", registry=r)

        # --- DeviceMemory: capacity truth ---
        self.device_total_bytes = G(
            "st_device_total_bytes", "Device total bytes",
            ["device_uuid"], registry=r)
        self.device_free_bytes = G(
            "st_device_free_bytes", "Driver-truth free bytes",
            ["device_uuid"], registry=r)
        self.device_used_bytes = G(
            "st_device_used_bytes", "Driver-truth used bytes",
            ["device_uuid"], registry=r)
        self.device_unattributed_bytes = G(
            "st_device_unattributed_bytes",
            "Used bytes not attributed to any registered consumer pool "
            "(CUDA contexts, non-torch workspaces, other processes)",
            ["device_uuid"], registry=r)
        self.consumer_reserved_bytes = G(
            "st_consumer_reserved_bytes", "Per-consumer framework pool, reserved",
            ["device_uuid", "consumer"], registry=r)
        self.consumer_allocated_bytes = G(
            "st_consumer_allocated_bytes", "Per-consumer framework pool, allocated",
            ["device_uuid", "consumer"], registry=r)
        self.device_snapshot_stale = G(
            "st_device_snapshot_stale",
            "1 when the last snapshot substituted last-known values after a "
            "consumer fan-out timeout (i.e. a consumer is not answering)",
            ["device_uuid"], registry=r)

    def _declare_noop(self):
        for name in (
            "queue_depth", "jobs_in_flight", "job_queue_wait_seconds",
            "job_execution_seconds", "job_terminal_total", "wait_expired_total",
            "mode_load_seconds", "mode_switch_total", "demand_reload_total",
            "unload_total", "worker_recovery_total", "mode_active",
            "resolution_epoch", "device_total_bytes", "device_free_bytes",
            "device_used_bytes", "device_unattributed_bytes",
            "consumer_reserved_bytes", "consumer_allocated_bytes",
            "device_snapshot_stale",
        ):
            setattr(self, name, _NoopMetric())
```

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: ALL PASS.

- [x] **Step 5: Commit**

```bash
git add server/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): declare baseline Governor + DeviceMemory families (STABL-asawxgvp)

Explicit histogram buckets: prometheus defaults stop at 10s and would bin every
real generation and mode load into +Inf. Tests pin that no family labels on
job_id or pid — pid looks bounded but the subprocess handle mints a new one per
kill+respawn.

next: Task 3 JobRecord.enqueued_at + terminal observation"
```

---

## Task 3: `JobRecord.enqueued_at` and terminal observation

**Files:**
- Modify: `backends/governor.py` — `JobRecord` (`:164-178`), `_register_job` (`:585`), `_finalize_job_record` (`:592`)
- Test: `tests/test_governor_metrics.py` (create)

**Interfaces:**
- Consumes: `get_metrics()` from Task 1; `job_terminal_total`, `job_queue_wait_seconds`, `job_execution_seconds` from Task 2.
- Produces: `JobRecord.enqueued_at: Optional[float]`; `Governor._observe_job_terminal(record) -> None` (never raises); module function `_terminal_outcome(fut) -> str`.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_governor_metrics.py
import time
from concurrent.futures import Future, CancelledError
from unittest.mock import Mock

import pytest

from server import metrics as m
from backends.governor import Governor, GenerationJob, JobRecord, _terminal_outcome


@pytest.fixture(autouse=True)
def _metrics_on(monkeypatch):
    monkeypatch.setenv("METRICS_ENABLED", "1")
    m.reset_metrics()
    yield
    m.reset_metrics()


def _sample(body: bytes, name: str) -> list[str]:
    return [ln for ln in body.decode().splitlines() if ln.startswith(name)]


def test_register_job_stamps_enqueued_at():
    job = GenerationJob(req=Mock(), resolution_epoch=1)
    rec = JobRecord(job_id=job.job_id, state="queued", job=job)
    assert hasattr(rec, "enqueued_at")
    assert rec.enqueued_at is None      # default; _register_job stamps it


def test_terminal_outcome_ok():
    fut = Future()
    fut.set_result("png")
    assert _terminal_outcome(fut) == "ok"


def test_terminal_outcome_cancelled_from_exception():
    fut = Future()
    fut.set_exception(CancelledError())
    assert _terminal_outcome(fut) == "cancelled"


def test_terminal_outcome_cancelled_future_does_not_raise():
    """fut.exception() RAISES CancelledError on a cancelled future, so the
    cancelled() check must come first."""
    fut = Future()
    fut.cancel()
    assert _terminal_outcome(fut) == "cancelled"


def test_terminal_outcome_oom():
    fut = Future()
    fut.set_exception(RuntimeError("CUDA out of memory. Tried to allocate 96.00 MiB"))
    assert _terminal_outcome(fut) == "oom"


def test_terminal_outcome_error():
    fut = Future()
    fut.set_exception(ValueError("nope"))
    assert _terminal_outcome(fut) == "error"


def test_observe_emits_durations_and_outcome(governor_with_stub_handle):
    gov = governor_with_stub_handle
    job = GenerationJob(req=Mock(), resolution_epoch=1)
    job.fut.set_result("png")
    now = time.monotonic()
    rec = JobRecord(job_id=job.job_id, state="running", job=job)
    rec.enqueued_at = now - 10.0
    rec.executing_since = now - 4.0

    gov._observe_job_terminal(rec)

    body, _ = m.get_metrics().render()
    assert _sample(body, "st_governor_job_terminal_total")
    assert any("outcome=\"ok\"" in ln for ln in _sample(body, "st_governor_job_terminal_total"))
    assert _sample(body, "st_governor_job_queue_wait_seconds_count")
    assert _sample(body, "st_governor_job_execution_seconds_count")


def test_observe_skips_a_job_that_never_ran(governor_with_stub_handle):
    """The queue.Full rollback in submit_job finalizes a record whose future was
    never resolved. That is not a terminal and must not be counted."""
    gov = governor_with_stub_handle
    job = GenerationJob(req=Mock(), resolution_epoch=1)
    rec = JobRecord(job_id=job.job_id, state="queued", job=job)
    rec.enqueued_at = time.monotonic()

    gov._observe_job_terminal(rec)      # future not done

    body, _ = m.get_metrics().render()
    assert not [ln for ln in _sample(body, "st_governor_job_terminal_total")
                if not ln.startswith("# ")]


def test_observe_cannot_raise(governor_with_stub_handle, monkeypatch):
    """STABL-hdzggeir: nothing added here may be able to kill the dispatch loop."""
    gov = governor_with_stub_handle

    class _Explodes:
        def labels(self, **kw):
            raise RuntimeError("metrics backend died")

    met = m.get_metrics()
    monkeypatch.setattr(met, "job_terminal_total", _Explodes())
    job = GenerationJob(req=Mock(), resolution_epoch=1)
    job.fut.set_result("png")
    rec = JobRecord(job_id=job.job_id, state="running", job=job)
    rec.enqueued_at = time.monotonic() - 1
    rec.executing_since = time.monotonic() - 0.5

    gov._observe_job_terminal(rec)      # must return normally


def test_finalize_observes_once_and_pops(governor_with_stub_handle):
    gov = governor_with_stub_handle
    job = GenerationJob(req=Mock(), resolution_epoch=1)
    job.fut.set_result("png")
    gov._register_job(job)
    rec = gov._get_job_record(job.job_id)
    assert rec.enqueued_at is not None, "_register_job must stamp enqueued_at"
    rec.executing_since = time.monotonic()

    gov._finalize_job_record(job.job_id)

    assert gov._get_job_record(job.job_id) is None
    body, _ = m.get_metrics().render()
    counted = [ln for ln in _sample(body, "st_governor_job_terminal_total")
               if not ln.startswith("# ")]
    assert len(counted) == 1
```

Add the shared fixture to the same file:

```python
from tests.test_governor import StubHandle, _make_multi_mode_config, _make_mock_registry


@pytest.fixture
def governor_with_stub_handle():
    gov = Governor(
        handle=StubHandle(),
        mode_config=_make_multi_mode_config("mode-a", default="mode-a"),
        registry=_make_mock_registry(),
    )
    yield gov
    gov.shutdown()
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_governor_metrics.py -v`
Expected: FAIL — `ImportError: cannot import name '_terminal_outcome'`.

- [x] **Step 3: Implement**

In `backends/governor.py`, add to the imports near `:41`:

```python
from server.metrics import get_metrics
```

Add the field to `JobRecord` (after `executing_since`):

```python
    # STABL-asawxgvp: monotonic enqueue time, stamped in _register_job. Paired with
    # executing_since this splits the two budgets into two observable durations —
    # queue wait and execution — which nothing could derive before.
    enqueued_at: Optional[float] = None
```

Add the module-level classifier next to `_is_oom` (`:181`):

```python
def _terminal_outcome(fut) -> str:
    """Terminal outcome from a RESOLVED future.

    fut.cancelled() is checked FIRST because fut.exception() raises
    CancelledError on a cancelled future — the obvious ordering is a bug.
    """
    if fut.cancelled():
        return "cancelled"
    exc = fut.exception()
    if exc is None:
        return "ok"
    if isinstance(exc, CancelledError):
        return "cancelled"
    if _is_oom(exc):
        return "oom"
    return "error"
```

Stamp in `_register_job`:

```python
    def _register_job(self, job: Job):
        if isinstance(job, GenerationJob):
            with self._job_lock:
                self._job_records[job.job_id] = JobRecord(
                    job_id=job.job_id, state="queued", job=job,
                    enqueued_at=time.monotonic(),
                )
```

Replace `_finalize_job_record` and add the observer:

```python
    def _finalize_job_record(self, job_id: str):
        with self._job_lock:
            record = self._job_records.pop(job_id, None)
        # Observed OUTSIDE _job_lock: the backplane's Subscriber<->lock invariant,
        # and there is no reason to hold the lock across a metrics write.
        if record is not None:
            self._observe_job_terminal(record)

    def _observe_job_terminal(self, record: JobRecord) -> None:
        """Emit terminal metrics for a finished job.

        This is the SINGLE instrumentation point for job terminals: every branch
        of the dispatch loop — early cancel, in-proc success, post-execute cancel,
        subprocess terminal, and _deliver_job_failure — funnels through
        _finalize_job_record, and by then the future is resolved on all of them.

        MUST NOT raise (STABL-hdzggeir): a throw here reaches the dispatch loop's
        try, kills the thread, and permanently deadens the queue.
        """
        try:
            fut = record.job.fut
            if record.enqueued_at is None or not fut.done():
                return          # e.g. the submit_job queue.Full rollback
            mode = self._current_mode or "unknown"
            met = get_metrics()
            if record.executing_since is not None:
                met.job_queue_wait_seconds.labels(mode=mode).observe(
                    max(0.0, record.executing_since - record.enqueued_at))
                met.job_execution_seconds.labels(mode=mode).observe(
                    max(0.0, time.monotonic() - record.executing_since))
            met.job_terminal_total.labels(
                mode=mode, outcome=_terminal_outcome(fut)).inc()
        except Exception:
            logger.debug("[Governor] terminal metrics failed", exc_info=True)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_governor_metrics.py -v`
Expected: ALL PASS.

- [x] **Step 5: Run the existing Governor suite for regressions**

Run: `python -m pytest tests/test_governor.py tests/test_worker_pool.py -q`
Expected: same pass count as before this task. `_finalize_job_record` changed shape;
this proves no caller depended on its old return.

- [x] **Step 6: Commit**

```bash
git add backends/governor.py tests/test_governor_metrics.py
git commit -m "feat(metrics): job terminal outcome + queue-wait/execution histograms (STABL-asawxgvp)

Instruments the single choke point: every dispatch-loop terminal branch funnels
through _finalize_job_record, and the future is resolved on all of them, so the
outcome is derived rather than passed from five scattered sites. Adds
JobRecord.enqueued_at — executing_since existed, the enqueue stamp did not, so
queue wait was not derivable at all.

_observe_job_terminal cannot raise (STABL-hdzggeir). _terminal_outcome checks
fut.cancelled() before fut.exception(), which raises on a cancelled future.

next: Task 4 Governor lifecycle counters"
```

---

## Task 4: Governor lifecycle counters

**Files:**
- Modify: `backends/governor.py` — `_load_mode` (`:488`), `_reload_from_snapshot` (`:556`), `_expire` (`:716`), `_unload_current_worker` (`:780`), the subprocess recovery branch (`:1016-1026`), `_reserve_and_enqueue_switch` (`:1164`)
- Test: `tests/test_governor_metrics.py` (extend)

**Interfaces:**
- Consumes: `get_metrics()`; families from Task 2.
- Produces: no new symbols — call sites only.

- [x] **Step 1: Write the failing tests**

Append to `tests/test_governor_metrics.py`:

```python
def test_mode_load_observes_duration_and_sets_active(governor_with_stub_handle, monkeypatch):
    from unittest.mock import patch
    gov = governor_with_stub_handle
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov._load_mode("mode-a")
    body, _ = m.get_metrics().render()
    assert _sample(body, "st_governor_mode_load_seconds_count")
    assert any('mode="mode-a"' in ln and ln.rstrip().endswith("1.0")
               for ln in _sample(body, "st_governor_mode_active"))
    assert _sample(body, "st_governor_resolution_epoch")


def test_expire_counts_the_budget_that_blew(governor_with_stub_handle):
    gov = governor_with_stub_handle
    with pytest.raises(TimeoutError):
        gov._expire(None, "execution", 120.0, 121.0)
    body, _ = m.get_metrics().render()
    assert any('budget="execution"' in ln
               for ln in _sample(body, "st_governor_wait_expired_total"))


def test_subprocess_recovery_counter_labels_the_cause(governor_with_stub_handle):
    gov = governor_with_stub_handle
    gov._count_worker_recovery(oom=True)
    body, _ = m.get_metrics().render()
    assert any('reason="oom"' in ln
               for ln in _sample(body, "st_governor_worker_recovery_total"))
    gov._count_worker_recovery(oom=False)
    body, _ = m.get_metrics().render()
    assert any('reason="dead"' in ln
               for ln in _sample(body, "st_governor_worker_recovery_total"))


def test_lifecycle_counters_cannot_raise(governor_with_stub_handle, monkeypatch):
    gov = governor_with_stub_handle

    class _Explodes:
        def labels(self, **kw):
            raise RuntimeError("boom")

    monkeypatch.setattr(m.get_metrics(), "worker_recovery_total", _Explodes())
    gov._count_worker_recovery(oom=True)     # must return normally
```

Import `_resolve_by_path` alongside the other test helpers at the top of the file:

```python
from tests.test_governor import (
    StubHandle, _make_multi_mode_config, _make_mock_registry, _resolve_by_path,
)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_governor_metrics.py -k "load or expire or recovery or lifecycle" -v`
Expected: FAIL — `AttributeError: 'Governor' object has no attribute '_count_worker_recovery'`.

- [x] **Step 3: Implement**

Add one non-raising helper to `Governor` (place it directly after `_observe_job_terminal`):

```python
    def _metric(self, fn) -> None:
        """Run a metrics side effect that must never reach the dispatch loop.

        Every lifecycle counter goes through here rather than growing its own
        try/except at the call site — one place to be certain about, and the call
        sites stay one line (STABL-hdzggeir).
        """
        try:
            fn(get_metrics())
        except Exception:
            logger.debug("[Governor] metrics side effect failed", exc_info=True)

    def _count_worker_recovery(self, *, oom: bool) -> None:
        self._metric(lambda met: met.worker_recovery_total.labels(
            reason="oom" if oom else "dead").inc())
```

In `_load_mode`, capture the start time immediately after the entry log line:

```python
        logger.info(f"[Governor] Loading mode: {mode_name}")
        _load_started = time.monotonic()
```

and observe just before the trailing `self._start_dispatch_thread()`:

```python
        logger.info(f"[Governor] Mode '{mode_name}' loaded (epoch={reservation.resolution_epoch})")
        self._metric(lambda met: (
            met.mode_load_seconds.labels(mode=mode_name).observe(
                time.monotonic() - _load_started),
            met.resolution_epoch.set(reservation.resolution_epoch),
            self._publish_mode_active(met, mode_name),
        ))

        # Start the dispatch thread (same as WorkerPool._start_worker_thread at :428)
        self._start_dispatch_thread()
```

A load that raises never reaches this line, which is correct: a failed load has no
duration to report. Failed loads are visible as `mode_active` staying 0 everywhere.

Add the mode-set publisher next to `_count_worker_recovery`:

```python
    def _publish_mode_active(self, met, active: Optional[str]) -> None:
        """0/1 for EVERY configured mode.

        A single gauge labelled with the current mode leaves a stale 1 on the
        previous mode's series forever. conf/modes.yml holds 4 modes, so
        reporting all of them is cheap (spec resolved question 3).
        """
        for name in self._mode_config.list_modes():
            met.mode_active.labels(mode=name).set(1 if name == active else 0)
```

`ModeConfigManager.list_modes()` (`server/mode_config.py:1002`) returns every configured
mode name; `Governor` holds the manager as `self._mode_config` (`:370`).

In `_expire`, count before raising:

```python
    def _expire(self, record: Optional[JobRecord], budget: str, limit_s: float, waited_s: float):
        knob = "ADMISSION_TIMEOUT_S" if budget == "admission" else "DEFAULT_TIMEOUT"
        who = f"job {record.job_id}" if record is not None else "job"
        self._metric(lambda met: met.wait_expired_total.labels(budget=budget).inc())
        if record is not None:
            self.cancel_job(record.job_id)
        raise TimeoutError(...)      # unchanged
```

In `_reload_from_snapshot`, after the reload succeeds:

```python
        self._metric(lambda met: met.demand_reload_total.labels(
            mode=self._current_mode or "unknown").inc())
```

In `_unload_current_worker`:

```python
        self._metric(lambda met: (
            met.unload_total.labels(
                mode=self._current_mode or "unknown", reason="explicit").inc(),
            self._publish_mode_active(met, None),
        ))
```

In `_reserve_and_enqueue_switch`, after the reservation is appended:

```python
        self._metric(lambda met: met.mode_switch_total.labels(mode=mode_name).inc())
```

Target mode only — a `{from,to}` pair squares the cardinality for a transition matrix
nobody asked for.

In the dispatch loop's subprocess recovery branch, directly after the existing
`logger.warning("[Governor] Subprocess needs recovery ...")`:

```python
                                self._count_worker_recovery(oom=oom)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_governor_metrics.py -v`
Expected: ALL PASS.

- [x] **Step 5: Run the full Governor suite**

Run: `python -m pytest tests/test_governor.py tests/test_worker_pool.py tests/test_device_memory.py -q`
Expected: no new failures.

- [x] **Step 6: Commit**

```bash
git add backends/governor.py tests/test_governor_metrics.py
git commit -m "feat(metrics): Governor lifecycle counters — load, churn, recovery, wait expiry (STABL-asawxgvp)

All sites go through Governor._metric(), one non-raising wrapper, so no counter
can reach the dispatch loop's try (STABL-hdzggeir). mode_active publishes 0/1
for every configured mode rather than labelling one gauge with the current mode,
which would leave a stale 1 on the previous mode forever.

Wait expiry is counted at _expire as its own series rather than as a terminal
outcome: a timeout is a waiter-side event and what reaches the dispatch loop
afterwards is a cancel. Counting it as a terminal would double-count.

next: Task 5 MetricsSampler"
```

---

## Task 5: The sampler — the only thing that fans out

**Files:**
- Create: `server/metrics_sampler.py`
- Test: `tests/test_metrics_sampler.py` (create)

**Interfaces:**
- Consumes: `get_metrics()`; the `DeviceMemorySnapshot` / `ConsumerMemory` shapes from `backends/device_memory.py` (duck-typed — the sampler imports nothing from `backends/`).
- Produces: `MetricsSampler(snapshot_fn, runtime_stats_fn=None, interval_s=None)` with `.start()`, `.stop()`, `.sample_once()`. `runtime_stats_fn` returns `{"queue_depth": int, "jobs_in_flight": int}` or `None`.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_metrics_sampler.py
import threading
import time

import pytest

from server import metrics as m
from server.metrics_sampler import MetricsSampler
from backends.device_memory import ConsumerMemory, DeviceMemorySnapshot, MemoryTopology


@pytest.fixture(autouse=True)
def _metrics_on(monkeypatch):
    monkeypatch.setenv("METRICS_ENABLED", "1")
    m.reset_metrics()
    yield
    m.reset_metrics()


def _snap(stale=False):
    return DeviceMemorySnapshot(
        device_uuid="GPU-fake",
        topology=MemoryTopology.DISCRETE,
        total_bytes=24 * 1024**3,
        free_bytes=20 * 1024**3,
        consumers=(
            ConsumerMemory(label="server", pid=1,
                           allocated_bytes=1 * 1024**3,
                           reserved_bytes=2 * 1024**3, stale=stale),
            ConsumerMemory(label="worker", pid=154,
                           allocated_bytes=1 * 1024**3,
                           reserved_bytes=1 * 1024**3, stale=stale),
        ),
    )


def _lines(prefix):
    body, _ = m.get_metrics().render()
    return [ln for ln in body.decode().splitlines()
            if ln.startswith(prefix) and not ln.startswith("# ")]


def test_sample_once_writes_device_gauges():
    MetricsSampler(snapshot_fn=_snap).sample_once()
    assert any("2.5769803776e+010" in ln or "25769803776" in ln
               for ln in _lines("st_device_total_bytes"))
    assert _lines("st_device_free_bytes")
    assert _lines("st_device_used_bytes")
    # used 4GiB - reserved 3GiB = 1GiB unattributed
    assert _lines("st_device_unattributed_bytes")


def test_sample_once_writes_per_consumer_gauges_by_label_not_pid():
    MetricsSampler(snapshot_fn=_snap).sample_once()
    reserved = _lines("st_consumer_reserved_bytes")
    assert any('consumer="server"' in ln for ln in reserved)
    assert any('consumer="worker"' in ln for ln in reserved)
    assert not any("pid" in ln for ln in reserved), (
        "pid must not reach a label: the subprocess handle mints a new one per respawn"
    )


def test_stale_snapshot_sets_the_stale_gauge():
    MetricsSampler(snapshot_fn=lambda: _snap(stale=True)).sample_once()
    assert any(ln.rstrip().endswith("1.0") for ln in _lines("st_device_snapshot_stale"))


def test_runtime_stats_feed_queue_gauges():
    MetricsSampler(
        snapshot_fn=_snap,
        runtime_stats_fn=lambda: {"queue_depth": 3, "jobs_in_flight": 1},
    ).sample_once()
    assert any(ln.rstrip().endswith("3.0") for ln in _lines("st_governor_queue_depth"))
    assert any(ln.rstrip().endswith("1.0") for ln in _lines("st_governor_jobs_in_flight"))


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
    assert not s._thread.is_alive()


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


def test_interval_from_env(monkeypatch):
    monkeypatch.setenv("METRICS_SAMPLE_INTERVAL_S", "42")
    assert MetricsSampler(snapshot_fn=_snap).interval_s == 42.0


def test_default_interval_is_15(monkeypatch):
    monkeypatch.delenv("METRICS_SAMPLE_INTERVAL_S", raising=False)
    assert MetricsSampler(snapshot_fn=_snap).interval_s == 15.0
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics_sampler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.metrics_sampler'`.

- [x] **Step 3: Implement**

```python
# server/metrics_sampler.py
"""Background sampler — the ONLY thing that fans out to DeviceMemory consumers.

DeviceMemory.snapshot() round-trips every registered consumer, and under
subprocess isolation the worker consumer's pool_stats() is a request/reply over
the child's control pipe. Putting that on the /metrics render path would tie
pipe traffic to scrape cadence and put a blocking round-trip inside an ASGI
handler — the event-loop starvation shape that already makes /status time out
during a job. So: this thread samples on a fixed interval and writes gauges;
/metrics renders those gauges and touches nothing else.

Sampling DURING a job is safe by prior design: the control channel is a separate
pipe from the data pipe precisely because drain_to_subscriber reads the data pipe
concurrently while a job runs.

Imports nothing from backends/ — readers arrive as callables.
"""
import logging
import os
import threading
from typing import Callable, Optional

from server.metrics import get_metrics

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = 15.0


def _coerce_interval(value, source: str) -> float:
    """Added during execution — see the note in __init__ below.

    Rejects non-numbers (a config typo must not stop the server starting) and
    non-positive values (Event.wait(<=0) returns immediately, spinning the sampler
    against the worker control pipe). Small POSITIVE values remain legal.
    """
    if value is None:
        return DEFAULT_INTERVAL_S
    try:
        interval = float(value)
    except (TypeError, ValueError):
        logger.warning(
            "%s=%r is not a number; using %.1fs", source, value, DEFAULT_INTERVAL_S)
        return DEFAULT_INTERVAL_S
    if interval <= 0:
        logger.warning(
            "%s=%r must be positive (a non-positive interval spins the sampler "
            "against the worker control pipe); using %.1fs",
            source, value, DEFAULT_INTERVAL_S,
        )
        return DEFAULT_INTERVAL_S
    return interval


class MetricsSampler:
    def __init__(
        self,
        snapshot_fn: Callable[[], object],
        runtime_stats_fn: Optional[Callable[[], Optional[dict]]] = None,
        interval_s: Optional[float] = None,
    ):
        self._snapshot_fn = snapshot_fn
        self._runtime_stats_fn = runtime_stats_fn
        # CORRECTED DURING EXECUTION — the version above shipped two defects:
        #   1. float() raises on a malformed env value, so a config typo killed
        #      startup instead of degrading;
        #   2. nothing rejected a NON-POSITIVE interval, and Event.wait(0)/wait(<0)
        #      return immediately — measured at ~12,000 snapshot_fn calls in 0.2s,
        #      i.e. a self-inflicted DoS on the worker's control pipe that presents
        #      as "the server is busy" rather than as an error.
        # Both paths are validated because the explicit argument bypasses the env
        # parser. Small POSITIVE values stay legal (the suite samples at 0.01s).
        self.interval_s = (
            _coerce_interval(interval_s, "interval_s")
            if interval_s is not None
            else _coerce_interval(
                os.getenv("METRICS_SAMPLE_INTERVAL_S"), "METRICS_SAMPLE_INTERVAL_S")
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not get_metrics().enabled:
            logger.debug("[Metrics] sampler not started: metrics disabled")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="metrics-sampler", daemon=True)
        self._thread.start()
        logger.info("[Metrics] sampler started (interval=%.1fs)", self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval_s)

    def sample_once(self) -> None:
        """One sampling pass. MUST NOT raise: a sampler that dies on one bad read
        stops every device metric permanently and does it silently."""
        met = get_metrics()
        try:
            snap = self._snapshot_fn()
        except Exception:
            logger.debug("[Metrics] device snapshot failed", exc_info=True)
            snap = None
        if snap is not None:
            try:
                uuid = snap.device_uuid
                met.device_total_bytes.labels(device_uuid=uuid).set(snap.total_bytes)
                met.device_free_bytes.labels(device_uuid=uuid).set(snap.free_bytes)
                met.device_used_bytes.labels(device_uuid=uuid).set(snap.used_bytes)
                met.device_unattributed_bytes.labels(device_uuid=uuid).set(
                    snap.unattributed_bytes)
                stale = any(c.stale for c in snap.consumers)
                met.device_snapshot_stale.labels(device_uuid=uuid).set(1 if stale else 0)
                for c in snap.consumers:
                    # label ONLY — c.pid is deliberately dropped, see module docstring
                    # of server/metrics.py and spec §5.
                    met.consumer_reserved_bytes.labels(
                        device_uuid=uuid, consumer=c.label).set(c.reserved_bytes)
                    met.consumer_allocated_bytes.labels(
                        device_uuid=uuid, consumer=c.label).set(c.allocated_bytes)
            except Exception:
                logger.debug("[Metrics] device gauge write failed", exc_info=True)

        if self._runtime_stats_fn is None:
            return
        try:
            stats = self._runtime_stats_fn()
            if stats:
                met.queue_depth.set(stats.get("queue_depth", 0))
                met.jobs_in_flight.set(stats.get("jobs_in_flight", 0))
        except Exception:
            logger.debug("[Metrics] runtime stats failed", exc_info=True)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics_sampler.py -v`
Expected: ALL PASS.

- [x] **Step 5: Commit**

```bash
git add server/metrics_sampler.py tests/test_metrics_sampler.py
git commit -m "feat(metrics): background sampler owns all DeviceMemory fan-out (STABL-asawxgvp)

snapshot() round-trips the child control pipe under subprocess isolation, so it
must not sit on the scrape path: fan-out cadence is now decoupled from scrape
cadence and ten scrapers cost the same as one. Consumer gauges key on the
consumer LABEL, never pid — respawn mints a new pid and would leak a dead series
per recovery.

sample_once() cannot raise; a raising snapshot_fn is proven not to kill the
thread. Disabled sampler never calls snapshot_fn at all.

next: Task 6 /metrics route + app wiring"
```

---

## Task 6: The `/metrics` route and app wiring

**Files:**
- Create: `server/metrics_routes.py`
- Modify: `server/lcm_sr_server.py` — router include near `/health` (`:946`), lifespan (`:506` region)
- Test: `tests/test_metrics_route.py` (create)

**Interfaces:**
- Consumes: `get_metrics()`, `MetricsSampler`.
- Produces: `server.metrics_routes.router` serving `GET /metrics`; `server.metrics_routes.build_runtime_stats_fn(pool_getter)`.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_metrics_route.py
import os

import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from server import metrics as m
from server.metrics_routes import router as metrics_router


@pytest.fixture(autouse=True)
def _fresh():
    m.reset_metrics()
    yield
    m.reset_metrics()


def test_metrics_endpoint_renders(monkeypatch):
    monkeypatch.setenv("METRICS_ENABLED", "1")
    m.get_metrics().job_terminal_total.labels(mode="SDXL", outcome="ok").inc()
    app = FastAPI()
    app.include_router(metrics_router)
    with TestClient(app) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "st_governor_job_terminal_total" in resp.text


def test_metrics_endpoint_404s_when_disabled(monkeypatch):
    monkeypatch.delenv("METRICS_ENABLED", raising=False)
    app = FastAPI()
    app.include_router(metrics_router)
    with TestClient(app) as client:
        resp = client.get("/metrics")
    assert resp.status_code == 404


def test_metrics_resolves_with_a_ui_static_mount_present(tmp_path, monkeypatch):
    """THE trap this test exists for: app.mount('/', StaticFiles(...)) matches
    everything, Starlette matches routes in registration order, and the UI dist
    is absent on a dev box — so a late-registered /metrics passes locally and
    404s only in the deployed image.
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


def test_registration_order_in_the_real_app():
    """Pin the ordering in lcm_sr_server itself, not just in a synthetic app."""
    from server import lcm_sr_server

    paths = []
    for r in lcm_sr_server.app.router.routes:
        paths.append(getattr(r, "path", None))
    assert "/metrics" in paths, "/metrics is not mounted on the app"
    if "/" in paths:
        assert paths.index("/metrics") < paths.index("/"), (
            "/metrics must be registered before the catch-all static mount"
        )


def test_scrape_does_not_fan_out(monkeypatch):
    """The render path must touch gauges only — never DeviceMemory.snapshot()."""
    monkeypatch.setenv("METRICS_ENABLED", "1")
    from backends import device_memory

    def _explode(*a, **kw):
        raise AssertionError("/metrics MUST NOT fan out to consumers")

    monkeypatch.setattr(device_memory._ConsumerRegistry, "snapshot", _explode)
    app = FastAPI()
    app.include_router(metrics_router)
    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 200
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics_route.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.metrics_routes'`.

- [x] **Step 3: Write the route module**

```python
# server/metrics_routes.py
"""The /metrics scrape endpoint.

Renders in-memory gauges written by MetricsSampler. It performs NO device or
consumer round-trip — see server/metrics_sampler.py for why that separation is
load-bearing.
"""
import logging
from typing import Callable, Optional

from fastapi import APIRouter, HTTPException, Response

from server.metrics import get_metrics

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/metrics")
def metrics_endpoint() -> Response:
    met = get_metrics()
    if not met.enabled:
        raise HTTPException(404, detail="metrics disabled")
    body, content_type = met.render()
    return Response(content=body, media_type=content_type,
                    headers={"Cache-Control": "no-store"})


def build_runtime_stats_fn(pool_getter: Callable[[], object]) -> Callable[[], Optional[dict]]:
    """Adapt the worker pool/Governor into the sampler's stats callable.

    Kept here rather than in the sampler so server/metrics_sampler.py stays free
    of any backends/ knowledge.
    """
    def _stats() -> Optional[dict]:
        try:
            pool = pool_getter()
        except Exception:
            return None
        if pool is None:
            return None
        try:
            # WorkerPool holds its Governor on _governor (backends/worker_pool.py:71);
            # the fallback keeps a rename degrading to "no queue gauges" rather than
            # an exception, and also lets a bare Governor be passed directly in tests.
            gov = getattr(pool, "_governor", pool)
            with gov._job_lock:
                in_flight = sum(
                    1 for r in gov._job_records.values()
                    if r.executing_since is not None
                )
            return {"queue_depth": gov.get_queue_size(), "jobs_in_flight": in_flight}
        except Exception:
            logger.debug("[Metrics] runtime stats unavailable", exc_info=True)
            return None

    return _stats
```

- [x] **Step 4: Wire it into the app**

In `server/lcm_sr_server.py`, add to the imports:

```python
from server.metrics_routes import router as metrics_router, build_runtime_stats_fn
from server.metrics_sampler import MetricsSampler
```

Register the router **immediately after the `/health` endpoint (`:946-948`)** — that is
well above the static mount at `:971`:

```python
@app.get("/health")
def health():
    return {"status": "ok"}

# STABL-asawxgvp: /metrics MUST be registered before the UI static mount below.
# app.mount("/", StaticFiles(...)) matches every path and Starlette routes in
# registration order, so anything registered after it is unreachable whenever the
# UI dist is present — which is true in the deployed image and false on a dev box.
app.include_router(metrics_router)
```

In the `lifespan` handler (`:506` region), start the sampler on entry and stop it on exit:

```python
    from backends.device_memory import get_device_memory
    from backends.worker_pool import get_worker_pool

    _sampler = MetricsSampler(
        snapshot_fn=lambda: get_device_memory().snapshot(),
        runtime_stats_fn=build_runtime_stats_fn(get_worker_pool),
    )
    _sampler.start()
    try:
        yield
    finally:
        _sampler.stop()
```

> Match the existing `lifespan` body's shape — if it already has a `try/finally` around its
> `yield`, add `_sampler.stop()` to that `finally` rather than nesting another one. Read
> `lcm_sr_server.py` around `:480-510` before editing.

- [x] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics_route.py -v`
Expected: ALL PASS.

- [x] **Step 6: Verify by hand, both gate positions**

```bash
conda activate stability-toys
METRICS_ENABLED=1 python -c "
from fastapi.testclient import TestClient
from server.lcm_sr_server import app
with TestClient(app) as c:
    r = c.get('/metrics')
    print(r.status_code, len(r.text))
    print([l for l in r.text.splitlines() if l.startswith('st_')][:5])
"
```

Expected: `200`, non-zero length, `st_`-prefixed sample lines.

```bash
python -c "
from fastapi.testclient import TestClient
from server.lcm_sr_server import app
with TestClient(app) as c:
    print(c.get('/metrics').status_code)
"
```

Expected: `404` — inert by default.

- [x] **Step 7: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: no new failures against the pre-task baseline. Capture the counts for the FP comment.

- [x] **Step 8: Commit**

```bash
git add server/metrics_routes.py server/lcm_sr_server.py tests/test_metrics_route.py
git commit -m "feat(metrics): /metrics endpoint, registered ahead of the static mount (STABL-asawxgvp)

app.mount('/', StaticFiles) matches every path and Starlette routes in
registration order, so /metrics goes in beside /health. The failure mode is
invisible on a dev box (no ui-dist -> mount skipped) and 404s only in the
deployed image, so there is a test that builds an app WITH a static mount and a
test that pins the ordering in lcm_sr_server itself.

Sampler starts/stops with the app lifespan. Scrape path proven not to fan out.

next: Task 7 observability contract doc"
```

---

## Task 7: The cross-repo contract doc

**Files:**
- Create: `docs/observability-contract.md`
- Test: `tests/test_metrics.py` (extend — the doc is verified, not just written)

**Interfaces:**
- Consumes: the family names from Task 2.
- Produces: the document `../continuous` reads to write scrape config and dashboards.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_metrics.py`:

```python
def test_every_family_is_documented_in_the_contract(monkeypatch):
    """The contract doc is the cross-repo interface. A family that ships without
    an entry is invisible to whoever writes the dashboards in ../continuous."""
    import pathlib
    monkeypatch.setenv("METRICS_ENABLED", "1")
    body, _ = m.get_metrics().render()
    shipped = {
        ln.split("{")[0].split(" ")[0]
           .removesuffix("_bucket").removesuffix("_count").removesuffix("_sum")
           .removesuffix("_created").removesuffix("_total")
        for ln in body.decode().splitlines()
        if ln.startswith("st_")
    }
    doc = pathlib.Path("docs/observability-contract.md").read_text()
    missing = sorted(n for n in shipped if n not in doc)
    assert not missing, f"undocumented metric families: {missing}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py::test_every_family_is_documented_in_the_contract -v`
Expected: FAIL — `FileNotFoundError: docs/observability-contract.md`

- [ ] **Step 3: Write the contract doc**

```markdown
# Observability contract — metrics exported by Stability-Toys

**Issue:** STABL-asawxgvp (umbrella STABL-oxbwjwvu)
**Spec:** `docs/superpowers/specs/2026-08-03-server-observability-seams-design.md`

This repo owns **emission only**. Scrape config, collectors, dashboards, alert
rules and retention live in `../continuous/docs` — see `AGENTS.md`. This document
is the interface between the two: it is what `../continuous` reads instead of
guessing or reverse-engineering a scrape.

## Endpoint

`GET /metrics` — Prometheus text format, `text/plain; version=0.0.4`.

- Gated by `METRICS_ENABLED` (default **off**). Disabled returns **404**.
- Single-process only. `prometheus_client`'s registry is process-local; the server
  runs one uvicorn worker (`server/run.py`). With `WEB_CONCURRENCY > 1` the
  endpoint reports whichever worker answered and counters appear to go backwards.
  The facade logs a WARNING in that case.
- Device and queue gauges are refreshed by a background sampler every
  `METRICS_SAMPLE_INTERVAL_S` (default **15s**), independent of scrape cadence.
  **Scrape interval is not yet aligned across repos** — when `../continuous`
  declares one, revisit this default.

## Label policy

Allowed: `device_uuid`, `mode`, plus `outcome`, `reason`, `consumer`, `budget`
where declared below.

**There is deliberately no `hostname` label.** Host identity arrives as the
`instance` label from the scrape target. A second host label on every series
would duplicate it and go stale when the container moves. Do not add one; do not
go looking for one.

**`job_id` and `pid` never appear as labels.** `job_id` is unbounded; `pid` looks
bounded and is not, because the subprocess worker mints a new pid on every
kill+respawn. Both are available in structured logs (`STABL-bpsfmoke`).

## Families

### Governor

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `st_governor_queue_depth` | gauge | — | jobs queued |
| `st_governor_jobs_in_flight` | gauge | — | jobs past the admission barrier |
| `st_governor_job_queue_wait_seconds` | histogram | `mode` | enqueue → execution start |
| `st_governor_job_execution_seconds` | histogram | `mode` | execution start → terminal |
| `st_governor_job_terminal_total` | counter | `mode`, `outcome` | `ok`/`cancelled`/`oom`/`error` |
| `st_governor_wait_expired_total` | counter | `budget` | `admission`/`execution` budget blown |
| `st_governor_mode_load_seconds` | histogram | `mode` | successful loads only |
| `st_governor_mode_switch_total` | counter | `mode` | target mode |
| `st_governor_demand_reload_total` | counter | `mode` | reload after idle eviction |
| `st_governor_unload_total` | counter | `mode`, `reason` | model unloads |
| `st_governor_worker_recovery_total` | counter | `reason` | `oom`/`dead` kill+respawn |
| `st_governor_mode_active` | gauge | `mode` | 1 for the loaded mode, 0 for all others |
| `st_governor_resolution_epoch` | gauge | — | current authority epoch |

**`wait_expired_total` is not the same as `job_terminal_total{outcome="cancelled"}`.**
A timeout is a waiter-side event; the job that follows it is reaped as a cancel.
"How often do requests time out" is the first series; "what happened to the job"
is the second.

**`mode_load_seconds` counts successful loads only.** A failed load raises before
observation. Failed loads show up as `mode_active` being 0 for every mode.

### DeviceMemory

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `st_device_total_bytes` | gauge | `device_uuid` | device total |
| `st_device_free_bytes` | gauge | `device_uuid` | driver-truth free |
| `st_device_used_bytes` | gauge | `device_uuid` | driver-truth used |
| `st_device_unattributed_bytes` | gauge | `device_uuid` | used minus all consumer reserved pools |
| `st_consumer_reserved_bytes` | gauge | `device_uuid`, `consumer` | per-consumer pool, reserved |
| `st_consumer_allocated_bytes` | gauge | `device_uuid`, `consumer` | per-consumer pool, allocated |
| `st_device_snapshot_stale` | gauge | `device_uuid` | 1 when a consumer fan-out timed out |

`consumer` takes the values `server` (the parent process, which hosts superres)
and `worker` (the process hosting the generation worker). The spelling `worker`
is load-bearing elsewhere in the codebase and will not change.

`st_device_unattributed_bytes` is the per-process CUDA context plus non-torch
workspaces plus anything unregistered. On a single-consumer discrete GPU it is
approximately the CUDA context (~300 MiB–1.5 GiB). **On UNIFIED topology it
includes the whole host's RAM usage and must never be alerted on.**

`st_device_snapshot_stale = 1` means a consumer stopped answering its control
pipe — an early wedged-worker signal with no other surface.

## Stability

Metric names and label sets are treated as a stable interface. Additions are
compatible; renames and label changes are breaking and must be announced to
`../continuous` before landing. Every family shipped here has a test asserting
it appears in this document.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Check drift bindings**

Run: `drift refs docs/observability-contract.md && drift check`

The new doc has no anchors, so it binds nothing. `drift check` currently reports **28
pre-existing stale anchors** repo-wide and exits 1 — that is unrelated to this work
(tracked separately). Confirm the count did not grow.

- [ ] **Step 6: Commit**

```bash
git add docs/observability-contract.md tests/test_metrics.py
git commit -m "docs(metrics): publish the cross-repo metric-name contract (STABL-asawxgvp)

Names and label sets are the interface ../continuous consumes to write scrape
config and dashboards; unpublished, that repo guesses. A test asserts every
family the facade actually renders has an entry here, so the doc cannot silently
fall behind the code.

next: FP comment + drift check, then STABL-xmsrxvto (HTTP/WS metrics)"
```

---

## Closeout

- [ ] **Run the full suite and record the numbers**

```bash
conda activate stability-toys && python -m pytest tests/ -q 2>&1 | tail -5
```

- [ ] **Update FP**

```bash
fp issue assign STABL-asawxgvp --rev <sha>
fp comment STABL-asawxgvp "<what landed / decisions / next step per Rule 2>"
```

Two amendments were **ratified by Sigma at plan review (2026-08-03)** and are already folded
into the spec — they are settled, not open, and the comment should record them as decided:

1. **`timeout` is not a terminal outcome.** `job_terminal_total{outcome}` is
   `ok|cancelled|oom|error`; wait expiry is its own
   `st_governor_wait_expired_total{budget}`. A timeout is a waiter-side event and the job
   that follows it is reaped as a cancel, so keeping it in the enum would double-count.
2. **No `hostname` label** on any family. Host identity is Prometheus's `instance` label
   from the scrape target. `hostname` remains a structured-log field under
   `STABL-bpsfmoke`.

- [ ] **Report ready for review.** Do not self-advance waveplan state or call `fin`.

---

## Deferred (NOT in this issue)

- HTTP and WebSocket metrics — `STABL-xmsrxvto`, already dependent on this issue.
- Leaked OS-resource gauges — `STABL-cxbwwgly`, now a sibling under the umbrella and
  dependent on this facade.
- Structured JSON logging and `job_id` correlation — `STABL-bpsfmoke`.
- Tracing — `STABL-qnlaclof`.
- Multiprocess `prometheus_client` registry — explicit non-goal until someone needs
  multiple uvicorn workers.
- Routing `/api/models/status` through the Governor bytes-shape builder — pre-existing
  deferral from PR #24, untouched here.
