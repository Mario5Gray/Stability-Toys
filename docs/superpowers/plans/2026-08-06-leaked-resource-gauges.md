# Leaked OS Resource Gauges — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, or execute it directly. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Do NOT use superpowers:subagent-driven-development.** `AGENTS.md` forbids sub-agent driven development in this repo.

**FP:** STABL-cxbwwgly (child of STABL-oxbwjwvu, built on STABL-asawxgvp — merged `dd633e6`)
**Contract:** `docs/observability-contract.md` — extended by this issue

**Goal:** Make the accepted-but-unbounded semaphore leak *visible*, so the eventual failure points back at its cause.

**Architecture:** One small stdlib+psutil probe module, three gauges on the existing facade, sampled by the existing `MetricsSampler`. No new dependency, no new infrastructure.

**Tech Stack:** Python 3, `psutil` (already pinned at `>=5.9.0`), `prometheus_client` (already pinned), pytest.

## Global Constraints

- **Absent, never zero.** Where a source cannot be read, the gauge must not be set at all. A `0` for "leaked semaphores" on a host with no `/dev/shm` is indistinguishable from a healthy Linux box — a lie that looks like health. Same principle as `NullDeviceMemory`: degrade, never borrow.
- **Availability is PER-SOURCE, not all-or-nothing.** Verified on this dev box: `/dev/shm` is absent (macOS) while `psutil.Process().num_fds()` works fine. One unavailable source must not suppress the others.
- **`METRICS_ENABLED` still gates everything** — the facade's no-op objects make every new call site inert by default.
- **The probe must never raise into the sampler.** It runs on the same pass that carries the device gauges.
- **`server/metrics.py` remains the only module importing `prometheus_client`.**
- Python env: `conda activate stability-toys`, then `python` (not `python3`).

---

## Why this design satisfies the issue's two stated requirements

The issue names two properties that "matter more than the specific numbers". Both are
satisfied structurally rather than by discipline, which is the point:

**1. "Sample inside the container."** `/dev/shm` is per-mount-namespace, and a host-side
check reads a different one — a mistake that already cost time during the
`STABL-nstyyrhh` investigation. Putting the probe in `MetricsSampler` satisfies this **by
construction**: the sampler thread runs inside the server process, which runs inside the
container. There is no configuration to get wrong and no way to accidentally sample the
host.

**2. "Report the trend, not the instant."** A count of 4 means nothing; 4 growing by one
per mode switch is the signal. The ratio is already derivable from metrics that shipped in
`STABL-asawxgvp` — no new counter is needed:

```promql
increase(st_process_leaked_semaphores[1h])
  / increase(st_governor_mode_load_seconds_count[1h])
```

A value near 1 reproduces `STABL-nstyyrhh`'s one-semaphore-per-model-load finding. Task 3
puts this query in the contract so `../continuous` does not have to derive it.

---

## File Structure

| File | Responsibility |
|---|---|
| `server/resource_probe.py` (create) | count sems / shm segments / fds; report unavailability honestly |
| `server/metrics.py` (modify) | three new gauges |
| `server/metrics_sampler.py` (modify) | a third independently-guarded reader |
| `docs/observability-contract.md` (modify) | the three entries + the ratio query |
| `tests/test_resource_probe.py` (create) | probe behaviour incl. the absent-source path |
| `tests/test_metrics.py` (modify) | family list; the contract test already covers the rest |
| `tests/test_metrics_sampler.py` (modify) | sampler wiring + independence of the third reader |

---

## Task 1: The probe

**Files:**
- Create: `server/resource_probe.py`
- Test: `tests/test_resource_probe.py`

**Interfaces:**
- Consumes: stdlib + `psutil`. Imports nothing from `backends/` and nothing from `server/metrics*`.
- Produces: `ResourceCounts(leaked_semaphores: Optional[int], shm_segments: Optional[int], open_fds: Optional[int])` and `probe_resources(shm_root: str = "/dev/shm") -> ResourceCounts`. `None` means "could not be read here", never `0`.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_resource_probe.py
"""OS resource counts (STABL-cxbwwgly).

Plan: docs/superpowers/plans/2026-08-06-leaked-resource-gauges.md Task 1.

The load-bearing test is test_missing_shm_root_reports_none_not_zero: a 0 for
"leaked semaphores" on a host with no /dev/shm is indistinguishable from a
healthy Linux box.
"""
import pytest

from server.resource_probe import ResourceCounts, probe_resources


def _make_shm(tmp_path, sems=0, segments=0):
    for i in range(sems):
        (tmp_path / f"sem.mp-abc{i}").write_text("")
    for i in range(segments):
        (tmp_path / f"psm_{i:08x}").write_text("")
    return str(tmp_path)


def test_counts_semaphores_by_the_sem_prefix(tmp_path):
    counts = probe_resources(shm_root=_make_shm(tmp_path, sems=3, segments=2))
    assert counts.leaked_semaphores == 3


def test_counts_shm_segments_as_everything_else(tmp_path):
    """/dev/shm holds both; the sem.* prefix is what separates them."""
    counts = probe_resources(shm_root=_make_shm(tmp_path, sems=3, segments=2))
    assert counts.shm_segments == 2


def test_an_empty_shm_root_is_zero_not_none(tmp_path):
    """Readable and empty is a real measurement — distinct from unreadable."""
    counts = probe_resources(shm_root=str(tmp_path))
    assert counts.leaked_semaphores == 0
    assert counts.shm_segments == 0


def test_missing_shm_root_reports_none_not_zero(tmp_path):
    """THE distinction this issue turns on. A host-side check once reported 0
    while the container held several (STABL-nstyyrhh); reporting 0 for an
    unreadable source repeats that mistake in metric form."""
    counts = probe_resources(shm_root=str(tmp_path / "does-not-exist"))
    assert counts.leaked_semaphores is None
    assert counts.shm_segments is None


def test_fds_are_counted_even_when_shm_is_unavailable(tmp_path):
    """Availability is PER-SOURCE. macOS has no /dev/shm but num_fds() works —
    one unavailable source must not suppress the others."""
    counts = probe_resources(shm_root=str(tmp_path / "does-not-exist"))
    assert counts.open_fds is not None and counts.open_fds > 0


def test_probe_never_raises(monkeypatch):
    """It runs on the sampler pass that also carries the device gauges."""
    import server.resource_probe as rp

    monkeypatch.setattr(rp, "_count_fds", lambda: (_ for _ in ()).throw(OSError("nope")))
    counts = probe_resources(shm_root="/definitely/not/here")
    assert counts == ResourceCounts(None, None, None)


def test_an_unreadable_shm_root_reports_none(tmp_path, monkeypatch):
    """Permission denied is unreadable, not empty."""
    import server.resource_probe as rp

    monkeypatch.setattr(rp.os, "listdir",
                        lambda p: (_ for _ in ()).throw(PermissionError("nope")))
    counts = probe_resources(shm_root=str(tmp_path))
    assert counts.leaked_semaphores is None


def test_module_imports_nothing_from_backends_or_metrics():
    import importlib.util

    spec = importlib.util.find_spec("server.resource_probe")
    assert spec is not None and spec.origin is not None
    with open(spec.origin) as fh:
        head = [ln for ln in fh if ln.startswith(("import ", "from "))]
    assert not any("backends" in ln or "server.metrics" in ln for ln in head)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `conda activate stability-toys && python -m pytest tests/test_resource_probe.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.resource_probe'`.

- [x] **Step 3: Implement**

```python
# server/resource_probe.py
"""OS resource counts for the leak watch (STABL-cxbwwgly).

STABL-nstyyrhh established that the server leaks one POSIX named semaphore per
MODEL LOAD — linear, never reclaimed — and accepted that risk specifically
BECAUSE it is cheap to watch. This is the watch. Closing that issue without it
would have retracted half the bargain: the eventual failure (a server that cannot
create a semaphore) does not point back at its cause.

`None` means "could not be read here" and is never rendered as a metric. `0`
means "read it, found none". Conflating the two would report a healthy-looking
zero on any host without /dev/shm — which is exactly the mistake a host-side
check made during the original investigation, when it read a different mount
namespace than the container and reported 0.

Imports nothing from backends/ and nothing from server/metrics — it is a pure
measurement, and the sampler decides what to do with it.
"""
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_SHM_ROOT = "/dev/shm"

# multiprocessing.synchronize.SemLock names its POSIX semaphores sem.mp-*;
# the sem. prefix is what separates them from shared-memory segments in the
# same directory.
_SEM_PREFIX = "sem."


@dataclass(frozen=True)
class ResourceCounts:
    leaked_semaphores: Optional[int]
    shm_segments: Optional[int]
    open_fds: Optional[int]


def _count_fds() -> Optional[int]:
    try:
        import psutil
        return int(psutil.Process().num_fds())
    except Exception:
        # Windows has no num_fds, psutil may be absent, /proc may be restricted.
        return None


def _count_shm(shm_root: str) -> tuple[Optional[int], Optional[int]]:
    """(semaphores, segments). Both None when the root cannot be listed."""
    try:
        entries = os.listdir(shm_root)
    except Exception:
        return None, None
    sems = sum(1 for e in entries if e.startswith(_SEM_PREFIX))
    return sems, len(entries) - sems


def probe_resources(shm_root: str = DEFAULT_SHM_ROOT) -> ResourceCounts:
    """One measurement. MUST NOT raise — it runs on the sampler pass that also
    carries the device gauges, and a wedged probe must not blank those."""
    try:
        sems, segments = _count_shm(shm_root)
    except Exception:
        logger.debug("[ResourceProbe] shm count failed", exc_info=True)
        sems, segments = None, None
    try:
        fds = _count_fds()
    except Exception:
        logger.debug("[ResourceProbe] fd count failed", exc_info=True)
        fds = None
    return ResourceCounts(leaked_semaphores=sems, shm_segments=segments, open_fds=fds)
```

- [x] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_resource_probe.py -q`
Expected: ALL PASS.

- [x] **Step 5: Commit**

```bash
git add server/resource_probe.py tests/test_resource_probe.py
git commit -m "feat(metrics): OS resource probe for the semaphore leak watch (STABL-cxbwwgly)

None means 'could not be read here' and is never rendered; 0 means 'read it,
found none'. Conflating them would report a healthy-looking zero on any host
without /dev/shm — the exact mistake a host-side check made during the
STABL-nstyyrhh investigation, when it read a different mount namespace than the
container and reported 0.

Availability is per-source: this dev box has no /dev/shm but num_fds() works, so
one unavailable source must not suppress the others.

next: Task 2 gauges + sampler wiring"
```

---

## Task 2: Gauges and sampler wiring

**Files:**
- Modify: `server/metrics.py` (three gauges)
- Modify: `server/metrics_sampler.py` (third reader)
- Test: `tests/test_metrics.py`, `tests/test_metrics_sampler.py`

**Interfaces:**
- Consumes: `probe_resources`, `ResourceCounts` from Task 1.
- Produces: `Metrics.process_leaked_semaphores`, `.process_shm_segments`, `.process_open_fds`; `MetricsSampler(..., resource_probe_fn=None)` defaulting to `probe_resources`.

- [x] **Step 1: Write the failing tests**

Extend `_ALL_FAMILIES` in `tests/test_metrics.py`:

```python
    # STABL-cxbwwgly
    "process_leaked_semaphores", "process_shm_segments", "process_open_fds",
```

and add to `tests/test_metrics_sampler.py`:

```python
from server.resource_probe import ResourceCounts


def _counts(sems=2, segments=1, fds=42):
    return ResourceCounts(leaked_semaphores=sems, shm_segments=segments, open_fds=fds)


def test_sample_once_writes_resource_gauges():
    MetricsSampler(snapshot_fn=_snap, resource_probe_fn=_counts).sample_once()

    assert _value("st_process_leaked_semaphores") == 2.0
    assert _value("st_process_shm_segments") == 1.0
    assert _value("st_process_open_fds") == 42.0


def test_an_unavailable_source_leaves_its_series_ABSENT():
    """Absent, never zero — a 0 here reads as 'no leak' on a host that simply
    cannot see /dev/shm."""
    probe = lambda: ResourceCounts(leaked_semaphores=None, shm_segments=None, open_fds=17)
    MetricsSampler(snapshot_fn=_snap, resource_probe_fn=probe).sample_once()

    assert not _lines("st_process_leaked_semaphores")
    assert not _lines("st_process_shm_segments")
    assert _value("st_process_open_fds") == 17.0      # per-source, not all-or-nothing


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
    """Wiring check: the sampler must not silently sample nothing in production."""
    from server.resource_probe import probe_resources

    s = MetricsSampler(snapshot_fn=_snap)
    assert s._resource_probe_fn is probe_resources


def test_disabled_sampler_never_probes(monkeypatch):
    monkeypatch.delenv("METRICS_ENABLED", raising=False)
    m.reset_metrics()

    def _must_not_run():
        raise AssertionError("disabled sampler MUST NOT probe")

    MetricsSampler(snapshot_fn=_snap, resource_probe_fn=_must_not_run).sample_once()
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py tests/test_metrics_sampler.py -q`
Expected: FAIL — `AttributeError` on the three new names and `TypeError` on the
unexpected `resource_probe_fn` keyword.

- [x] **Step 3: Declare the gauges**

In `server/metrics.py` `_declare`, after the WebSocket block:

```python
        # --- OS resources (STABL-cxbwwgly) ---
        self.process_leaked_semaphores = G(
            "st_process_leaked_semaphores",
            "POSIX named semaphores visible to this process "
            "(one per model load is the known leak — STABL-nstyyrhh)",
            ["process"], registry=r)
        self.process_shm_segments = G(
            "st_process_shm_segments",
            "Shared-memory segments visible to this process, excluding semaphores",
            ["process"], registry=r)
        self.process_open_fds = G(
            "st_process_open_fds", "Open file descriptors for this process",
            ["process"], registry=r)
```

and add the three names to the `_declare_noop` tuple.

> **CORRECTED DURING EXECUTION — the `["process"]` label is load-bearing, and this
> plan originally specified these gauges without it.** An **unlabelled** `Gauge`
> renders its default `0.0` from the moment it is declared, so
> `test_an_unavailable_source_leaves_its_series_ABSENT` failed with
> `['st_process_leaked_semaphores 0.0']` — the whole absent-never-zero rule is
> unachievable that way. Verified:
>
> ```text
> before any set:  bare_g 0.0          <- renders immediately
>                  (labelled: nothing)
> after set:       lab_g{process="server"} 7.0
> ```
>
> A labelled family emits nothing until a child is created, which is what makes
> absence real. The label is not a workaround dressed as design: it answers the
> question the contract would otherwise answer only in prose — whose counts these
> are — and leaves room for the deferred worker-side probe.

- [x] **Step 4: Wire the sampler**

In `server/metrics_sampler.py`, import at module top:

```python
from server.resource_probe import probe_resources
```

Add the constructor parameter, defaulting to the real probe:

```python
        resource_probe_fn: Optional[Callable[[], object]] = None,
    ):
        self._snapshot_fn = snapshot_fn
        self._runtime_stats_fn = runtime_stats_fn
        # Defaults to the real probe: an injectable with no default would sample
        # nothing in production the day someone forgets to pass it.
        self._resource_probe_fn = resource_probe_fn or probe_resources
```

and add a third **independently guarded** block at the end of `sample_once`:

```python
        try:
            counts = self._resource_probe_fn()
        except Exception:
            logger.debug("[Metrics] resource probe failed", exc_info=True)
            counts = None
        if counts is not None:
            try:
                # ABSENT, never zero: a source that could not be read leaves its
                # series unset, because 0 leaked semaphores and "this host has no
                # /dev/shm" are opposite findings that must not render alike.
                for gauge_name, value in (
                    ("process_leaked_semaphores", counts.leaked_semaphores),
                    ("process_shm_segments", counts.shm_segments),
                    ("process_open_fds", counts.open_fds),
                ):
                    if value is not None:
                        getattr(met, gauge_name).labels(
                            process=SELF_PROCESS_LABEL).set(value)
            except Exception:
                logger.debug("[Metrics] resource gauge write failed", exc_info=True)
```

> Place this block **after** the runtime-stats block, and make sure the existing
> `if self._runtime_stats_fn is None: return` early-return does not skip it — that
> `return` must become a guarded branch, or the resource gauges silently never fire
> whenever no runtime stats reader is injected. **This is the easiest thing in the task
> to get wrong**, and `test_sample_once_writes_resource_gauges` (which passes no
> `runtime_stats_fn`) is what catches it.

- [x] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py tests/test_metrics_sampler.py -q`
Expected: all pass **except** `test_every_family_is_documented_in_the_contract`, which
stays RED until Task 3.

- [x] **Step 6: Verify against the real app**

```bash
conda activate stability-toys
METRICS_ENABLED=1 CONTROLNET_REGISTRY_VALIDATION=off BACKEND=cpu python -c "
from fastapi.testclient import TestClient
from server.lcm_sr_server import app
with TestClient(app) as c:
    app.state.metrics_sampler.sample_once()
    body = c.get('/metrics').text
for l in body.splitlines():
    if l.startswith('st_process_'): print(l)
"
```

Expected **on this macOS dev box**: `st_process_open_fds` only. The semaphore and
segment series are **absent**, which is the correct rendering of "no `/dev/shm` here" —
if you see them at 0, the absent-never-zero rule has been broken.

- [x] **Step 7: Commit**

```bash
git add server/metrics.py server/metrics_sampler.py tests/test_metrics.py tests/test_metrics_sampler.py
git commit -m "feat(metrics): leaked semaphore, shm segment and fd gauges (STABL-cxbwwgly)

Sampled by MetricsSampler, which satisfies the issue's 'sample inside the
container' requirement BY CONSTRUCTION — the sampler thread runs in the server
process, so there is no configuration to get wrong and no way to accidentally
read the host's mount namespace.

A source that cannot be read leaves its series ABSENT, never 0: those are
opposite findings and must not render alike. Availability is per-source, so a
macOS dev box reports fds and omits the two /dev/shm series.

Third reader is independently guarded — a wedged probe must not blank the device
gauges it shares a pass with.

next: Task 3 contract doc"
```

---

## Task 3: Extend the contract

**Files:**
- Modify: `docs/observability-contract.md`
- Test: `tests/test_metrics.py` (the existing bidirectional test)

- [x] **Step 1: Confirm the contract test is RED**

Run: `python -m pytest tests/test_metrics.py::test_every_family_is_documented_in_the_contract -q`
Expected: FAIL listing the three undocumented families.

- [x] **Step 2: Add the entries**

Append to `docs/observability-contract.md`, before `## A note on _created series`:

```markdown
## OS resource families

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `st_process_leaked_semaphores` | gauge | — | POSIX named semaphores visible to this process |
| `st_process_shm_segments` | gauge | — | shared-memory segments, excluding semaphores |
| `st_process_open_fds` | gauge | — | open file descriptors for this process |

**These exist to watch an accepted risk, not to alert on an absolute number.**
`STABL-nstyyrhh` established that the server leaks one POSIX named semaphore per
**model load** — linear and never reclaimed — and accepted that risk precisely
because it is cheap to watch. A count of 4 means nothing. The signal is growth
per model load:

```promql
increase(st_process_leaked_semaphores[1h])
  / increase(st_governor_mode_load_seconds_count[1h])
```

A value near 1 reproduces the original finding. Alert on that ratio, or on
absolute exhaustion approaching the host's limit — not on the raw count.

**An ABSENT series is not zero.** Where a source cannot be read the series is not
emitted at all. `/dev/shm` is per-mount-namespace and does not exist on macOS, so
a developer machine emits `st_process_open_fds` and neither of the other two.
Treat absence as "not measurable here", never as "no leak" — a host-side check
reporting 0 while the container held several is the exact mistake that cost time
in the original investigation.

**Counts are per-PROCESS and per-NAMESPACE — the server's own.** They do not
cover the subprocess worker, which has its own fd table. `/dev/shm` is shared
within the container's mount namespace, so the semaphore and segment counts DO
include anything the worker child created there.
```

- [x] **Step 3: Widen the contract test to accept histogram children**

**Verified before writing this step: the PromQL above FAILS the contract test as it
stands.** `_registry_family_names()` adds `family.name` plus a `_total` variant for
counters, so `st_governor_mode_load_seconds_count` is reported as documenting a metric
that does not exist:

```text
st_governor_mode_load_seconds        in known: True
st_governor_mode_load_seconds_count  in known: False
```

That is a gap in the test, not in the query. `_count` is a genuinely emitted series and
the normal way to query a histogram — confirmed against a real observation, whose children
are `_bucket`, `_count`, `_sum` and `_created`. A contract that cannot reference them
cannot describe how to use its own histograms, and the tempting "fix" when Task 3 fails is
to delete the PromQL or loosen the check. Widen the known set instead, in
`tests/test_metrics.py`:

```python
    for family in met._registry.collect():
        names.add(family.name)
        if family.type == "counter":
            names.add(f"{family.name}_total")
        elif family.type == "histogram":
            # real emitted children — a contract that cannot name _count cannot
            # explain how to query a histogram
            for suffix in ("_bucket", "_count", "_sum"):
                names.add(f"{family.name}{suffix}")
```

Apply the same widening to the forward-direction loop inside
`test_every_family_is_documented_in_the_contract`, which builds its `spellings` set the
same way.

Add a test pinning it:

```python
def test_the_contract_may_reference_histogram_children(monkeypatch):
    """_count is how you query a histogram. The contract documents a leak ratio
    built on st_governor_mode_load_seconds_count; if the known-set does not
    accept histogram children, that documentation cannot exist."""
    monkeypatch.setenv("METRICS_ENABLED", "1")
    known = _registry_family_names(m.get_metrics())
    assert "st_governor_mode_load_seconds" in known
    assert "st_governor_mode_load_seconds_count" in known
    assert "st_governor_mode_load_seconds_sum" in known
    # _created stays OUT: it is a client-library artifact, not part of the contract
    assert "st_governor_mode_load_seconds_created" not in known
```

- [x] **Step 4: Run the contract test**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: ALL PASS, both directions.

- [x] **Step 4: Commit**

```bash
git add docs/observability-contract.md
git commit -m "docs(metrics): document the OS resource gauges and the leak ratio (STABL-cxbwwgly)

Ships the PromQL that makes the number meaningful, so ../continuous does not have
to derive it: leaked semaphores per model load, near 1 reproducing
STABL-nstyyrhh's finding. Also states that an absent series means 'not measurable
here' rather than 'no leak', and that the counts are the server process's own.

next: closeout"
```

---

## Closeout

- [x] **Run the full suite.** Baseline before this issue: **1302 passed, 9 skipped, 1 xfailed.**

- [x] **Check drift.** `drift refs` the touched files; read each binding's prose BEFORE relinking. Baseline: 18 stale, none attributable to the metrics work.

- [x] **Update FP.** Assign each commit as you make it, in chronological order — `fp issue diff` derives its baseline from the first-listed revision.

- [x] **Report ready for review.** Do not self-advance state or call `fin`.

---

## Deferred (NOT in this issue)

- **Fixing the leak.** This is the watch, not the cure. `spikes/sem_creator_trace.py` names
  the owning library in about a minute by patching
  `multiprocessing.synchronize.SemLock.__init__`, if anyone wants to revisit the
  underlying bug — the creator was never identified.
- **Surfacing these on `/api/models/status`.** That endpoint is a DeviceMemory view;
  adding OS-resource counts widens its contract for no operator gain now that Prometheus
  carries them. The issue's "wherever runtime health is already surfaced" is answered by
  the metrics endpoint.
- **Counting the subprocess worker's fds.** Would need a control-pipe round trip per
  sample, for a number whose trend the parent's own count already approximates. Revisit
  only if fd exhaustion is actually observed.
- **Alert rules and dashboards** — `../continuous` owns those.
