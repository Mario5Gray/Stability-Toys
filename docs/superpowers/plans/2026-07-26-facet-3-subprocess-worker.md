# Facet-3: SubprocessWorkerHandle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Project policy forbids sub-agent-driven development (AGENTS.md) — do NOT use superpowers:subagent-driven-development.**

**Goal:** Host the real `CudaWorker` in a **spawn** subprocess behind the merged `WorkerHandle` interface, delivering durable OOM recovery (kill + respawn) that in-process `empty_cache`/`del` cannot achieve.

**Architecture:** A new `backends/worker_handle_subprocess.py` adds `SubprocessWorkerHandle` — spawn a child hosting the `CudaWorker`, drive it over the merged backplane IPC transport, and expose liveness through a `LivenessSource` seam behind `health()`. The Governor flips its GenerationJob dispatch to `handle.submit()` for out-of-proc handles (in-proc path unchanged), and gains a kill+respawn recovery path keyed on `health().state`.

**Tech Stack:** Python 3.11, stdlib `multiprocessing` (spawn context) + `multiprocessing.Connection` + `shared_memory`, the merged `backends/backplane/` transport, `pytest`. No new runtime dependency.

**Spec:** `docs/superpowers/specs/2026-07-26-facet-3-subprocess-worker-design.md` (approved).

**FP:** `STABL-rgvxuedo` (child of umbrella `STABL-nvmieaxh`).

## Plan reconciliations (design decisions the spec left to the plan)

1. **Dual-path dispatch preserves the in-proc no-op.** The Governor selects the dispatch path by locality: `self._handle.worker is not None` → the existing in-proc direct-`record.sink` drive (UNCHANGED — the frozen suite must stay green); `self._handle.worker is None` → the subprocess path (`handle.submit()` → `Publisher` → subscribe `_FutureBridge`). Unifying both localities onto `handle.submit()` (which would make in-proc cancel eventually-consistent) is explicitly **deferred** — it would change shipped v1 cancel behavior and risk the frozen cancel tests. Two coexisting paths during facet-3; unification is a later cleanup.

2. **`_FutureBridge` attachment moves for the subprocess path only.** In-proc: `submit_job` opens the `InProcBackplane` channel and attaches `_FutureBridge` before enqueue (v1, unchanged). Subprocess: the IPC channel is owned by `handle.submit()`, so `submit_job` does NOT open a channel; the dispatch loop subscribes `_FutureBridge` to the `Publisher` returned by `handle.submit()`. This is the spec §4.2 "channel ownership moves to the handle" change, scoped to out-of-proc.

3. **Liveness-read flip via one helper, semantics-preserving for in-proc — FIVE sites, not three.** All `self._handle.worker`-as-liveness predicates are replaced by `self._worker_available()`, where `_worker_available()` returns `self._handle.health().state in ("ready", "busy")`. For `InProcessWorkerHandle` this is **exactly equivalent** to `worker is not None` (start→"ready", unload→"dead", job→"busy"), so the frozen suite is unaffected; for the subprocess handle (whose `worker` is always `None`, even when alive) it reads true liveness. The five sites (review-verified):
   - `:529` (`_idle_watchdog_loop`, "is there a worker to evict?") → `if not self._worker_available(): continue`
   - `:551` (`_evict_if_idle`, "already unloaded?") → `if not self._worker_available(): return already_unloaded`
   - `:594` (dispatch **demand-reload trigger**, "respawn then run") → `if not self._worker_available() and self._active_snapshot is not None:`
   - `:756` (`is_model_loaded`, "loaded?") → `return self._worker_available()` — **without this, `/models/status` reports `is_loaded: false` for a loaded subprocess** (the frozen `test_model_routes` mocks `is_model_loaded`, so only Task 8 live acceptance catches it)
   - `:482` (`_unload_current_worker` unregister guard) → `if self._worker_available() and self._current_mode:` — **without this, a subprocess unload never unregisters** (`worker` is always `None`), leaking a stale registry entry on every kill. See recon #4 for the OOM-death complement.

   The spec §9.2 semantic split is preserved: `:529`/`:551` mean "nothing to evict"; `:594` means "demand-reload/respawn, then run".

4. **Kill seam — two unregister paths (clean vs dirty).** Registry `unregister_model` is Governor authority (`unregister_model` is idempotent in both `ModelRegistry` impls — `pop(name, None)` / `if name in self._loaded`). Two cases:
   - **Clean unload** (subprocess *alive* — `unload_current_model`, `_load_mode` replace, `_cleanup_vram`, `_evict_if_idle`): routes through `_unload_current_worker`, whose unregister guard is flipped to `self._worker_available()` (recon #3 site `:482`). For an alive subprocess that reads true → unregister fires. → `handle.start()` re-registers on respawn. No new seam.
   - **Dirty death** (subprocess *already dead* — OOM/frameless, Task 7 recovery): `_worker_available()` is `False`, so `_unload_current_worker`'s guard would skip the unregister and **leak the entry**. The recovery path therefore unregisters **explicitly** before respawn: `if self._current_mode: self._registry.unregister_model(self._current_mode)` (idempotent → safe even if already gone), then `handle.stop()` → `_reload_from_snapshot()` re-registers. This is the OOM-death complement to the `:482` flip.

## Global Constraints

- **Python env:** `conda activate stability-toys` before pytest; use `python`, not `python3`.
- **Spawn, never fork:** all subprocess creation uses `multiprocessing.get_context("spawn")`. CUDA contexts do not survive fork.
- **Existing suite stays green:** `tests/test_worker_pool.py`, `tests/test_model_lifecycle.py`, `tests/test_ws_routes.py`, `tests/test_model_routes.py`, and all backplane/governor tests. The in-proc dispatch path is UNCHANGED.
- **Zero `server/` route diff** unless a caller intentionally migrates (the dispatch flip is Governor-internal).
- **Commit discipline (AGENTS.md / stopping-point policy):** every commit message includes the FP id `STABL-rgvxuedo`, what changed, and the exact next step.
- **TDD mandatory:** RED → GREEN → COMMIT for every task. No implementation code without a failing test first.
- **No real GPU in unit tests:** tests cross a real spawn boundary but use a **fault-injecting fake worker** (raises `torch.cuda.OutOfMemoryError` on command; can die frameless). Real-CudaWorker acceptance is a live run on the RTX-3090 box, out of the unit suite.
- **Versioned wire-form:** the job envelope carries a leading `schema_version` byte (backplane discipline); post-M1 additions are additive.

---

## File Structure

**New files:**

| File | Responsibility |
|---|---|
| `backends/worker_handle_subprocess.py` | `SubprocessWorkerHandle` (spawn/kill/respawn, `start`/`submit`/`health`/`unload`/`stop`), the child entrypoint `_worker_main`, and the `LivenessSource` wiring |
| `backends/liveness.py` | `LivenessSource` protocol + `SubprocessLiveness` (heartbeat staleness + EOF/dead-process) |
| `backends/job_envelope.py` | Versioned job wire-form: `encode_job` / `decode_job` (schema_version byte + `{req, job_id, resolution_epoch}`) |
| `tests/test_generaterequest_serialization.py` | M0 prerequisite: `GenerateRequest` round-trips the spawn boundary |
| `tests/test_liveness.py` | `SubprocessLiveness` unit tests |
| `tests/test_job_envelope.py` | wire-form codec tests |
| `tests/test_subprocess_worker_handle.py` | `SubprocessWorkerHandle` across a real spawn boundary (M1 + M2) |
| `tests/_fault_worker.py` | fault-injecting fake worker (module-level so spawn can import it) |

**Modified files:**

| File | Change |
|---|---|
| `backends/worker_handle.py` | `WorkerHealth`: replace `vram_bytes` with `vram_free_bytes` + `vram_total_bytes` (driver truth); update `InProcessWorkerHandle.health()` |
| `backends/backplane/ipc.py` | `IpcJobSink(conn, job_id)`; EOF guard in `drain_to_subscriber` (synthesize `on_error` instead of silent break) |
| `backends/governor.py` | `_worker_available()` helper; flip 3 liveness reads; dual-path dispatch + `submit_job`; OOM/dead → kill+respawn |
| `tests/test_governor.py` | `StubHandle.health()` returns the new `WorkerHealth` fields |

---

## Task 0 (M0 — prerequisite): `GenerateRequest` round-trips the spawn boundary

**Goal:** Confirm — RED-first — whether `GenerateRequest` (a pydantic `BaseModel`, `server/lcm_sr_server.py:136`) survives the spawn pickle. This gates the M1 wire-form: if raw pickle is clean, the envelope carries the instance; if not, it carries `model_dump()` and reconstructs with `model_validate()`. **This is the #1 M1 schedule risk — resolve it before M1.**

**Files:**
- Test: `tests/test_generaterequest_serialization.py`

**Interfaces:**
- Produces: the decision consumed by Task 3 (`job_envelope.py`) — either "pickle the instance" or "carry a dict via `model_dump`/`model_validate`".

- [ ] **Step 1: Write the failing test — round-trip through a spawn child**

```python
# tests/test_generaterequest_serialization.py
import multiprocessing as mp
from server.lcm_sr_server import GenerateRequest

def _echo(conn, payload):
    # Runs in a spawn child: receive, send back — proves cross-process transport.
    conn.send(payload)
    conn.close()

def test_generaterequest_round_trips_spawn_boundary():
    req = GenerateRequest(prompt="a cat", steps=4, width=512, height=512)
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    p = ctx.Process(target=_echo, args=(child, req))
    p.start()
    got = parent.recv()
    p.join(timeout=10)
    assert isinstance(got, GenerateRequest)
    assert got.prompt == "a cat"
    assert got.steps == 4
```

- [ ] **Step 2: Run it**

Run: `conda activate stability-toys && python -m pytest tests/test_generaterequest_serialization.py -q`
Expected: **PASS** if `GenerateRequest` pickles cleanly (pydantic v2 `BaseModel` is picklable by default). **FAIL** (PicklingError/AttributeError) if a field holds an unpicklable value.

- [ ] **Step 3: If it FAILED, add the `model_dump` boundary test and adopt it as the envelope contract**

```python
def test_generaterequest_round_trips_via_model_dump():
    req = GenerateRequest(prompt="a cat", steps=4, width=512, height=512)
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    p = ctx.Process(target=_echo, args=(child, req.model_dump()))
    p.start()
    got = GenerateRequest.model_validate(parent.recv())
    p.join(timeout=10)
    assert got.prompt == "a cat"
```

Record the outcome in the commit message: **"GenerateRequest pickles cleanly"** OR **"GenerateRequest requires model_dump boundary"**. Task 3 consumes this.

- [ ] **Step 4: Commit**

```bash
git add tests/test_generaterequest_serialization.py
git commit -m "test(facet-3): M0 — GenerateRequest round-trips the spawn boundary (STABL-rgvxuedo)

<record the outcome: pickles cleanly | requires model_dump boundary>. Gates the
Task 3 job wire-form. Next: Task 1 WorkerHealth driver-truth fields."
```

---

## Task 1 (early): `WorkerHealth` driver-truth VRAM fields

**Goal:** Resolve the v1 dataclass mismatch (spec §8.1) BEFORE the subprocess handle implements `health()`: `WorkerHealth.vram_bytes` (torch allocator) → `vram_free_bytes` + `vram_total_bytes` (driver truth via `mem_get_info`). This touches merged v1 code; settling it first keeps the health contract stable for Tasks 2/5.

**Files:**
- Modify: `backends/worker_handle.py` (`WorkerHealth`, `InProcessWorkerHandle.health()`)
- Modify: `tests/test_governor.py` (`StubHandle.health()`)

**Interfaces:**
- Produces: `WorkerHealth(state: str, vram_free_bytes: int, vram_total_bytes: int, mode: str | None)`. Consumed by Tasks 2, 3, 5, and the Governor helper in Task 6.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_worker_handle.py
def test_health_reports_driver_truth_vram_fields():
    handle, _ = _make_handle()
    handle.start(Mock(), Mock(), _make_mode())
    h = handle.health()
    assert hasattr(h, "vram_free_bytes")
    assert hasattr(h, "vram_total_bytes")
    assert not hasattr(h, "vram_bytes")
    assert isinstance(h.vram_free_bytes, int)
    assert isinstance(h.vram_total_bytes, int)
```

- [ ] **Step 2: Run it**

Run: `conda activate stability-toys && python -m pytest tests/test_worker_handle.py::test_health_reports_driver_truth_vram_fields -q`
Expected: FAIL — `WorkerHealth` has `vram_bytes`, not the new fields.

- [ ] **Step 3: Change `WorkerHealth` + `InProcessWorkerHandle.health()`**

In `backends/worker_handle.py`, replace the dataclass field and the health method:

```python
@dataclass
class WorkerHealth:
    """Liveness + readiness snapshot the Governor reads for admission/status.
    VRAM is DRIVER TRUTH (mem_get_info), aligning with STABL-sqqlkmdl — not the
    torch allocator."""
    state: str                 # starting | ready | busy | draining | dead
    vram_free_bytes: int       # driver-truth free (what admission needs); 0 if N/A
    vram_total_bytes: int      # driver-truth total; 0 if N/A
    mode: str | None           # loaded mode name, or None
```

```python
    def health(self) -> WorkerHealth:
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
        else:
            free_b, total_b = 0, 0
        return WorkerHealth(
            state=self._state,
            vram_free_bytes=int(free_b),
            vram_total_bytes=int(total_b),
            mode=None,  # mode is the Governor's authority
        )
```

- [ ] **Step 4: Update `StubHandle.health()` in `tests/test_governor.py`**

```python
    def health(self):
        return WorkerHealth(state=self._state, vram_free_bytes=0, vram_total_bytes=0, mode=None)
```

- [ ] **Step 5: Run health + governor + handle suites**

Run: `conda activate stability-toys && python -m pytest tests/test_worker_handle.py tests/test_governor.py -q`
Expected: PASS. (No other consumer reads `vram_bytes` — confirmed by grep before this change.)

- [ ] **Step 6: Commit**

```bash
git add backends/worker_handle.py tests/test_worker_handle.py tests/test_governor.py
git commit -m "feat(facet-3): WorkerHealth driver-truth VRAM fields (STABL-rgvxuedo)

Resolve spec §8.1: vram_bytes (torch allocator) -> vram_free_bytes +
vram_total_bytes (mem_get_info driver truth, aligning with STABL-sqqlkmdl).
InProcessWorkerHandle.health() + StubHandle updated. Next: Task 2 LivenessSource."
```

---

## Task 2: `LivenessSource` abstraction

**Goal:** A transport-agnostic liveness seam (spec §7) the Governor consumes only through `health()`. Subprocess impl = heartbeat staleness + EOF/dead-process; a future rsocket/remote handle backs the same protocol with KEEPALIVE.

**Files:**
- Create: `backends/liveness.py`
- Test: `tests/test_liveness.py`

**Interfaces:**
- Produces: `class LivenessSource(Protocol)` with `state() -> str` (`"live"` | `"dead"`) and `note_heartbeat() -> None`; `class SubprocessLiveness(process, stale_after_s: float)` implementing it. Consumed by `SubprocessWorkerHandle` (Task 5).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_liveness.py
import time
from backends.liveness import SubprocessLiveness

class _FakeProc:
    def __init__(self, alive=True): self._alive = alive
    def is_alive(self): return self._alive

def test_live_when_process_alive_and_heartbeat_fresh():
    liv = SubprocessLiveness(_FakeProc(alive=True), stale_after_s=1.0)
    liv.note_heartbeat()
    assert liv.state() == "live"

def test_dead_when_process_exited():
    liv = SubprocessLiveness(_FakeProc(alive=False), stale_after_s=1.0)
    liv.note_heartbeat()
    assert liv.state() == "dead"

def test_dead_when_heartbeat_stale():
    liv = SubprocessLiveness(_FakeProc(alive=True), stale_after_s=0.05)
    liv.note_heartbeat()
    time.sleep(0.1)
    assert liv.state() == "dead"
```

- [ ] **Step 2: Run it** — Run: `python -m pytest tests/test_liveness.py -q` — Expected: FAIL (module missing).

- [ ] **Step 3: Implement `backends/liveness.py`**

```python
from __future__ import annotations
import time
from typing import Protocol, runtime_checkable

@runtime_checkable
class LivenessSource(Protocol):
    def state(self) -> str: ...          # "live" | "dead"
    def note_heartbeat(self) -> None: ...

class SubprocessLiveness:
    """Liveness = process alive AND last heartbeat within stale_after_s. When the
    transport becomes rsocket, a KeepaliveLiveness backs the same protocol with no
    Governor change."""
    def __init__(self, process, stale_after_s: float = 10.0):
        self._process = process
        self._stale_after_s = stale_after_s
        self._last_heartbeat = time.monotonic()

    def note_heartbeat(self) -> None:
        self._last_heartbeat = time.monotonic()

    def state(self) -> str:
        if not self._process.is_alive():
            return "dead"
        if time.monotonic() - self._last_heartbeat > self._stale_after_s:
            return "dead"
        return "live"
```

- [ ] **Step 4: Run it** — Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add backends/liveness.py tests/test_liveness.py
git commit -m "feat(facet-3): LivenessSource seam + SubprocessLiveness (STABL-rgvxuedo)

Transport-agnostic liveness (process-alive + heartbeat staleness) behind a
Protocol; rsocket KEEPALIVE backs the same contract later. Next: Task 3 wire-form."
```

---

## Task 3: Versioned job wire-form

**Goal:** `encode_job`/`decode_job` carrying `{req, job_id, resolution_epoch}` behind a leading `schema_version` byte (spec §4.2). Uses the Task 0 outcome for `req` (instance vs `model_dump`).

**Files:**
- Create: `backends/job_envelope.py`
- Test: `tests/test_job_envelope.py`

**Interfaces:**
- Consumes: Task 0 decision (pickle instance vs model_dump).
- Produces: `encode_job(job) -> bytes`, `decode_job(raw: bytes) -> DecodedJob` where `DecodedJob` has `.req`, `.job_id: str`, `.resolution_epoch: int`. Consumed by `SubprocessWorkerHandle.submit` + `_worker_main` (Task 5).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_job_envelope.py
from backends.job_envelope import encode_job, decode_job, JOB_SCHEMA_VERSION
from backends.governor import GenerationJob
from unittest.mock import Mock

def test_job_envelope_round_trips_minimal_fields():
    from server.lcm_sr_server import GenerateRequest
    req = GenerateRequest(prompt="x", steps=4, width=512, height=512)
    job = GenerationJob(req=req, resolution_epoch=7)
    raw = encode_job(job)
    assert raw[0] == JOB_SCHEMA_VERSION           # leading version byte
    d = decode_job(raw)
    assert d.job_id == job.job_id
    assert d.resolution_epoch == 7
    assert d.req.prompt == "x"

def test_job_envelope_rejects_unknown_version():
    import pytest
    with pytest.raises(ValueError):
        decode_job(bytes([99]) + b"garbage")
```

- [ ] **Step 2: Run it** — Expected: FAIL (module missing).

- [ ] **Step 3: Implement `backends/job_envelope.py`**

Use the Task 0 outcome. If `GenerateRequest` pickles cleanly, the body pickles `(req, job_id, resolution_epoch)`; otherwise it pickles `(req.model_dump(), job_id, resolution_epoch)` and `decode_job` reconstructs via `GenerateRequest.model_validate`. Version-gate on the first byte.

```python
from __future__ import annotations
import pickle
from dataclasses import dataclass

JOB_SCHEMA_VERSION = 1

@dataclass
class DecodedJob:
    req: object
    job_id: str
    resolution_epoch: int

def encode_job(job) -> bytes:
    body = pickle.dumps((job.req, job.job_id, job.resolution_epoch))
    return bytes([JOB_SCHEMA_VERSION]) + body

def decode_job(raw: bytes) -> DecodedJob:
    version = raw[0]
    if version != JOB_SCHEMA_VERSION:
        raise ValueError(f"unknown job schema_version {version}")
    req, job_id, resolution_epoch = pickle.loads(raw[1:])
    return DecodedJob(req=req, job_id=job_id, resolution_epoch=resolution_epoch)
```

(If Task 0 required the `model_dump` boundary: `pickle.dumps((job.req.model_dump(), ...))` in encode, and `GenerateRequest.model_validate(req_dict)` in decode. Keep `init_image`/`controlnet_bindings` OUT — additive later.)

- [ ] **Step 4: Run it** — Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backends/job_envelope.py tests/test_job_envelope.py
git commit -m "feat(facet-3): versioned job wire-form {req,job_id,resolution_epoch} (STABL-rgvxuedo)

schema_version-byte gated; init_image/controlnet deferred (additive). Next:
Task 4 IPC job_id + EOF guard hardening."
```

---

## Task 4: IPC hardening — `job_id` threading + EOF guard

**Goal:** Two backplane facet-3 debts the subprocess handle needs: `IpcJobSink(conn, job_id)` (currently hardcodes `"job"`) and the frameless-death guard in `drain_to_subscriber` (currently a silent `break` on `EOFError` — spec §6(b)).

**Files:**
- Modify: `backends/backplane/ipc.py`
- Test: `tests/test_backplane_ipc.py` (add cases)

**Interfaces:**
- Produces: `IpcJobSink(conn, job_id: str)`; `drain_to_subscriber(conn, subscriber)` calls `subscriber.on_error(BackplaneError(...))` on EOF before a terminal.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_backplane_ipc.py
from backends.backplane.ipc import IpcJobSink, drain_to_subscriber
from backends.backplane.frames import BackplaneError, BackplaneErrorCode

def test_ipc_sink_stamps_job_id():
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe()
    sink = IpcJobSink(a, job_id="abc123")
    sink.ack(0)
    from backends.backplane.blob import decode_frame
    frame = decode_frame(b.recv_bytes())
    assert frame.job_id == "abc123"

class _CollectingSub:
    def __init__(self): self.error = None; self.completed = False
    def on_subscribe(self, s): pass
    def on_next(self, v): pass
    def on_error(self, e): self.error = e
    def on_complete(self): self.completed = True

def test_drain_synthesizes_error_on_frameless_eof():
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe()
    b.close()                       # frameless death: producer end closed, no terminal
    sub = _CollectingSub()
    drain_to_subscriber(a, sub)
    assert sub.error is not None
    assert sub.error.code == BackplaneErrorCode.GENERIC
    assert not sub.completed
```

- [ ] **Step 2: Run it** — Expected: FAIL (`IpcJobSink` takes no `job_id`; drain breaks silently, `error` stays None).

- [ ] **Step 3: Thread `job_id` and add the EOF guard**

`IpcJobSink.__init__(self, conn, job_id: str = "job")`; replace the hardcoded `"job"` in `ack`/`progress`/`result` frames with `self._job_id`. In `drain_to_subscriber`, replace the silent `except EOFError: break`:

```python
    while True:
        try:
            raw = conn.recv_bytes()
        except EOFError:
            # Frameless death (spec §6b): the producer end closed before a terminal.
            # Synthesize a failure terminal so a waiting Future never hangs.
            subscriber.on_error(BackplaneError(BackplaneErrorCode.GENERIC, "worker connection closed"))
            break
```

- [ ] **Step 4: Run it** — Run: `python -m pytest tests/test_backplane_ipc.py -q` — Expected: PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add backends/backplane/ipc.py tests/test_backplane_ipc.py
git commit -m "feat(facet-3): IPC job_id threading + frameless-death EOF guard (STABL-rgvxuedo)

IpcJobSink(conn, job_id); drain_to_subscriber synthesizes on_error on EOF before
a terminal so a waiting Future never hangs (spec §6b). Next: Task 5 handle.start."
```

---

## Task 5: `SubprocessWorkerHandle` — spawn, load, submit, drive (M1)

**Goal:** The core M1 handle: `start()` spawns a child that builds the worker via the factory and loads the model, then signals READY; `submit(job)` sends the wire-form and returns a `Publisher` driven by the child; `health()` reads the `LivenessSource`; `stop()`/`unload()` kill the process.

**Files:**
- Create: `backends/worker_handle_subprocess.py`
- Create: `tests/_fault_worker.py` (importable by the spawn child)
- Test: `tests/test_subprocess_worker_handle.py`

**Interfaces:**
- Consumes: `WorkerHandle` ABC + `WorkerHealth` (Task 1); `encode_job`/`decode_job` (Task 3); `IpcJobSink`/`drain_to_subscriber` (Task 4); `SubprocessLiveness` (Task 2).
- Produces: `SubprocessWorkerHandle(worker_factory_ref: str)` implementing `WorkerHandle`; `worker` property returns `None`; `health().state` ∈ {starting, ready, busy, dead}.

- [ ] **Step 1: Write the fault worker (module-level, spawn-importable)**

```python
# tests/_fault_worker.py
class FaultWorker:
    """Fake PipelineWorker for spawn-boundary tests. run_job echoes; can be told to
    OOM or die frameless via the request payload's prompt sentinel."""
    def __init__(self, *args, **kwargs): pass
    def run_job(self, job):
        prompt = getattr(job.req, "prompt", "")
        if prompt == "__OOM__":
            import torch
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (injected)")
        if prompt == "__DIE__":
            import os, signal
            os.kill(os.getpid(), signal.SIGKILL)   # frameless death
        return b"PNG:" + prompt.encode()

def make_fault_worker(worker_id, resolved, binding):
    return FaultWorker()
```

- [ ] **Step 2: Write the failing test — start + submit a succeeding job across spawn**

```python
# tests/test_subprocess_worker_handle.py
from concurrent.futures import Future
from unittest.mock import Mock
from backends.worker_handle_subprocess import SubprocessWorkerHandle
from backends.governor import GenerationJob, _FutureBridge
from server.lcm_sr_server import GenerateRequest

def _req(prompt="hello"):
    return GenerateRequest(prompt=prompt, steps=4, width=512, height=512)

def test_subprocess_handle_runs_a_job_end_to_end():
    h = SubprocessWorkerHandle("tests._fault_worker.make_fault_worker")
    h.start(Mock(), Mock(), Mock())            # spawns child, loads FaultWorker
    assert h.health().state == "ready"
    assert h.worker is None                     # no in-proc worker
    job = GenerationJob(req=_req("hello"), resolution_epoch=0)
    pub = h.submit(job)
    fut = Future()
    pub.subscribe(_FutureBridge(fut))
    assert fut.result(timeout=15) == b"PNG:hello"
    h.stop()
    assert h.health().state == "dead"
```

- [ ] **Step 3: Run it** — Expected: FAIL (module missing).

- [ ] **Step 4: Implement `SubprocessWorkerHandle` + `_worker_main`**

The child `_worker_main(conn, factory_ref, resolved, binding, mode)`: import the factory by dotted ref, build the worker, send `READY`, then loop — receive an encoded job, `decode_job`, build a `GenerationJob`-shaped object, run `worker.run_job`, and drive an `IpcJobSink(conn, job_id)` (`result` + `complete`, or `error` on exception). Heartbeats piggyback on a periodic `Ack`/dedicated frame. The parent `submit()` sends the envelope and returns a `Publisher` whose `subscribe` starts a reader thread running `drain_to_subscriber`.

```python
from __future__ import annotations
import importlib, threading, multiprocessing as mp
from typing import Optional
from backends.worker_handle import WorkerHandle, WorkerHealth
from backends.liveness import SubprocessLiveness
from backends.job_envelope import encode_job, decode_job
from backends.backplane.ipc import IpcJobSink, drain_to_subscriber
from backends.backplane.reactivestreams import Publisher

_READY = b"\x00READY"

def _resolve_ref(dotted: str):
    mod, name = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)

def _worker_main(conn, factory_ref, resolved, binding, mode):
    factory = _resolve_ref(factory_ref)
    worker = factory(0, resolved, binding)
    conn.send_bytes(_READY)
    while True:
        try:
            raw = conn.recv_bytes()
        except EOFError:
            break
        d = decode_job(raw)
        from backends.governor import GenerationJob
        job = GenerationJob(req=d.req, resolution_epoch=d.resolution_epoch, job_id=d.job_id)
        sink = IpcJobSink(conn, job_id=d.job_id)
        try:
            result = worker.run_job(job)
            sink.result(0, result)
            sink.complete()
        except Exception as e:   # noqa: BLE001 — rides the sink terminal
            from backends.backplane.frames import BackplaneError
            sink.error(BackplaneError.from_exc(e))

class _SubprocPublisher(Publisher):
    def __init__(self, conn): self._conn = conn
    def subscribe(self, subscriber):
        t = threading.Thread(target=drain_to_subscriber, args=(self._conn, subscriber), daemon=True)
        t.start()

class SubprocessWorkerHandle(WorkerHandle):
    def __init__(self, worker_factory_ref: str):
        self._factory_ref = worker_factory_ref
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._parent_conn = None
        self._liveness: Optional[SubprocessLiveness] = None
        self._state = "starting"

    @property
    def worker(self): return None

    def start(self, resolved_mode, binding, mode) -> None:
        self._parent_conn, child_conn = self._ctx.Pipe()
        self._proc = self._ctx.Process(
            target=_worker_main,
            args=(child_conn, self._factory_ref, resolved_mode, binding, mode),
            daemon=True,
        )
        self._proc.start()
        self._liveness = SubprocessLiveness(self._proc)
        if self._parent_conn.recv_bytes() != _READY:   # blocks until READY
            raise RuntimeError("subprocess worker failed to signal READY")
        self._liveness.note_heartbeat()
        self._state = "ready"

    def submit(self, job) -> Publisher:
        self._state = "busy"
        self._parent_conn.send_bytes(encode_job(job))
        self._liveness.note_heartbeat()
        return _SubprocPublisher(self._parent_conn)

    def health(self) -> WorkerHealth:
        state = "dead" if (self._liveness is None or self._liveness.state() == "dead") else self._state
        return WorkerHealth(state=state, vram_free_bytes=0, vram_total_bytes=0, mode=None)

    def unload(self) -> None: self.stop()

    def stop(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._proc.kill()
            self._proc.join(timeout=5.0)
        self._state = "dead"
```

(VRAM reporting via heartbeat — `vram_free_bytes`/`vram_total_bytes` from the child's `mem_get_info` — is wired when the child sends periodic health frames; M1 reports 0 and M2/heartbeat fills them. Keep this note; do not block M1 on it.)

- [ ] **Step 5: Run it** — Run: `python -m pytest tests/test_subprocess_worker_handle.py::test_subprocess_handle_runs_a_job_end_to_end -q` — Expected: PASS. If flaky on `recv`, confirm the reader thread + `_FutureBridge` unbounded demand deliver the buffered Result.

- [ ] **Step 6: Commit**

```bash
git add backends/worker_handle_subprocess.py tests/_fault_worker.py tests/test_subprocess_worker_handle.py
git commit -m "feat(facet-3): SubprocessWorkerHandle — spawn+load+submit across the boundary (STABL-rgvxuedo)

M1 core: start() spawns a child hosting the worker, submit() drives IpcJobSink and
returns a Publisher, stop() kills the process. Next: Task 6 Governor dispatch flip."
```

---

## Task 6: Governor dual-path dispatch + liveness flip (M1 — the finding-B change)

**Goal:** Make the Governor drive a subprocess handle: `_worker_available()` helper + the 3 liveness-read flips (reconciliation #3), and the dual-path GenerationJob dispatch (reconciliation #1/#2) — subprocess uses `handle.submit()` + `_FutureBridge`; in-proc unchanged.

**Files:**
- Modify: `backends/governor.py`
- Test: `tests/test_governor.py` (add a subprocess-path test using `SubprocessWorkerHandle`)

**Interfaces:**
- Consumes: `SubprocessWorkerHandle` (Task 5); `_worker_available()` (defined here).
- Produces: a Governor that dispatches GenerationJobs to an out-of-proc handle via `handle.submit()`.

- [ ] **Step 1: Write the failing test — Governor drives a subprocess handle**

```python
# add to tests/test_governor.py
def test_governor_dispatches_generation_to_subprocess_handle():
    from backends.worker_handle_subprocess import SubprocessWorkerHandle
    from server.lcm_sr_server import GenerateRequest
    handle = SubprocessWorkerHandle("tests._fault_worker.make_fault_worker")
    mode_config = Mock(); mode = Mock()
    mode.model_path = "/models/test.safetensors"; mode.loras = []
    from backends.conditioning.contracts import ConditioningConfig
    mode.conditioning = ConditioningConfig()
    mode_config.get_mode.return_value = mode
    mode_config.get_default_mode.return_value = "test-mode"
    registry = Mock(); registry.get_total_vram.return_value = 0
    registry.get_used_vram.return_value = 0; registry.get_allocated_vram.return_value = 0
    with patch("backends.governor.resolve_model", return_value=(Mock(), Mock())):
        gov = Governor(handle=handle, mode_config=mode_config, registry=registry)
    job = GenerationJob(req=GenerateRequest(prompt="hi", steps=4, width=512, height=512),
                        resolution_epoch=gov.current_resolution_epoch())
    fut = gov.submit_job(job)
    assert fut.result(timeout=15) == b"PNG:hi"
    gov.shutdown()
```

- [ ] **Step 2: Run it** — Expected: FAIL — the in-proc dispatch calls `job.execute(self._handle.worker)` with `worker=None` → "No worker available".

- [ ] **Step 3: Add `_worker_available()` and flip all FIVE liveness reads**

In `backends/governor.py`, add the helper and replace **five** predicates (recon #3 — review-verified sites). For `InProcessWorkerHandle`, `_worker_available()` is exactly equivalent to `worker is not None`, so the frozen suite is unaffected.

```python
    def _worker_available(self) -> bool:
        """Locality-agnostic 'is a live, loaded worker present?'. For InProcess this
        equals worker-is-not-None (start->ready, unload->dead); for Subprocess it
        reads true liveness (its .worker is always None). Preserves the spec §9.2
        semantic split at each call site."""
        return self._handle.health().state in ("ready", "busy")
```

- `:529` (`_idle_watchdog_loop`) `if self._handle.worker is None:` → `if not self._worker_available():`
- `:551` (`_evict_if_idle`) `if self._handle.worker is None:` → `if not self._worker_available():`
- `:594` (demand-reload) `if self._handle.worker is None and self._active_snapshot is not None:` → `if not self._worker_available() and self._active_snapshot is not None:`
- `:756` (`is_model_loaded`) `return self._handle.worker is not None` → `return self._worker_available()` — **without this, `/models/status` reports `is_loaded: false` for a loaded subprocess** (only Task 8 live acceptance catches it; the frozen `test_model_routes` mocks `is_model_loaded`).
- `:482` (`_unload_current_worker` unregister guard) `if self._handle.worker is not None and self._current_mode:` → `if self._worker_available() and self._current_mode:` — **without this, a clean subprocess unload never unregisters** (`worker` is always `None`). The OOM-*death* complement is the explicit unregister in Task 7 Step 3 (recon #4).

- [ ] **Step 4: Dual-path the GenerationJob dispatch + `submit_job`**

In `submit_job`, open the InProcBackplane channel + attach `_FutureBridge` ONLY for the in-proc path (guard on `self._handle.worker is not None`); for the subprocess path, just register + enqueue (the bridge attaches in the dispatch loop). In the dispatch loop's GenerationJob branch, replace the single `result = job.execute(self._handle.worker)` block with:

```python
                    if isinstance(job, GenerationJob):
                        if self._handle.worker is not None:
                            # --- IN-PROC PATH (v1, unchanged) ---
                            result = job.execute(self._handle.worker)
                            sink = job_record.sink if job_record is not None else None
                            if job_record is not None and job_record.cancel_requested:
                                job_record.state = "cancelled"
                                if sink is not None:
                                    sink.error(BackplaneError(BackplaneErrorCode.CANCELLED, "cancelled"))
                                elif not job.fut.done():
                                    job.fut.set_exception(CancelledError())
                                self._finalize_job_record(job.job_id)
                            elif sink is not None:
                                sink.result(0, InProcBlob(result)); sink.complete()
                                self._finalize_job_record(job.job_id)
                            elif not job.fut.done():
                                job.fut.set_result(result)
                        else:
                            # --- SUBPROCESS PATH (facet-3): the handle owns the channel ---
                            publisher = self._handle.submit(job)
                            publisher.subscribe(_FutureBridge(job.fut))
                            if job_record is not None:
                                self._finalize_job_record(job.job_id)
```

Guard the in-proc channel-open in `submit_job`:

```python
            if isinstance(job, GenerationJob) and self._handle.worker is not None:
                sink, publisher = InProcBackplane(job.job_id).open()
                record = self._get_job_record(job.job_id)
                if record is not None:
                    record.sink = sink
                publisher.subscribe(_FutureBridge(job.fut))
```

- [ ] **Step 5: Run it** — Run: `python -m pytest tests/test_governor.py -q` — Expected: PASS (subprocess path + all existing governor tests). Then the frozen in-proc suite:

Run: `python -m pytest tests/test_worker_pool.py tests/test_model_lifecycle.py -q`
Expected: PASS unmodified (in-proc path untouched; `_worker_available()` is equivalent to `worker is None` for in-proc).

- [ ] **Step 6: Commit**

```bash
git add backends/governor.py tests/test_governor.py
git commit -m "feat(facet-3): Governor dual-path dispatch + liveness-read flip (STABL-rgvxuedo)

Finding B realized: _worker_available() (health().state in ready/busy) replaces the
3 worker-is-None reads (semantics-identical in-proc); GenerationJob dispatch branches
by locality — subprocess uses handle.submit()+_FutureBridge, in-proc unchanged. M1
complete: a subprocess job runs end-to-end. Next: Task 7 OOM kill+respawn."
```

---

## Task 7: Durable OOM recovery — kill + respawn (M2)

**Goal:** The thesis proof. An OOM error frame from the child → Governor kills + respawns via the existing seam → next job succeeds. Plus the frameless-death path (Task 4's EOF guard already synthesizes the terminal; here the Governor reacts to a `dead` handle).

**Files:**
- Modify: `backends/governor.py` (recovery on OOM/dead in the dispatch exception path + demand-reload respawn)
- Test: `tests/test_subprocess_worker_handle.py` + `tests/test_governor.py`

**Interfaces:**
- Consumes: `_worker_available()`, `_unload_current_worker`, `handle.start()`, `handle.stop()`.
- Produces: `Governor` recovers a subprocess after OOM/death and the next job succeeds.

- [ ] **Step 1: Write the failing test — OOM then success**

```python
# add to tests/test_governor.py
def test_governor_recovers_from_subprocess_oom_and_next_job_succeeds():
    from backends.worker_handle_subprocess import SubprocessWorkerHandle
    from server.lcm_sr_server import GenerateRequest
    handle = SubprocessWorkerHandle("tests._fault_worker.make_fault_worker")
    # ... same Governor construction as the Task 6 test (mock mode_config/registry,
    #     patch backends.governor.resolve_model) ...
    gov = _make_subprocess_governor(handle)     # helper factoring the Task 6 setup
    oom = GenerationJob(req=GenerateRequest(prompt="__OOM__", steps=4, width=512, height=512),
                        resolution_epoch=gov.current_resolution_epoch())
    with pytest.raises(Exception):
        gov.submit_job(oom).result(timeout=15)   # OOM surfaces as the job's exception
    # recovery: the handle was killed + respawned; a fresh job succeeds
    ok = GenerationJob(req=GenerateRequest(prompt="after", steps=4, width=512, height=512),
                       resolution_epoch=gov.current_resolution_epoch())
    assert gov.submit_job(ok).result(timeout=15) == b"PNG:after"
    gov.shutdown()
```

- [ ] **Step 2: Run it** — Expected: FAIL — after OOM the dead subprocess is never respawned; the second job hangs or errors.

- [ ] **Step 3: Add subprocess recovery to the dispatch exception path + demand-reload**

For the subprocess path, when the job terminates in an OOM error (or the handle goes `dead`), the Governor must kill+respawn before the next job. Two touch-points:

(a) In the dispatch loop's exception handler, after the subprocess job fails, if the handle is dead, kill + **explicitly unregister** (the recon #4 dirty-death complement — `_worker_available()` is `False` here, so `_unload_current_worker`'s guard would skip the unregister and leak), then respawn:

```python
            if not self._worker_available():
                logger.warning("[Governor] Subprocess dead post-job; killing + respawning")
                if self._current_mode:
                    self._registry.unregister_model(self._current_mode)   # idempotent; dead subprocess never self-unregistered
                self._handle.stop()                     # ensure the process is killed
                if self._active_snapshot is not None:
                    self._reload_from_snapshot()        # -> handle.start() respawns + re-registers
```

(The `self._handle.worker is None` conjunct is dropped — `not self._worker_available()` already implies the worker is unavailable regardless of locality.)

(b) The demand-reload trigger (the flipped `:594`) already respawns *before* a job when `not self._worker_available()`. **No change needed to `_reload_from_snapshot`** — it already calls `self._handle.start(snapshot.resolved, snapshot.binding, snapshot.mode)` (`governor.py:387`), so both localities respawn identically. (Confirmed against source during review — do not "generalize" it; it is already locality-agnostic.)

- [ ] **Step 4: Run it** — Run: `python -m pytest tests/test_governor.py::test_governor_recovers_from_subprocess_oom_and_next_job_succeeds -q` — Expected: PASS.

- [ ] **Step 5: Add the frameless-death test**

```python
def test_governor_recovers_from_frameless_subprocess_death():
    # prompt="__DIE__" SIGKILLs the child mid-job; the EOF guard synthesizes the
    # failure terminal, the Governor respawns, next job succeeds.
    ... assert next job == b"PNG:after"
```

Run it; expected PASS (the Task 4 EOF guard + Step 3 recovery cover this).

- [ ] **Step 6: Full regression + commit**

Run: `conda activate stability-toys && python -m pytest tests/ --ignore=tests/test_backplane_ipc.py -q` (then the ipc suite separately)
Expected: only the pre-existing `test_mode_config` hunyuandit failure; everything else green.

```bash
git add backends/governor.py tests/test_governor.py tests/test_subprocess_worker_handle.py
git commit -m "feat(facet-3): durable OOM recovery — kill+respawn after subprocess OOM/death (STABL-rgvxuedo)

M2 thesis proof: subprocess OOM -> error frame -> _unload_current_worker (kill +
unregister) -> _reload_from_snapshot (respawn via handle.start) -> next job succeeds.
Frameless death (SIGKILL) covered by the EOF guard's synthesized terminal. Next:
live acceptance on the RTX-3090 box + STOP/NEXT."
```

---

## Task 8: Live acceptance + docs

**Goal:** Prove the thesis on real hardware (unit tests use a fake worker across a real boundary; this is the real-CudaWorker run) and record the outcome.

**Files:**
- Modify: `project-forward-notes.md` (facet-3 landed; VRAM-from-child accounting shift; dual-path dispatch note)

- [ ] **Step 1: Live run on the RTX-3090 box** — load a real mode (e.g. SDXL txt2img), run a job that succeeds; then force an OOM (oversized request or a low `PYTORCH_CUDA_ALLOC_CONF`), confirm the subprocess is killed + respawned and the next job succeeds. Capture `nvidia-smi` before/after showing the per-process context is reclaimed on kill (the ~0.5–1.5 GB that `empty_cache` could not free).

- [ ] **Step 2: Update `project-forward-notes.md`** — move facet-3 from "todo/no issue" to "landed"; record: VRAM accounting source moved to the child (spec §8); dual-path dispatch (reconciliation #1) with unification deferred; the backplane facet-3 debts now discharged (`job_id`, EOF guard) vs still-deferred (`STALE_EPOCH` registry, `request(n)` backpressure).

- [ ] **Step 3: Commit + FP STOP/NEXT**

```bash
git add project-forward-notes.md
git commit -m "docs(facet-3): forward register — subprocess worker landed, VRAM-from-child (STABL-rgvxuedo)"
```
Then `fp comment STABL-rgvxuedo` with the live-acceptance result and the deferred-sibling status; `fp issue update --status done STABL-rgvxuedo` after review.

---

## Deferred (tracked, NOT in this plan)

Multi-GPU/UUID (`STABL-cchxvuhs`); `superres` migration (parent keeps its context for it); full `ControlNetBinding` + `init_image` wire-forms (additive to Task 3's envelope); `CustomJob` redesign (D4); reap *policy* (`STABL-qvmdayhb`); mode-switch races (`STABL-ltefhpkk`/`STABL-iuiwzthc`); dispatch **unification** onto `handle.submit()` for both localities (reconciliation #1); `STALE_EPOCH` consumer-injected reconstruction registry + IPC `request(n)` backpressure (backplane facet-3 debts not needed for M1/M2).

---

## Self-review

- **Spec coverage:** §2 scope → Tasks 0–8; §4.1 start/load → Task 5; §4.2 versioned wire-form + GenerateRequest prereq → Tasks 0,3; §5 dispatch flip + eventual-consistency cancel → Task 6 (recon #1/#2); §6 OOM (a)+(b) → Tasks 4,7; §7 LivenessSource → Task 2; §8 VRAM driver-truth → Task 1 (heartbeat VRAM fill noted in Task 5); §9 finding-B enumerations → Task 6 (helper + **5 liveness sites**: `:529`/`:551`/`:594`/`is_model_loaded:756`/`_unload_current_worker:482`) + Task 7 (kill seam, clean + dirty-death unregister). All covered.
- **Placeholder scan:** the one deliberate branch is Task 0's outcome feeding Task 3 (both concrete forms given); Task 5 notes heartbeat-VRAM fill as a non-blocking follow-on with the value stated (0 for M1). No TBDs.
- **Type consistency:** `WorkerHealth(state, vram_free_bytes, vram_total_bytes, mode)` (Task 1) used consistently in Tasks 2/5/6; `DecodedJob(req, job_id, resolution_epoch)` (Task 3) consumed in Task 5; `_worker_available()` (Task 6) referenced in Task 7; `IpcJobSink(conn, job_id)` (Task 4) used in Task 5.
