# Timed-Out Job Reap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **This repo forbids subagent-driven development (`AGENTS.md`) — execute inline.**

**Goal:** A generation that exceeds its execution budget stops within roughly one denoise step, in both isolation modes, without killing the worker or reloading the model.

**Architecture:** One predicate — `should_cancel()` — consulted inside the denoise loop at the per-step callback `STABL-zueslhah` already installs. In-proc it reads `record.cancel_requested`; in the subprocess child it reads a `threading.Event` set by a `_CANCEL_JOB` message on the **dedicated control pipe**. The raise is `concurrent.futures.CancelledError`, which the existing classifier and the existing dispatch cancel branch already terminate correctly.

**Tech Stack:** Python 3.12, pytest, `multiprocessing` spawn context, diffusers step callbacks.

**Spec:** `docs/superpowers/specs/2026-08-03-timed-out-job-reap-design.md`
**Issue:** STABL-jredufxb

## Global Constraints

- **The cancel check must NOT go inside `_emit`.** `_emit` swallows every exception on purpose ("a bad consumer must never break generation"); a cancel raised there is silently eaten.
- **The exception is `concurrent.futures.CancelledError`.** Not a bespoke class: `classify_exception()` maps only that type or a class literally named `CancelledError` to `BackplaneErrorCode.CANCELLED`, and the subprocess parent path does no `cancel_requested` remap.
- **The exception message must never contain the substring `"out of memory"`.** The dispatch loop's `_oom` test is a substring match on `str(e)` and runs *before* the cancel branch.
- **`should_cancel` is read lock-free.** The worker must never acquire `_job_lock` (backplane `Subscriber`↔lock invariant).
- **Subprocess cancel travels on the control pipe, never the data pipe.** The child blocks on the data pipe awaiting its next job; a cancel there is decoded as a job frame.
- **Cancel sends take `_control_lock`, and are issued OUTSIDE `_job_lock`.**
- Every new parameter is keyword-with-default-`None`, so existing callers and the RKNN worker (`run_job(self, job)`) are unaffected.
- Run tests with the repo env: `conda activate stability-toys`.

---

### Task 1: The cancel predicate in the step callback

**Files:**
- Modify: `backends/step_progress.py`
- Test: `tests/test_step_progress_cancel.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `inject_step_progress(pipe, pipe_kwargs, progress, total, should_cancel=None) -> None`. Raises `concurrent.futures.CancelledError` from inside the installed callback when `should_cancel()` returns true.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_step_progress_cancel.py`:

```python
"""STABL-jredufxb: the cancel predicate at the denoise step boundary."""
import pytest
from concurrent.futures import CancelledError

from backends.step_progress import inject_step_progress


class ModernPipe:
    """Pipeline exposing the modern `callback_on_step_end` hook."""

    def __call__(self, callback_on_step_end=None, **kwargs):
        for step in range(10):
            self.seen.append(step)
            callback_on_step_end(self, step, None, {})
        return "done"

    def __init__(self):
        self.seen = []


class LegacyPipe:
    """Pipeline exposing only the legacy `callback` / `callback_steps` pair."""

    def __call__(self, callback=None, callback_steps=1, **kwargs):
        for step in range(10):
            self.seen.append(step)
            callback(step, None, None)
        return "done"

    def __init__(self):
        self.seen = []


def test_predicate_stops_the_modern_loop():
    """The predicate flips mid-run; the loop must stop at the NEXT step, not run
    to completion and not stop before it was asked to."""
    flag = {"cancel": False}
    pipe = ModernPipe()
    kwargs = {}

    def should_cancel():
        # Flip on the fourth interrogation, i.e. after three steps have run.
        should_cancel.calls += 1
        return should_cancel.calls > 3
    should_cancel.calls = 0

    inject_step_progress(pipe, kwargs, None, 10, should_cancel=should_cancel)

    with pytest.raises(CancelledError):
        pipe(**kwargs)
    assert pipe.seen == [0, 1, 2, 3]


def test_predicate_stops_the_legacy_loop():
    flag = {"cancel": True}
    pipe = LegacyPipe()
    kwargs = {}
    inject_step_progress(pipe, kwargs, None, 10,
                         should_cancel=lambda: flag["cancel"])
    with pytest.raises(CancelledError):
        pipe(**kwargs)
    assert pipe.seen == [0]


def test_cancel_message_never_mentions_out_of_memory():
    """The dispatch loop's OOM test is a substring match on str(e) and runs
    BEFORE the cancel branch — a cancel that says 'out of memory' would route
    into OOM recovery."""
    pipe = ModernPipe()
    kwargs = {}
    inject_step_progress(pipe, kwargs, None, 10, should_cancel=lambda: True)
    with pytest.raises(CancelledError) as excinfo:
        pipe(**kwargs)
    assert "out of memory" not in str(excinfo.value).lower()


def test_a_raising_progress_consumer_still_cannot_break_generation():
    """Regression guard on the _emit swallow — proves the cancel check sits
    OUTSIDE it."""
    def boom(step, total, stage):
        raise RuntimeError("bad consumer")

    pipe = ModernPipe()
    kwargs = {}
    inject_step_progress(pipe, kwargs, boom, 10)
    assert pipe(**kwargs) == "done"
    assert pipe.seen == list(range(10))


def test_no_progress_and_no_predicate_installs_nothing():
    pipe = ModernPipe()
    kwargs = {}
    inject_step_progress(pipe, kwargs, None, 10)
    assert kwargs == {}


def test_predicate_alone_installs_the_callback():
    """progress=None must no longer short-circuit: a reap with no progress
    consumer attached still needs the callback installed."""
    pipe = ModernPipe()
    kwargs = {}
    inject_step_progress(pipe, kwargs, None, 10, should_cancel=lambda: False)
    assert "callback_on_step_end" in kwargs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_step_progress_cancel.py -v`
Expected: FAIL — `TypeError: inject_step_progress() got an unexpected keyword argument 'should_cancel'`

- [ ] **Step 3: Implement**

In `backends/step_progress.py`, add the import at module top:

```python
from concurrent.futures import CancelledError
```

Replace the body of `inject_step_progress` with:

```python
def inject_step_progress(
    pipe: Any,
    pipe_kwargs: dict,
    progress: Optional[ProgressEmitter],
    total: int,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    if progress is None and should_cancel is None:
        return
    try:
        params = inspect.signature(pipe.__call__).parameters
    except (ValueError, TypeError):
        return

    total_i = int(total)

    def _emit(step_index: int) -> None:
        if progress is None:
            return
        try:
            progress(step_index + 1, total_i, "denoise")  # 1-based
        except Exception:
            pass  # a bad consumer must never break generation

    def _check_cancel() -> None:
        # STABL-jredufxb. Deliberately NOT inside _emit: that swallows every
        # exception so a misbehaving progress consumer cannot break generation,
        # and a cancel raised there would be eaten with it.
        #
        # concurrent.futures.CancelledError, not a bespoke type: classify_exception()
        # maps only that (or a class named CancelledError) to CANCELLED, and the
        # subprocess parent path does no cancel_requested remap. The message must
        # never contain "out of memory" — the dispatch loop's _oom substring test
        # runs before its cancel branch.
        if should_cancel is not None and should_cancel():
            raise CancelledError("job cancelled at step boundary")

    if "callback_on_step_end" in params:
        def _modern(_pipe, step, _timestep, callback_kwargs):
            _check_cancel()
            _emit(step)
            return callback_kwargs
        pipe_kwargs["callback_on_step_end"] = _modern
    elif "callback" in params and "callback_steps" in params:
        def _legacy(step, _timestep, _latents):
            _check_cancel()
            _emit(step)
        pipe_kwargs["callback"] = _legacy
        pipe_kwargs["callback_steps"] = 1
```

Also update the module docstring's first paragraph to mention the cancel predicate:

```python
"""Map a diffusers step callback to a backplane Progress emitter (STABL-zueslhah)
and to a cancellation predicate (STABL-jredufxb).
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_step_progress_cancel.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Run the existing progress suite for regressions**

Run: `python -m pytest tests/ -k "step_progress or progress" -v`
Expected: PASS, no failures

- [ ] **Step 6: Commit**

```bash
git add backends/step_progress.py tests/test_step_progress_cancel.py
git commit -m "feat(reap): cancel predicate at the denoise step boundary (STABL-jredufxb) — next: thread should_cancel through run_job"
```

---

### Task 2: Thread `should_cancel` from the job to the pipeline

**Files:**
- Modify: `backends/base.py:37`
- Modify: `backends/governor.py:150-156` (`GenerationJob.execute`)
- Modify: `backends/cuda_worker.py:1030`, `:1133`, `:1406`, `:1519`, `:1757`, `:1854`
- Test: `tests/test_reap_threading.py` (create)

**Interfaces:**
- Consumes: `inject_step_progress(..., should_cancel=None)` from Task 1.
- Produces:
  - `GenerationJob.execute(worker, progress=None, should_cancel=None) -> Any`
  - `PipelineWorker.run_job(job, progress=None, should_cancel=None) -> Tuple[bytes, int]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_reap_threading.py`:

```python
"""STABL-jredufxb: should_cancel reaches the worker from the job."""
from backends.governor import GenerationJob


class SpyWorker:
    def __init__(self):
        self.seen = None

    def run_job(self, job, progress=None, should_cancel=None):
        self.seen = {"progress": progress, "should_cancel": should_cancel}
        return b"PNG"


def _job():
    class Req:
        prompt = "x"
        num_inference_steps = 4
    return GenerationJob(req=Req(), resolution_epoch=1)


def test_execute_threads_should_cancel_to_run_job():
    worker = SpyWorker()
    predicate = lambda: False
    _job().execute(worker, progress=None, should_cancel=predicate)
    assert worker.seen["should_cancel"] is predicate


def test_execute_still_threads_progress():
    worker = SpyWorker()
    emitter = lambda s, t, stage: None
    _job().execute(worker, progress=emitter)
    assert worker.seen["progress"] is emitter
    assert worker.seen["should_cancel"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reap_threading.py -v`
Expected: FAIL — `TypeError: execute() got an unexpected keyword argument 'should_cancel'`

- [ ] **Step 3: Implement — `GenerationJob.execute`**

In `backends/governor.py`, replace `GenerationJob.execute`:

```python
    def execute(self, worker: Optional[PipelineWorker], progress=None,
                should_cancel=None) -> Any:
        """Execute generation job. `progress(step, total, stage)` is the sink's
        Progress emitter, threaded into run_job so the diffusion step callback can
        report (STABL-zueslhah); None when no consumer is attached.

        `should_cancel()` is the reap predicate (STABL-jredufxb), consulted at the
        same step boundary; None when the job is not cancellable."""
        if worker is None:
            raise RuntimeError("No worker available for generation")
        return worker.run_job(self, progress=progress, should_cancel=should_cancel)  # type: ignore[arg-type]
```

- [ ] **Step 4: Implement — the worker protocol**

In `backends/base.py`, change line 37:

```python
    def run_job(self, job: Any, progress=None, should_cancel=None) -> Tuple[bytes, int]:
```

- [ ] **Step 5: Implement — the three CUDA `run_job` methods**

In `backends/cuda_worker.py`, change each of the three signatures at `:1030`, `:1406`, `:1757` from:

```python
    def run_job(self, job, progress=None) -> tuple[bytes, int]:
```

to:

```python
    def run_job(self, job, progress=None, should_cancel=None) -> tuple[bytes, int]:
```

and each of the three `inject_step_progress` calls at `:1133`, `:1519`, `:1854` from:

```python
                    inject_step_progress(pipe, pipe_kwargs, progress, req.num_inference_steps)
```

to:

```python
                    inject_step_progress(pipe, pipe_kwargs, progress,
                                         req.num_inference_steps,
                                         should_cancel=should_cancel)
```

(Preserve each call site's existing indentation — `:1854` is one level shallower than the other two.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_reap_threading.py -v`
Expected: PASS (2 tests)

- [ ] **Step 7: Run the worker and governor suites for regressions**

Run: `python -m pytest tests/ -k "worker or governor" -q`
Expected: no new failures

- [ ] **Step 8: Commit**

```bash
git add backends/base.py backends/governor.py backends/cuda_worker.py tests/test_reap_threading.py
git commit -m "feat(reap): thread should_cancel job->worker->pipeline (STABL-jredufxb) — next: in-proc reap in the dispatch loop"
```

---

### Task 3: In-proc reap — dispatch wiring and pool trim

**Files:**
- Modify: `backends/governor.py:905-914` (in-proc execute call), `:987-992` (cancel terminal branch)
- Test: `tests/test_reap_inproc.py` (create)

**Interfaces:**
- Consumes: `GenerationJob.execute(..., should_cancel=...)` from Task 2.
- Produces: nothing new; behavior only. The in-proc dispatch path passes a predicate reading `JobRecord.cancel_requested`, and the existing cancel terminal branch additionally calls `self._dm.reclaim()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_reap_inproc.py`:

```python
"""STABL-jredufxb: the in-proc reap, end to end through the dispatch loop."""
import time

import pytest

from backends.backplane.frames import BackplaneErrorCode


class CancellableWorker:
    """run_job polls should_cancel like a denoise loop polls it at each step."""

    def __init__(self):
        self.steps_run = 0

    def run_job(self, job, progress=None, should_cancel=None):
        from concurrent.futures import CancelledError
        for _ in range(200):
            if should_cancel is not None and should_cancel():
                raise CancelledError("job cancelled at step boundary")
            self.steps_run += 1
            time.sleep(0.01)
        return b"PNG:finished"


def test_inproc_job_stops_when_cancel_requested(governor_with_worker):
    """The worker must observe the flag mid-run, not at a job boundary."""
    gov, worker = governor_with_worker(CancellableWorker())
    fut = gov.submit_job(_generation_job(gov))
    _wait_until(lambda: worker.steps_run > 2, timeout=5.0)

    job_id = _only_job_id(gov)
    gov.cancel_job(job_id)

    with pytest.raises(Exception):
        fut.result(timeout=5.0)
    assert worker.steps_run < 200, "worker ran to completion — it never saw the flag"


def test_inproc_reap_trims_the_pool(governor_with_worker, monkeypatch):
    gov, worker = governor_with_worker(CancellableWorker())
    calls = []
    monkeypatch.setattr(gov._dm, "reclaim", lambda: calls.append(1))

    fut = gov.submit_job(_generation_job(gov))
    _wait_until(lambda: worker.steps_run > 2, timeout=5.0)
    gov.cancel_job(_only_job_id(gov))
    with pytest.raises(Exception):
        fut.result(timeout=5.0)

    _wait_until(lambda: calls, timeout=5.0)
    assert calls, "a reaped job must trim the allocator pool"
```

Add these helpers at the top of the same file, below the imports:

```python
def _wait_until(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _only_job_id(gov):
    with gov._job_lock:
        ids = list(gov._job_records.keys())
    assert len(ids) == 1, f"expected exactly one job record, got {ids}"
    return ids[0]


def _generation_job(gov):
    from backends.governor import GenerationJob

    class Req:
        prompt = "a quiet harbour"
        num_inference_steps = 8
    snapshot = gov._active_snapshot
    epoch = snapshot.resolution_epoch if snapshot is not None else 0
    return GenerationJob(req=Req(), resolution_epoch=epoch)
```

**Fixture:** `governor_with_worker` does not exist yet. Add it to `tests/conftest.py`:

```python
@pytest.fixture
def governor_with_worker():
    """A Governor whose in-proc handle serves a caller-supplied fake worker.

    Yields a factory so the test picks the worker; tears the Governor down so the
    dispatch thread does not outlive the test (see the _freeze_dispatch note in
    test_governor.py — _stop.set() alone does NOT stop the loop)."""
    built = []

    def _build(worker):
        from backends.governor import Governor
        gov = Governor(worker_factory=lambda *a, **k: worker)
        gov.start()
        built.append(gov)
        return gov, worker

    yield _build
    for gov in built:
        try:
            gov.shutdown()
        except Exception:
            pass
```

> **If `Governor(worker_factory=...)` / `gov.start()` do not match the current constructor**, read `tests/test_governor.py`'s existing Governor construction and copy its exact fixture shape rather than inventing one. `shutdown()` begins with `q.join()`, so any job left queued must be drained first — `test_governor.py` has `_drain_queue` for this.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_reap_inproc.py -v`
Expected: FAIL — `test_inproc_job_stops_when_cancel_requested` fails on `worker.steps_run < 200` (the worker runs to completion), and `test_inproc_reap_trims_the_pool` fails on the empty `calls` list.

- [ ] **Step 3: Implement — pass the predicate**

In `backends/governor.py`, replace the in-proc execute call:

```python
                            _sink = job_record.sink if job_record is not None else None
                            _rec = job_record
                            result = job.execute(
                                self._handle.worker,
                                progress=(_sink.progress if _sink is not None else None),
                                # STABL-jredufxb: the reap predicate. Read LOCK-FREE by
                                # design — cancel_job writes the flag under _job_lock, but
                                # the worker must never acquire it (backplane
                                # Subscriber<->lock invariant). A bool read is atomic under
                                # the GIL and a one-step-late read is harmless.
                                should_cancel=(
                                    (lambda: _rec.cancel_requested) if _rec is not None
                                    else None
                                ),
                            )
```

- [ ] **Step 4: Implement — trim the pool on the cancel terminal**

In the dispatch loop's `except Exception` handler, in the `elif job_record.cancel_requested:` branch, add the reclaim after the state assignment:

```python
                        elif job_record.cancel_requested:
                            if sink is not None:
                                sink.error(BackplaneError(BackplaneErrorCode.CANCELLED, "cancelled"))
                            elif not job.fut.done():
                                job.fut.set_exception(CancelledError())
                            job_record.state = "cancelled"
                            # STABL-jredufxb: unwinding returned the intermediates to
                            # torch's caching allocator, not to the driver. Trim so the
                            # reaped bytes are visible as free VRAM again. (No-op for a
                            # subprocess consumer by design — see the spec.)
                            self._dm.reclaim()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_reap_inproc.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Run the governor suite for regressions**

Run: `python -m pytest tests/test_governor.py -q`
Expected: no new failures

- [ ] **Step 7: Commit**

```bash
git add backends/governor.py tests/test_reap_inproc.py tests/conftest.py
git commit -m "feat(reap): in-proc reap via lock-free cancel predicate + pool trim (STABL-jredufxb) — next: subprocess child control-pipe cancel"
```

---

### Task 4: Subprocess child — cancel over the control pipe

**Files:**
- Modify: `backends/worker_handle_subprocess.py` (`_serve_stats` → `_serve_control`, `_worker_main` job loop, module constants)
- Modify: `tests/_fault_worker.py` (add a cancellable worker)
- Test: `tests/test_reap_subprocess.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks (the child calls `worker.run_job` directly).
- Produces:
  - module constant `_CANCEL_JOB = b"\x00CANCELJOB"`
  - `_serve_control(control_conn, cancel_event) -> None`
  - `tests/_fault_worker.py::make_cancellable_worker(worker_id, resolved, binding)`

- [ ] **Step 1: Add the cancellable fake worker**

Append to `tests/_fault_worker.py`:

```python
class CancellableWorker:
    """Polls should_cancel the way a denoise loop polls it at each step, so a
    spawn-boundary test can prove the child stops MID-JOB rather than at a job
    boundary. Echoes how many steps it managed before the cancel landed."""

    def __init__(self, *args, **kwargs):
        pass

    def run_job(self, job, progress=None, should_cancel=None):
        import time
        from concurrent.futures import CancelledError

        for step in range(500):
            if should_cancel is not None and should_cancel():
                raise CancelledError("job cancelled at step boundary")
            time.sleep(0.01)
        return b"PNG:finished"


def make_cancellable_worker(worker_id, resolved, binding):
    return CancellableWorker()
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_reap_subprocess.py`:

```python
"""STABL-jredufxb: the subprocess reap, across a REAL spawn boundary.

Mocked transport is what produced STABL-spxwqlan — these must spawn.
"""
import time

import pytest

from backends.backplane.frames import BackplaneErrorCode
from backends.backplane.ipc import drain_to_subscriber
from backends.worker_handle_subprocess import SubprocessWorkerHandle


def _wait_until(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _job(prompt="x", job_id="j1"):
    from backends.governor import GenerationJob

    class Req:
        pass
    req = Req()
    req.prompt = prompt
    req.num_inference_steps = 4
    return GenerationJob(req=req, resolution_epoch=1, job_id=job_id)


class _Collector:
    """Minimal Subscriber that records the terminal."""

    def __init__(self):
        self.error_code = None
        self.completed = False
        self.result = None

    def on_subscribe(self, subscription):
        self.subscription = subscription

    def on_next(self, frame):
        self.result = frame

    def on_error(self, err):
        self.error_code = err.code

    def on_complete(self):
        self.completed = True


@pytest.mark.timeout(60)
def test_child_stops_mid_job_and_survives():
    h = SubprocessWorkerHandle("tests._fault_worker.make_cancellable_worker")
    h.start(None, None, None)
    pid_before = h._proc.pid
    try:
        publisher = h.submit(_job())
        collector = _Collector()
        time.sleep(0.5)              # let the child get well into the job
        h.cancel_current()
        publisher.subscribe(collector)

        assert collector.error_code == BackplaneErrorCode.CANCELLED, (
            "a bespoke exception type would arrive as GENERIC — this is the test "
            "that pins the concurrent.futures.CancelledError decision"
        )
        assert h._proc.pid == pid_before, "the child was respawned; the reap must not kill"
        assert h._proc.is_alive()
    finally:
        h.stop()


@pytest.mark.timeout(60)
def test_cancel_between_jobs_does_not_corrupt_the_job_stream():
    """The test that justifies the control pipe. On the DATA pipe this cancel
    would be handed to decode_job() as if it were a job frame."""
    h = SubprocessWorkerHandle("tests._fault_worker.make_payload_echo_worker")
    h.start(None, None, None)
    try:
        h.cancel_current()           # nothing is running; the child is in recv_bytes()
        time.sleep(0.2)

        publisher = h.submit(_job(prompt="hello", job_id="j2"))
        collector = _Collector()
        publisher.subscribe(collector)

        assert collector.error_code is None, "the stray cancel corrupted the stream"
        assert collector.completed
    finally:
        h.stop()


@pytest.mark.timeout(60)
def test_cancel_does_not_leak_into_the_next_job():
    """The child's event must be CLEARED at each job start, or the cancel that
    reaped job 1 silently kills job 2."""
    h = SubprocessWorkerHandle("tests._fault_worker.make_cancellable_worker")
    h.start(None, None, None)
    try:
        publisher = h.submit(_job(job_id="j1"))
        time.sleep(0.3)
        h.cancel_current()
        first = _Collector()
        publisher.subscribe(first)
        assert first.error_code == BackplaneErrorCode.CANCELLED

        h2 = SubprocessWorkerHandle("tests._fault_worker.make_payload_echo_worker")
        h2.start(None, None, None)
        try:
            pub2 = h2.submit(_job(prompt="second", job_id="j2"))
            second = _Collector()
            pub2.subscribe(second)
            assert second.error_code is None
        finally:
            h2.stop()
    finally:
        h.stop()
```

> **Two shapes to confirm against the existing suite before running these.** `h.start(None, None, None)` must match the current `start()` signature, and `_Collector`'s method names (`on_subscribe` / `on_next` / `on_error` / `on_complete`) must match the vendored `Subscriber` ABC in `backends/backplane/reactivestreams/`. `tests/test_subprocess_worker_handle.py` constructs this handle and subscribes to its publisher several times — copy its shapes rather than adjusting production code to fit the test.

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_reap_subprocess.py -v`
Expected: FAIL — `AttributeError: 'SubprocessWorkerHandle' object has no attribute 'cancel_current'`

- [ ] **Step 4: Implement — the control constant and the control server**

In `backends/worker_handle_subprocess.py`, add next to `_STATS`:

```python
_CANCEL_JOB = b"\x00CANCELJOB"
```

Rename `_serve_stats` to `_serve_control` and give it the cancel event:

```python
def _serve_control(control_conn, cancel_event):
    """Child-side control server: VRAM stats (STABL-xtkhoidu) and job cancel
    (STABL-jredufxb).

    Runs on its own daemon thread reading its OWN pipe. It cannot share the data
    pipe: `drain_to_subscriber` reads that concurrently while a job runs, and the
    child itself blocks on that pipe awaiting its next job — a cancel sent there
    would be decoded as a job frame.

    A separate thread is also what lets both verbs land DURING a generation — torch
    releases the GIL across CUDA calls, so the reply lands inside the fan-out
    budget instead of queueing behind the denoise.
    """
    import torch

    while True:
        try:
            msg = control_conn.recv_bytes()
            if msg == _CANCEL_JOB:
                cancel_event.set()          # no reply: cancel is fire-and-forget
                continue
            if msg != _STATS:
                continue
            control_conn.send_bytes(pickle.dumps({
                "pid": os.getpid(),
                "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
            }))
        except (EOFError, OSError):
            break                      # parent closed the pipe: the child is going away
        except Exception:              # noqa: BLE001 — never kill the child over control
            try:
                control_conn.send_bytes(pickle.dumps(None))
            except Exception:          # noqa: BLE001
                break
```

- [ ] **Step 5: Implement — the child's job loop**

In `_worker_main`, replace the control-thread start:

```python
    # Start the control server only after the worker is built: answering stats for a
    # half-constructed child would report numbers nobody can act on.
    cancel_event = threading.Event()
    if control_conn is not None:
        threading.Thread(target=_serve_control, args=(control_conn, cancel_event),
                         daemon=True).start()
```

and in the job loop, clear the event before the job and pass the predicate:

```python
        sink = IpcJobSink(conn, job_id=d.job_id)
        # STABL-jredufxb: clear BEFORE the job, or a cancel that reaped the previous
        # job silently kills this one.
        cancel_event.clear()
        try:
            result = worker.run_job(job, progress=sink.progress,
                                    should_cancel=cancel_event.is_set)
            sink.result(0, pickle.dumps(result))
            sink.complete()
        except Exception as e:   # noqa: BLE001 — rides the sink terminal
            from backends.backplane.frames import BackplaneError
            sink.error(BackplaneError.from_exc(e))
```

- [ ] **Step 6: Implement — the parent-side send**

Add to `SubprocessWorkerHandle`, directly below `request_stats`:

```python
    def cancel_current(self) -> bool:
        """Ask the child to abandon the job it is running (STABL-jredufxb).

        Fire-and-forget: there is no reply, and the terminal arrives on the DATA
        pipe as a CANCELLED error frame. Takes `_control_lock` because this pipe
        also carries the `_STATS` request/reply — an unserialized write is exactly
        the interleaving the lock exists to prevent.

        MUST be called OUTSIDE `_job_lock`: `_control_lock` can be held for up to
        `_STATS_REPLY_TIMEOUT_S` by an in-flight stats reply, and cancelling under
        `_job_lock` would let a /status fan-out stall the dispatch loop.
        """
        conn = self._control_conn
        if conn is None or self._proc is None or not self._proc.is_alive():
            return False
        with self._control_lock:
            try:
                conn.send_bytes(_CANCEL_JOB)
                return True
            except (EOFError, OSError):
                return False
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/test_reap_subprocess.py -v`
Expected: PASS (3 tests)

- [ ] **Step 8: Run the subprocess and attribution suites for regressions**

Run: `python -m pytest tests/test_subprocess_worker_handle.py tests/test_device_memory.py tests/test_worker_isolation.py -q`
Expected: no new failures. (The `_serve_stats` rename must not break the stats round-trip — `test_device_memory.py` and the concurrent-stats test in `test_subprocess_worker_handle.py` are the guards.)

- [ ] **Step 9: Commit**

```bash
git add backends/worker_handle_subprocess.py tests/_fault_worker.py tests/test_reap_subprocess.py
git commit -m "feat(reap): subprocess cancel over the control pipe (STABL-jredufxb) — next: wire cancel_job to the handle"
```

---

### Task 5: Wire `cancel_job` to the subprocess handle

**Files:**
- Modify: `backends/governor.py:690-700` (`cancel_job`)
- Test: `tests/test_reap_cancel_wiring.py` (create)

**Interfaces:**
- Consumes: `SubprocessWorkerHandle.cancel_current() -> bool` from Task 4.
- Produces: nothing new; `Governor.cancel_job(job_id)` now also signals the handle for a running job, after releasing `_job_lock`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_reap_cancel_wiring.py`:

```python
"""STABL-jredufxb: cancel_job reaches the handle, and does so outside _job_lock."""


class SpyHandle:
    """Records whether _job_lock was held at the moment cancel_current fired."""

    worker = None

    def __init__(self, gov_ref):
        self.gov_ref = gov_ref
        self.calls = 0
        self.lock_was_held = None

    def cancel_current(self):
        gov = self.gov_ref[0]
        acquired = gov._job_lock.acquire(blocking=False)
        self.lock_was_held = not acquired
        if acquired:
            gov._job_lock.release()
        self.calls += 1
        return True


def test_cancel_job_signals_a_running_job(governor_with_spy_handle):
    gov, handle, job_id = governor_with_spy_handle(state="running")
    assert gov.cancel_job(job_id) is True
    assert handle.calls == 1


def test_cancel_job_releases_job_lock_before_signalling(governor_with_spy_handle):
    """_control_lock can be held 0.25s by an in-flight stats reply; cancelling
    under _job_lock would stall the dispatch loop behind a /status fan-out."""
    gov, handle, job_id = governor_with_spy_handle(state="running")
    gov.cancel_job(job_id)
    assert handle.lock_was_held is False


def test_cancel_job_does_not_signal_a_queued_job(governor_with_spy_handle):
    """A queued job is taken off the queue outright; there is nothing running to
    cancel, and signalling would reap the job that IS running."""
    gov, handle, job_id = governor_with_spy_handle(state="queued")
    gov.cancel_job(job_id)
    assert handle.calls == 0


def test_cancel_job_tolerates_a_handle_without_cancel_current(governor_with_spy_handle):
    """InProcessWorkerHandle has no cancel_current — the in-proc reap goes through
    the predicate instead, and cancel_job must not raise."""
    gov, handle, job_id = governor_with_spy_handle(state="running", handle_can_cancel=False)
    assert gov.cancel_job(job_id) is True
```

Add the fixture to `tests/conftest.py`:

```python
@pytest.fixture
def governor_with_spy_handle():
    """A Governor with one job record in a chosen state and a handle that records
    cancel_current calls. Built without starting the dispatch thread — this
    fixture is about cancel_job's own logic, not about running work."""
    def _build(state="running", handle_can_cancel=True):
        from tests.test_reap_cancel_wiring import SpyHandle
        from backends.governor import Governor, GenerationJob, JobRecord

        gov_ref = []
        gov = Governor(worker_factory=lambda *a, **k: None)
        gov_ref.append(gov)
        handle = SpyHandle(gov_ref)
        if not handle_can_cancel:
            del type(handle).cancel_current
        gov._handle = handle

        class Req:
            prompt = "x"
            num_inference_steps = 4
        job = GenerationJob(req=Req(), resolution_epoch=0)
        record = JobRecord(job=job)
        record.state = state
        with gov._job_lock:
            gov._job_records[job.job_id] = record
        return gov, handle, job.job_id

    return _build
```

> **`JobRecord(job=job)` and `Governor(worker_factory=...)` must match the current constructors.** Read `backends/governor.py`'s `JobRecord` dataclass and `tests/test_governor.py`'s Governor construction and adjust the fixture rather than changing production code to fit the test. `del type(handle).cancel_current` mutates the class — if that proves awkward, define a second spy class without the method instead.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_reap_cancel_wiring.py -v`
Expected: FAIL — `handle.calls == 0` (cancel_job never signals the handle)

- [ ] **Step 3: Implement**

In `backends/governor.py`, replace `cancel_job`:

```python
    def cancel_job(self, job_id: str) -> bool:
        """Request cancellation. For a QUEUED job this takes it off the queue, so the
        work is never done. For a RUNNING job it flips the flag the in-proc reap
        predicate reads, and signals a subprocess child over the control pipe
        (STABL-jredufxb) — the child cannot see `cancel_requested`."""
        signal_handle = False
        with self._job_lock:
            record = self._job_records.get(job_id)
            if record is None or record.job.fut.done():
                return False
            record.cancel_requested = True
            if record.state == "queued" and record.job.fut.cancel():
                record.state = "cancelled"
                return True
            record.state = "running"
            signal_handle = True

        # OUTSIDE _job_lock, deliberately: cancel_current takes _control_lock, which
        # an in-flight stats reply can hold for _STATS_REPLY_TIMEOUT_S. Signalling
        # under _job_lock would let a /status fan-out stall the dispatch loop.
        if signal_handle:
            cancel_current = getattr(self._handle, "cancel_current", None)
            if callable(cancel_current):
                try:
                    cancel_current()
                except Exception:  # noqa: BLE001 — a failed signal must not fail the cancel
                    logger.warning("[Governor] cancel_current failed", exc_info=True)
        return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_reap_cancel_wiring.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full unit suite**

Run: `python -m pytest tests/ -m "not cuda" -q`
Expected: no new failures against the pre-task baseline. Record the counts.

- [ ] **Step 6: Commit**

```bash
git add backends/governor.py tests/test_reap_cancel_wiring.py tests/conftest.py
git commit -m "feat(reap): cancel_job signals the subprocess child outside _job_lock (STABL-jredufxb) — next: live acceptance on enigma"
```

---

### Task 6: Live acceptance on enigma

**Files:**
- Create: `spikes/reap_acceptance.py`
- Test: none — this is hardware verification, not a unit test.

**Interfaces:**
- Consumes: everything from Tasks 1-5, through the production `get_worker_pool()` path.
- Produces: a PASS/FAIL verdict block, in the style of `spikes/facet3_oom_acceptance.py`.

- [ ] **Step 1: Write the acceptance spike**

Create `spikes/reap_acceptance.py`:

```python
"""STABL-jredufxb — live reap acceptance.

Proves on real hardware that a generation exceeding its execution budget STOPS,
rather than running to completion with its result discarded.

The proof is WALL TIME plus the CHILD PID. If the call returns in roughly the
budget rather than the full generation time, the work stopped; if the pid is
unchanged, it stopped COOPERATIVELY rather than by kill+respawn — which is the
whole design.

Run inside the CUDA container, with the production models mounted:

    TEST_MODELS_HOST_PATH=/media/cold1/ComfyUI/models \
    docker compose -f docker-compose.test.yml run --rm \
      -e WORKER_ISOLATION=subprocess -e DEFAULT_TIMEOUT=5 test-cuda \
      python spikes/reap_acceptance.py lcm-general

Re-run with -e WORKER_ISOLATION=inproc to cover the other path; the same verdict
block applies except that the pid check is skipped.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("BACKEND", "cuda")


def child_pid(pool):
    proc = getattr(pool._governor._handle, "_proc", None)
    return getattr(proc, "pid", None)


def main() -> int:
    mode_name = sys.argv[1] if len(sys.argv) > 1 else "lcm-general"
    budget = float(os.environ.get("DEFAULT_TIMEOUT", "5"))

    from backends.governor import GenerationJob
    from backends.worker_pool import get_worker_pool
    from server.lcm_sr_server import GenerateRequest

    pool = get_worker_pool()
    handle = pool._governor._handle
    isolation = type(handle).__name__
    print(f"handle = {isolation}", flush=True)
    print(f"execution budget = {budget}s", flush=True)

    pool.switch_mode(mode_name, force=True).result(timeout=900.0)
    pid_before = child_pid(pool)
    print(f"child pid after load: {pid_before}", flush=True)

    # Enough steps that an un-reaped run takes MUCH longer than the budget —
    # otherwise a job that finished normally is indistinguishable from a reap.
    authority = pool.admit_generation(mode_name)
    req = GenerateRequest(prompt="a quiet harbour at dawn", size="1024x1024",
                          num_inference_steps=150, guidance_scale=2.0)
    job = GenerationJob(req=req, controlnet_bindings=[],
                        resolution_epoch=authority.resolution_epoch)

    t0 = time.monotonic()
    fut = pool.submit_job(job)
    try:
        pool._governor.wait_for_result(fut)
        elapsed = time.monotonic() - t0
        print(f"\nFAIL: the job COMPLETED in {elapsed:.1f}s — it was never reaped.",
              flush=True)
        print("Lower DEFAULT_TIMEOUT or raise num_inference_steps so the "
              "generation is comfortably longer than the budget.", flush=True)
        return 2
    except TimeoutError as exc:
        elapsed = time.monotonic() - t0
        print(f"timed out after {elapsed:.1f}s: {str(exc)[:160]}", flush=True)

    # The reap lands one step after the budget; give the terminal a moment to settle.
    time.sleep(3.0)
    pid_after = child_pid(pool)

    # A second job must succeed on the SAME worker — no reload, no respawn.
    t1 = time.monotonic()
    authority2 = pool.admit_generation(mode_name)
    req2 = GenerateRequest(prompt="a still lake", size="512x512",
                           num_inference_steps=8, guidance_scale=2.0)
    job2 = GenerationJob(req=req2, controlnet_bindings=[],
                         resolution_epoch=authority2.resolution_epoch)
    png, seed = pool.submit_job(job2).result(timeout=900.0)
    next_job_s = time.monotonic() - t1

    stopped_early = elapsed < budget * 4
    same_process = pid_before is None or pid_after == pid_before

    print("\n==================== VERDICT ====================", flush=True)
    print(f"isolation                  : {isolation}", flush=True)
    print(f"wall time to timeout       : {elapsed:.1f}s (budget {budget:g}s)", flush=True)
    print(f"stopped near the budget    : {stopped_early}   <-- the reap", flush=True)
    print(f"child pid {pid_before} -> {pid_after}", flush=True)
    print(f"cooperative, not killed    : {same_process}", flush=True)
    print(f"next job OK                : True ({len(png)} bytes, seed={seed}, "
          f"{next_job_s:.1f}s)", flush=True)

    if not stopped_early:
        print("\nFAIL: the timeout returned but the wall clock says the generation "
              "kept running. The predicate is not reaching the denoise loop.",
              flush=True)
        return 3
    if not same_process:
        print("\nFAIL: the child was respawned. The reap is supposed to be "
              "cooperative — check that the terminal is CANCELLED, not OOM or a "
              "frameless death.", flush=True)
        return 4
    print("\nPASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Sync the branch to enigma**

```bash
scripts/remote-worktree.sh --host enigma.lan --repo-path /home/hdd/workspace/Stability-Toys
```

- [ ] **Step 3: Run the subprocess acceptance**

```bash
ssh enigma.lan 'cd /home/hdd/workspace/Stability-Toys/.worktrees/fix/jredufxb-reap && \
  TEST_MODELS_HOST_PATH=/media/cold1/ComfyUI/models \
  docker compose -f docker-compose.test.yml run --rm \
    -e WORKER_ISOLATION=subprocess -e DEFAULT_TIMEOUT=5 test-cuda \
    python spikes/reap_acceptance.py lcm-general'
```

Expected: `PASS`, with the pid unchanged across the reap.

- [ ] **Step 4: Run the in-proc acceptance**

```bash
ssh enigma.lan 'cd /home/hdd/workspace/Stability-Toys/.worktrees/fix/jredufxb-reap && \
  TEST_MODELS_HOST_PATH=/media/cold1/ComfyUI/models \
  docker compose -f docker-compose.test.yml run --rm \
    -e WORKER_ISOLATION=inproc -e DEFAULT_TIMEOUT=5 test-cuda \
    python spikes/reap_acceptance.py lcm-general'
```

Expected: `PASS`. `child pid None -> None` is correct here — the pid check is skipped for the in-proc handle.

- [ ] **Step 5: Commit**

```bash
git add spikes/reap_acceptance.py
git commit -m "chore(spike): live reap acceptance, both isolation modes (STABL-jredufxb)"
```

- [ ] **Step 6: Close out**

- Update `project-forward-notes.md`: move the reap from "open" to landed, with the measured wall-time numbers from Step 3 and Step 4.
- `fp issue assign STABL-jredufxb --rev <sha>` and an FP comment covering what landed, what the coverage bound is, and what was left uncovered (no kill fallback; no reap without a waiter).
- `drift check` before declaring done.

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| Mechanism — predicate outside `_emit` | 1 |
| Coverage bound (between steps only) | documented in 1; asserted by the loop-position assertions |
| Exception type `concurrent.futures.CancelledError` | 1 (raise), 4 (terminal is `CANCELLED`, the pinning test) |
| No `"out of memory"` in the message | 1 |
| In-proc wiring, lock-free read | 3 |
| No change to the in-proc terminal path | 3 (verified by reusing the existing branch) |
| Subprocess wiring — control pipe, not data pipe | 4 (and the between-jobs test) |
| `IpcJobSink.cancelled` stays unused | 4 (nothing touches it) |
| Event cleared per job | 4 (`test_cancel_does_not_leak_into_the_next_job`) |
| `_control_lock` on cancel sends | 4 |
| Send outside `_job_lock` | 5 (`test_cancel_job_releases_job_lock_before_signalling`) |
| Recovery not triggered | 4 (`pid == pid_before`), 6 (live) |
| In-proc pool trim via `DeviceMemory.reclaim()` | 3 |
| Subprocess returns bytes only to the child pool | documented; no task — it is a stated limitation, not work |
| Deadline ownership already resolved | no task — nothing to change |

**Type consistency:** `should_cancel` is the parameter name in `inject_step_progress`, `GenerationJob.execute`, `PipelineWorker.run_job`, and all three `cuda_worker.run_job` methods. `cancel_current()` is the handle method name in Task 4 (definition), Task 5 (call), and both test files. `_CANCEL_JOB` is the constant name in the child server and the parent send.

**Known fixture risk:** Tasks 3 and 5 introduce `conftest.py` fixtures whose constructor calls (`Governor(...)`, `JobRecord(...)`, `SubprocessWorkerHandle.start(...)`) are written from the current source but not executed. Each is flagged inline with an instruction to copy the shape from the existing test file rather than to change production code to fit the fixture.
