# Timeout semantics: bound execution, not queue wait — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A generate queued behind a multi-minute model load is no longer timed out for waiting, while a generate whose *execution* runs long still fails promptly.

**Architecture:** `JobRecord` gains an `executing_since` timestamp stamped at the true execution point. A Governor-owned waiter polls the future and applies the admission budget before that stamp, the execution budget after it. Both transports call the same waiter.

**Tech Stack:** Python 3.12, `concurrent.futures`, pytest.

**Spec:** `docs/superpowers/specs/2026-07-31-execution-timeout-semantics-design.md`

## Global Constraints

- `DEFAULT_TIMEOUT` default stays **`120`**. Its meaning changes to execution-only; its value does not change.
- `ADMISSION_TIMEOUT_S` default **`900`**.
- `JobRecord.state` transitions and `cancel_job` semantics are **not** modified. The new timestamp is additive.
- The waiter raises `concurrent.futures.TimeoutError` — the type callers already handle.
- Record lookup is by **future identity**, never by job id (the HTTP path has no job id).
- TDD: every task writes a failing test first and runs it to confirm RED.
- Commits carry the FP id, what, and next; `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: `executing_since` stamped after the reload and the barrier

**Files:**
- Modify: `backends/governor.py` (`JobRecord`, dispatch loop ~`:766`)
- Test: `tests/test_governor.py`

**Interfaces:**
- Produces: `JobRecord.executing_since: Optional[float]` — `None` until the job truly begins executing, then `time.monotonic()`.

- [ ] **Step 1: Write the failing test**

```python
def test_executing_since_is_none_until_the_job_actually_executes():
    """The execution clock must not start while the job is queued.

    Guards the whole point of STABL-atzqpcte: state == "running" is set at
    governor.py:724, BEFORE the demand reload and the stale-epoch barrier, so it
    is the wrong signal. executing_since must still be None at that moment.
    """
    from backends.governor import JobRecord
    import dataclasses
    fields = {f.name: f for f in dataclasses.fields(JobRecord)}
    assert "executing_since" in fields, "JobRecord has no executing_since"
    assert fields["executing_since"].default is None
```

- [ ] **Step 2: Run it to confirm RED**

Run: `python -m pytest tests/test_governor.py -k executing_since -q`
Expected: FAIL — `JobRecord has no executing_since`.

- [ ] **Step 3: Add the field**

In `backends/governor.py`, on `JobRecord`:

```python
    # STABL-atzqpcte: monotonic timestamp of TRUE execution start — set after the
    # demand reload and after the stale-epoch barrier, so neither is charged to the
    # execution budget. Deliberately NOT `state == "running"`, which is set earlier
    # (:724) and whose transition cancel_job depends on.
    executing_since: Optional[float] = None
```

- [ ] **Step 4: Stamp it at the true execution point**

Immediately before `if isinstance(job, GenerationJob):` in the dispatch loop:

```python
                    if job_record is not None:
                        job_record.executing_since = time.monotonic()
```

- [ ] **Step 5: Write the ordering test**

```python
def test_demand_reload_is_not_charged_to_the_execution_budget(monkeypatch):
    """A demand reload runs BEFORE executing_since is stamped, so a slow reload
    cannot consume the execution budget. Asserted by observing the record from
    inside the reload itself."""
    # See Task 1 notes: drive a governor whose _reload_from_snapshot records
    # whether executing_since was still None when it ran.
```

- [ ] **Step 6: Run the full governor suite**

Run: `python -m pytest tests/test_governor.py -q`
Expected: all pass — `state`/cancel behaviour is untouched.

- [ ] **Step 7: Commit**

---

### Task 2: `wait_for_result` — the two-budget waiter

**Files:**
- Modify: `backends/governor.py`, `backends/worker_pool.py`
- Test: `tests/test_governor.py`

**Interfaces:**
- Consumes: `JobRecord.executing_since` (Task 1).
- Produces: `Governor.wait_for_result(fut, *, admission_timeout_s=None, execution_timeout_s=None, poll_interval_s=0.25)`; `WorkerPool.wait_for_result(...)` passthrough; module constants `DEFAULT_EXECUTION_TIMEOUT_S`, `DEFAULT_ADMISSION_TIMEOUT_S`.

- [ ] **Step 1: Write the four failing tests** — headline first: queued past the execution budget is NOT timed out; executing past it IS and sets `cancel_requested`; queued past the admission budget IS, naming admission; a normal result returns unchanged.

- [ ] **Step 2: Run to confirm RED** (`AttributeError: wait_for_result`).

- [ ] **Step 3: Implement**

```python
DEFAULT_EXECUTION_TIMEOUT_S: float = float(os.environ.get("DEFAULT_TIMEOUT", "120"))
DEFAULT_ADMISSION_TIMEOUT_S: float = float(os.environ.get("ADMISSION_TIMEOUT_S", "900"))
```

```python
    def _record_for_future(self, fut) -> Optional[JobRecord]:
        with self._job_lock:
            for record in self._job_records.values():
                if record.job.fut is fut:
                    return record
        return None
```

`wait_for_result` polls `fut.result(timeout=poll_interval_s)`, and on each miss reads the record: while `executing_since is None` it judges against the admission deadline; afterwards against `executing_since + execution_timeout_s`. A vanished record counts as executing from first observation. On expiry it calls `cancel_job(record.job_id)` and raises `concurrent.futures.TimeoutError` naming the budget and the elapsed time.

- [ ] **Step 4: Run to GREEN, then the whole governor + worker_pool suite.**

- [ ] **Step 5: Commit**

---

### Task 3: Both transports use the waiter

**Files:**
- Modify: `server/ws_routes.py:540-544`, `server/lcm_sr_server.py:350,666`
- Test: `tests/test_ws_routes.py`

**Interfaces:**
- Consumes: `WorkerPool.wait_for_result` (Task 2).

- [ ] **Step 1: Write the failing test** — assert neither wait site reads `DEFAULT_TIMEOUT` directly any more, and that the WS path resolves through `wait_for_result`. The HTTP assertion matters most: an id-based design would have fixed WS only, and this test is what catches that class of regression.

- [ ] **Step 2: Run to confirm RED.**

- [ ] **Step 3: Replace both wait sites** with `state.worker_pool.wait_for_result(fut)` / `pool.wait_for_result(fut)`, deleting the local `DEFAULT_TIMEOUT` reads.

- [ ] **Step 4: Run to GREEN**, then `tests/test_ws_routes.py` and the unit suite.

- [ ] **Step 5: Update `server/lcm_sr_server.py`'s env docstring** (`:25`) so `DEFAULT_TIMEOUT` is documented as the execution budget and `ADMISSION_TIMEOUT_S` appears.

- [ ] **Step 6: Commit**

---

## Self-review

- **Spec coverage:** two budgets (Task 2), the `executing_since` clock start (Task 1), both transports (Task 3), cancel-on-timeout (Task 2), unchanged `DEFAULT_TIMEOUT` default (Global Constraints). Spec tests 1-4 → Task 2; test 5 → Task 1 Step 5; test 6 → Task 3 Step 1.
- **Out of scope, deliberately:** the reap (`STABL-jredufxb`), a `running` backplane frame, `SR_REQUEST_TIMEOUT`.
