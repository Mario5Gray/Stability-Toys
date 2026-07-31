# Timeout semantics: bound execution, not queue wait — design

**Issue:** STABL-atzqpcte (child of STABL-nvmieaxh)
**Date:** 2026-07-31
**Status:** approved (approach A, bounded admission, cancel-on-timeout — 2026-07-31)

## Problem

`DEFAULT_TIMEOUT` is applied as `fut.result(timeout=120)` on a generate's future, at
`server/ws_routes.py:540` and `server/lcm_sr_server.py:350`. The future does not resolve
until everything ahead of it in the Governor queue has run. For a bundled
`st gen --mode X` that is a `ModeSwitchJob` (`_load_mode`, tens of seconds to minutes for
a cold checkpoint) and only then the generation.

So a timeout written to bound GENERATION also bounds QUEUE WAIT and MODEL LOAD — work
whose duration has nothing to do with the request being timed. Confirmed in the field on
enigma (2026-07-30): the first inline `--mode` generate against HunyuanDiT timed out during
the load. `DEFAULT_TIMEOUT=600` cleared it.

Raising the number is the wrong trade in the other direction: a genuinely hung generation
then hangs for ten minutes. Worse, per the umbrella cross-link, an abandoned job keeps
running and holds VRAM the whole time — so a bigger number makes the VRAM-pressure problem
worse while fixing the admission symptom.

## Approach: two budgets, split by observed execution start

One clock becomes two:

| budget | env | default | bounds |
|---|---|---|---|
| execution | `DEFAULT_TIMEOUT` | `120` (unchanged) | the generation itself |
| admission | `ADMISSION_TIMEOUT_S` | `900` | queue wait + everything ahead of it |

The waiter polls the future in short slices. While the job has not begun executing it is
judged against the admission budget; once execution starts, against the execution budget,
measured from the moment execution actually began.

`DEFAULT_TIMEOUT` keeps its name, its value, and finally acquires the meaning its name
always claimed. No deployment that is working today changes behaviour, except that jobs
which were failing for waiting now succeed.

Bounded rather than unbounded admission: a job wedged behind a hung `ModeSwitchJob` must
still fail eventually instead of pinning a WebSocket connection forever.

## Where the clock starts — and why not `state == "running"`

`JobRecord.state` is set to `"running"` at `governor.py:724`, which is BEFORE the demand
reload at `:727-731` and before the stale-epoch barrier at `:738-764`. Starting the
execution clock there would put a full model reload inside the execution budget — the same
defect this issue exists to remove, in miniature.

The marker is NOT moved. `cancel_job` (`governor.py:573-583`) branches on
`state == "queued"` to cancel the future outright; moving the transition later would widen
the window in which a cancel cancels a future that the dispatch loop is about to fulfil,
trading a timeout bug for a cancel bug.

Instead `JobRecord` gains:

```python
executing_since: Optional[float] = None      # time.monotonic() at true execution start
```

set immediately before the generation dispatch (`governor.py:766`), after the reload and
after the barrier. `state` and cancellation are untouched.

## The waiter

On the Governor, since it owns job state:

```python
def wait_for_result(self, fut, *, admission_timeout_s=None, execution_timeout_s=None,
                    poll_interval_s=0.25): ...
```

`None` means the module default read from env, so both transports call
`pool.wait_for_result(fut)` and neither carries policy.

The record is found by **future identity** (`record.job.fut is fut`). This is deliberate:
`runtime.submit_generate()` on the HTTP path returns only a future and never exposes the
job id, so an id-based API would fix one transport and not the other. The scan is over
live job records only.

Raises `concurrent.futures.TimeoutError` on either budget — the same type callers already
handle, so the blast radius is the message, not the control flow. The message names which
budget expired and how long the job actually waited.

### Record disappears while the future is pending

`_finalize_job_record` pops the record on completion, so a lookup can return `None` while
the future is briefly unresolved. Treated as executing, with the clock starting when the
waiter first observed the absence. Never treated as still-queued — that would grant the
generous admission budget to a job that has already run.

## Cancel on timeout

On either timeout the waiter calls `cancel_job(job_id)` before raising. If the job is still
queued this cancels it outright — real work avoided. If it is executing, this sets
`cancel_requested`, and the existing post-execute check at `governor.py:772` discards the
result.

**It does not stop a running generation.** `cancel_requested` is checked only at job
boundaries (`:718`, `:772`, `:841`); `run_job` never checks it and the CUDA worker has no
cancel path. A timed-out generation runs to completion and holds VRAM until it does. This
is stated plainly rather than implied away — the honest reap needs `handle.stop()`, which
facet-3 now makes possible, and is deferred (see below).

## Non-goals

- **Reaping the abandoned job** — deferred to its own issue. Approach A stops *false*
  timeouts; the reap stops the *abandonment* that follows a true one. Different problems.
  Note the umbrella comment cites `STABL-qvmdayhb` for this and that ID does not resolve,
  so the reap is currently tracked nowhere.
- **A `running` backplane frame** (the issue's option 2). It would let the client display
  the load, which is the other half of the operator complaint, but WS holds only a future
  today and never subscribes to the stream. Deferred to the Reactor work rather than
  building plumbing on its eve.
- `SR_REQUEST_TIMEOUT` — a second ceiling of the same shape, on a path with no queue
  behind it. Out of scope.

## Tests

Unit, deterministic, no worker or GPU — the waiter is driven against hand-built records and
futures with sub-second budgets.

1. **The headline:** a job still queued past the execution budget, inside the admission
   budget, is NOT timed out.
2. A job executing past the execution budget IS timed out, and `cancel_requested` is set.
3. A job queued past the admission budget is timed out, with a message naming admission.
4. A result arriving normally is returned unchanged.
5. The execution clock excludes the demand reload: `executing_since` is `None` throughout a
   reload and set only after it, asserted at the dispatch loop.
6. Both transports resolve through the same helper — asserted by call site, so the HTTP
   path cannot silently keep the old semantics.
