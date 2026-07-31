# Bounded subprocess startup handshake — design

**Issue:** STABL-wotsqcjb (child of STABL-rgvxuedo, umbrella STABL-nvmieaxh)
**Date:** 2026-07-31
**Status:** approved (A+B, 2026-07-31)

## Problem

`SubprocessWorkerHandle.start()` blocks on an unbounded `recv_bytes()` waiting for the
child's `_READY` byte string. The child is `daemon=True` and sends `_READY` only after
importing its worker module, rebuilding the resolution, constructing the worker, and
configuring conditioning. Any failure before that line kills the child silently; nobody
closes the pipe in a way that raises `EOFError` parent-side, so the parent blocks
forever.

Note the asymmetry with the job path, which already has an EOF guard for frameless
death (backplane Task 4). Only startup is unguarded.

This defect is a diagnosis-cost multiplier, not a correctness bug in its own right. It
was found (2026-07-31) because a one-line wire-form mistake in the child presented as a
300s pytest timeout instead of a traceback.

## Approach: A + B

Two independent guards, because neither covers the other's cases.

**A — parent-side bounded wait.** Replace the blocking `recv_bytes()` with a
`poll(interval)` loop that also checks `self._proc.is_alive()` each iteration. Covers
every death mode, including `SIGKILL` and the kernel OOM-killer.

**B — child-side failure frame.** The child wraps its startup in `try/except
BaseException`, sends `_FAILED + traceback` on the pipe, then re-raises. Gives the
parent the child's actual traceback instead of an exit code.

B alone is unsound: a hard kill leaves nothing to send the frame. A alone reports only
"the child exited with code N", which is exactly the diagnosis cost this issue exists to
remove. A is the safety net; B is the diagnostics.

The order matters in the loop: `poll()` is checked before `is_alive()`, so a child that
writes `_FAILED` and immediately exits still has its frame read — buffered data stays
readable after the writer dies.

## Decisions

| Decision | Value | Why |
|---|---|---|
| Timeout | `WORKER_START_TIMEOUT_S`, default `300.0` | Module constant read via `os.environ.get`, matching `DEFAULT_QUEUE_TIMEOUT_S` (governor.py:43). Generous because the spawn child imports torch + diffusers cold before `_READY`. |
| Constructor override | `SubprocessWorkerHandle(ref, start_timeout_s=None)` | Tests inject a sub-second timeout without touching process-global env. `None` means "use the module constant", so the env var stays the production lever. |
| Poll interval | `0.1s` | Bounds the post-death detection lag. Not load-bearing anywhere. |
| On timeout | `self.stop()` then raise | Never leave an orphan child holding a CUDA context. |
| On any failure | `_state = "dead"` (via `stop()`) | `health()` must not report `starting` for a handle that will never be ready. |
| Error type | `WorkerStartError(RuntimeError)` | Subclass, so `governor.py:447`'s `except Exception` catches it unchanged and still runs `_drop_reservation(reservation, dead=True)`. No Governor change. |
| Traceback cap | 8 KiB, `decode(errors="replace")` | A truncated multibyte sequence must not turn a diagnostic into a `UnicodeDecodeError`. |

## Non-goals

- Retrying a failed start. The Governor already owns recovery policy
  (`_reload_from_snapshot`); adding a second retry loop here would fight it.
- Heartbeats during startup. `SubprocessLiveness` is still wired with
  `stale_after_s=float("inf")` (recon #5B); the periodic-heartbeat follow-on is separate.
- The semaphore leak (STABL-nstyyrhh). Sibling issue, different mechanism.

## Tests

All three run on macOS with no GPU, using `tests/_fault_worker.py` factories imported by
dotted ref in the spawn child.

1. **Ordinary startup exception** — a factory that raises. `start()` raises
   `WorkerStartError`; the message contains the child's exception text. Proves B.
2. **Hard kill before READY** — a factory that `SIGKILL`s its own process. No `_FAILED`
   frame is possible; `start()` still raises, and the message names the exit code.
   Proves A's liveness check.
3. **Hung but alive** — a factory that sleeps far longer than the injected timeout.
   `start()` raises within the timeout, `health().state == "dead"`, and the child is no
   longer alive. Proves A's deadline and the no-orphan rule.

Each test drives `start()` on a thread with a bounded `join`, the pattern established at
`tests/test_subprocess_worker_handle.py:216-224`, so a regression FAILS rather than
hanging the suite for the 300s pytest timeout.

The existing four spawn-boundary tests are the regression gate: the happy path must not
change.
