"""STABL-jredufxb — live reap acceptance.

Proves on real hardware that a generation exceeding its execution budget STOPS,
rather than running to completion with its result discarded.

THREE signals, all required. Wall time alone cannot tell a reap from a job that
merely finished quickly, and it cannot tell a cooperative stop from a
kill+respawn — which would still "stop" the job while destroying the loaded
model, the outcome this design exists to avoid:

  1. wall time      — the wait returns near the budget, not near the full run
  2. terminal       — the job's future carries CancelledError, not a timeout or
                      a generic failure (the exception-type decision, live)
  3. child pid      — unchanged across the reap, so the child was NOT respawned
                      (subprocess only; the in-proc handle has no pid)

Plus: the next job must succeed on the same worker, which is what "no reload"
means in practice.

Run inside the CUDA container, with the production models mounted:

    TEST_MODELS_HOST_PATH=/media/cold1/ComfyUI/models \
    docker compose -f docker-compose.test.yml run --rm \
      -e WORKER_ISOLATION=subprocess -e DEFAULT_TIMEOUT=5 test-cuda \
      python spikes/reap_acceptance.py lcm-general

Re-run with -e WORKER_ISOLATION=inproc for the other path; the pid check is
skipped there and says so.

NOT a pytest test on purpose: it must own its process (STABL-sgdavnvz — a shared
pytest session can leave stub modules resident) and it needs a real model
directory.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("BACKEND", "cuda")

# GenerateRequest caps num_inference_steps at 50 (pydantic le=50), so the long
# job cannot simply be made arbitrarily long. The wall-time signal is therefore
# calibrated against a MEASURED un-reaped run rather than against the budget: on
# a fast LCM model a full 50-step run can finish inside `budget * 4` and would
# read as a pass with no reap at all.
LONG_STEPS = int(os.environ.get("REAP_LONG_STEPS", "50"))
LONG_SIZE = os.environ.get("REAP_LONG_SIZE", "1024x1024")


def child_pid(pool):
    proc = getattr(getattr(pool._governor, "_handle", None), "_proc", None)
    return getattr(proc, "pid", None)


def terminal_exception(fut, timeout_s: float = 30.0):
    """The reap's terminal rides the FUTURE, not the waiter: wait_for_result
    raises TimeoutError to the caller the moment the budget expires, while the
    job's own terminal arrives afterwards, once the worker unwinds."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if fut.done():
            try:
                fut.result(timeout=0)
                return None                     # completed — no terminal exception
            except Exception as exc:            # noqa: BLE001 — the terminal is the point
                return exc
        time.sleep(0.1)
    return TimeoutError("future never reached a terminal")


def main() -> int:
    mode_name = sys.argv[1] if len(sys.argv) > 1 else "lcm-general"
    budget = float(os.environ.get("DEFAULT_TIMEOUT", "5"))

    from backends.governor import GenerationJob
    from backends.worker_pool import get_worker_pool
    from server.lcm_sr_server import GenerateRequest

    pool = get_worker_pool()
    handle = pool._governor._handle
    isolation = type(handle).__name__
    subprocess_mode = child_pid(pool) is not None or "Subprocess" in isolation
    print(f"handle = {isolation}", flush=True)
    print(f"execution budget = {budget:g}s, long job = {LONG_STEPS} steps", flush=True)

    print(f"\n--- loading mode {mode_name!r} ---", flush=True)
    pool.switch_mode(mode_name, force=True).result(timeout=900.0)
    pid_before = child_pid(pool)
    print(f"child pid after load: {pid_before}", flush=True)

    def long_job():
        authority = pool.admit_generation(mode_name)
        req = GenerateRequest(prompt="a quiet harbour at dawn", size=LONG_SIZE,
                              num_inference_steps=LONG_STEPS, guidance_scale=2.0)
        return GenerationJob(req=req, controlnet_bindings=[],
                             resolution_epoch=authority.resolution_epoch)

    # --- baseline: how long does this job take when NOBODY reaps it? -----------
    # Self-calibration, not decoration. num_inference_steps is capped at 50, so on
    # a fast model the full run can finish inside any fixed multiple of the budget
    # — and a "stopped early" verdict measured against the budget alone would then
    # be true of a job that was never reaped.
    print("\n--- baseline: the same job, un-reaped ---", flush=True)
    tb = time.monotonic()
    pool.wait_for_result(pool.submit_job(long_job()), execution_timeout_s=600.0)
    full_run_s = time.monotonic() - tb
    print(f"un-reaped run: {full_run_s:.1f}s", flush=True)
    if full_run_s < budget * 2:
        print(f"\nFAIL: the un-reaped run ({full_run_s:.1f}s) is not comfortably "
              f"longer than the {budget:g}s budget, so a reap cannot be "
              f"distinguished from normal completion.", flush=True)
        print("Lower DEFAULT_TIMEOUT, or raise REAP_LONG_SIZE (steps are capped "
              "at 50).", flush=True)
        return 6

    print("\n--- a job that must exceed its execution budget ---", flush=True)
    job = long_job()

    t0 = time.monotonic()
    fut = pool.submit_job(job)
    timed_out = False
    try:
        pool.wait_for_result(fut)
        elapsed = time.monotonic() - t0
        print(f"\nFAIL: the job COMPLETED in {elapsed:.1f}s — it was never reaped.",
              flush=True)
        print(f"The budget is {budget:g}s and the un-reaped baseline was "
              f"{full_run_s:.1f}s, so the waiter should have expired.", flush=True)
        return 2
    except TimeoutError as exc:
        elapsed = time.monotonic() - t0
        timed_out = True
        print(f"waiter raised after {elapsed:.1f}s: {str(exc)[:160]}", flush=True)

    terminal = terminal_exception(fut)
    reap_seen_at = time.monotonic() - t0
    print(f"job terminal after {reap_seen_at:.1f}s: {type(terminal).__name__}: "
          f"{str(terminal)[:160]}", flush=True)
    pid_after = child_pid(pool)

    print("\n--- the next job must succeed on the SAME worker ---", flush=True)
    t1 = time.monotonic()
    authority2 = pool.admit_generation(mode_name)
    req2 = GenerateRequest(prompt="a still lake", size="512x512",
                           num_inference_steps=8, guidance_scale=2.0)
    job2 = GenerationJob(req=req2, controlnet_bindings=[],
                         resolution_epoch=authority2.resolution_epoch)
    png, seed = pool.submit_job(job2).result(timeout=900.0)
    next_job_s = time.monotonic() - t1

    # --- verdict ---------------------------------------------------------------
    # Half the un-reaped run is a wide margin that still cannot be satisfied by a
    # job that ran to completion.
    stopped_early = elapsed < full_run_s / 2
    terminal_cancelled = type(terminal).__name__ == "CancelledError"
    same_process = (not subprocess_mode) or (pid_after is not None
                                             and pid_after == pid_before)

    print("\n==================== VERDICT ====================", flush=True)
    print(f"isolation                  : {isolation}", flush=True)
    print(f"waiter raised TimeoutError  : {timed_out}", flush=True)
    print(f"un-reaped baseline          : {full_run_s:.1f}s", flush=True)
    print(f"wall time to timeout        : {elapsed:.1f}s (budget {budget:g}s)", flush=True)
    print(f"stopped well short of a run : {stopped_early}", flush=True)
    print(f"terminal is CancelledError  : {terminal_cancelled} "
          f"({type(terminal).__name__})", flush=True)
    if subprocess_mode:
        print(f"child pid {pid_before} -> {pid_after}", flush=True)
        print(f"cooperative, NOT respawned  : {same_process}", flush=True)
    else:
        print(f"child pid check             : skipped (in-proc handle has no pid)",
              flush=True)
    print(f"next job OK on same worker  : True ({len(png)} bytes, seed={seed}, "
          f"{next_job_s:.1f}s)", flush=True)

    if not stopped_early:
        print(f"\nFAIL: the wait returned after {elapsed:.1f}s against an un-reaped "
              f"baseline of {full_run_s:.1f}s — the generation kept running. The "
              f"predicate is not reaching the denoise loop.", flush=True)
        return 3
    if not terminal_cancelled:
        print("\nFAIL: the job's terminal is not CancelledError. If it is a generic "
              "failure, the exception type has drifted and classify_exception() no "
              "longer maps it to CANCELLED.", flush=True)
        return 4
    if not same_process:
        print("\nFAIL: the child was respawned. A cooperative reap must not kill the "
              "worker — check that the terminal is CANCELLED rather than OOM or a "
              "frameless death.", flush=True)
        return 5
    print("\nPASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
