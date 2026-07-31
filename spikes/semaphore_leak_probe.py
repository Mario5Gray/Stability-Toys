"""STABL-nstyyrhh probe: is the semaphore leak per-respawn, and is it OURS?

The issue records "3 leaked semaphore objects" after one kill+respawn cycle in the
Task 8 live run, and SUSPECTS (explicitly unconfirmed) that it is linear in respawn
count. That distinction decides the severity:

  linear in N  -> a long-running server eventually cannot spawn a worker at all,
                  with an error that will not point back here. Real bug.
  constant     -> a fixed one-time cost of using spawn. Not a leak; close the issue.
  absent       -> not the handle. The real worker's own libraries (torch/CUDA/
                  diffusers) create them in the child, and the fix is elsewhere.

Each cycle is a full spawn -> (optional job) -> kill, i.e. the production OOM
recovery path, driven with the FaultWorker so no GPU or model is required.

On Linux this samples /dev/shm/sem.* per cycle, which is direct evidence rather
than the interpreter's shutdown warning. macOS has no /dev/shm, so there the
shutdown warning on stderr is the only signal (run with N=1 and N=6 and compare).

Run:

    PYTHONPATH=$PWD python spikes/semaphore_leak_probe.py 6 --jobs

Grep the tail of stderr for "leaked semaphore objects".
"""
from __future__ import annotations

import glob
import sys
from concurrent.futures import Future

from backends.governor import GenerationJob
from backends.worker_handle_subprocess import (
    SubprocessWorkerHandle,
    _SubprocessFutureBridge,
)
from server.lcm_sr_server import GenerateRequest


def sem_count() -> int | None:
    """POSIX named semaphores visible on Linux; None where /dev/shm is absent."""
    entries = glob.glob("/dev/shm/sem.*")
    if not entries and not glob.glob("/dev/shm/*"):
        return None
    return len(entries)


def _run_job(handle):
    job = GenerationJob(
        req=GenerateRequest(prompt="probe", num_inference_steps=4, size="512x512"),
        resolution_epoch=0,
    )
    fut = Future()
    handle.submit(job).subscribe(_SubprocessFutureBridge(fut))
    return fut.result(timeout=30)


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    with_jobs = "--jobs" in sys.argv

    baseline = sem_count()
    print(f"platform sem tracking: "
          f"{'/dev/shm' if baseline is not None else 'unavailable (macOS)'}", flush=True)
    print(f"baseline semaphores: {baseline}", flush=True)

    for i in range(n):
        h = SubprocessWorkerHandle("tests._fault_worker.make_fault_worker")
        h.start(None, None, None)
        pid = h._proc.pid
        detail = f", job -> {_run_job(h)!r}" if with_jobs else ""
        h.stop()                       # kill + join: the production recovery path
        print(f"cycle {i + 1}/{n}: pid {pid} killed, semaphores now "
              f"{sem_count()}{detail}", flush=True)

    final = sem_count()
    print(f"\ncompleted {n} cycles (jobs={with_jobs})", flush=True)
    if baseline is not None and final is not None:
        grew = final - baseline
        print(f"semaphores: {baseline} -> {final} (delta {grew:+d})", flush=True)
        print(f"VERDICT: {'PER-RESPAWN LEAK' if grew >= n else 'not linear in respawns'}"
              f" — delta {grew} over {n} cycles", flush=True)
    else:
        print("VERDICT: compare the 'leaked semaphore objects' warning below "
              "across two runs with different N.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
