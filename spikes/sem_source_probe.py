"""STABL-nstyyrhh: WHICH import or initialisation creates the mp semaphores?

The leak is NOT caused by kill+respawn — the 2026-07-31 acceptance run leaked 2
semaphores having performed ZERO respawns (its OOM never fired; resolution
binning folded 3072x3072 to 1280x1280). And 6 spawn->job->kill cycles of the
handle alone leak none. So something else in the process creates them.

The names are the clue: /dev/shm/sem.mp-* is multiprocessing.synchronize.SemLock,
i.e. a Lock / RLock / Event / Condition / Semaphore / Queue. Our code creates
none of those anywhere, so a library does.

This walks the stack one stage at a time and prints the delta after each, so the
creator is named rather than guessed. It never loads a model and needs no GPU
memory beyond CUDA init.

Run INSIDE the container (/dev/shm is per-mount-namespace — the host cannot see
the container's semaphores, which is why an earlier host-side check showed 0):

    PYTHONPATH=$PWD python spikes/sem_source_probe.py
"""
from __future__ import annotations

import glob
import os


def sems() -> set[str]:
    if not os.path.isdir("/dev/shm"):
        return set()
    return {os.path.basename(p) for p in glob.glob("/dev/shm/sem.*")}


class Tracker:
    def __init__(self):
        self.seen = sems()
        print(f"baseline: {len(self.seen)} semaphores {sorted(self.seen) or ''}",
              flush=True)

    def check(self, label: str) -> None:
        now = sems()
        added = now - self.seen
        removed = self.seen - now
        self.seen = now
        mark = "  <-- CREATED HERE" if added else ""
        print(f"{label:<48} total={len(now):<3} +{len(added)} -{len(removed)}{mark}",
              flush=True)
        for name in sorted(added):
            print(f"    + {name}", flush=True)


def main() -> int:
    t = Tracker()

    import torch                                                  # noqa: F401
    t.check("import torch")

    import diffusers                                              # noqa: F401
    t.check("import diffusers")

    import transformers                                           # noqa: F401
    t.check("import transformers")

    if torch.cuda.is_available():
        torch.cuda.init()
        t.check("torch.cuda.init()")
        torch.zeros(1, device="cuda")
        t.check("first cuda allocation")
        torch.cuda.mem_get_info()
        t.check("torch.cuda.mem_get_info() (DeviceMemory snapshot)")
    else:
        print("cuda unavailable — skipping cuda stages", flush=True)

    try:
        import xformers                                           # noqa: F401
        t.check("import xformers")
    except Exception as exc:                    # noqa: BLE001 — optional dep
        print(f"xformers unavailable: {exc}", flush=True)

    from backends.device_memory import get_device_memory
    get_device_memory().snapshot()
    t.check("get_device_memory().snapshot()  (nvml/psutil)")

    from backends.worker_pool import get_worker_pool, reset_worker_pool
    t.check("import worker_pool")

    os.environ.setdefault("WORKER_ISOLATION", "subprocess")
    os.environ.setdefault("BACKEND", "cuda" if torch.cuda.is_available() else "cpu")
    try:
        get_worker_pool()
        t.check("get_worker_pool() (governor + handle construction)")
    except Exception as exc:                    # noqa: BLE001 — keep walking the stack
        print(f"get_worker_pool() unavailable here: {type(exc).__name__}: {exc}",
              flush=True)

    from backends.worker_handle_subprocess import SubprocessWorkerHandle
    h = SubprocessWorkerHandle("tests._fault_worker.make_fault_worker")
    h.start(None, None, None)
    t.check("spawn one child (FaultWorker)")
    h.stop()
    t.check("kill that child")

    reset_worker_pool()
    t.check("reset_worker_pool()")

    print("\nWhatever line says CREATED HERE owns the semaphores. If nothing "
          "does, they predate this script — check the interpreter's own startup.",
          flush=True)
    print(f"final: {len(sems())} semaphores {sorted(sems()) or ''}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
