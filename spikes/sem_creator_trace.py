"""STABL-nstyyrhh: print the STACK that creates each multiprocessing semaphore.

Established on enigma: one semaphore per MODEL LOAD, linear (4 = 1 default load
+ 3 forced switches), and reset_worker_pool() reclaims none. Not respawn — six
spawn/kill cycles with no model leak nothing. So a library on the model-load path
creates a multiprocessing SemLock (Lock / RLock / Event / Condition / Semaphore /
Queue) and never releases it.

`get_worker_pool()` is a STAGE, not a culprit. This names the culprit: it patches
multiprocessing.synchronize.SemLock.__init__ — the common base of every one of
those primitives — records the creating stack, then does one model load.

Run in-container, uses the small lcm model:

    PYTHONPATH=$PWD python spikes/sem_creator_trace.py --mode lcm-general

Read the deepest non-stdlib frame in each trace: that is the owner.
"""
from __future__ import annotations

import glob
import os
import sys
import traceback


def _arg_str(flag: str, default: str) -> str:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def sem_names() -> set[str]:
    if not os.path.isdir("/dev/shm"):
        return set()
    return {os.path.basename(p) for p in glob.glob("/dev/shm/sem.*")}


def main() -> int:
    import multiprocessing.synchronize as sync

    captured: list[str] = []
    original = sync.SemLock.__init__

    def traced(self, *args, **kwargs):
        # Drop this frame; keep the rest so the caller is the last line.
        captured.append("".join(traceback.format_stack()[:-1]))
        return original(self, *args, **kwargs)

    sync.SemLock.__init__ = traced                      # type: ignore[method-assign]

    before = sem_names()
    print(f"semaphores before: {len(before)}", flush=True)

    os.environ.setdefault("WORKER_ISOLATION", "subprocess")
    os.environ.setdefault("BACKEND", "cuda")
    mode = _arg_str("--mode", "")

    from backends.worker_pool import get_worker_pool
    try:
        pool = get_worker_pool()                        # loads the default mode
        print("get_worker_pool() done", flush=True)
        if mode:
            pool.switch_mode(mode, force=True).result(timeout=900.0)
            print(f"switch_mode({mode!r}, force=True) done", flush=True)
    except Exception as exc:            # noqa: BLE001 — still print what was captured
        print(f"load failed ({type(exc).__name__}: {exc}); "
              f"reporting whatever was captured before the failure", flush=True)

    after = sem_names()
    print(f"semaphores after: {len(after)} (+{len(after - before)})", flush=True)
    print(f"SemLock.__init__ calls captured: {len(captured)}", flush=True)

    for i, stack in enumerate(captured, 1):
        print(f"\n{'=' * 70}\nSemLock #{i} created by:\n{'=' * 70}", flush=True)
        # Only frames outside the stdlib matter; keep the tail, it is the caller.
        for line in stack.rstrip().splitlines()[-14:]:
            print(line, flush=True)

    if not captured:
        print("\nNo SemLock constructions observed. The semaphores are created in "
              "a CHILD process, or before this patch was installed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
