"""STABL-rgvxuedo Task 8 — live subprocess OOM recovery acceptance.

Proves the facet-3 thesis on real hardware: the CUDA context lives in a child
process, so a poisoned context is dropped by KILLING the process — which
in-process empty_cache()/del provably cannot do.

The proof is the CHILD PID. If it changes across the OOM, the process was killed
and respawned; if it does not, recovery did not happen and any subsequent success
is just the same poisoned context getting lucky.

Run inside the CUDA test container, with the production models mounted. The OOM
must come from EXTERNAL VRAM pressure — an oversized `size` cannot work, because
HunyuanDiT bins every request to a supported shape (see step 2):

    TEST_MODELS_HOST_PATH=/media/cold1/ComfyUI/models \
    docker compose -f docker-compose.test.yml run --rm \
      -e WORKER_ISOLATION=subprocess test-cuda \
      bash -c 'python spikes/vram_hog.py 14 > /tmp/hog.log 2>&1 &
               until grep -q HOG_READY /tmp/hog.log; do sleep 1; done
               python spikes/facet3_oom_acceptance.py <mode-name>'

Tune the hog to leave a WINDOW: enough free VRAM for the 1024x1024 baseline job,
not enough for the binned 1280x1280. On a 24GB card 14 GiB works and 15 does not
(15 OOMs the baseline job itself, which proves nothing — see step 1).

NOT a pytest test on purpose: it must own its process (see STABL-sgdavnvz — a
shared pytest session can leave stub modules resident) and it needs a real
model directory, so it cannot run in CI as-is.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

# Must be set before get_worker_pool() builds the singleton.
os.environ.setdefault("WORKER_ISOLATION", "subprocess")
os.environ.setdefault("BACKEND", "cuda")


def nvidia_smi(label: str) -> None:
    print(f"\n=== nvidia-smi ({label}) ===", flush=True)
    try:
        print(subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=pid,used_memory",
             "--format=csv"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip(), flush=True)
    except Exception as exc:  # noqa: BLE001 — diagnostics only
        print(f"nvidia-smi unavailable: {exc}", flush=True)


def child_pid(pool) -> int | None:
    proc = getattr(pool._governor._handle, "_proc", None)
    return getattr(proc, "pid", None)


def submit(pool, mode_name: str, size: str, *, timeout: float):
    from backends.governor import GenerationJob
    from server.lcm_sr_server import GenerateRequest

    authority = pool.admit_generation(mode_name)
    req = GenerateRequest(prompt="a forest scene", size=size,
                          num_inference_steps=8, guidance_scale=5.0)
    job = GenerationJob(req=req, controlnet_bindings=[],
                        resolution_epoch=authority.resolution_epoch)
    return pool.submit_job(job).result(timeout=timeout)


def main() -> int:
    mode_name = sys.argv[1] if len(sys.argv) > 1 else "HunyuanDiT"
    oom_size = os.environ.get("OOM_SIZE", "3072x3072")

    from backends.worker_handle_subprocess import SubprocessWorkerHandle
    from backends.worker_pool import get_worker_pool

    pool = get_worker_pool()
    handle = pool._governor._handle
    print(f"handle = {type(handle).__name__}", flush=True)
    if not isinstance(handle, SubprocessWorkerHandle):
        print("FAIL: not a SubprocessWorkerHandle — is WORKER_ISOLATION=subprocess set?")
        return 1

    print(f"\n--- loading mode {mode_name!r} ---", flush=True)
    pool.switch_mode(mode_name, force=True).result(timeout=900.0)
    pid_before = child_pid(pool)
    print(f"child pid after load: {pid_before}", flush=True)
    nvidia_smi("model loaded")

    print("\n--- step 1: a job that should SUCCEED ---", flush=True)
    try:
        png, seed = submit(pool, mode_name, "1024x1024", timeout=900.0)
    except Exception as exc:            # noqa: BLE001 — turn this into guidance
        oom = "out of memory" in str(exc).lower()
        print(f"\nSTEP 1 FAILED: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        if oom:
            # Too much external pressure: the baseline job needs headroom ABOVE the
            # resident model, and the acceptance needs a window where 1024x1024 fits
            # but the binned 1280x1280 does not. Squeezing past both proves nothing.
            print("\nThe baseline job OOMed, so the acceptance cannot distinguish "
                  "'recovery works' from 'there was never enough VRAM'.", flush=True)
            print("Lower the hog and re-run — you need a window where 1024x1024 "
                  "fits but the binned 1280x1280 does not:", flush=True)
            print("    python spikes/vram_hog.py 14     # 15 is too much on a 24GB card",
                  flush=True)
        return 4
    print(f"OK: {len(png)} bytes, seed={seed}", flush=True)

    print(f"\n--- step 2: force an OOM at {oom_size} ---", flush=True)
    t0 = time.monotonic()
    try:
        submit(pool, mode_name, oom_size, timeout=900.0)
        print(f"NO OOM at {oom_size} after {time.monotonic() - t0:.0f}s.", flush=True)
        print("Do NOT just raise OOM_SIZE — for HunyuanDiT it is INOPERABLE: the "
              "pipeline sets use_resolution_binning=True and folds every request to "
              "a supported shape (3072x3072 -> 1280x1280), so size cannot provoke "
              "an OOM at all.", flush=True)
        print("Force the OOM from OUTSIDE instead — hold the VRAM from another "
              "process, then re-run this script while it is held:", flush=True)
        print("    python spikes/vram_hog.py 14   # in a second shell", flush=True)
        return 2
    except Exception as exc:  # noqa: BLE001 — the OOM is the point
        print(f"job failed as intended: {type(exc).__name__}: "
              f"{str(exc)[:200]}", flush=True)

    # The Governor's recovery runs on the dispatch thread after the terminal
    # frame; give it a moment to kill + respawn before reading the pid.
    time.sleep(5.0)
    pid_after = child_pid(pool)
    print(f"\nchild pid before OOM: {pid_before}", flush=True)
    print(f"child pid after  OOM: {pid_after}", flush=True)
    nvidia_smi("after OOM recovery")

    print("\n--- step 3: the next job must SUCCEED on the fresh process ---", flush=True)
    png2, seed2 = submit(pool, mode_name, "1024x1024", timeout=900.0)
    print(f"OK: {len(png2)} bytes, seed={seed2}", flush=True)

    respawned = pid_after is not None and pid_after != pid_before
    print("\n==================== VERDICT ====================", flush=True)
    print(f"subprocess handle in use : True", flush=True)
    print(f"job succeeded before OOM : True", flush=True)
    print(f"OOM observed             : True", flush=True)
    print(f"child KILLED + RESPAWNED : {respawned}   <-- the thesis", flush=True)
    print(f"job succeeded after OOM  : True", flush=True)
    if not respawned:
        print("\nFAIL: pid unchanged. The next job succeeding does NOT prove "
              "recovery — the same context survived. Check the Governor's "
              "post-OOM kill/reload branch.", flush=True)
        return 3
    print("\nPASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
