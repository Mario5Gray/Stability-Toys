"""STABL-rgvxuedo Task 8 — live subprocess OOM recovery acceptance.

Proves the facet-3 thesis on real hardware: the CUDA context lives in a child
process, so a poisoned context is dropped by KILLING the process — which
in-process empty_cache()/del provably cannot do.

The proof is the CHILD PID. If it changes across the OOM, the process was killed
and respawned; if it does not, recovery did not happen and any subsequent success
is just the same poisoned context getting lucky.

Run inside the CUDA test container, with the production models mounted:

    TEST_MODELS_HOST_PATH=/media/cold1/ComfyUI/models \
    docker compose -f docker-compose.test.yml run --rm \
      -e WORKER_ISOLATION=subprocess test-cuda \
      python spikes/facet3_oom_acceptance.py <mode-name>

HOG_GIB (default 14) sizes the synthetic VRAM pressure used to provoke the
OOM. An oversized request does NOT work for HunyuanDiT: use_resolution_binning
bins everything to 1280x1280, so the size is normalised away before it can
exhaust anything.

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


def spawn_hog(gib: float):
    """Hold `gib` GiB of VRAM from a SEPARATE process so the worker's next
    allocation genuinely fails.

    An oversized `size` does NOT work for this family: HunyuanDiT sets
    use_resolution_binning=True and bins every request to 1280x1280, so the
    request is normalised away before it can exhaust anything. Squeezing free
    VRAM from outside is the only lever that reaches the child.
    """
    proc = subprocess.Popen(
        [sys.executable, "spikes/vram_hog.py", str(gib)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"  [hog] {line.rstrip()}", flush=True)
        if line.startswith("HOG_READY"):
            return proc, True
        if line.startswith("HOG_FAILED"):
            return proc, False
    return proc, False


def main() -> int:
    mode_name = sys.argv[1] if len(sys.argv) > 1 else "HunyuanDiT"
    hog_gib = float(os.environ.get("HOG_GIB", "14"))

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
    png, seed = submit(pool, mode_name, "1024x1024", timeout=900.0)
    print(f"OK: {len(png)} bytes, seed={seed}", flush=True)

    print(f"\n--- step 2: force an OOM by holding {hog_gib} GiB elsewhere ---",
          flush=True)
    hog, ready = spawn_hog(hog_gib)
    if not ready:
        print(f"WARNING: hog did not reach {hog_gib} GiB; the OOM may not fire. "
              f"Lower HOG_GIB if the model itself is now starved.", flush=True)
    nvidia_smi("hog holding")
    t0 = time.monotonic()
    try:
        try:
            submit(pool, mode_name, "1024x1024", timeout=900.0)
        finally:
            hog.kill()          # release the pressure before recovery is judged
            hog.wait(timeout=30)
        print(f"NO OOM after {time.monotonic() - t0:.0f}s — raise HOG_GIB and "
              f"re-run (free VRAM was still sufficient)", flush=True)
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
