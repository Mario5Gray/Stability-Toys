"""STABL-xtkhoidu — live per-process VRAM attribution acceptance.

Proves on real hardware what PR #36 claims: under `WORKER_ISOLATION=subprocess`
the DeviceMemory snapshot carries TWO consumers with DISTINCT pids — `"worker"`
(the child, holding the generation model) and `"server"` (the parent, holding
superres) — and the bytes those two hold are no longer counted in
`unattributed_bytes`.

The proof is the PID PAIR plus the DEREGISTRATION DELTA. Two consumers alone
only show that something registered; the delta shows how much of
`unattributed_bytes` they actually explain, by closing the child's registration
and re-reading — which is exactly the pre-#36 state, measured rather than
argued.

`/api/models/status` cannot show this: it surfaces only the `"worker"` entry's
reserved bytes (`ModelRegistry._worker_entry`) and never the consumer list,
pids, or `unattributed_bytes`. The snapshot has to be read in the server's own
process, so this script owns the process and builds the pool through the
production `get_worker_pool()` path.

Superres runs with lifecycle=sticky regardless of `CUDA_SR_LIFECYCLE`: the
deployed default is `per_request`, which frees the upscaler before a snapshot
can see it, so a per_request run would show the parent holding nothing and prove
nothing about attribution.

Run inside the CUDA container, with the production models mounted:

    docker compose -f docker-cuda.yml run --rm --no-deps \
      -e WORKER_ISOLATION=subprocess stability-toys \
      python spikes/xtkhoidu_attribution_acceptance.py lcm-general

NOT a pytest test on purpose: it must own its process (a shared pytest session
can leave stub modules resident — STABL-sgdavnvz) and it needs a real model
directory, so it cannot run in CI as-is.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys

# Must be set before get_worker_pool() builds the singleton.
os.environ.setdefault("WORKER_ISOLATION", "subprocess")
os.environ.setdefault("BACKEND", "cuda")

GB = 1024 ** 3


def gb(n: int) -> float:
    return n / GB


def print_snapshot(label: str, snap) -> None:
    print(f"\n=== DeviceMemorySnapshot ({label}) ===", flush=True)
    print(f"  device       : {snap.device_uuid} [{snap.topology.value}]", flush=True)
    print(f"  total / used : {gb(snap.total_bytes):.2f} / {gb(snap.used_bytes):.2f} GiB",
          flush=True)
    print(f"  consumers    : {len(snap.consumers)}", flush=True)
    for c in snap.consumers:
        print(f"    - label={c.label!r:<10} pid={c.pid} "
              f"allocated={gb(c.allocated_bytes):.2f} GiB "
              f"reserved={gb(c.reserved_bytes):.2f} GiB stale={c.stale}", flush=True)
    print(f"  unattributed : {gb(snap.unattributed_bytes):.2f} GiB", flush=True)


def nvidia_smi_apps() -> dict[int, int]:
    """pid -> MiB, straight from the driver. Cross-check only: the driver's
    per-process figure includes the CUDA context and non-torch workspaces, so it
    is always ABOVE a consumer's torch reserved pool, never equal to it."""
    out: dict[int, int] = {}
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
        for line in res.stdout.strip().splitlines():
            pid_s, mib_s = (p.strip() for p in line.split(","))
            out[int(pid_s)] = int(mib_s)
    except Exception as exc:  # noqa: BLE001 — diagnostics only
        print(f"nvidia-smi unavailable: {exc}", flush=True)
    return out


def consumer(snap, label: str):
    return next((c for c in snap.consumers if c.label == label), None)


def generate_png(pool, mode_name: str) -> bytes:
    """A real generation, so the child is holding a model the way production
    holds one. Falls back to a synthetic image: the acceptance is about
    attribution, and a generation failure should not mask that."""
    try:
        from backends.governor import GenerationJob
        from server.lcm_sr_server import GenerateRequest

        authority = pool.admit_generation(mode_name)
        req = GenerateRequest(prompt="a quiet harbour at dawn", size="512x512",
                              num_inference_steps=8, guidance_scale=2.0)
        job = GenerationJob(req=req, controlnet_bindings=[],
                            resolution_epoch=authority.resolution_epoch)
        png, seed = pool.submit_job(job).result(timeout=900.0)
        print(f"generated {len(png)} bytes, seed={seed}", flush=True)
        return png
    except Exception as exc:  # noqa: BLE001
        print(f"generation failed ({type(exc).__name__}: {str(exc)[:200]}); "
              f"falling back to a synthetic image", flush=True)
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (512, 512), (90, 120, 160)).save(buf, format="PNG")
        return buf.getvalue()


def run_superres(png: bytes):
    """Load superres IN THIS PROCESS — the parent — which is the whole point:
    it is the second in-parent GPU consumer STABL-xtkhoidu was filed about."""
    from dataclasses import replace

    from server.superres_service import CudaSuperResService, load_cuda_superres_config

    config = replace(load_cuda_superres_config(), lifecycle="sticky")
    print(f"superres model: {config.model_path or '<unset>'} "
          f"(lifecycle forced to sticky)", flush=True)
    service = CudaSuperResService(config=config)
    out = service.submit(png, out_format="png", quality=92,
                         magnitude=1, timeout_s=5.0).result(timeout=600.0)
    print(f"upscaled to {len(out)} bytes", flush=True)
    return service


def main() -> int:
    mode_name = sys.argv[1] if len(sys.argv) > 1 else "lcm-general"

    from backends.device_memory import get_device_memory
    from backends.worker_handle_subprocess import SubprocessWorkerHandle
    from backends.worker_pool import get_worker_pool

    pool = get_worker_pool()
    handle = pool._governor._handle
    parent_pid = os.getpid()
    print(f"handle     = {type(handle).__name__}", flush=True)
    print(f"parent pid = {parent_pid}", flush=True)
    if not isinstance(handle, SubprocessWorkerHandle):
        print("FAIL: not a SubprocessWorkerHandle — is WORKER_ISOLATION=subprocess set?")
        return 1

    dm = get_device_memory()

    # --- phase 1: before anything is loaded ---------------------------------
    print_snapshot("phase 1 — pool built, nothing loaded", dm.snapshot())

    # --- phase 2: the child loads a model -----------------------------------
    print(f"\n--- phase 2: loading mode {mode_name!r} in the child ---", flush=True)
    pool.switch_mode(mode_name, force=True).result(timeout=900.0)
    child_pid = getattr(getattr(handle, "_proc", None), "pid", None)
    print(f"child pid = {child_pid}", flush=True)
    png = generate_png(pool, mode_name)
    snap_worker = dm.snapshot()
    print_snapshot("phase 2 — model resident in the child", snap_worker)

    # --- phase 3: superres loads in the PARENT ------------------------------
    print("\n--- phase 3: superres in the parent process ---", flush=True)
    sr_service = None
    try:
        sr_service = run_superres(png)
    except Exception as exc:  # noqa: BLE001 — report, do not mask
        print(f"SUPERRES FAILED: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
    snap_both = dm.snapshot()
    print_snapshot("phase 3 — worker in the child, superres in the parent", snap_both)

    # --- phase 4: driver cross-check ----------------------------------------
    apps = nvidia_smi_apps()
    print("\n=== nvidia-smi compute apps (pid -> MiB) ===", flush=True)
    for pid, mib in sorted(apps.items()):
        who = ("parent" if pid == parent_pid else
               "child" if pid == child_pid else "other")
        print(f"    pid={pid} used={mib} MiB   [{who}]", flush=True)

    # --- phase 5: the deregistration delta (the pre-#36 state, measured) ----
    print("\n--- phase 5: closing the child's registration (pre-#36 state) ---",
          flush=True)
    worker_before = consumer(snap_both, "worker")
    registration = getattr(handle, "_registration", None)
    if registration is not None:
        registration.close()
    snap_unregistered = dm.snapshot()
    print_snapshot("phase 5 — child deregistered", snap_unregistered)
    delta = snap_unregistered.unattributed_bytes - snap_both.unattributed_bytes

    # --- verdict -------------------------------------------------------------
    server_c = consumer(snap_both, "server")
    worker_c = worker_before
    two_consumers = len(snap_both.consumers) == 2
    distinct_pids = (server_c is not None and worker_c is not None
                     and server_c.pid == parent_pid
                     and worker_c.pid == child_pid
                     and server_c.pid != worker_c.pid)
    worker_holds = worker_c is not None and worker_c.reserved_bytes > 0
    server_holds = server_c is not None and server_c.reserved_bytes > 0
    delta_matches = (worker_c is not None
                     and abs(delta - worker_c.reserved_bytes) < 64 * 1024 * 1024)
    fresh = all(not c.stale for c in snap_both.consumers)

    print("\n==================== VERDICT ====================", flush=True)
    print(f"subprocess handle in use          : True", flush=True)
    print(f"exactly two consumers             : {two_consumers} "
          f"({len(snap_both.consumers)})", flush=True)
    print(f"distinct pids, parent + child     : {distinct_pids} "
          f"(server={server_c.pid if server_c else None}, "
          f"worker={worker_c.pid if worker_c else None})", flush=True)
    print(f"child 'worker' holds VRAM         : {worker_holds} "
          f"({gb(worker_c.reserved_bytes) if worker_c else 0:.2f} GiB)", flush=True)
    print(f"parent 'server' holds superres    : {server_holds} "
          f"({gb(server_c.reserved_bytes) if server_c else 0:.2f} GiB)", flush=True)
    print(f"neither consumer stale            : {fresh}", flush=True)
    print(f"unattributed WITH the child       : "
          f"{gb(snap_both.unattributed_bytes):.2f} GiB", flush=True)
    print(f"unattributed WITHOUT it (pre-#36) : "
          f"{gb(snap_unregistered.unattributed_bytes):.2f} GiB", flush=True)
    print(f"delta == child's reserved pool    : {delta_matches} "
          f"(delta={gb(delta):.2f} GiB)", flush=True)

    ok = two_consumers and distinct_pids and worker_holds and server_holds and fresh
    if not ok:
        print("\nFAIL: attribution is not what STABL-xtkhoidu claims. Read the "
              "snapshots above — a missing 'server' entry means the parent "
              "registration did not happen; a stale 'worker' entry means the "
              "child's control pipe did not answer inside the fan-out budget.",
              flush=True)
    else:
        print("\nPASS", flush=True)

    if sr_service is not None:
        sr_service.shutdown()
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
