"""Synthetic VRAM pressure — hold N GiB on the GPU from a SEPARATE process.

Purpose: force a real OOM inside the facet-3 subprocess worker without touching
production code. HunyuanDiT sets use_resolution_binning=True and bins every
request to 1280x1280, so an oversized `size` is silently normalised away and
cannot be used to provoke an OOM. Squeezing free VRAM from outside can.

A SEPARATE PROCESS is deliberate. Allocating from the parent would give the
parent its own CUDA context — changing the exact topology facet-3 exists to
test, and confusing any nvidia-smi reading of which process holds what.

    python spikes/vram_hog.py 14          # hold 14 GiB until killed
    python spikes/vram_hog.py 14 --seconds 300

Prints `HOG_READY <bytes>` on stdout once the target is held, so a driver can
wait on it, then `HOG_FAILED <bytes>` if it could not reach the target.
"""
from __future__ import annotations

import argparse
import os
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gib", type=float, help="GiB of VRAM to hold")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="hold for N seconds then exit (0 = until killed)")
    ap.add_argument("--chunk-mib", type=int, default=256)
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("HOG_FAILED 0  (no CUDA)", flush=True)
        return 1

    torch.cuda.set_device(0)
    free_before, total = torch.cuda.mem_get_info()
    print(f"pid={os.getpid()} device=0 free_before={free_before} total={total}",
          flush=True)

    target = int(args.gib * (1024 ** 3))
    chunk = args.chunk_mib * (1024 ** 2)
    blocks: list = []           # hold references; never let them be collected
    held = 0

    try:
        while held < target:
            take = min(chunk, target - held)
            # uint8 so bytes == elements: no dtype arithmetic to get wrong.
            blocks.append(torch.empty(take, dtype=torch.uint8, device="cuda"))
            held += take
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print(f"HOG_FAILED {held}  (wanted {target}; {exc.__class__.__name__})",
              flush=True)
        # Keep what we got — partial pressure is still useful.

    free_after, _ = torch.cuda.mem_get_info()
    if held >= target:
        print(f"HOG_READY {held}", flush=True)
    print(f"held={held} free_after={free_after}", flush=True)

    if args.seconds > 0:
        time.sleep(args.seconds)
    else:
        # Hold until killed. The blocks stay referenced for the process
        # lifetime; exiting is what releases them (and the CUDA context).
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
