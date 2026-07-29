#!/usr/bin/env python3
"""VRAM watcher for STABL-hjldxurg T10 behavioral-no-op acceptance.

Polls the running server's model-status endpoint on a fixed interval and, for
each poll, captures nvidia-smi driver-truth memory alongside it. Every sample is
appended as ONE JSON object per line (NDJSON) — append-only, crash-safe, and
queryable after the fact with jq / pandas / this script's --summarize mode.

The point: line up the server's DeviceMemory view (/status vram + the new
`stale` field, sourced from NVML) against raw driver truth (nvidia-smi) over
time, so the NVML-vs-mem_get_info delta and admission behavior can be replayed.

Stdlib only (urllib, subprocess, json, argparse) — no torch, no requests, no jq
dependency. Runs in any python3.

WATCH:
    ST_BASE_URL=http://localhost python scripts/vram_watch.py -n 5 -r 1024x1024
    # -> appends to vram_watch.jsonl every 5s until Ctrl-C

QUERY (the file is a list of results):
    jq -s 'map(.gpus[0].memory_used_mib) | {min:min, max:max}' vram_watch.jsonl
    jq 'select(.status.vram != null) | {ts, free: .gpus[0].memory_free_mib}' vram_watch.jsonl
    python scripts/vram_watch.py --summarize vram_watch.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

STATUS_PATH = "/api/model/status"
SMI_FIELDS = ["index", "uuid", "memory.total", "memory.free", "memory.used"]
# JSON-safe keys for the parsed nvidia-smi row (memory.* -> *_mib, values in MiB).
SMI_KEYS = ["index", "uuid", "memory_total_mib", "memory_free_mib", "memory_used_mib"]


def _now() -> tuple[str, float]:
    t = time.time()
    return datetime.now(timezone.utc).isoformat(), t


def _build_url(base: str, port: int) -> str:
    base = base.rstrip("/")
    if not base.startswith(("http://", "https://")):
        base = "http://" + base
    return f"{base}:{port}{STATUS_PATH}"


def poll_status(url: str, timeout: float) -> dict:
    """GET the status endpoint. Never raises — errors ride in the record."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            code = resp.getcode()
            raw = resp.read().decode("utf-8", "replace")
        try:
            return {"http": code, "body": json.loads(raw), "error": None}
        except json.JSONDecodeError:
            # Endpoint answered but not with JSON — keep the raw text for triage.
            return {"http": code, "body": None, "raw": raw[:2000], "error": "non-json body"}
    except urllib.error.HTTPError as e:
        return {"http": e.code, "body": None, "error": f"http {e.code}"}
    except Exception as e:  # noqa: BLE001 — connection refused, timeout, DNS, ...
        return {"http": None, "body": None, "error": f"{type(e).__name__}: {e}"}


def poll_nvidia_smi(timeout: float) -> tuple[list | None, str | None]:
    """Run nvidia-smi and parse per-GPU memory into JSON. MiB ints, no units.
    Returns (rows, error). rows is None on any failure (off-GPU host, no driver).
    """
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(SMI_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=True
        ).stdout
    except FileNotFoundError:
        return None, "nvidia-smi not found"
    except subprocess.CalledProcessError as e:
        return None, f"nvidia-smi exit {e.returncode}: {e.stderr.strip()[:200]}"
    except subprocess.TimeoutExpired:
        return None, "nvidia-smi timeout"

    rows = []
    for line in out.strip().splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != len(SMI_KEYS):
            continue
        row = {}
        for key, val in zip(SMI_KEYS, cells):
            if key.endswith("_mib") or key == "index":
                try:
                    row[key] = int(val)
                except ValueError:
                    row[key] = val
            else:
                row[key] = val
        rows.append(row)
    return (rows or None), (None if rows else "nvidia-smi produced no rows")


def sample(url: str, resolution: str, wh: tuple[int, int] | None, http_timeout: float) -> dict:
    iso, epoch = _now()
    status = poll_status(url, http_timeout)
    gpus, smi_err = poll_nvidia_smi(http_timeout)
    rec = {
        "ts": iso,
        "epoch": epoch,
        "resolution": resolution,
        "status": status,
        "gpus": gpus,
    }
    if wh is not None:
        rec["width"], rec["height"] = wh
    if smi_err is not None:
        rec["nvidia_smi_error"] = smi_err
    return rec


def _parse_resolution(s: str) -> tuple[int, int] | None:
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        return None  # free-form label allowed; width/height just omitted


def watch(args: argparse.Namespace) -> int:
    url = _build_url(args.base_url, args.port)
    wh = _parse_resolution(args.resolution)
    stop = {"flag": False}

    def _handle(signum, _frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    print(f"[vram_watch] polling {url} every {args.interval}s -> {args.out}", file=sys.stderr)
    print(f"[vram_watch] resolution={args.resolution}  Ctrl-C to stop", file=sys.stderr)

    n = 0
    with open(args.out, "a", encoding="utf-8") as f:
        while not stop["flag"]:
            rec = sample(url, args.resolution, wh, args.http_timeout)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
            n += 1
            g0 = (rec["gpus"] or [{}])[0]
            print(
                f"[{rec['ts']}] #{n} "
                f"status={'ok' if rec['status']['error'] is None else rec['status']['error']} "
                f"free={g0.get('memory_free_mib', '?')}MiB "
                f"used={g0.get('memory_used_mib', '?')}MiB",
                file=sys.stderr,
            )
            if args.count and n >= args.count:
                break
            # Interruptible sleep so Ctrl-C is responsive mid-interval.
            slept = 0.0
            while slept < args.interval and not stop["flag"]:
                time.sleep(min(0.25, args.interval - slept))
                slept += 0.25

    print(f"[vram_watch] wrote {n} samples to {args.out}", file=sys.stderr)
    return 0


def summarize(path: str) -> int:
    """Read back the NDJSON and print free/used min/max/delta per GPU index —
    the behavioral-no-op delta at a glance. Proof the file is queryable."""
    per_gpu: dict[int, dict[str, list]] = {}
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            for g in rec.get("gpus") or []:
                idx = g.get("index", 0)
                b = per_gpu.setdefault(idx, {"free": [], "used": []})
                if isinstance(g.get("memory_free_mib"), int):
                    b["free"].append(g["memory_free_mib"])
                if isinstance(g.get("memory_used_mib"), int):
                    b["used"].append(g["memory_used_mib"])

    out = {"samples": total, "gpus": {}}
    for idx, b in sorted(per_gpu.items()):
        entry = {}
        for k in ("free", "used"):
            vals = b[k]
            if vals:
                entry[k + "_mib"] = {
                    "min": min(vals), "max": max(vals),
                    "delta": max(vals) - min(vals), "last": vals[-1],
                }
        out["gpus"][idx] = entry
    print(json.dumps(out, indent=2))
    return 0


def main(argv=None) -> int:
    default_base = os.environ.get("ST_BASE_URL", "http://localhost")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summarize", metavar="FILE",
                   help="read a prior NDJSON log, print free/used min/max/delta per GPU, and exit")
    p.add_argument("-n", "--interval", type=float, default=5.0, help="seconds between polls (default 5)")
    p.add_argument("-r", "--resolution", default="1024x1024", help="resolution label recorded per sample")
    p.add_argument("-o", "--out", default="vram_watch.jsonl", help="NDJSON output file (append mode)")
    p.add_argument("--base-url", default=default_base, help="server base URL (default $ST_BASE_URL or http://localhost)")
    p.add_argument("--port", type=int, default=4200, help="server port (default 4200)")
    p.add_argument("--count", type=int, default=0, help="stop after N samples (default 0 = run until Ctrl-C)")
    p.add_argument("--http-timeout", type=float, default=5.0, help="per-poll timeout seconds (default 5)")
    args = p.parse_args(argv)

    if args.summarize:
        return summarize(args.summarize)
    return watch(args)


if __name__ == "__main__":
    sys.exit(main())
