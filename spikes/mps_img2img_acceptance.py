"""STABL-tukrgrjb - live MPS SD1.5 LCM img2img acceptance.

Proves on a real Mac that an init image reaches the MPS worker, that the worker
renders from it, and that `denoise_strength` controls how much of the init image
survives.

FOUR signals, all required. A valid PNG alone proves nothing: the server returns
a valid PNG for text-to-image too, so a dropped init image would still pass a
format check.

  1. admitted     - the WebSocket returns `job:complete`, not `job:error`. The
                    `sd15` cell admits img2img.
  2. not txt2img  - the img2img bytes differ from a text-to-image render with the
                    same prompt and seed. The init image changed the output.
  3. valid        - each render is a 768x768 PNG, not black, not flat, not
                    corrupt.
  4. strength     - a low strength stays closer to the init image than a high
                    strength. The request field reaches the pipeline.

This script is a client. It starts no server and imports no server module.
Start the server first, in the same shell family as the isolation mode under
test:

    conda activate stability-toys
    export BACKEND=mps MODE_CONFIG_PATH=<conf copy> STORAGE_PROVIDER=MEMORY
    export WORKER_ISOLATION=inproc          # then repeat with subprocess
    python -m uvicorn server.lcm_sr_server:app --host 127.0.0.1 --port 4200

Then:

    python spikes/mps_img2img_acceptance.py --out /tmp/mps_img2img_inproc

`python -m uvicorn ...`, never `python server/lcm_sr_server.py`: the module uses
a relative import and the script form raises ImportError.

NOT a pytest test on purpose. It needs a live Metal runtime, a real model
directory, and a running server.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import statistics
import sys
import time
import urllib.request
import uuid

from PIL import Image, ImageDraw, ImageStat

PROMPT = "a bright red sailboat on a blue lake beneath a green mountain"
SIZE = "768x768"
STEPS = 8
GUIDANCE = 3.0
SEED = 7
LOW_STRENGTH = 0.35
HIGH_STRENGTH = 0.9

# Validity limits, the same ones the slice 1 acceptance used.
MIN_DISTINCT_COLORS = 32
MIN_GRAYSCALE_STDDEV = 5.0


def build_init_image() -> bytes:
    """One deterministic init image. Hard shapes, so a surviving structure is
    visible by eye and measurable by difference."""
    image = Image.new("RGB", (768, 768), (18, 120, 54))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 520, 768, 768], fill=(24, 60, 170))
    draw.ellipse([180, 120, 420, 360], fill=(220, 40, 40))
    draw.polygon([(540, 480), (660, 180), (700, 480)], fill=(235, 220, 90))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def post_multipart(url: str, filename: str, data: bytes, fields: dict) -> dict:
    boundary = uuid.uuid4().hex
    parts = []
    for key, value in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode()
        )
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{filename}\"\r\nContent-Type: image/png\r\n\r\n".encode()
    )
    parts.append(data)
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read())


def get_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def render(base_url: str, ws_url: str, params: dict, timeout_s: float = 600.0) -> dict:
    """Submit one generate job and return the PNG bytes plus the wall time."""
    from websockets.sync.client import connect

    job_params = {
        "prompt": PROMPT,
        "size": SIZE,
        "num_inference_steps": STEPS,
        "guidance_scale": GUIDANCE,
        "seed": SEED,
        "mode": "lcm-general",
    }
    job_params.update(params)

    started = time.monotonic()
    with connect(ws_url, open_timeout=30, close_timeout=5) as ws:
        json.loads(ws.recv(timeout=30))  # consume status
        ws.send(json.dumps({
            "type": "job:submit",
            "id": uuid.uuid4().hex[:8],
            "jobType": "generate",
            "params": job_params,
        }))
        frames = []
        while True:
            frame = json.loads(ws.recv(timeout=timeout_s))
            frames.append(frame)
            if frame.get("type") in ("job:complete", "job:error", "error"):
                break

    elapsed = time.monotonic() - started
    terminal = frames[-1]
    if terminal.get("type") != "job:complete":
        return {"ok": False, "terminal": terminal, "wall_s": round(elapsed, 2)}

    outputs = terminal.get("outputs") or []
    if not outputs:
        return {
            "ok": False,
            "terminal": {"type": "job:complete", "error": "no outputs; set STORAGE_PROVIDER=MEMORY"},
            "wall_s": round(elapsed, 2),
        }
    png = get_bytes(base_url + outputs[0]["url"])
    return {
        "ok": True,
        "png": png,
        "wall_s": round(elapsed, 2),
        "seed": terminal.get("meta", {}).get("seed"),
        "backend": terminal.get("meta", {}).get("backend"),
        "progress_frames": sum(1 for f in frames if f.get("type") == "job:progress"),
    }


def describe(png: bytes) -> dict:
    with Image.open(io.BytesIO(png)) as image:
        fmt = image.format
        size = image.size
        metadata = image.info.get("lcm")
        rgb = image.convert("RGB")
        small = rgb.resize((64, 64))
        distinct = len(set(small.getdata()))
        stddev = ImageStat.Stat(rgb.convert("L")).stddev[0]
    return {
        "format": fmt,
        "size": f"{size[0]}x{size[1]}",
        "bytes": len(png),
        "sha256": hashlib.sha256(png).hexdigest(),
        "distinct_colors_64x64": distinct,
        "grayscale_stddev": round(stddev, 2),
        "lcm_chunk": json.loads(metadata) if metadata else None,
        "valid": (
            fmt == "PNG"
            and size == (768, 768)
            and distinct > MIN_DISTINCT_COLORS
            and stddev > MIN_GRAYSCALE_STDDEV
        ),
    }


def mean_abs_difference(first: bytes, second: bytes) -> float:
    """Mean per-channel distance between two renders, 0 to 255."""
    with Image.open(io.BytesIO(first)) as a, Image.open(io.BytesIO(second)) as b:
        left = list(a.convert("RGB").resize((128, 128)).getdata())
        right = list(b.convert("RGB").resize((128, 128)).getdata())
    deltas = [
        abs(p[channel] - q[channel])
        for p, q in zip(left, right)
        for channel in range(3)
    ]
    return round(statistics.fmean(deltas), 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:4200")
    parser.add_argument("--out", default="/tmp/mps_img2img")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/v1/ws"

    report: dict = {
        "schema": "mps_img2img_acceptance/v1",
        "issue": "STABL-tukrgrjb",
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "mac_ver": platform.mac_ver()[0],
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "request": {
            "prompt": PROMPT, "size": SIZE, "steps": STEPS,
            "guidance": GUIDANCE, "seed": SEED,
        },
    }

    try:
        report["models_status"] = get_json(base_url + "/api/models/status")
    except Exception as exc:  # noqa: BLE001 - a report field, not control flow
        report["models_status"] = {"error": str(exc)}

    init_png = build_init_image()
    init_path = f"{args.out}_init.png"
    with open(init_path, "wb") as handle:
        handle.write(init_png)
    report["init_image"] = {
        "path": init_path,
        "sha256": hashlib.sha256(init_png).hexdigest(),
        "bytes": len(init_png),
    }

    upload = post_multipart(base_url + "/v1/upload", "init.png", init_png, {"type": "image"})
    report["upload"] = upload
    init_ref = upload["fileRef"]

    runs = {
        "txt2img": {},
        "img2img_low": {"init_image_ref": init_ref, "denoise_strength": LOW_STRENGTH},
        "img2img_high": {"init_image_ref": init_ref, "denoise_strength": HIGH_STRENGTH},
    }

    results = {}
    for name, params in runs.items():
        outcome = render(base_url, ws_url, params)
        if not outcome["ok"]:
            report["runs"] = results
            report[name] = outcome
            report["status"] = "failed"
            print(json.dumps(report, indent=2, sort_keys=True))
            return 1
        path = f"{args.out}_{name}.png"
        with open(path, "wb") as handle:
            handle.write(outcome["png"])
        results[name] = {
            "path": path,
            "wall_s": outcome["wall_s"],
            "seed": outcome["seed"],
            "backend": outcome["backend"],
            "progress_frames": outcome["progress_frames"],
            "strength": params.get("denoise_strength"),
            **describe(outcome["png"]),
        }
        results[name]["_png"] = outcome["png"]

    low = results["img2img_low"]["_png"]
    high = results["img2img_high"]["_png"]
    txt = results["txt2img"]["_png"]
    distances = {
        "low_vs_init": mean_abs_difference(low, init_png),
        "high_vs_init": mean_abs_difference(high, init_png),
        "low_vs_txt2img": mean_abs_difference(low, txt),
    }
    for value in results.values():
        value.pop("_png")

    checks = {
        "admitted": True,
        "all_valid": all(value["valid"] for value in results.values()),
        "img2img_differs_from_txt2img": (
            results["img2img_low"]["sha256"] != results["txt2img"]["sha256"]
        ),
        "low_strength_stays_closer_to_init": distances["low_vs_init"] < distances["high_vs_init"],
    }

    report["runs"] = results
    report["distances"] = distances
    report["checks"] = checks
    report["status"] = "pass" if all(checks.values()) else "failed"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
