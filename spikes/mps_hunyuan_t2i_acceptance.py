"""STABL-appbjuws - live MPS HunyuanDiT text-to-image acceptance client.

Submit one HunyuanDiT text-to-image job through the normal WebSocket path and
record the evidence: request, host, elapsed, PNG, PNG metadata, and validity.

Start the server first:

    export BACKEND=mps MODE_CONFIG_PATH=<conf copy> STORAGE_PROVIDER=MEMORY
    export WORKER_ISOLATION=inproc CONTROLNET_REGISTRY_VALIDATION=off
    python -m uvicorn server.lcm_sr_server:app --host 127.0.0.1 --port 4200

Then:

    python spikes/mps_hunyuan_t2i_acceptance.py --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import time
import urllib.request
import uuid

from PIL import Image, ImageStat

PROMPT = "a red apple on a wooden table, soft light, photo"
NEGATIVE = "blurry, watermark, text, low quality"
SIZE = "1024x1024"
STEPS = 6
GUIDANCE = 6.0
SEED = 4242
MODE = "HunyuanDiT"


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


def get_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def render(base_url: str, ws_url: str, timeout_s: float = 900.0) -> dict:
    from websockets.sync.client import connect

    job_params = {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE,
        "size": SIZE,
        "num_inference_steps": STEPS,
        "guidance_scale": GUIDANCE,
        "seed": SEED,
        "mode": MODE,
    }

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
            "terminal": {"type": "job:complete", "error": "no outputs"},
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
        metadata = image.info.get("lcm")
        rgb = image.convert("RGB")
        return {
            "format": image.format,
            "size": list(image.size),
            "metadata": json.loads(metadata) if metadata else None,
            "distinct_colors": len(rgb.getcolors(maxcolors=1_000_000) or []),
            "grayscale_stddev": round(ImageStat.Stat(rgb.convert("L")).stddev[0], 2),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:4200")
    parser.add_argument("--out", default="/tmp/hunyuan_mps_acceptance")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/v1/ws"

    report: dict = {
        "schema": "hunyuan_mps_acceptance/v1",
        "issue": "STABL-appbjuws",
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "mac_ver": platform.mac_ver()[0],
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "request": {
            "mode": MODE, "prompt": PROMPT, "negative_prompt": NEGATIVE,
            "size": SIZE, "steps": STEPS, "guidance": GUIDANCE, "seed": SEED,
        },
    }

    try:
        report["models_status"] = get_json(base_url + "/api/models/status")
    except Exception as exc:  # noqa: BLE001 - a report field, not control flow
        report["models_status"] = {"error": str(exc)}

    result = render(base_url, ws_url)
    if not result.get("ok"):
        report["terminal"] = result.get("terminal")
        report["wall_s"] = result.get("wall_s")
        print(json.dumps(report, indent=2))
        return 1

    png = result["png"]
    path = f"{args.out}.png"
    with open(path, "wb") as handle:
        handle.write(png)

    report["wall_s"] = result["wall_s"]
    report["seed"] = result["seed"]
    report["backend"] = result["backend"]
    report["progress_frames"] = result["progress_frames"]
    report["output"] = {
        "path": path,
        "sha256": hashlib.sha256(png).hexdigest(),
        "bytes": len(png),
        **describe(png),
    }
    report["checks"] = {
        "png": report["output"]["format"] == "PNG",
        "size": report["output"]["size"] == [1024, 1024],
        "not_flat": report["output"]["distinct_colors"] >= 32
        and report["output"]["grayscale_stddev"] >= 5.0,
        "metadata_seed": (report["output"]["metadata"] or {}).get("seed") == SEED,
        "metadata_steps": (report["output"]["metadata"] or {}).get("steps") == STEPS,
    }
    report["ok"] = all(report["checks"].values())
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
