"""STABL-sulxavyo - live MPS HunyuanDiT img2img acceptance client.

Proves on the real M1 Max that an init image reaches the MPS HunyuanDiT worker,
that the repo-owned `backends/hunyuandit_img2img.py` loop renders from it, and
that `denoise_strength` controls how much of the init image survives.

FIVE signals, all required:

  1. admitted       - all three jobs return `job:complete`, on backend `mps`,
                      at the requested seed.
  2. differs        - each img2img render differs from a text-to-image render
                      at the same prompt, size, steps, and seed.
  3. valid          - each render is a 1024x1024 PNG, not black, not flat, not
                      corrupt. This is where float16 on MPS is settled.
  4. strength       - the low-strength render is closer to the init image than
                      the high-strength render.
  5. metadata       - the PNG `lcm` chunk records the requested strength and
                      seed.

Start the server first:

    export BACKEND=mps MODE_CONFIG_PATH=<conf copy> STORAGE_PROVIDER=MEMORY
    export WORKER_ISOLATION=inproc CONTROLNET_REGISTRY_VALIDATION=off
    python -m uvicorn server.lcm_sr_server:app --host 127.0.0.1 --port 4200

Then:

    python spikes/mps_hunyuan_img2img_acceptance.py --out /tmp/mps_hunyuan_img2img
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

from PIL import Image, ImageDraw, ImageStat

from mps_hunyuan_scope import admitted_check

PROMPT = "a red apple on a wooden table, soft light, photo"
NEGATIVE = "blurry, watermark, text, low quality"
MODE = "HunyuanDiT"
SIZE = "1024x1024"
STEPS = 6
GUIDANCE = 6.0
SEED = 4242
POSITIVE_RUNS = ("txt2img", "img2img_low", "img2img_high")
LOW_STRENGTH = 0.35
HIGH_STRENGTH = 0.9
MIN_DISTINCT_COLORS = 32
MIN_GRAYSCALE_STDDEV = 5.0


def build_init_image() -> bytes:
    image = Image.new("RGB", (1024, 1024), (18, 120, 54))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 700, 1024, 1024], fill=(24, 60, 170))
    draw.ellipse([300, 180, 720, 600], fill=(220, 40, 40))
    draw.polygon([(760, 660), (900, 300), (980, 660)], fill=(235, 220, 90))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def post_multipart(url: str, filename: str, data: bytes, fields: dict) -> dict:
    boundary = uuid.uuid4().hex
    body = b""
    for name, value in fields.items():
        body += (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
            f"{value}\r\n"
        ).encode()
    body += (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{filename}\"\r\nContent-Type: image/png\r\n\r\n"
    ).encode()
    body += data + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


def get_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def render(base_url: str, ws_url: str, params: dict, timeout_s: float = 900.0) -> dict:
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
    job_params.update(params)

    started = time.monotonic()
    with connect(ws_url, open_timeout=30, close_timeout=5) as ws:
        json.loads(ws.recv(timeout=30))
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

    terminal = frames[-1]
    elapsed = round(time.monotonic() - started, 2)
    if terminal.get("type") != "job:complete":
        return {"ok": False, "terminal": terminal, "wall_s": elapsed}
    outputs = terminal.get("outputs") or []
    if not outputs:
        return {"ok": False, "terminal": "no outputs", "wall_s": elapsed}
    return {
        "ok": True,
        "png": get_bytes(base_url + outputs[0]["url"]),
        "wall_s": elapsed,
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


def distance(a: bytes, b: bytes) -> float:
    with Image.open(io.BytesIO(a)) as ia, Image.open(io.BytesIO(b)) as ib:
        ra = ia.convert("RGB").resize((128, 128))
        rb = ib.convert("RGB").resize((128, 128))
    pa, pb = ra.tobytes(), rb.tobytes()
    return round(sum(abs(x - y) for x, y in zip(pa, pb)) / len(pa), 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:4200")
    parser.add_argument("--out", default="/tmp/mps_hunyuan_img2img")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/v1/ws"

    init_png = build_init_image()
    upload = post_multipart(base_url + "/v1/upload", "init.png", init_png, {"type": "image"})
    init_ref = upload["fileRef"]

    report: dict = {
        "schema": "mps_hunyuan_img2img_acceptance/v1",
        "issue": "STABL-sulxavyo",
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
            "low_strength": LOW_STRENGTH, "high_strength": HIGH_STRENGTH,
        },
        "init_image": {
            "sha256": hashlib.sha256(init_png).hexdigest(),
            "bytes": len(init_png),
            "upload": upload,
        },
    }

    runs = {
        "txt2img": {},
        "img2img_low": {"init_image_ref": init_ref, "denoise_strength": LOW_STRENGTH},
        "img2img_high": {"init_image_ref": init_ref, "denoise_strength": HIGH_STRENGTH},
    }
    results: dict[str, dict] = {}
    for label, params in runs.items():
        result = render(base_url, ws_url, params)
        if not result.get("ok"):
            report["failed_run"] = {"label": label, **result}
            print(json.dumps(report, indent=2))
            return 1
        png = result["png"]
        path = f"{args.out}_{label}.png"
        with open(path, "wb") as handle:
            handle.write(png)
        results[label] = {
            "ok": result["ok"],
            "wall_s": result["wall_s"],
            "seed": result["seed"],
            "backend": result["backend"],
            "progress_frames": result["progress_frames"],
            "path": path,
            "sha256": hashlib.sha256(png).hexdigest(),
            "bytes": len(png),
            **describe(png),
        }

    low_to_init = distance(open(results["img2img_low"]["path"], "rb").read(), init_png)
    high_to_init = distance(open(results["img2img_high"]["path"], "rb").read(), init_png)

    checks = {
        "admitted": admitted_check(results, POSITIVE_RUNS, SEED),
        "differs_from_txt2img": (
            distance(open(results["img2img_low"]["path"], "rb").read(),
                     open(results["txt2img"]["path"], "rb").read()) > 0
            and distance(open(results["img2img_high"]["path"], "rb").read(),
                         open(results["txt2img"]["path"], "rb").read()) > 0
        ),
        "valid": all(
            r["format"] == "PNG"
            and r["size"] == [1024, 1024]
            and r["distinct_colors"] >= MIN_DISTINCT_COLORS
            and r["grayscale_stddev"] >= MIN_GRAYSCALE_STDDEV
            for r in results.values()
        ),
        "strength": low_to_init < high_to_init,
        "metadata": all(
            (results[label]["metadata"] or {}).get("denoise_strength") == strength
            and (results[label]["metadata"] or {}).get("seed") == SEED
            for label, strength in (
                ("img2img_low", LOW_STRENGTH),
                ("img2img_high", HIGH_STRENGTH),
            )
        ),
    }

    report["runs"] = results
    report["distances"] = {
        "low_to_init": low_to_init,
        "high_to_init": high_to_init,
        "low_to_txt2img": distance(open(results["img2img_low"]["path"], "rb").read(),
                                   open(results["txt2img"]["path"], "rb").read()),
        "high_to_txt2img": distance(open(results["img2img_high"]["path"], "rb").read(),
                                    open(results["txt2img"]["path"], "rb").read()),
    }
    report["checks"] = checks
    report["ok"] = all(checks.values())
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
