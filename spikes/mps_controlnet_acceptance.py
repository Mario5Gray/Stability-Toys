"""STABL-xomvolgw - live MPS SD1.5 LCM ControlNet acceptance.

Proves on a real Mac that one Canny attachment reaches the MPS worker, that the
worker renders through the ControlNet pipeline, and that the control map steers
the result. Also proves that the platform scope refuses what the mode permits.

SEVEN signals, all required. A valid PNG alone proves nothing: the server
returns a valid PNG for text-to-image too, so a dropped control map would still
pass a format check.

  1. admitted     - the WebSocket returns `job:complete` for one Canny
                    attachment. The `sd15` cell admits ControlNet.
  2. not txt2img  - the ControlNet bytes differ from a text-to-image render with
                    the same prompt and seed. The control map changed the output.
  3. content      - TWO DIFFERENT maps at the SAME strength, prompt, and seed
                    produce different renders. This is the signal that the map's
                    CONTENT steers the output. Comparing one map at two
                    strengths cannot prove it: a worker that passed a fixed or
                    wrong image while honouring `strength` would pass that
                    comparison.
  4. strength     - a strength 0.0 render differs from a strength 1.0 render, so
                    the request field reaches the pipeline.
  5. valid        - each render is a 768x768 PNG, not black, not flat, not
                    corrupt. This is where float16 on MPS is settled: a NaN
                    pipeline returns a black image and fails here.
  6. chunk        - the generation PNG carries the `controlnet` text chunk with
                    the attachment id, the control type, and the model id.
  7. scope        - Depth is refused, and two attachments are refused, EACH with
                    its own platform message, while conf/modes.yml permits
                    canny, depth, pose and max_attachments 2. A refusal that
                    carries another message, or no `job:error` at all, fails:
                    a timeout and a mode-policy rejection are not this signal.

This script is a client. It starts no server and imports no server module.
Start the server first, in the same shell family as the isolation mode under
test:

    conda activate stability-toys
    export BACKEND=mps MODE_CONFIG_PATH=<conf copy> STORAGE_PROVIDER=MEMORY
    export CONTROLNET_REGISTRY_PATH=/tmp/controlnets-host.yaml
    export WORKER_ISOLATION=inproc          # then repeat with subprocess
    python -m uvicorn server.lcm_sr_server:app --host 127.0.0.1 --port 4200

Then:

    python spikes/mps_controlnet_acceptance.py --out /tmp/mps_controlnet_inproc

`CONTROLNET_REGISTRY_PATH` is read once, when the registry singleton is built,
so it must be in the server's environment before the first request. Leave
`CONTROLNET_REGISTRY_VALIDATION` unset: strict is the default, and strict is
what proves the host paths resolve.

`python -m uvicorn ...`, never `python server/lcm_sr_server.py`: the module uses
a relative import and the script form raises ImportError.

NOT a pytest test on purpose. It needs a live Metal runtime, a real model
directory, real ControlNet weights, and a running server.
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
FULL_STRENGTH = 1.0
ZERO_STRENGTH = 0.0

# Validity limits, the same ones the slice 1 and slice 2 acceptances used.
MIN_DISTINCT_COLORS = 32
MIN_GRAYSCALE_STDDEV = 5.0

# Two renders count as different above this mean per-channel distance, 0 to 255.
# JPEG-free PNG renders of the same seed are bit-stable, so any real difference
# sits far above it.
MIN_RENDER_DISTANCE = 1.0

# The platform messages, copied from backends/mps_worker.py and
# server/controlnet_execution.py. The scope signal requires THESE, not merely a
# failure: a timeout, a dropped socket, and a mode-policy rejection are all
# failures that prove nothing about the platform scope.
SCOPE_MESSAGES = {
    "depth": (
        "backend 'mps' does not support control_type 'depth' for family 'sd15'; "
        "allowed: canny"
    ),
    "two_attachments": (
        "backend 'mps' admits at most 1 controlnet attachment(s) for family "
        "'sd15', got 2"
    ),
}


def _to_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def build_source_image() -> bytes:
    """One deterministic source image. Hard edges, so Canny has something to
    find and a surviving structure is visible by eye."""
    image = Image.new("RGB", (768, 768), (18, 120, 54))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 520, 768, 768], fill=(24, 60, 170))
    draw.ellipse([180, 120, 420, 360], fill=(220, 40, 40))
    draw.polygon([(540, 480), (660, 180), (700, 480)], fill=(235, 220, 90))
    return _to_png(image)


def build_alt_source_image() -> bytes:
    """A second deterministic source image whose edges sit elsewhere.

    Its Canny map must differ from the first one's in CONTENT, not just in edge
    count, so a render that depends on the map cannot match both.
    """
    image = Image.new("RGB", (768, 768), (240, 240, 230))
    draw = ImageDraw.Draw(image)
    for offset in range(0, 768, 96):
        draw.line([(offset, 0), (offset, 768)], fill=(20, 20, 20), width=6)
        draw.line([(0, offset), (768, offset)], fill=(20, 20, 20), width=6)
    draw.rectangle([96, 480, 672, 672], fill=(30, 30, 30))
    return _to_png(image)


def build_canny_map(source_png: bytes) -> bytes:
    """The control map, derived here rather than by the server.

    The spike uploads a finished map and sets no `preprocess` block, so the
    render depends on no preprocessing dependency. Canny needs OpenCV only.
    """
    import cv2
    import numpy as np

    with Image.open(io.BytesIO(source_png)) as image:
        array = np.array(image.convert("RGB"))
    edges = cv2.Canny(array, 100, 200)
    stacked = np.stack([edges, edges, edges], axis=-1)
    return _to_png(Image.fromarray(stacked))


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


def attachment(map_ref: str, *, control_type: str = "canny", strength: float = FULL_STRENGTH,
               attachment_id: str = "cn_1", model_id: str | None = None) -> dict:
    return {
        "attachment_id": attachment_id,
        "control_type": control_type,
        "model_id": model_id or f"sd15-{'openpose' if control_type == 'pose' else control_type}",
        "map_asset_ref": map_ref,
        "strength": strength,
    }


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
        controlnet_chunk = image.info.get("controlnet")
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
        "controlnet_chunk": json.loads(controlnet_chunk) if controlnet_chunk else None,
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
    parser.add_argument("--out", default="/tmp/mps_controlnet")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/v1/ws"

    report: dict = {
        "issue": "STABL-xomvolgw",
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "renders": {},
        "scope": {},
    }
    try:
        report["models_status"] = get_json(base_url + "/api/models/status")
    except Exception as exc:
        report["models_status"] = {"error": str(exc)}

    source_png = build_source_image()
    canny_png = build_canny_map(source_png)
    alt_source_png = build_alt_source_image()
    alt_canny_png = build_canny_map(alt_source_png)
    assert canny_png != alt_canny_png, "the two control maps must differ"
    for name, data in (
        ("source", source_png),
        ("canny_map", canny_png),
        ("alt_source", alt_source_png),
        ("alt_canny_map", alt_canny_png),
    ):
        path = f"{args.out}_{name}.png"
        with open(path, "wb") as handle:
            handle.write(data)
        report[name] = {"path": path, "sha256": hashlib.sha256(data).hexdigest()}
    report["map_distance"] = mean_abs_difference(canny_png, alt_canny_png)

    # type=canny routes each map to the durable control_map bucket (STABL-kcjkrpry).
    upload = post_multipart(base_url + "/v1/upload", "canny.png", canny_png, {"type": "canny"})
    alt_upload = post_multipart(
        base_url + "/v1/upload", "canny_alt.png", alt_canny_png, {"type": "canny"}
    )
    report["upload"] = upload
    report["alt_upload"] = alt_upload
    map_ref = upload["fileRef"]
    alt_map_ref = alt_upload["fileRef"]
    assert map_ref != alt_map_ref, "the two maps must be separate assets"

    jobs = {
        "txt2img": {},
        "controlnet_full": {"controlnets": [attachment(map_ref, strength=FULL_STRENGTH)]},
        # Same strength, same prompt, same seed. Only the map's content differs,
        # so a difference here is the content steering the render.
        "controlnet_alt_map": {
            "controlnets": [attachment(alt_map_ref, strength=FULL_STRENGTH)]
        },
        "controlnet_zero": {"controlnets": [attachment(map_ref, strength=ZERO_STRENGTH)]},
    }
    pngs: dict[str, bytes] = {}
    for name, params in jobs.items():
        result = render(base_url, ws_url, params)
        if not result.get("ok"):
            report["renders"][name] = result
            print(json.dumps(report, indent=2))
            print(f"FAIL: render '{name}' did not complete", file=sys.stderr)
            return 1
        png = result.pop("png")
        pngs[name] = png
        path = f"{args.out}_{name}.png"
        with open(path, "wb") as handle:
            handle.write(png)
        report["renders"][name] = {**result, "path": path, **describe(png)}

    # Signal 6. Each of these must be refused, and refused by the platform
    # scope rather than by the mode policy.
    refusals = {
        "depth": {"controlnets": [attachment(map_ref, control_type="depth")]},
        "two_attachments": {
            "controlnets": [
                attachment(map_ref, attachment_id="cn_1"),
                attachment(map_ref, attachment_id="cn_2"),
            ]
        },
    }
    for name, params in refusals.items():
        result = render(base_url, ws_url, params)
        terminal = result.get("terminal") or {}
        message = terminal.get("error", "")
        expected = SCOPE_MESSAGES[name]
        report["scope"][name] = {
            "terminal_type": terminal.get("type"),
            "message": message,
            "expected": expected,
            # Not "did it fail". A timeout or a mode-policy rejection also fails.
            "refused_by_the_platform": (
                not result.get("ok")
                and terminal.get("type") == "job:error"
                and expected in message
            ),
        }

    full = report["renders"]["controlnet_full"]
    report["distances"] = {
        "full_vs_txt2img": mean_abs_difference(pngs["controlnet_full"], pngs["txt2img"]),
        "full_vs_alt_map": mean_abs_difference(
            pngs["controlnet_full"], pngs["controlnet_alt_map"]
        ),
        "full_vs_zero": mean_abs_difference(pngs["controlnet_full"], pngs["controlnet_zero"]),
        "zero_vs_txt2img": mean_abs_difference(pngs["controlnet_zero"], pngs["txt2img"]),
        "alt_map_vs_txt2img": mean_abs_difference(
            pngs["controlnet_alt_map"], pngs["txt2img"]
        ),
    }
    distances = report["distances"]
    checks = {
        "admitted": full["sha256"] is not None,
        "not_txt2img": distances["full_vs_txt2img"] > MIN_RENDER_DISTANCE,
        # The content signal. Two maps, one strength, one seed.
        "map_content_steers": distances["full_vs_alt_map"] > MIN_RENDER_DISTANCE,
        "strength_reaches_pipeline": distances["full_vs_zero"] > MIN_RENDER_DISTANCE,
        "valid_renders": all(report["renders"][name]["valid"] for name in jobs),
        "controlnet_chunk": (
            full["controlnet_chunk"] is not None
            and full["controlnet_chunk"][0]["control_type"] == "canny"
            and full["controlnet_chunk"][0]["generation"]["model_id"] == "sd15-canny"
        ),
        "scope_refuses_depth": report["scope"]["depth"]["refused_by_the_platform"],
        "scope_refuses_two": report["scope"]["two_attachments"]["refused_by_the_platform"],
    }
    report["checks"] = checks
    report["result"] = "PASS" if all(checks.values()) else "FAIL"
    print(json.dumps(report, indent=2))
    if report["result"] != "PASS":
        failed = [name for name, ok in checks.items() if not ok]
        print(f"FAIL: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"PASS: two maps at strength {FULL_STRENGTH} differ by "
          f"{distances['full_vs_alt_map']}; strength {FULL_STRENGTH} differs "
          f"from {ZERO_STRENGTH} by {distances['full_vs_zero']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
