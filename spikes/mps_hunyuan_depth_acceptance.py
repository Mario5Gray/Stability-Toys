"""STABL-wucttopg - live MPS HunyuanDiT Depth ControlNet acceptance client.

Proves on the real M1 Max that a normal request path carries one Depth
attachment to the MPS HunyuanDiT worker, that the repository Depth preprocessor
builds the control map from a source image, and that the control map steers the
render.

SIX signals, all required. A valid PNG alone proves nothing: the server returns
a valid PNG for text-to-image too, so a dropped control map would still pass a
format check.

  1. admitted     - the WebSocket returns `job:complete` for the preprocessed
                    Depth request. The `hunyuandit` cell admits plain ControlNet.
  2. preprocess   - the completion frame carries `controlnet_artifacts` with
                    `preprocessor_id: "depth"`, the source ref, and a new
                    control-map ref.
  3. not txt2img  - the ControlNet render differs from a text-to-image render
                    with the same prompt, size, steps, and seed.
  4. map content  - TWO DIFFERENT source images at the SAME strength, prompt,
                    and seed produce different renders. This is the signal that
                    the map's CONTENT steers the output.
  5. strength     - a strength 0.0 render differs from a strength 1.0 render, so
                    the request field reaches the pipeline.
  6. valid        - each render is a 1024x1024 PNG, not black, not flat, not
                    corrupt. This is where float16 on MPS is settled.
  7. chunk        - the generation PNG carries the `controlnet` text chunk with
                    the control type and the model id.
  8. scope        - Pose is refused, and two attachments are refused, EACH with
                    its own platform message, while conf/modes.yml permits
                    canny, depth, pose and one attachment.

The map-format prerequisite is settled before this script runs. The Depth
checkpoint declares `conditioning_channels: 3`. The preprocessor emits a
three-channel RGB image with R = G = B, per-image normalized to 0-255, at the
source size. The worker resizes it to the request size. Brighter means nearer
in both the preprocessor output and the checkpoint author's shipped sample map.
See the STABL-wucttopg comments for the measurements.

The zero-strength run is a cross-check. A strength of 0.0 removes the ControlNet
contribution, so its render should match the text-to-image render at the same
seed. `distances.zero_vs_txt2img` reports that distance.

This script is a client. It starts no server and imports no server module.
Start the server first:

    conda activate stability-toys
    export BACKEND=mps MODE_CONFIG_PATH=<conf copy with host model paths> STORAGE_PROVIDER=MEMORY
    export CONTROLNET_REGISTRY_PATH=/tmp/controlnets-host.yaml
    export WORKER_ISOLATION=inproc
    python -m uvicorn server.lcm_sr_server:app --host 127.0.0.1 --port 4200

Then:

    python spikes/mps_hunyuan_depth_acceptance.py --out /tmp/mps_hunyuan_depth

`CONTROLNET_REGISTRY_PATH` is read once, when the registry singleton is built,
so it must be in the server environment before the first request. Leave
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

PROMPT = "a red apple on a wooden table, soft light, photo"
NEGATIVE = "blurry, watermark, text, low quality"
MODE = "HunyuanDiT"
SIZE = "1024x1024"
STEPS = 6
GUIDANCE = 6.0
SEED = 4242
FULL_STRENGTH = 1.0
ZERO_STRENGTH = 0.0

MODEL_ID = "hunyuandit-depth"

# Validity limits, the same ones the earlier MPS acceptances used.
MIN_DISTINCT_COLORS = 32
MIN_GRAYSCALE_STDDEV = 5.0

# Two renders count as different above this mean per-channel distance, 0 to 255.
MIN_RENDER_DISTANCE = 1.0

# The platform messages, copied from server/controlnet_execution.py.
SCOPE_MESSAGES = {
    "pose": (
        "backend 'mps' does not support control_type 'pose' for family "
        "'hunyuandit'; allowed: canny, depth"
    ),
    "two_attachments": (
        "backend 'mps' admits at most 1 controlnet attachment(s) for family "
        "'hunyuandit', got 2"
    ),
}


def _to_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def build_source_image() -> bytes:
    """One deterministic source with a clear near-to-far depth order.

    A big near shape sits over a mid table, over a far wall. Depth-Anything
    reads that ordering, so the map carries structure.
    """
    image = Image.new("RGB", (1024, 1024), (200, 205, 210))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 720, 1024, 1024], fill=(150, 105, 60))
    draw.ellipse([320, 220, 700, 700], fill=(210, 40, 40))
    draw.rectangle([760, 300, 960, 700], fill=(60, 90, 160))
    return _to_png(image)


def build_alt_source_image() -> bytes:
    """A second source with a different depth layout, so the maps differ."""
    image = Image.new("RGB", (1024, 1024), (225, 225, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 1024, 380], fill=(40, 60, 90))
    draw.rectangle([120, 460, 900, 900], fill=(180, 175, 165))
    draw.ellipse([420, 560, 620, 780], fill=(230, 230, 235))
    return _to_png(image)


def map_stats(png: bytes) -> dict:
    """Describe a control map: channel count, range, and near/far polarity."""
    import numpy as np

    with Image.open(io.BytesIO(png)) as image:
        array = np.array(image.convert("RGB"))
    gray = array[:, :, 0].astype("float32")
    height = gray.shape[0]
    return {
        "size": f"{image_size(png)[0]}x{image_size(png)[1]}",
        "channels_equal": bool(
            (array[:, :, 0] == array[:, :, 1]).all()
            and (array[:, :, 1] == array[:, :, 2]).all()
        ),
        "min": float(gray.min()),
        "max": float(gray.max()),
        "mean": round(float(gray.mean()), 2),
        "top_band": round(float(gray[: height // 5].mean()), 2),
        "bottom_band": round(float(gray[int(height * 0.8) :].mean()), 2),
    }


def image_size(png: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(png)) as image:
        return image.size


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
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


def get_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def attachment(
    *,
    source_ref: str | None = None,
    control_type: str = "depth",
    strength: float = FULL_STRENGTH,
    attachment_id: str = "cn_1",
    model_id: str = MODEL_ID,
) -> dict:
    """One ControlNet attachment in the server's request shape."""
    entry: dict = {
        "attachment_id": attachment_id,
        "control_type": control_type,
        "model_id": model_id,
        "strength": strength,
    }
    if source_ref is not None:
        entry["source_asset_ref"] = source_ref
        entry["preprocess"] = {"id": "depth", "options": {}}
    return entry


def render(base_url: str, ws_url: str, params: dict, timeout_s: float = 900.0) -> dict:
    """Submit one generate job and return the PNG bytes plus the wall time."""
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
    artifacts = terminal.get("controlnet_artifacts") or []
    if terminal.get("type") != "job:complete":
        return {
            "ok": False,
            "terminal": terminal,
            "artifacts": artifacts,
            "wall_s": round(elapsed, 2),
        }
    outputs = terminal.get("outputs") or []
    if not outputs:
        return {
            "ok": False,
            "terminal": {"type": "job:complete", "error": "no outputs; set STORAGE_PROVIDER=MEMORY"},
            "artifacts": artifacts,
            "wall_s": round(elapsed, 2),
        }
    return {
        "ok": True,
        "png": get_bytes(base_url + outputs[0]["url"]),
        "wall_s": round(elapsed, 2),
        "seed": terminal.get("meta", {}).get("seed"),
        "backend": terminal.get("meta", {}).get("backend"),
        "artifacts": artifacts,
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
            and size == (1024, 1024)
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
    parser.add_argument("--out", default="/tmp/mps_hunyuan_depth")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/v1/ws"

    report: dict = {
        "schema": "mps_hunyuan_depth_acceptance/v1",
        "issue": "STABL-wucttopg",
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
            "model_id": MODEL_ID, "full_strength": FULL_STRENGTH,
            "preprocess": {"id": "depth", "options": {}},
        },
        "renders": {},
        "scope": {},
    }
    try:
        report["models_status"] = get_json(base_url + "/api/models/status")
    except Exception as exc:
        report["models_status"] = {"error": str(exc)}

    source_png = build_source_image()
    alt_source_png = build_alt_source_image()
    if source_png == alt_source_png:
        print("FAIL: the two source images must differ", file=sys.stderr)
        return 1
    for name, data in (
        ("source", source_png),
        ("alt_source", alt_source_png),
    ):
        path = f"{args.out}_{name}.png"
        with open(path, "wb") as handle:
            handle.write(data)
        report[name] = {"path": path, "sha256": hashlib.sha256(data).hexdigest()}

    # type=image routes the source to the durable ref_image bucket; the server
    # preprocessor reads it and writes the control map to the control_map bucket.
    source_upload = post_multipart(base_url + "/v1/upload", "source.png", source_png, {"type": "image"})
    alt_upload = post_multipart(
        base_url + "/v1/upload", "source_alt.png", alt_source_png, {"type": "image"}
    )
    report["uploads"] = {"source": source_upload, "alt_source": alt_upload}
    source_ref = source_upload["fileRef"]
    alt_ref = alt_upload["fileRef"]

    jobs = {
        "txt2img": {},
        "controlnet_preprocessed": {"controlnets": [attachment(source_ref=source_ref)]},
        "controlnet_alt_source": {"controlnets": [attachment(source_ref=alt_ref)]},
        "controlnet_zero": {
            "controlnets": [attachment(source_ref=source_ref, strength=ZERO_STRENGTH)]
        },
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

    # The server-side control maps, for the format record.
    artifacts = report["renders"]["controlnet_preprocessed"]["artifacts"]
    report["control_maps"] = {}
    for entry in artifacts:
        asset_ref = entry.get("asset_ref")
        if not asset_ref:
            continue
        try:
            map_png = get_bytes(base_url + f"/storage/{asset_ref}")
        except Exception as exc:
            report["control_maps"]["error"] = str(exc)
            continue
        inner = asset_ref.rsplit("/", 1)[-1]
        path = f"{args.out}_map_{inner}.png"
        with open(path, "wb") as handle:
            handle.write(map_png)
        report["control_maps"][asset_ref] = {"path": path, **map_stats(map_png)}

    refusals = {
        "pose": {
            "controlnets": [
                attachment(
                    source_ref=source_ref,
                    control_type="pose",
                    model_id="hunyuandit-pose",
                )
            ]
        },
        "two_attachments": {
            "controlnets": [
                attachment(source_ref=source_ref, attachment_id="cn_1"),
                attachment(source_ref=alt_ref, attachment_id="cn_2"),
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

    full = report["renders"]["controlnet_preprocessed"]
    report["distances"] = {
        "full_vs_txt2img": mean_abs_difference(
            pngs["controlnet_preprocessed"], pngs["txt2img"]
        ),
        "full_vs_alt_source": mean_abs_difference(
            pngs["controlnet_preprocessed"], pngs["controlnet_alt_source"]
        ),
        "full_vs_zero": mean_abs_difference(
            pngs["controlnet_preprocessed"], pngs["controlnet_zero"]
        ),
        "zero_vs_txt2img": mean_abs_difference(
            pngs["controlnet_zero"], pngs["txt2img"]
        ),
        "alt_source_vs_txt2img": mean_abs_difference(
            pngs["controlnet_alt_source"], pngs["txt2img"]
        ),
    }
    distances = report["distances"]
    artifacts = full["artifacts"]
    checks = {
        "admitted": full["sha256"] is not None,
        "preprocess": (
            len(artifacts) == 1
            and artifacts[0]["preprocessor_id"] == "depth"
            and artifacts[0]["control_type"] == "depth"
            and artifacts[0]["source_asset_ref"] == source_ref
            and artifacts[0]["asset_ref"] != source_ref
        ),
        "not_txt2img": distances["full_vs_txt2img"] > MIN_RENDER_DISTANCE,
        # The content signal. Two source images, one strength, one seed.
        "map_content_steers": distances["full_vs_alt_source"] > MIN_RENDER_DISTANCE,
        "strength_reaches_pipeline": distances["full_vs_zero"] > MIN_RENDER_DISTANCE,
        "valid_renders": all(report["renders"][name]["valid"] for name in jobs),
        "controlnet_chunk": (
            full["controlnet_chunk"] is not None
            and full["controlnet_chunk"][0]["control_type"] == "depth"
            and full["controlnet_chunk"][0]["generation"]["model_id"] == MODEL_ID
        ),
        "scope_refuses_pose": report["scope"]["pose"]["refused_by_the_platform"],
        "scope_refuses_two": report["scope"]["two_attachments"]["refused_by_the_platform"],
    }
    report["checks"] = checks
    report["result"] = "PASS" if all(checks.values()) else "FAIL"
    print(json.dumps(report, indent=2))
    if report["result"] != "PASS":
        failed = [name for name, ok in checks.items() if not ok]
        print(f"FAIL: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(
        f"PASS: map content differs by {distances['full_vs_alt_source']}; "
        f"strength {FULL_STRENGTH} differs from {ZERO_STRENGTH} by "
        f"{distances['full_vs_zero']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
