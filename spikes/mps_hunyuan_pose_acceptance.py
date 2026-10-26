"""STABL-ziorpomq - live MPS HunyuanDiT Pose ControlNet acceptance client.

Proves on the real M1 Max that a normal request path carries one Pose
attachment to the MPS HunyuanDiT worker, that the server Pose preprocessor
builds the map on CPU from a source image, and that the map steers the render.

TEN signals, all required. A valid PNG alone proves nothing: the server returns
a valid PNG for text-to-image too, so a dropped control map would still pass a
format check.

  1. admitted       - the WebSocket returns `job:complete` for the preprocessed
                      Pose request. The `hunyuandit` cell admits plain ControlNet.
  2. preprocess     - the completion frame carries `controlnet_artifacts` with
                      `preprocessor_id: "pose"`, the source ref, a new control
                      map ref, and `metadata.device: "cpu"`.
  3. not txt2img    - the ControlNet render differs from a text-to-image render
                      with the same prompt, size, steps, and seed.
  4. map content    - a source image and its horizontal mirror at the SAME
                      strength, prompt, and seed produce different renders.
                      This is the signal that the map's CONTENT steers the
                      output, not that the attachment merely arrived.
  5. strength       - a strength 0.0 render differs from a strength 1.0 render,
                      so the request field reaches the pipeline.
  6. valid          - each render is a 1024x1024 PNG, not black, not flat, not
                      corrupt.
  7. chunk          - the generation PNG carries the `controlnet` text chunk
                      with control type `pose` and model id `hunyuandit-pose`.
  8. map is a pose map - the map is mostly black, carries colored skeleton
                      pixels, and does not match a grayscale depth profile.
                      This separates a real pose map from a blank one. The
                      asset store has no HTTP surface, so `/storage/{ref}`
                      returns 404 for a map ref. Supply the map with `--map`:
                      re-run PosePreprocessor on the same source, on CPU.
  9. scope, two     - two attachments are refused by the platform message,
                      while conf/modes.yml permits two.
 10. scope, combined - img2img plus Pose is refused, because the third
                      capability flag on the cell stays false.

The source image MUST contain at least one person. The script checks that the
operator-supplied `--map` has pose-map shape and reports
`local_map_is_pose_map` as a failed signal when it does not. The check makes
no claim about the server's control map, because the asset store has no HTTP
surface. A source with no detected body therefore does not read as a pipeline
defect, and the local map's shape does not prove the delivered map's bytes.

This script is a client. It starts no server and imports no server module.
Start the server first:

    conda activate stability-toys
    export BACKEND=mps MODE_CONFIG_PATH=<conf copy with host model paths> STORAGE_PROVIDER=MEMORY
    export CONTROLNET_REGISTRY_PATH=/tmp/controlnets-host.yaml
    export WORKER_ISOLATION=inproc
    python -m uvicorn server.lcm_sr_server:app --host 127.0.0.1 --port 4200

Then:

    python spikes/mps_hunyuan_pose_acceptance.py --source /path/to/person.jpg \
        --map /path/to/pose_map.png --out /tmp/mps_hunyuan_pose

`CONTROLNET_REGISTRY_PATH` is read once, when the registry singleton is built,
so it must be in the server environment before the first request.

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

from PIL import Image, ImageOps, ImageStat

from mps_hunyuan_scope import (
    FAMILY_BY_MODE,
    admitted_check,
    run_controls,
    status_error_report,
    validate_status,
)

PROMPT = "a person standing in a sunlit room, photo, full body"
NEGATIVE = "blurry, watermark, text, low quality"
MODE = "HunyuanDiT"
SIZE = "1024x1024"
STEPS = 6
GUIDANCE = 6.0
SEED = 4242
FULL_STRENGTH = 1.0
ZERO_STRENGTH = 0.0

MODEL_ID = "hunyuandit-pose"
CONTROL_TYPE = "pose"
POSITIVE_RUNS = (
    "txt2img",
    "controlnet_preprocessed",
    "controlnet_alt_source",
    "controlnet_zero",
)

# Validity limits, the same ones the earlier MPS acceptances used.
MIN_DISTINCT_COLORS = 32
MIN_GRAYSCALE_STDDEV = 5.0

# Two renders count as different above this mean per-channel distance, 0 to 255.
MIN_RENDER_DISTANCE = 1.0

# A pose map is a black field with a thin colored skeleton.
MIN_MAP_BLACK_FRACTION = 0.5
MIN_MAP_COLORED_PIXELS = 200
COLORED_SATURATION = 60


def _to_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def load_source(path: str) -> bytes:
    with Image.open(path) as image:
        return _to_png(image.convert("RGB"))


def mirror_source(png: bytes) -> bytes:
    """The same photo, horizontally flipped, so the skeleton mirrors."""
    with Image.open(io.BytesIO(png)) as image:
        return _to_png(ImageOps.mirror(image.convert("RGB")))


def map_stats(png: bytes) -> dict:
    """Describe a pose map: black field, colored skeleton, and size."""
    import numpy as np

    with Image.open(io.BytesIO(png)) as image:
        array = np.array(image.convert("RGB")).astype("int16")
        size = image.size
    top = array[:, :, 0]
    mid = array[:, :, 1]
    bottom = array[:, :, 2]
    saturation = array.max(axis=2) - array.min(axis=2)
    nonblack = array.sum(axis=2) > 30
    return {
        "size": f"{size[0]}x{size[1]}",
        "nonblack_px": int(nonblack.sum()),
        "black_fraction": round(float(1 - nonblack.mean()), 4),
        "colored_px": int((saturation > COLORED_SATURATION).sum()),
        "white_px": int(((top > 200) & (mid > 200) & (bottom > 200)).sum()),
        "is_grayscale": bool((saturation.max() if saturation.size else 0) <= 8),
    }


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
    strength: float = FULL_STRENGTH,
    attachment_id: str = "cn_1",
) -> dict:
    """One Pose attachment in the server's request shape."""
    entry: dict = {
        "attachment_id": attachment_id,
        "control_type": CONTROL_TYPE,
        "model_id": MODEL_ID,
        "strength": strength,
    }
    if source_ref is not None:
        entry["source_asset_ref"] = source_ref
        entry["preprocess"] = {"id": "pose", "options": {"max_res": 1024}}
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
            "terminal": {"type": "job:complete", "error": "no outputs. Set STORAGE_PROVIDER=MEMORY."},
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


def build_refusals(source_ref: str, alt_ref: str) -> dict:
    """The negative controls this client can build, one per platform rule."""
    return {
        "two_attachments": {
            "controlnets": [
                attachment(source_ref=source_ref, attachment_id="cn_1"),
                attachment(source_ref=alt_ref, attachment_id="cn_2"),
            ]
        },
        "combined_pose": {
            "init_image_ref": source_ref,
            "controlnets": [attachment(source_ref=source_ref)],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:4200")
    parser.add_argument("--source", required=True, help="A photo with at least one person")
    parser.add_argument("--out", default="/tmp/mps_hunyuan_pose")
    parser.add_argument(
        "--map",
        required=True,
        help=(
            "A pose map produced by PosePreprocessor on the same source, on CPU. "
            "The asset store has no HTTP route, so the server map cannot be "
            "downloaded."
        ),
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/v1/ws"

    report: dict = {
        "schema": "mps_hunyuan_pose_acceptance/v1",
        "issue": "STABL-ziorpomq",
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
            "preprocess": {"id": "pose", "options": {"max_res": 1024}},
        },
        "renders": {},
        "scope": {},
    }

    # Required inputs are checked before any network call. A missing --map used
    # to raise after four MPS renders, minutes in, with no report printed
    # (STABL-vsypokqy).
    try:
        source_png = load_source(args.source)
    except Exception as exc:
        report["source"] = {"path": args.source, "error": str(exc)}
        print(json.dumps(report, indent=2))
        print(f"FAIL: --source: {exc}", file=sys.stderr)
        return 1

    try:
        with open(args.map, "rb") as handle:
            local_map = handle.read()
    except OSError as exc:
        report["local_map"] = {"path": args.map, "error": str(exc)}
        print(json.dumps(report, indent=2))
        print(f"FAIL: --map: {exc}", file=sys.stderr)
        return 1

    alt_source_png = mirror_source(source_png)
    if source_png == alt_source_png:
        print("FAIL: the mirrored source must differ", file=sys.stderr)
        return 1
    for name, data in (("source", source_png), ("alt_source", alt_source_png)):
        path = f"{args.out}_{name}.png"
        with open(path, "wb") as handle:
            handle.write(data)
        report[name] = {"path": path, "sha256": hashlib.sha256(data).hexdigest()}

    # type=image routes the source to the durable ref_image bucket. The server
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
    scope_cell: dict | None = None
    supports_combined = False
    for name, params in jobs.items():
        result = render(base_url, ws_url, params)
        if not result.get("ok"):
            report["renders"][name] = result
            print(json.dumps(report, indent=2))
            print(f"FAIL: render '{name}' did not complete", file=sys.stderr)
            return 1
        # The status route reports the ACTIVE family's scope. Read it only after
        # a HunyuanDiT render has activated that family (STABL-dvozpehl).
        if scope_cell is None:
            status_payload = None
            try:
                status_payload = get_json(base_url + "/api/models/status")
                report["models_status"] = status_payload
                scope_cell, supports_combined = validate_status(
                    status_payload, MODE
                )
                report["active_scope"] = scope_cell
                report["supports_img2img_and_controlnet"] = supports_combined
            except Exception as exc:
                report["models_status"] = status_error_report(status_payload, exc)
                print(json.dumps(report, indent=2))
                print(f"FAIL: active family status: {exc}", file=sys.stderr)
                return 1
        png = result.pop("png")
        pngs[name] = png
        path = f"{args.out}_{name}.png"
        with open(path, "wb") as handle:
            handle.write(png)
        report["renders"][name] = {**result, "path": path, **describe(png)}

    # The server-side control map. The asset store has no HTTP route, so this
    # probe is expected to fail. Record the failure rather than hide it.
    artifacts = report["renders"]["controlnet_preprocessed"]["artifacts"]
    map_ref = artifacts[0].get("asset_ref") if artifacts else None
    report["server_map_probe"] = {"asset_ref": map_ref}
    if map_ref:
        try:
            get_bytes(base_url + f"/storage/{map_ref}")
        except Exception as exc:
            report["server_map_probe"]["status"] = "not retrievable"
            report["server_map_probe"]["error"] = str(exc)
        else:
            report["server_map_probe"]["status"] = "retrievable"

    # The format record comes from the local re-run, because the server bytes
    # are not reachable. `local_map` was read before the render loop.
    with open(f"{args.out}_map_local.png", "wb") as handle:
        handle.write(local_map)
    report["local_map"] = {
        "path": f"{args.out}_map_local.png",
        "bytes": len(local_map),
        "sha256": hashlib.sha256(local_map).hexdigest(),
        **map_stats(local_map),
    }

    refusals = build_refusals(source_ref, alt_ref)
    report["scope"] = run_controls(
        scope_cell=scope_cell,
        supports_combined=supports_combined,
        controls=refusals,
        submit=lambda _name, params: render(base_url, ws_url, params),
        family=FAMILY_BY_MODE[MODE],
    )

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
    local_map = report["local_map"]
    checks = {
        "admitted": admitted_check(report["renders"], POSITIVE_RUNS, SEED),
        "preprocess": (
            len(artifacts) == 1
            and artifacts[0]["preprocessor_id"] == "pose"
            and artifacts[0]["control_type"] == CONTROL_TYPE
            and artifacts[0]["source_asset_ref"] == source_ref
            and artifacts[0]["asset_ref"] != source_ref
            and (artifacts[0].get("metadata") or {}).get("device") == "cpu"
        ),
        "not_txt2img": distances["full_vs_txt2img"] > MIN_RENDER_DISTANCE,
        # The content signal. One photo and its mirror, one strength, one seed.
        "map_content_steers": distances["full_vs_alt_source"] > MIN_RENDER_DISTANCE,
        "strength_reaches_pipeline": distances["full_vs_zero"] > MIN_RENDER_DISTANCE,
        "valid_renders": all(report["renders"][name]["valid"] for name in jobs),
        "controlnet_chunk": (
            full["controlnet_chunk"] is not None
            and full["controlnet_chunk"][0]["control_type"] == CONTROL_TYPE
            and full["controlnet_chunk"][0]["generation"]["model_id"] == MODEL_ID
        ),
        "local_map_is_pose_map": (
            local_map["black_fraction"] >= MIN_MAP_BLACK_FRACTION
            and local_map["colored_px"] >= MIN_MAP_COLORED_PIXELS
            and not local_map["is_grayscale"]
        ),
    }
    for name, entry in report["scope"].items():
        if entry["decision"] == "applies":
            key = (
                "scope_refuses_two"
                if name == "two_attachments"
                else f"scope_refuses_{name}"
            )
            checks[key] = entry["refused_by_the_platform"]
    report["checks"] = checks
    report["result"] = "PASS" if all(checks.values()) else "FAIL"
    print(json.dumps(report, indent=2))
    if report["result"] != "PASS":
        failed = [name for name, ok in checks.items() if not ok]
        print(f"FAIL: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(
        f"PASS: map content differs by {distances['full_vs_alt_source']}. "
        f"Strength {FULL_STRENGTH} differs from {ZERO_STRENGTH} by "
        f"{distances['full_vs_zero']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
