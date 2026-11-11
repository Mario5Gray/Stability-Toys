#!/usr/bin/env python3
"""Generate a pose map from an image for use with ControlNet."""

import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

from cn_metadata import build_map_metadata, save_with_metadata

try:
    import torch
except ModuleNotFoundError as exc:
    # torch is not a global dependency. Only the pose extra installs it.
    # A missing dependency of an installed torch is a different failure. Re-raise it.
    if exc.name != "torch":
        raise
    raise SystemExit(
        "st-pose-map needs torch. Install the pose extra: pip install \"./scripts[pose]\""
    ) from exc
if not hasattr(torch, "float8_e8m0fnu"):
  setattr(torch, "float8_e8m0fnu", torch.float32)


MEDIAPIPE_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)


def load_image(path: Path, max_res: int | None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max_res:
        w, h = img.size
        scale = max_res / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def openpose(img: Image.Image, parts: set[str]) -> Image.Image:
    from controlnet_aux import OpenposeDetector

    detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    return detector(
        img,
        include_body="body" in parts,
        include_face="face" in parts,
        include_hand="hands" in parts,
    )


def dwpose(img: Image.Image) -> Image.Image:
    try:
        from controlnet_aux import DWposeDetector

        detector = DWposeDetector()
    except (ImportError, NameError) as exc:
        raise SystemExit(
            "st-pose-map: dwpose needs mmdet and mmpose. "
            "Install them, or use --model openpose or mediapipe."
        ) from exc
    return detector(img)


def mediapipe_model_path() -> Path:
    configured_path = os.environ.get("ST_MEDIAPIPE_POSE_MODEL")
    if configured_path:
        model_path = Path(configured_path).expanduser()
        if model_path.is_file():
            return model_path
        raise SystemExit(
            f"st-pose-map: ST_MEDIAPIPE_POSE_MODEL does not name a file: {model_path}"
        )

    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    model_path = cache_root / "st-controlnet-helpers" / "pose_landmarker_heavy.task"
    if model_path.is_file():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = model_path.with_suffix(".task.download")
    try:
        with urllib.request.urlopen(MEDIAPIPE_POSE_MODEL_URL, timeout=60) as response:
            with temporary_path.open("wb") as model_file:
                shutil.copyfileobj(response, model_file)
        temporary_path.replace(model_path)
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise SystemExit(
            "st-pose-map: unable to download the MediaPipe pose model. "
            "Set ST_MEDIAPIPE_POSE_MODEL to a local pose_landmarker_heavy.task file."
        ) from exc
    return model_path


def draw_mediapipe_landmarks(
    img: Image.Image,
    landmarks: list[list[object]],
    connections: list[object],
    overlay: bool,
    show_keypoints_only: bool,
) -> Image.Image:
    canvas = img.copy() if overlay else Image.new("RGB", img.size)
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    line_width = max(2, round(min(width, height) / 256))
    radius = max(3, line_width + 1)

    def point(landmark: object) -> tuple[int, int]:
        x = min(max(round(getattr(landmark, "x") * (width - 1)), 0), width - 1)
        y = min(max(round(getattr(landmark, "y") * (height - 1)), 0), height - 1)
        return x, y

    for pose_landmarks in landmarks:
        if not show_keypoints_only:
            for connection in connections:
                start = getattr(connection, "start")
                end = getattr(connection, "end")
                draw.line(
                    (point(pose_landmarks[start]), point(pose_landmarks[end])),
                    fill="white",
                    width=line_width,
                )
        for landmark in pose_landmarks:
            x, y = point(landmark)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="white")
    return canvas


def mediapipe(img: Image.Image, overlay: bool, show_keypoints_only: bool) -> Image.Image:
    import mediapipe as mp
    import numpy as np

    tasks = mp.tasks
    vision = tasks.vision
    options = vision.PoseLandmarkerOptions(
        base_options=tasks.BaseOptions(model_asset_path=str(mediapipe_model_path()))
    )
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(img))
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        results = landmarker.detect(mp_image)
    return draw_mediapipe_landmarks(
        img,
        results.pose_landmarks,
        vision.PoseLandmarksConnections.POSE_LANDMARKS,
        overlay,
        show_keypoints_only,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a pose map for ControlNet.")
    parser.add_argument("source", type=Path, help="Input image path")
    parser.add_argument("destination", type=Path, help="Output pose map path (PNG recommended)")
    parser.add_argument(
        "--model",
        choices=["openpose", "dwpose", "mediapipe"],
        default="dwpose",
        help="Pose estimation model (default: dwpose)",
    )
    parser.add_argument(
        "--parts",
        default="body,face,hands",
        metavar="LIST",
        help="Comma-separated parts to include for openpose: body,face,hands (default: all)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Compute device. Only cpu is supported: the detectors run on cpu "
        "and this script passes no device to them (default: cpu)",
    )
    parser.add_argument(
        "--max-res",
        type=int,
        default=None,
        metavar="PX",
        help="Cap longest edge before inference (e.g. 768)",
    )
    parser.add_argument(
        "--show-keypoints",
        action="store_true",
        help="Draw raw keypoint dots only, no limb connections (mediapipe only)",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Draw skeleton on the original image instead of a black background (mediapipe only)",
    )
    args = parser.parse_args()

    if not args.source.exists():
        print(f"error: source not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    device = args.device.strip().lower()
    if device != "cpu":
        print(
            f"error: --device {args.device}: pose detection runs on cpu. "
            "This script passes no device to the detectors.",
            file=sys.stderr,
        )
        sys.exit(1)

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    parts = {p.strip().lower() for p in args.parts.split(",")}

    print(f"loading  {args.source}")
    img = load_image(args.source, args.max_res)
    print(f"running  {args.model}")

    if args.model == "openpose":
        result = openpose(img, parts)
    elif args.model == "dwpose":
        result = dwpose(img)
    else:
        result = mediapipe(img, args.overlay, args.show_keypoints)

    payload = build_map_metadata(
        tool="pose_map",
        control_type="pose",
        source_size=img.size,
        params={
            "model": args.model,
            "parts": sorted(parts),
            "device": device,
            "max_res": args.max_res,
        },
    )
    save_with_metadata(result, args.destination, payload)
    print(f"saved    {args.destination}")


if __name__ == "__main__":
    main()
