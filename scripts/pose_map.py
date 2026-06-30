#!/usr/bin/env python3
"""Generate a pose map from an image for use with ControlNet."""

import argparse
import sys
from pathlib import Path

from PIL import Image

import torch
if not hasattr(torch, "float8_e8m0fnu"):
  setattr(torch, "float8_e8m0fnu", torch.float32)


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
    from controlnet_aux import DWposeDetector

    detector = DWposeDetector()
    return detector(img)


def mediapipe(img: Image.Image, overlay: bool, show_keypoints_only: bool) -> Image.Image:
    import mediapipe as mp
    import numpy as np

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    arr = np.array(img)
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        results = pose.process(arr)

    canvas = arr.copy() if overlay else np.zeros_like(arr)

    if results.pose_landmarks:
        if show_keypoints_only:
            spec = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3)
            mp_draw.draw_landmarks(canvas, results.pose_landmarks, None, spec, spec)
        else:
            mp_draw.draw_landmarks(
                canvas,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_styles.get_default_pose_landmarks_style(),
            )

    return Image.fromarray(canvas)


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
        help="Compute device: cpu, cuda, mps (default: cpu)",
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

    result.save(args.destination)
    print(f"saved    {args.destination}")


if __name__ == "__main__":
    main()
