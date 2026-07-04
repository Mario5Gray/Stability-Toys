#!/usr/bin/env python3
"""Generate a canny edge map from an image for use with ControlNet."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from cn_metadata import build_map_metadata, save_with_metadata


def load_image(path: Path, max_res: int | None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max_res:
        w, h = img.size
        scale = max_res / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def validate_blur(value: str) -> int:
    ivalue = int(value)
    if ivalue == 0:
        return ivalue
    if ivalue < 0 or ivalue % 2 == 0:
        raise argparse.ArgumentTypeError("--blur must be 0 or a positive odd integer")
    return ivalue


def canny_edges(
    img: Image.Image,
    *,
    low_threshold: int,
    high_threshold: int,
    blur: int,
    invert: bool,
) -> Image.Image:
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    if blur > 0:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    if invert:
        edges = 255 - edges
    return Image.fromarray(edges, mode="L")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a canny edge map for ControlNet.")
    parser.add_argument("source", type=Path, help="Input image path")
    parser.add_argument("destination", type=Path, help="Output canny map path (PNG recommended)")
    parser.add_argument(
        "--low-threshold",
        type=int,
        default=100,
        help="Low hysteresis threshold for Canny (default: 100)",
    )
    parser.add_argument(
        "--high-threshold",
        type=int,
        default=200,
        help="High hysteresis threshold for Canny (default: 200)",
    )
    parser.add_argument(
        "--blur",
        type=validate_blur,
        default=0,
        help="Optional Gaussian blur kernel size; use 0 to disable blur (default: 0)",
    )
    parser.add_argument(
        "--max-res",
        type=int,
        default=None,
        metavar="PX",
        help="Cap longest edge before processing (e.g. 1024)",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert edge polarity after Canny",
    )
    args = parser.parse_args()

    if not args.source.exists():
        print(f"error: source not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    args.destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading  {args.source}")
    img = load_image(args.source, args.max_res)
    print(
        "running  canny"
        f" (low={args.low_threshold}, high={args.high_threshold}, blur={args.blur})"
    )

    result = canny_edges(
        img,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        blur=args.blur,
        invert=args.invert,
    )
    payload = build_map_metadata(
        tool="canny_map",
        control_type="canny",
        source_size=img.size,
        params={
            "low_threshold": args.low_threshold,
            "high_threshold": args.high_threshold,
            "blur": args.blur,
            "invert": args.invert,
            "max_res": args.max_res,
        },
    )
    save_with_metadata(result, args.destination, payload)
    print(f"saved    {args.destination}")


if __name__ == "__main__":
    main()
