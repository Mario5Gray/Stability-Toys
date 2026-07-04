"""Generate a depth map from an image for use with ControlNet."""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from cn_metadata import build_map_metadata, save_with_metadata

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


def depth_anything(img: Image.Image, size: str, device: str) -> Image.Image:
    from transformers import pipeline

    model_id = f"depth-anything/Depth-Anything-V2-{size.capitalize()}-hf"
    pipe = pipeline("depth-estimation", model=model_id, device=device)
    return pipe(img)["depth"]


def midas(img: Image.Image, device: str) -> Image.Image:
    from controlnet_aux import MidasDetector

    detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
    return detector(img)


def zoe(img: Image.Image, device: str) -> Image.Image:
    import torch

    repo = "isl-org/ZoeDepth"
    model = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
    model = model.to(device).eval()
    tensor = (
        torch.from_numpy(np.array(img).transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    )
    tensor = tensor.to(device)
    with torch.no_grad():
        depth = model.infer(tensor).squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return Image.fromarray((depth * 255).astype(np.uint8))


def to_grayscale(depth: Image.Image) -> Image.Image:
    if depth.mode != "L":
        return depth.convert("L")
    return depth


def invert_image(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    return Image.fromarray(255 - arr)


def colorize(img: Image.Image) -> Image.Image:
    import matplotlib.cm as cm

    arr = np.array(img.convert("L")).astype(np.float32) / 255.0
    colored = (cm.jet(arr)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a depth map for ControlNet.")
    parser.add_argument("source", type=Path, help="Input image path")
    parser.add_argument("destination", type=Path, help="Output depth map path (PNG recommended)")
    parser.add_argument(
        "--model",
        choices=["depth-anything", "midas", "zoe"],
        default="depth-anything",
        help="Depth estimation model (default: depth-anything)",
    )
    parser.add_argument(
        "--size",
        choices=["small", "base", "large"],
        default="small",
        help="Model variant for depth-anything (default: small)",
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
        "--invert",
        action="store_true",
        help="Invert depth polarity (white=far instead of white=near)",
    )
    parser.add_argument(
        "--colorize",
        action="store_true",
        help="Also save a jet-colormap visualization alongside the grayscale output",
    )
    args = parser.parse_args()

    if not args.source.exists():
        print(f"error: source not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    args.destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading  {args.source}")
    img = load_image(args.source, args.max_res)
    print(f"running  {args.model}" + (f" ({args.size})" if args.model == "depth-anything" else ""))

    if args.model == "depth-anything":
        depth = depth_anything(img, args.size, args.device)
    elif args.model == "midas":
        depth = midas(img, args.device)
    else:
        depth = zoe(img, args.device)

    depth = to_grayscale(depth)

    if args.invert:
        depth = invert_image(depth)

    payload = build_map_metadata(
        tool="depth_map",
        control_type="depth",
        source_size=img.size,
        params={
            "model": args.model,
            "size": args.size,
            "device": args.device,
            "invert": args.invert,
            "max_res": args.max_res,
        },
    )
    save_with_metadata(depth, args.destination, payload)
    print(f"saved    {args.destination}")

    if args.colorize:
        color_path = args.destination.with_stem(args.destination.stem + "_color")
        colorize(depth).save(color_path)
        print(f"saved    {color_path}")


if __name__ == "__main__":
    main()
