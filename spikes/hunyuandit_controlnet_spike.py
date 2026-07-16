#!/usr/bin/env python
"""Throwaway spike: prove HunyuanDiT ControlNet imports, composes via from_pipe,
and generates a Canny-conditioned image on CUDA.

FP: STABL-ichgkgno
Spec: docs/superpowers/specs/2026-07-15-hunyuandit-controlnet-spike-design.md

NOT production code. Must not import from server/ or backends/.
Exit codes: 0 success, 2 import gate failed, 3 no CUDA device.
"""
from __future__ import annotations

import argparse
import sys

BASE_REPO = "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers"
CANNY_REPO = "Tencent-Hunyuan/HunyuanDiT-v1.1-ControlNet-Diffusers-Canny"


def stage(msg: str) -> None:
    print(f"[spike] {msg}", flush=True)


def import_gate() -> dict:
    """Record pins, then import the HunyuanDiT ControlNet stack."""
    import diffusers
    import transformers

    stage(f"diffusers={diffusers.__version__} transformers={transformers.__version__}")
    try:
        from diffusers import (  # noqa: F401
            HunyuanDiT2DControlNetModel,
            HunyuanDiTControlNetPipeline,
            HunyuanDiTPipeline,
        )
        from transformers import BertModel, T5EncoderModel, T5Tokenizer  # noqa: F401

        try:
            tokenizer_loader = getattr(T5Tokenizer, "from_pretrained")
        except Exception as exc:
            raise RuntimeError(
                "T5Tokenizer is not loadable; install the SentencePiece dependency"
            ) from exc
        if not callable(tokenizer_loader):
            raise RuntimeError(
                "T5Tokenizer is not loadable; install the SentencePiece dependency"
            )
    except Exception as exc:
        stage(f"IMPORT GATE FAILED: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
    stage("import gate: OK")
    return {
        "base_pipeline_cls": HunyuanDiTPipeline,
        "controlnet_model_cls": HunyuanDiT2DControlNetModel,
        "controlnet_pipeline_cls": HunyuanDiTControlNetPipeline,
    }


def load_control_map(path: str | None, size: tuple[int, int] = (1024, 1024)):
    """Load the Canny map, or synthesize a geometric edge image (white lines on
    black — a valid canny-style conditioning input) when no path is given."""
    from PIL import Image, ImageDraw

    if path is not None:
        img = Image.open(path).convert("RGB").resize(size)
        stage(f"control map: {path} -> {img.size}")
        return img
    img = Image.new("L", size, 0)
    draw = ImageDraw.Draw(img)
    w, h = size
    draw.rectangle([w // 4, h // 4, 3 * w // 4, 3 * h // 4], outline=255, width=4)
    draw.ellipse([w // 3, h // 3, 2 * w // 3, 2 * h // 3], outline=255, width=4)
    draw.line([w // 4, h // 4, 3 * w // 4, 3 * h // 4], fill=255, width=4)
    stage("control map: synthesized geometric edges (rect + ellipse + diagonal)")
    return img.convert("RGB")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imports-only",
        action="store_true",
        help="run the import gate and exit",
    )
    parser.add_argument(
        "--control-map",
        default=None,
        help="path to a Canny control-map PNG (default: synthesized)",
    )
    parser.add_argument(
        "--prompt",
        default="a photograph of a cat, high quality, detailed",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="spike_hunyuandit_out.png")
    return parser.parse_args(argv)


def run(classes: dict, args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available():
        stage("no CUDA device — spike requires the test-cuda container on an NVIDIA host")
        return 3

    device = "cuda"
    dtype = torch.float16

    stage(f"loading base: {BASE_REPO} (fp16)")
    base = classes["base_pipeline_cls"].from_pretrained(BASE_REPO, torch_dtype=dtype)
    base.to(device)

    stage(f"loading controlnet: {CANNY_REPO} (fp16)")
    controlnet = classes["controlnet_model_cls"].from_pretrained(
        CANNY_REPO, torch_dtype=dtype
    )

    stage("composing via from_pipe (production load shape)")
    pipe = classes["controlnet_pipeline_cls"].from_pipe(base, controlnet=controlnet)
    pipe.to(device)

    control_image = load_control_map(args.control_map)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    torch.cuda.reset_peak_memory_stats()
    stage(f"generating: steps={args.steps} seed={args.seed} 1024x1024 binning=True")
    result = pipe(
        prompt=args.prompt,
        control_image=control_image,  # the one per-family kwarg divergence
        height=1024,
        width=1024,
        num_inference_steps=args.steps,
        use_resolution_binning=True,
        generator=generator,
    )
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    result.images[0].save(args.out)
    stage(f"saved: {args.out}")
    stage(f"peak VRAM: {peak_gib:.2f} GiB")
    return 0


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    classes = import_gate()
    if args.imports_only:
        return 0
    return run(classes, args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
