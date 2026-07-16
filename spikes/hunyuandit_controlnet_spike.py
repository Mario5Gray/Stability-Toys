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
        from transformers import BertModel, T5EncoderModel  # noqa: F401
    except Exception as exc:
        stage(f"IMPORT GATE FAILED: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
    stage("import gate: OK")
    return {
        "base_pipeline_cls": HunyuanDiTPipeline,
        "controlnet_model_cls": HunyuanDiT2DControlNetModel,
        "controlnet_pipeline_cls": HunyuanDiTControlNetPipeline,
    }


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


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    import_gate()
    if args.imports_only:
        return 0
    stage("generation path not implemented yet (Task 2)")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
