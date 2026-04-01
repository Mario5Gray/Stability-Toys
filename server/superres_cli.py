import argparse
import os
import sys
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

from server.superres_http import (
    initialize_superres_service,
    load_superres_runtime_settings,
    submit_superres,
)
from server.superres_service import resolve_superres_backend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m server.superres_cli",
        description="Run shared RKNN/CUDA super-resolution directly from the command line.",
    )
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--magnitude", type=int, default=2, help="Super-resolution passes (1..3)")
    parser.add_argument("--format", dest="out_format", default="png", choices=("png", "jpeg"))
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality (1..100)")
    return parser


def run_once(
    *,
    input_path: Path,
    output_path: Path,
    magnitude: int,
    out_format: str,
    quality: int,
    environ: Optional[Mapping[str, str]] = None,
) -> int:
    if not input_path.is_file():
        print(f"input file not found: {input_path}", file=sys.stderr)
        return 2
    if magnitude < 1 or magnitude > 3:
        print("magnitude must be 1..3", file=sys.stderr)
        return 2
    if quality < 1 or quality > 100:
        print("quality must be 1..100", file=sys.stderr)
        return 2

    env = environ or os.environ
    settings = load_superres_runtime_settings(env)
    if not settings.enabled:
        print("super-resolution is disabled (SR_ENABLED=0)", file=sys.stderr)
        return 2

    service = None
    start = time.perf_counter()
    try:
        service = initialize_superres_service(
            enabled=settings.enabled,
            backend=settings.backend,
            use_cuda=settings.use_cuda,
            sr_model_path=settings.sr_model_path,
            sr_num_workers=settings.sr_num_workers,
            sr_queue_max=settings.sr_queue_max,
            sr_input_size=settings.sr_input_size,
            sr_output_size=settings.sr_output_size,
            sr_max_pixels=settings.sr_max_pixels,
            environ=env,
        )
        if service is None:
            print("super-resolution service unavailable", file=sys.stderr)
            return 2

        output_bytes = submit_superres(
            sr_service=service,
            image_bytes=input_path.read_bytes(),
            out_format=out_format,
            quality=quality,
            magnitude=magnitude,
            queue_timeout_s=0.25,
            request_timeout_s=settings.sr_request_timeout,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(output_bytes)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        model_name = os.path.basename(getattr(service, "model_path", ""))
        backend_kind = resolve_superres_backend(backend=settings.backend, use_cuda=settings.use_cuda)
        print(
            f"superres complete backend={backend_kind} model={model_name} "
            f"magnitude={magnitude} format={out_format} output={output_path} elapsed_ms={elapsed_ms:.1f}"
        )
        return 0
    except Exception as exc:
        print(f"superres failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if service is not None:
            service.shutdown()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_once(
        input_path=Path(args.input),
        output_path=Path(args.output),
        magnitude=int(args.magnitude),
        out_format=str(args.out_format),
        quality=int(args.quality),
    )


if __name__ == "__main__":
    raise SystemExit(main())
