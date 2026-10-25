#!/usr/bin/env python
"""Throwaway spike: real-request-path probe for HunyuanDiT combined img2img + Depth and Pose.

FP: STABL-poidjrjv.

Purpose: prove what the real server request path does with a Depth or a Pose
ControlNet on the combined img2img operation. The production path rejects both
today. Two gates hold the same Canny-only check:
  - server/controlnet_execution.py::admit_generation_operation
  - backends/cuda_worker.py::DiffusersHunyuanDiTCudaWorker.run_job (defense in depth)

This probe patches BOTH checks in this process only. No file on disk changes.
The probe runs the production sequence: build request, admit, apply mode
defaults, enforce policy, preprocess the map, resolve bindings, then submit a
GenerationJob to the WorkerPool.

Not production code. No test coverage. Exit codes:
  0 success, 2 import gate failed, 3 no CUDA device, 4 probe assertion failed.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

ALLOWED_PROBE_TYPES = ("canny", "depth", "pose")


def stage(msg: str) -> None:
    print(f"[probe] {msg}", flush=True)


def emit_json(label: str, payload: dict[str, Any]) -> None:
    print(f"[probe-json] {label} {json.dumps(payload, sort_keys=True)}", flush=True)


def import_gate() -> None:
    import diffusers
    import transformers

    stage(f"diffusers={diffusers.__version__} transformers={transformers.__version__}")
    try:
        from backends.cuda_worker import DiffusersHunyuanDiTCudaWorker  # noqa: F401
        from backends.model_registry import ModelRegistry  # noqa: F401
        from backends.platforms.cuda import CUDAProvider  # noqa: F401
        from backends.worker_pool import GenerationJob, WorkerPool  # noqa: F401
        from server.asset_store import get_store  # noqa: F401
        from server.controlnet_execution import admit_generation_operation  # noqa: F401
        from server.controlnet_execution import resolve_controlnet_bindings  # noqa: F401
        from server.controlnet_preprocessors import DEFAULT_REGISTRY  # noqa: F401
        from server.mode_config import ModeConfigManager  # noqa: F401
        from server.ws_routes import _build_generate_request  # noqa: F401
    except Exception as exc:
        stage(f"IMPORT GATE FAILED: {type(exc).__name__}: {exc}")
        raise SystemExit(2)
    stage("import gate: OK")


def patch_canny_gate(container, function_name: str) -> Any:
    """Relax one gate's `!= "canny"` check inside this process.

    `container` is a module or a class. Returns a callable that restores the
    original function. The replacement compiles a source copy of the function.
    The only textual change is the control-type comparison. A missing expected
    pattern aborts the probe.
    """
    original = getattr(container, function_name)
    # A class method's source keeps its class-level indent. A module-level
    # function has none. Dedent both so the exec copy parses either way.
    source = textwrap.dedent(inspect.getsource(original))
    expected = '.control_type != "canny"'
    replacement = f".control_type not in {ALLOWED_PROBE_TYPES!r}"
    if source.count(expected) != 1:
        raise SystemExit(
            f"probe assertion failed: expected exactly one {expected!r} in {function_name}"
        )
    if "super(" in source:
        raise SystemExit(
            f"probe assertion failed: {function_name} uses zero-arg super. "
            "An exec copy has no __class__ cell."
        )
    patched_source = source.replace(expected, replacement)
    module = inspect.getmodule(original)
    namespace = dict(module.__dict__)
    label = f"{module.__name__}.{getattr(container, '__name__', 'module')}.{function_name}"
    exec(compile(patched_source, f"<probe-patch:{label}>", "exec"), namespace)
    patched = namespace[function_name]
    setattr(container, function_name, patched)

    def restore() -> None:
        setattr(container, function_name, original)

    stage(f"patched in-process: {label}")
    return restore


def self_check() -> int:
    """Exercise both gate patches with stubs. No CUDA, no model, no preprocessor.

    Proves three facts before any GPU time is spent:
      1. the unpatched admission gate refuses Depth,
      2. the patched admission gate admits Depth,
      3. the patched worker gate no longer raises the Canny-only error.
    """
    from server.controlnet_execution import ControlNetBinding
    from server.controlnet_models import ControlNetAttachment
    from tests.snapshot_test_helpers import make_active_snapshot, make_family_provider

    import backends.cuda_worker as cw
    import server.controlnet_execution as cne

    mode = type("Mode", (), {"name": "hunyuandit-mode"})()
    snapshot = make_active_snapshot(mode, family_id="hunyuandit")
    provider = make_family_provider(family_id="hunyuandit", supports_combined=True)
    attachment = ControlNetAttachment(
        attachment_id="cn", control_type="depth", map_asset_ref="map"
    )
    req = type("Req", (), {"controlnets": [attachment]})()

    admission_refusal = ""
    try:
        cne.admit_generation_operation(
            req, snapshot=snapshot, provider=provider, has_init_image=True
        )
        stage("SELF CHECK FAILED: unpatched admission gate admitted Depth")
        return 4
    except ValueError as exc:
        admission_refusal = str(exc)
    stage(f"unpatched admission refusal: {admission_refusal}")

    restore = patch_canny_gate(cne, "admit_generation_operation")
    try:
        admitted = cne.admit_generation_operation(
            req, snapshot=snapshot, provider=provider, has_init_image=True
        )
    finally:
        restore()
    stage(f"patched admission result: {admitted!r}")

    binding = ControlNetBinding(
        attachment_id="cn",
        control_type="depth",
        model_id="hunyuandit-depth",
        model_path="/nonexistent",
        control_image_bytes=b"x",
        strength=1.0,
        start_percent=0.0,
        end_percent=1.0,
    )
    job = type(
        "Job",
        (),
        {"req": req, "init_image": b"x", "controlnet_bindings": [binding]},
    )()

    worker_refusal = ""
    try:
        cw.DiffusersHunyuanDiTCudaWorker.run_job(object(), job)
        worker_refusal = "NO ERROR (unexpected)"
    except RuntimeError as exc:
        worker_refusal = str(exc)
    stage(f"unpatched worker refusal: {worker_refusal}")

    restore_worker = patch_canny_gate(cw.DiffusersHunyuanDiTCudaWorker, "run_job")
    after_patch_error = ""
    try:
        cw.DiffusersHunyuanDiTCudaWorker.run_job(object(), job)
        after_patch_error = "NO ERROR"
    except Exception as exc:
        after_patch_error = f"{type(exc).__name__}: {exc}"
    finally:
        restore_worker()
    stage(f"patched worker error after the gate: {after_patch_error}")

    if "Canny only" in after_patch_error:
        stage("SELF CHECK FAILED: patched worker gate still rejects Depth")
        return 4
    if "Canny only" not in worker_refusal:
        stage("SELF CHECK FAILED: unpatched worker gate did not reject Depth")
        return 4

    emit_json(
        "self_check",
        {
            "admission_unpatched": admission_refusal,
            "admission_patched": admitted,
            "worker_unpatched": worker_refusal,
            "worker_patched_then": after_patch_error,
        },
    )
    stage("self check: OK")
    return 0


def preprocess_only(args: argparse.Namespace) -> int:
    """Run the real preprocessor on the source image. No model and no CUDA."""
    from server.controlnet_preprocessors import DEFAULT_REGISTRY

    preprocessor = DEFAULT_REGISTRY.get(args.control_type)
    if preprocessor is None:
        stage(f"no preprocessor registered for {args.control_type!r}")
        return 4
    source = Path(args.source_image)
    if not source.exists():
        stage(f"source image missing: {source}")
        return 4
    started = time.monotonic()
    result = preprocessor.run(source.read_bytes(), {})
    elapsed = time.monotonic() - started
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    map_path = out_dir / f"{args.control_type}-map-preprocess-only.png"
    map_path.write_bytes(result.image_bytes)
    stage(f"saved map: {map_path}")
    emit_json(
        "preprocess",
        {
            "control_type": args.control_type,
            "preprocessor_id": result.preprocessor_id,
            "width": result.width,
            "height": result.height,
            "media_type": result.media_type,
            "metadata": result.metadata,
            "elapsed_s": round(elapsed, 2),
            "map_path": str(map_path),
        },
    )
    return 0


def _attachment_payload(control_type: str, source_ref: str, model_id: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "attachment_id": f"probe-{control_type}",
        "control_type": control_type,
        "source_asset_ref": source_ref,
        "preprocess": {"id": control_type, "options": {}},
    }
    if model_id is not None:
        payload["model_id"] = model_id
    return payload


def _write_png(path: Path, png_bytes: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes)


def run_probe(args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available():
        stage("no CUDA device. Run this stage in the test-cuda container on an NVIDIA host.")
        return 3

    from backends.cuda_worker import DiffusersHunyuanDiTCudaWorker
    from backends.model_registry import ModelRegistry
    from backends.platforms.cuda import CUDAProvider
    from backends.worker_pool import GenerationJob, WorkerPool, reset_worker_pool
    from server.asset_store import get_store
    from server.controlnet_constraints import enforce_controlnet_policy
    from server.controlnet_execution import admit_generation_operation, resolve_controlnet_bindings
    from server.controlnet_preprocessing import preprocess_controlnet_attachments
    from server.controlnet_registry import reset_controlnet_registry
    from server.generation_constraints import finalize_mode_generate_request
    from server.mode_config import ModeConfigManager
    from server.upload_routes import resolve_file_ref
    from server.ws_routes import _build_generate_request

    conf_dir = Path(args.conf_dir)
    # Override, do not setdefault: the test image mounts conf/modes-test.yml at
    # /conf, and that file has no HunyuanDiT mode.
    os.environ["MODE_CONFIG_PATH"] = str(conf_dir)
    os.environ["CONTROLNET_REGISTRY_PATH"] = str(conf_dir / "controlnets.yaml")
    os.environ["CONTROLNET_REGISTRY_VALIDATION"] = "strict"
    reset_controlnet_registry()
    reset_worker_pool()

    source_path = Path(args.source_image)
    if not source_path.exists():
        stage(f"source image missing: {source_path}")
        return 4
    map_source_path = Path(args.map_source_image) if args.map_source_image else source_path

    mode_config = ModeConfigManager(str(conf_dir))
    provider = CUDAProvider()
    pool = WorkerPool(queue_max=2, mode_config=mode_config, registry=ModelRegistry())
    store = get_store()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    restores: list[Any] = []
    try:
        stage(f"switching mode: {args.mode_name} (force)")
        pool.switch_mode(args.mode_name, force=True).result(timeout=900.0)
        snapshot = pool.admit_generation(args.mode_name)
        if snapshot is None:
            stage("admit_generation returned no snapshot")
            return 4

        init_ref = store.write("upload", source_path.read_bytes())
        map_source_ref = store.write("upload", map_source_path.read_bytes())

        params = {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "mode": args.mode_name,
            "size": args.size,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "denoise_strength": args.strength,
            "controlnets": [
                _attachment_payload(args.control_type, map_source_ref, args.model_id)
            ],
        }
        req = _build_generate_request(params)

        refusal = ""
        try:
            admit_generation_operation(
                req, snapshot=snapshot, provider=provider, has_init_image=True
            )
        except Exception as exc:
            refusal = f"{type(exc).__name__}: {exc}"
        stage(f"unpatched admission result: {refusal}")
        emit_json("admission_unpatched", {"control_type": args.control_type, "refusal": refusal})

        import server.controlnet_execution as cne

        restores.append(patch_canny_gate(cne, "admit_generation_operation"))
        # Call through the module. The local import above still points at the
        # unpatched function.
        admitted = cne.admit_generation_operation(
            req, snapshot=snapshot, provider=provider, has_init_image=True
        )
        stage(f"patched admission result: {admitted!r}")

        finalize_mode_generate_request(
            req,
            snapshot.mode,
            env_default_size=os.environ.get("DEFAULT_SIZE", "512x512"),
            env_default_steps=int(os.environ.get("DEFAULT_STEPS", "4")),
            env_default_guidance=float(os.environ.get("DEFAULT_GUIDANCE", "1.0")),
        )
        enforce_controlnet_policy(req, snapshot.mode)

        artifacts = preprocess_controlnet_attachments(req, store)
        if not artifacts:
            stage("no controlnet artifact produced. The attachment did not run the preprocessor.")
            return 4
        artifact = artifacts[0]
        map_bytes = store.resolve(artifact.asset_ref).data
        map_path = out_dir / f"hunyuandit-{args.control_type}-probe-map.png"
        _write_png(map_path, map_bytes)
        stage(f"saved map: {map_path}")
        emit_json(
            "preprocess_request_path",
            {
                "control_type": args.control_type,
                "preprocessor_id": artifact.preprocessor_id,
                "map_path": str(map_path),
            },
        )

        bindings = resolve_controlnet_bindings(
            req,
            mode=snapshot.mode,
            store=store,
            active_family=snapshot.resolved.profile.family_id,
        )
        stage(f"resolved bindings: {[(b.control_type, b.model_id) for b in bindings]}")
        if len(bindings) != 1:
            stage("probe assertion failed: expected exactly one binding")
            return 4

        init_bytes = resolve_file_ref(init_ref)
        job = GenerationJob(
            req=req,
            init_image=init_bytes,
            controlnet_bindings=bindings,
            resolution_epoch=snapshot.resolution_epoch,
        )

        worker_refusal = ""
        try:
            DiffusersHunyuanDiTCudaWorker.run_job(object(), job)
        except Exception as exc:
            worker_refusal = f"{type(exc).__name__}: {exc}"
        stage(f"unpatched worker result: {worker_refusal}")
        emit_json("worker_unpatched", {"control_type": args.control_type, "refusal": worker_refusal})

        import backends.cuda_worker as cw

        restores.append(patch_canny_gate(cw.DiffusersHunyuanDiTCudaWorker, "run_job"))

        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        png_bytes, seed = pool.submit_job(job).result(timeout=1800.0)
        elapsed = time.monotonic() - started
        peak_bytes = int(torch.cuda.max_memory_allocated())

        output_path = out_dir / f"hunyuandit-{args.control_type}-probe-{args.strength:.2f}.png"
        _write_png(output_path, png_bytes)
        stage(f"saved output: {output_path}")

        from io import BytesIO

        from PIL import Image

        image = Image.open(BytesIO(png_bytes))
        image.load()
        png_text = dict(image.text)
        controlnet_meta = png_text.get("controlnet")
        emit_json(
            "generation",
            {
                "control_type": args.control_type,
                "admitted": admitted,
                "seed": seed,
                "requested_seed": args.seed,
                "strength": args.strength,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "size": image.size,
                "elapsed_s": round(elapsed, 2),
                "peak_allocated_bytes": peak_bytes,
                "output_path": str(output_path),
                "controlnet_chunk": controlnet_meta,
            },
        )
        return 0
    finally:
        for restore in reversed(restores):
            restore()
        pool.shutdown()
        reset_worker_pool()
        reset_controlnet_registry()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports-only", action="store_true", help="run the import gate and exit")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="prove both gate patches with stubs. No CUDA needed.",
    )
    parser.add_argument(
        "--preprocess-only", action="store_true", help="run the preprocessor only. No CUDA needed."
    )
    parser.add_argument("--control-type", choices=("depth", "pose"), default="depth")
    parser.add_argument("--mode-name", default="HunyuanDiT")
    parser.add_argument("--model-id", default=None, help="ControlNet model id. Default: mode policy")
    parser.add_argument("--source-image", default="/app/logs/probe-source.png")
    parser.add_argument("--map-source-image", default=None, help="Default: --source-image")
    parser.add_argument("--prompt", default="a photograph of a person, natural light, high detail")
    parser.add_argument("--negative-prompt", default="blurry, low quality")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--strength", type=float, default=0.5)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--conf-dir", default="/app/conf")
    parser.add_argument("--out-dir", default="/app/logs/hunyuandit-depth-pose-probe")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    import_gate()
    if args.imports_only:
        return 0
    if args.self_check:
        return self_check()
    if args.preprocess_only:
        return preprocess_only(args)
    return run_probe(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
