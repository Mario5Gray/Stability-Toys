"""Throwaway spike: Z-Image Qwen text-encoder placement on one 24 GB GPU.

FP issue: STABL-ruxdbohc (parent STABL-kbzlfbts). No spec yet. The result feeds
the Z-Image design spec. Nothing here is production code.

Run one arm per process so that each peak-VRAM figure starts from zero.

Arms:
  resident  All modules bf16 on the GPU. Writes reference embeddings.
  int8      Qwen encoder quantized to int8 (optimum-quanto), all on the GPU.
            Saves the int8 encoder to disk. Compares embeddings with `resident`.
  reload    Reloads the saved int8 encoder. Compares embeddings with `int8`.
  swap      Encoder encodes on the GPU, then moves to host RAM before denoise.

Each arm writes `<out>/<arm>.json` and images `<out>/<arm>_<size>.png`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import torch

PROMPT = (
    'A weathered wooden sign beside a mountain trail. The sign reads '
    '"SUMMIT 2.4 KM" in white painted letters, and below it in smaller letters '
    '"Keep to the marked path". Morning fog, pine trees, photorealistic.'
)
SEED = 42
STEPS = 9  # 8 transformer forwards for Turbo
GUIDANCE = 0.0
DEV = torch.device("cuda")
DTYPE = torch.bfloat16
GIB = 1024**3


def gib(n: int) -> float:
    return round(n / GIB, 2)


def vram() -> dict:
    free, total = torch.cuda.mem_get_info()
    return {
        "peak_allocated_gib": gib(torch.cuda.max_memory_allocated()),
        "peak_reserved_gib": gib(torch.cuda.max_memory_reserved()),
        "now_allocated_gib": gib(torch.cuda.memory_allocated()),
        "device_used_gib": gib(total - free),
    }


def reset_peak() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


class Timer:
    def __enter__(self):
        torch.cuda.synchronize()
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        self.s = round(time.perf_counter() - self.t, 2)


def encode(pipe, device) -> list[torch.Tensor]:
    out = pipe.encode_prompt(PROMPT, device=device, do_classifier_free_guidance=False)
    embeds = out[0] if isinstance(out, tuple) else out
    return embeds


def generate(pipe, embeds, size: int, out: Path, arm: str) -> dict:
    """Run latent-only first, then the full call, to split denoise from decode."""
    res: dict = {"size": size}
    try:
        common = dict(
            prompt_embeds=embeds,
            height=size,
            width=size,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
        )
        reset_peak()
        with Timer() as t:
            pipe(**common, generator=torch.Generator("cuda").manual_seed(SEED), output_type="latent")
        res["denoise_s"] = t.s
        res["denoise"] = vram()
        reset_peak()
        with Timer() as t:
            image = pipe(**common, generator=torch.Generator("cuda").manual_seed(SEED)).images[0]
        res["full_s"] = t.s
        res["full"] = vram()
        image.save(out / f"{arm}_{size}.png")
    except Exception as e:  # record, do not stop the arm
        res["error"] = repr(e)[:400]
        torch.cuda.empty_cache()
    return res


def embed_compare(a: list[torch.Tensor], b: list[torch.Tensor]) -> dict:
    x, y = a[0].float().cpu(), b[0].float().cpu()
    if x.shape != y.shape:
        return {"shape_a": list(x.shape), "shape_b": list(y.shape)}
    cos = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    return {
        "tokens": x.shape[0],
        "cosine_mean": round(cos.mean().item(), 5),
        "cosine_min": round(cos.min().item(), 5),
        "max_abs_diff": round((x - y).abs().max().item(), 5),
        "ref_abs_mean": round(x.abs().mean().item(), 5),
    }


def load_pipe(model: str, **kw):
    from diffusers import ZImagePipeline

    return ZImagePipeline.from_pretrained(model, torch_dtype=DTYPE, **kw)


def arm_resident(model: str, out: Path, sizes: list[int]) -> dict:
    r: dict = {}
    with Timer() as t:
        pipe = load_pipe(model).to(DEV)
    r["load_s"], r["after_load"] = t.s, vram()
    with Timer() as t:
        embeds = encode(pipe, DEV)
    r["encode_s"], r["after_encode"] = t.s, vram()
    with Timer() as t:
        encode(pipe, DEV)
    r["encode_warm_s"] = t.s
    torch.save([e.cpu() for e in embeds], out / "embeds_resident.pt")
    r["runs"] = [generate(pipe, embeds, s, out, "resident") for s in sizes]
    return r


def _load_encoder(model: str):
    from transformers import AutoModel

    return AutoModel.from_pretrained(Path(model) / "text_encoder", torch_dtype=DTYPE)


def arm_int8(model: str, out: Path, sizes: list[int]) -> dict:
    from optimum.quanto import freeze, qint8, quantization_map, quantize
    from safetensors.torch import save_file

    r: dict = {}
    te = _load_encoder(model)
    with Timer() as t:
        quantize(te, weights=qint8)
        te.to(DEV)
        freeze(te)
    r["quantize_s"] = t.s
    r["encoder_only"] = vram()

    qdir = out / "qwen_int8"
    qdir.mkdir(exist_ok=True)
    with Timer() as t:
        save_file({k: v.cpu().contiguous() for k, v in te.state_dict().items()}, qdir / "model.safetensors")
        (qdir / "quanto_qmap.json").write_text(json.dumps(quantization_map(te)))
        te.config.save_pretrained(qdir)
    r["save_s"] = t.s
    r["saved_gib"] = gib((qdir / "model.safetensors").stat().st_size)

    pipe = load_pipe(model, text_encoder=te)
    pipe.transformer.to(DEV)
    pipe.vae.to(DEV)
    r["after_load"] = vram()
    with Timer() as t:
        embeds = encode(pipe, DEV)
    r["encode_s"] = t.s
    with Timer() as t:
        encode(pipe, DEV)
    r["encode_warm_s"] = t.s
    torch.save([e.cpu() for e in embeds], out / "embeds_int8.pt")
    ref = out / "embeds_resident.pt"
    if ref.exists():
        r["vs_resident"] = embed_compare(torch.load(ref), embeds)
    r["runs"] = [generate(pipe, embeds, s, out, "int8") for s in sizes]
    return r


def arm_reload(model: str, out: Path, sizes: list[int]) -> dict:
    from accelerate import init_empty_weights
    from optimum.quanto import requantize
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel

    r: dict = {}
    qdir = out / "qwen_int8"
    with Timer() as t:
        config = AutoConfig.from_pretrained(qdir)
        with init_empty_weights():
            te = AutoModel.from_config(config)
        state = load_file(qdir / "model.safetensors")
        qmap = json.loads((qdir / "quanto_qmap.json").read_text())
        requantize(te, state, qmap, device=DEV)
        te.eval()
    r["reload_s"] = t.s
    r["encoder_only"] = vram()

    pipe = load_pipe(model, text_encoder=te)
    with Timer() as t:
        embeds = encode(pipe, DEV)
    r["encode_s"] = t.s
    ref = out / "embeds_int8.pt"
    if ref.exists():
        r["vs_int8"] = embed_compare(torch.load(ref), embeds)
    return r


def arm_swap(model: str, out: Path, sizes: list[int]) -> dict:
    r: dict = {}
    pipe = load_pipe(model)  # all in host RAM
    te = pipe.text_encoder

    moves = []
    for cycle in range(2):
        reset_peak()
        with Timer() as t_in:
            te.to(DEV)
        with Timer() as t_enc:
            embeds = encode(pipe, DEV)
        peak = vram()
        with Timer() as t_out:
            te.to("cpu")
            torch.cuda.empty_cache()
        moves.append({"cycle": cycle, "to_gpu_s": t_in.s, "encode_s": t_enc.s, "to_cpu_s": t_out.s, "encode_peak": peak})
    r["swap_cycles"] = moves
    r["after_encoder_left"] = vram()

    ref = out / "embeds_resident.pt"
    if ref.exists():
        r["vs_resident"] = embed_compare(torch.load(ref), embeds)

    # CPU encode cost, as the no-GPU-move alternative.
    try:
        t0 = time.perf_counter()
        encode(pipe, torch.device("cpu"))
        r["cpu_encode_s"] = round(time.perf_counter() - t0, 2)
    except Exception as e:
        r["cpu_encode_error"] = repr(e)[:300]
    # The pipeline reads its execution device from the first module. Detach the
    # CPU encoder so that the device is the GPU, as in the other arms.
    pipe.text_encoder = None

    with Timer() as t:
        pipe.transformer.to(DEV)
        pipe.vae.to(DEV)
    r["denoiser_to_gpu_s"], r["after_denoiser_load"] = t.s, vram()
    r["runs"] = [generate(pipe, embeds, s, out, "swap") for s in sizes]

    return r


ARMS = {"resident": arm_resident, "int8": arm_int8, "reload": arm_reload, "swap": arm_swap}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="512,1024,1536")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sizes = [int(s) for s in args.sizes.split(",")]
    free, total = torch.cuda.mem_get_info()
    result: dict = {
        "arm": args.arm,
        "model": args.model,
        "gpu": torch.cuda.get_device_name(0),
        "device_total_gib": gib(total),
        "device_used_before_gib": gib(total - free),
        "torch": torch.__version__,
        "steps": STEPS,
        "seed": SEED,
        "pid": os.getpid(),
    }
    try:
        with torch.inference_mode():
            result.update(ARMS[args.arm](args.model, out, sizes))
    except Exception as e:
        result["fatal"] = repr(e)[:600]
        result["trace_tail"] = traceback.format_exc()[-1500:]
    (out / f"{args.arm}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
