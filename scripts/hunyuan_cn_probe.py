"""Standalone HunyuanDiT ControlNet probe (STABL-ichgkgno).

Bypasses all app plumbing — no WorkerPool, no admission, no registry, no asset
store. Exercises only: base pipe load, ControlNet load, `from_pipe`, call
kwargs, output.

Baseline (synthetic control map, matches the validated reference run):

    docker compose -f docker-compose.test.yml run --rm test-cuda \
        python /app/scripts/hunyuan_cn_probe.py

Scale sweep — `0.0` coherent with nonzero garbled isolates the residual path:

    CONTROL_SCALE=0.0 python /app/scripts/hunyuan_cn_probe.py
    CONTROL_SCALE=0.1 python /app/scripts/hunyuan_cn_probe.py
    CONTROL_SCALE=1.0 python /app/scripts/hunyuan_cn_probe.py

Replay a live worker's dumped control image through this same path. Set
HUNYUAN_DEBUG_DUMP=1 on an acceptance run, then point CONTROL_IMAGE at the
resulting PNG. A clean replay clears the image bytes and implicates pipe state
(scheduler, conditioning seam, process history, cache); a garbled replay
convicts the image path:

    CONTROL_IMAGE=/app/logs/hunyuan_debug/<job_id>/control_image.png \
        python /app/scripts/hunyuan_cn_probe.py
"""

import os
import time
import warnings
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from diffusers import (
    HunyuanDiT2DControlNetModel,
    HunyuanDiTControlNetPipeline,
    HunyuanDiTPipeline,
)

MODEL = os.getenv("HUNYUAN_MODEL", "/models/diffusers/HunyuanDiT-v1.1-Diffusers")
CONTROLNET = os.getenv(
    "HUNYUAN_CONTROLNET", "/models/controlnets/HunyuanDiT-v1.1-ControlNet-Canny"
)
CONTROL_IMAGE = os.getenv("CONTROL_IMAGE")
OUT = Path(os.getenv("OUT_DIR", "/store"))
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda:0"
dtype = torch.float16
seed = int(os.getenv("SEED", "1337"))
width = height = 1024
steps = int(os.getenv("STEPS", "30"))
guidance = float(os.getenv("GUIDANCE", "5.0"))
scale = float(os.getenv("CONTROL_SCALE", "1.0"))
prompt = os.getenv("PROMPT", "person grinning wearing mesh cap, profile view")
negative_prompt = os.getenv("NEGATIVE_PROMPT", "blurry, noisy, low quality")


def synthetic_canny():
    img = Image.new("RGB", (width, height), "black")
    d = ImageDraw.Draw(img)
    d.rectangle((160, 180, 864, 850), outline="white", width=8)
    d.line((160, 850, 512, 180, 864, 850), fill="white", width=8)
    d.line((250, 720, 760, 720), fill="white", width=5)
    return img


if CONTROL_IMAGE:
    # Replay path: use the bytes exactly as dumped, no regeneration.
    control_image = Image.open(CONTROL_IMAGE).convert("RGB")
    print(f"control_image=replay:{CONTROL_IMAGE} size={control_image.size}")
    label = "replay"
else:
    control_image = synthetic_canny()
    print("control_image=synthetic")
    label = "synthetic"

control_image.save(OUT / f"control_image_{label}.png")

print("loading base", MODEL)
pipe = HunyuanDiTPipeline.from_pretrained(MODEL, torch_dtype=dtype)
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

# Critical Hunyuan behavior: do not enable xformers or attention slicing. Both
# substitute an attention processor that drops cross_attention_kwargs
# ["image_rotary_emb"], leaving the transformer without positional information.
pipe = pipe.to(device)

print("loading controlnet", CONTROLNET)
controlnet = HunyuanDiT2DControlNetModel.from_pretrained(
    CONTROLNET,
    torch_dtype=dtype,
    local_files_only=True,
).to(device)

cn_pipe = HunyuanDiTControlNetPipeline.from_pipe(pipe, controlnet=controlnet)

print(f"scheduler={type(cn_pipe.scheduler).__name__}")

gen = torch.Generator(device=device).manual_seed(seed)
torch.cuda.reset_peak_memory_stats()
started = time.monotonic()

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    image = cn_pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        use_resolution_binning=True,
        generator=gen,
        control_image=control_image,
        controlnet_conditioning_scale=scale,
    ).images[0]

elapsed = time.monotonic() - started
peak = torch.cuda.max_memory_allocated()

path = OUT / f"hunyuan_cn_{label}_scale_{scale}.png"
image.save(path)

print(f"output={path}")
print(f"elapsed_s={elapsed:.2f}")
print(f"peak_allocated_bytes={int(peak)}")
for w in caught:
    msg = str(w.message)
    if "ignored" in msg or "cross_attention_kwargs" in msg or "image_rotary_emb" in msg:
        print("WARNING:", msg)
