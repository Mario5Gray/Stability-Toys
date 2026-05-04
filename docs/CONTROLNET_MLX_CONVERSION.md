# Converting ControlNet Models to MLX (Apple Silicon)

This guide walks through producing MLX-compatible ControlNet artifacts for the four target combinations (canny + depth, SD1.5 + SDXL) so they can run natively on Apple Silicon (M1/M2/M3/M4).

The MLX path is fundamentally different from the RKNN path. RKNN converts a serialized graph (PyTorch → ONNX → `.rknn`) and runs it on a fixed-function NPU. MLX is a JIT array library: there is no graph-converter, you re-implement the model in MLX once and then load weights from any compatible source. So "convert ControlNet to MLX" is really "port ControlNet's architecture to `mlx.nn`, then load the Hugging Face weights into the ported module."

If you only need a weight-format change (e.g. PyTorch state_dict → MLX safetensors), that is one script and one short table at the bottom of this doc. If you need a working MLX ControlNet runtime, you also need the architecture port and a glue layer that wires its outputs into the UNet.

---

## Status and scope

**This guide covers:**

- The architecture-port + weight-load workflow for converting HF ControlNet checkpoints to MLX-runnable form.
- The naming / layout convention that aligns with `conf/controlnets.yaml`.
- Quantization options native to MLX (`mlx.core.quantize`, group-wise int4/int8).
- A validation checklist before trusting a converted model.

**This guide does not cover:**

- Wiring the MLX backend provider into the pipeline runtime. The current state of [backends/platforms/mlx.py](../backends/platforms/mlx.py) is a placeholder (`raise NotImplementedError("BACKEND=mlx worker factory is not implemented")`). Both the base SD/SDXL UNet *and* the ControlNet need MLX ports before any image actually generates.
- Running on Intel-Mac or non-Apple hardware. MLX is Apple Silicon only — there is no CPU/x86 fallback path.
- Distillation, LCM scheduling, or other quality optimizations. Get the conversion correct first; tune later.

If you only need CUDA inference today, stop reading; the CUDA path does not consume MLX artifacts.

---

## Conversion path

```
HF ControlNet repo ──► PyTorch / diffusers ControlNetModel
                       │
                       │  state_dict export + per-layer key remap
                       ▼
                  weights.safetensors (MLX-naming)
                       │
                       │  mlx.nn module tree built from a port of ControlNetModel
                       │  weights loaded via mx.utils.tree_unflatten
                       ▼
                  MLX module instance (in-memory) ──► optional mx.quantize ──► weights.q4.safetensors
```

You produce per-id artifacts:

| Registry id (already in `conf/controlnets.yaml`) | Source HF repo | Base family |
|---|---|---|
| `sd15-canny` | `lllyasviel/sd-controlnet-canny` | SD1.5 |
| `sd15-depth` | `lllyasviel/sd-controlnet-depth` (or `lllyasviel/control_v11f1p_sd15_depth`) | SD1.5 |
| `sdxl-canny` | `diffusers/controlnet-canny-sdxl-1.0` | SDXL |
| `sdxl-depth` | `diffusers/controlnet-depth-sdxl-1.0` | SDXL |

Pin a specific HF revision sha when you actually convert so the output is reproducible.

---

## Toolchain prerequisites

Apple Silicon Mac (M1 or newer). Convert and run on the same machine — there is no host/target split.

```bash
python3 -m venv .venv-mlx
source .venv-mlx/bin/activate

# MLX moves quickly. Pin to versions you have actually validated against.
# Numbers below are illustrative — check
# https://github.com/ml-explore/mlx/releases for the version you install.
pip install \
  "mlx>=0.15" \
  "mlx-data>=0.0.2" \
  "torch>=2.1" \
  "diffusers>=0.28" \
  "transformers>=4.41" \
  "safetensors>=0.4" \
  "huggingface_hub>=0.23" \
  "numpy<2"
```

You also want a working reference path to compare against. The two practical anchors:

- [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples) — Apple's own MLX ports, including a Stable Diffusion port. Read its `stable_diffusion/` directory before you write anything yourself; many of the layer porting decisions (Conv weight layout, attention shape, scheduler integration) are already solved there. ControlNet is not currently part of that repo, but the SD UNet port is the analogue you'll mimic.
- [huggingface/diffusers](https://github.com/huggingface/diffusers) — canonical PyTorch `ControlNetModel` reference. Use it as the architecture spec and as the numerical oracle during validation.

---

## Stage 1 — Port the ControlNet architecture to MLX

This is the big-rocks step. Without it, there's no module to load weights into.

### What to port

The diffusers `ControlNetModel` is structurally a copy of the base UNet's down-blocks plus a small "controlnet_cond_embedding" tower and per-block zero-conv heads. If you have already ported the SD/SDXL UNet to MLX (which you must have, to run the base model), porting ControlNet is mostly:

1. Reuse the ported `DownBlock2D` / `CrossAttnDownBlock2D` / `MidBlock2DCrossAttn` modules from your UNet port.
2. Add `controlnet_cond_embedding`: a small `Conv2d` tower that takes the `(1, 3, H, W)` conditioning image and projects it into the latent feature space.
3. Add `controlnet_down_blocks`: a list of zero-initialized `Conv2d` heads, one per down-block residual output, that gate the residuals.
4. Add `controlnet_mid_block`: a single zero-conv head for the mid-block residual.
5. The forward pass mirrors `ControlNetModel.forward` from diffusers exactly: down through the blocks (collecting residuals), apply the per-block zero-conv heads, return `(down_block_res_samples, mid_block_res_sample)`.

### MLX-specific layer mapping

MLX `nn` and PyTorch `nn` largely look the same, with a few traps:

| PyTorch | MLX | Notes |
|---|---|---|
| `nn.Conv2d` weight `(out, in, kh, kw)` | `mlx.nn.Conv2d` weight `(out, kh, kw, in)` | **Transpose during weight load**, not at inference time. Forgetting this produces silently wrong outputs. |
| `nn.Linear` weight `(out, in)` | `mlx.nn.Linear` weight `(out, in)` | Same layout — no transpose. |
| `nn.GroupNorm` | `mlx.nn.GroupNorm` | Same. |
| `nn.SiLU` | `mlx.nn.SiLU` | Same. |
| `F.scaled_dot_product_attention` | hand-written `mx.softmax(q @ k.T / sqrt(d)) @ v` | MLX doesn't (yet, at the time you read this) ship a fused SDPA. The naive form is fine for correctness; `mx.fast.scaled_dot_product_attention` exists in newer mlx versions — check `import mlx.core as mx; help(mx.fast)` against your installed version. |
| `torch.nn.functional.interpolate(mode='nearest')` | `mlx.core.repeat` or `mlx.nn.Upsample` | The exact API has shifted across MLX versions; verify before you commit. |

### Forward shape contract (must match the PyTorch reference)

The MLX module's forward must produce output shapes identical to diffusers' `ControlNetModel`:

| Inputs | Shape (SD1.5, 512×512) | Shape (SDXL, 1024×1024) |
|---|---|---|
| `sample` | (1, 4, 64, 64) | (1, 4, 128, 128) |
| `timestep` | (1,) int32 | (1,) int32 |
| `encoder_hidden_states` | (1, 77, 768) | (1, 77, 2048) |
| `controlnet_cond` | (1, 3, 512, 512), range [0, 1] | (1, 3, 1024, 1024), range [0, 1] |
| `text_embeds` (SDXL only) | — | (1, 1280) |
| `time_ids` (SDXL only) | — | (1, 6) |

| Outputs | SD1.5 | SDXL |
|---|---|---|
| `down_block_res_samples` (tuple) | length 12, descending pyramid | length 9 |
| `mid_block_res_sample` | (1, 1280, 8, 8) | (1, 1280, 16, 16) |

These outputs feed the ported UNet's `down_block_additional_residuals` and `mid_block_additional_residual` arguments at inference time — same wiring contract as the PyTorch path.

### Layout note on Conv2d

MLX uses **NHWC** internally for `Conv2d`, while PyTorch is **NCHW**. The simplest robust approach:

- Inside the MLX modules, accept and emit NHWC throughout.
- In the public forward, accept NCHW for compatibility with diffusers test fixtures, transpose once on entry and once on exit.
- Document this contract and stick to it; don't intersperse layouts.

If you copy the layer code from `mlx-examples/stable_diffusion`, it has already chosen a convention. Match it.

---

## Stage 2 — Load weights from Hugging Face

Once the MLX module exists, weight loading is mechanical: state_dict from diffusers → key-remap → load into MLX module via `mx.utils.tree_unflatten`.

```python
# scripts/convert_controlnet_mlx.py  (illustrative, not yet checked in)
import mlx.core as mx
import mlx.utils
import torch
from diffusers import ControlNetModel

REPO_ID = "lllyasviel/sd-controlnet-canny"
REVISION = "<pin a specific commit sha>"

# 1. Load the PyTorch ControlNet
torch_cn = ControlNetModel.from_pretrained(REPO_ID, revision=REVISION, torch_dtype=torch.float32)
sd = torch_cn.state_dict()  # {"down_blocks.0.resnets.0.norm1.weight": tensor, ...}

# 2. Remap keys to your MLX port's parameter tree.
#
# This is the hand-written part. MLX modules name parameters identically to
# PyTorch, EXCEPT Conv2d weights need transposing from (O, I, kH, kW) to
# (O, kH, kW, I). Other deltas: any layer your MLX port renames (e.g. if you
# folded two blocks into one) must be remapped here.
def to_mlx(name: str, t: torch.Tensor) -> "tuple[str, mx.array]":
    arr = mx.array(t.numpy())  # CPU copy through numpy
    if "conv" in name and arr.ndim == 4:
        arr = mx.transpose(arr, (0, 2, 3, 1))  # OIHW -> OHWI
    return name, arr

remapped = dict(to_mlx(k, v) for k, v in sd.items())

# 3. Build the MLX ControlNet module (your port from Stage 1).
from your_mlx_port import MLXControlNet  # not real, replace with your module
cn = MLXControlNet(config=...)            # config matches torch_cn.config

# 4. Load remapped weights into the module.
cn.update(mlx.utils.tree_unflatten(list(remapped.items())))
mx.eval(cn.parameters())  # force materialization

# 5. Save MLX-naming safetensors.
mx.save_safetensors("sd15-canny.safetensors", dict(mx.utils.tree_flatten(cn.parameters())))
```

The `safetensors` written here are not interchangeable with the original HF file — names are MLX-port names and Conv weights are transposed. Treat them as your build artifact, not a re-shareable HF model.

---

## Stage 3 — Quantization (optional, recommended for SDXL)

MLX has built-in group-wise quantization. For SDXL ControlNet (~2.5 GB fp32, ~1.2 GB fp16), int4 grouped quant brings the file under ~400 MB and the runtime memory accordingly.

```python
import mlx.core as mx
import mlx.nn as nn

# Quantize Linear/Conv layers in place. group_size=64 and bits=4 are the
# common settings; smaller group_size yields better quality at modest size cost.
nn.quantize(cn, group_size=64, bits=4)

# Verify the quantized module produces sane outputs (Stage 5).
# Save the quantized module's parameters.
mx.save_safetensors("sd15-canny.q4.safetensors", dict(mx.utils.tree_flatten(cn.parameters())))
```

For SD1.5 ControlNet (~1.4 GB fp32, ~700 MB fp16), int4 may be unnecessary — fp16 is often the better quality/size trade.

---

## Stage 4 — Calibration (only matters if you go below int4)

MLX's group-wise quantization is data-free: it does not need a calibration set. That is one of the reasons int4 is practical here in a way it isn't on RKNN. Skip Stage 4 unless you experiment with PTQ schemes that *do* need calibration (rare on MLX; not officially supported by `mlx.nn.quantize`).

---

## Stage 5 — Validation

Before shipping a converted file:

1. **State-dict completeness:** after `cn.update(...)`, confirm every parameter the MLX module exposes was populated. A common bug: a renamed PyTorch key gets dropped during remap and the MLX layer silently keeps its random init. Compare key counts:

   ```python
   torch_keys = set(sd.keys())
   mlx_keys = {k for k, _ in mx.utils.tree_flatten(cn.parameters())}
   print("In torch only:", torch_keys - mlx_keys)
   print("In mlx only:", mlx_keys - torch_keys)
   ```

   Both diffs should be empty (after accounting for any deliberate renames).

2. **Numerical sanity (fp32):** run identical inputs through PyTorch and the MLX port. Compare `mid` and the last few `down_*` outputs:

   ```python
   torch_out = torch_cn(sample, timestep, hidden, cond, return_dict=False)
   mlx_out = cn(sample_mx, timestep_mx, hidden_mx, cond_mx)
   # Expect max abs err < 1e-4 for fp32; < 1e-2 for fp16.
   ```

   If error is much larger, the most likely culprits are: Conv weight transpose missed in remap, GroupNorm `num_groups` mismatch, attention scale factor (`1/sqrt(d)` vs `1/sqrt(head_dim)`).

3. **Quantized comparison:** repeat the numerical test against the quantized module. Per-output max abs err for int4 typically lands in the 0.1–0.5 range; visually outputs are close.

4. **End-to-end image:** plumb the MLX ControlNet outputs into the MLX UNet at sample time and render with the same prompt/seed under the CUDA reference. Side-by-side, an int4-quantized SDXL ControlNet should be recognizably the same composition.

5. **Memory check:** `print(mx.metal.get_active_memory() / 1024 / 1024, "MB")` after a full forward. SDXL ControlNet + base UNet at fp16 is the difference between "fits comfortably on 16 GB unified memory" and "swap-thrash."

6. **Latency:** record per-step latency. ControlNet runs once per UNet step; if it doubles total step latency you may want a smaller group_size for the int4 quant or a different layer mix.

---

## Output layout

Same scheme as the RKNN guide so the V2 MLX provider can drop in alongside:

```
/models/controlnets/
  sd15-canny/
    sd15-canny.safetensors        # fp16 MLX weights
    sd15-canny.q4.safetensors     # optional int4-quantized MLX weights
    metadata.json                 # {"resolution":512, "base":"sd15", "control_type":"canny", "mlx_dtype":"fp16", "quant":"none|q4_g64", "mlx_version":"<x.y.z>", "source_revision":"<sha>", "transposed_conv2d": true}
  sd15-depth/
  sdxl-canny/
  sdxl-depth/
```

`transposed_conv2d: true` flag is informational — it pins the layout convention used at conversion time so a future loader can verify it matches the runtime port. The V2 MLX provider will read `metadata.json` to pick the right precision file and to refuse loads where the conversion convention disagrees with the runtime.

`conf/controlnets.yaml` already points `path` at `/models/controlnets/<id>` (directory, not file). The V2 provider should look inside that directory for `<id>.safetensors` (or `<id>.q4.safetensors` if quantized was preferred) plus `metadata.json`. No registry edits required.

---

## Known gotchas

- **Conv weight transpose**: PyTorch `(O, I, kH, kW)` → MLX `(O, kH, kW, I)`. This is the most common silent-wrong-output bug. Cover it in the validation diff above.
- **Attention head shape**: `(B, heads, seq, head_dim)` vs `(B, seq, heads, head_dim)`. Pick one in your port and stick to it.
- **GroupNorm num_groups**: diffusers' default is `32` for most norms. If you parameterize this on the config object, double-check the config's `norm_num_groups` matches your port's expectation.
- **`controlnet_cond` range**: must be `[0, 1]` fp32 before the model. Don't pre-multiply by 2 or shift to `[-1, 1]`. Mismatch silently produces washed-out conditioning.
- **`mx.eval`**: MLX is lazy. After loading weights, before timing or accuracy checks, call `mx.eval(cn.parameters())` to force materialization. Otherwise your first inference timing will include weight upload.
- **Unified memory**: there is no host/device copy on Apple Silicon. This is a feature, not a bug, but it means a leaked reference to a large array will show up as "swap pressure" not "GPU OOM" — diagnose with `mx.metal.get_active_memory()` rather than chasing CUDA-style errors.
- **bf16 vs fp16**: MLX supports both. SDXL is more numerically stable in bf16; SD1.5 is fine in either. Pick one per file and pin it in `metadata.json`.

---

## References

- [ml-explore/mlx](https://github.com/ml-explore/mlx) — core array library and `mlx.nn`.
- [ml-explore/mlx-examples](https://github.com/ml-explore/mlx-examples) — Apple's reference ports, especially `stable_diffusion/`. The single highest-leverage thing you can read before starting this conversion.
- [huggingface/diffusers — ControlNetModel](https://huggingface.co/docs/diffusers/api/models/controlnet) — canonical PyTorch reference for the I/O contract.
- [docs/superpowers/specs/2026-04-18-controlnet-design.md](superpowers/specs/2026-04-18-controlnet-design.md) — ControlNet design spec; the V2 MLX provider seam is the consumer of the artifacts this guide produces.
- [conf/controlnets.yaml](../conf/controlnets.yaml) — model id ↔ path registry.
- [docs/CONTROLNET_RKNN_CONVERSION.md](CONTROLNET_RKNN_CONVERSION.md) — sibling guide for the RKNN target. Same registry, different conversion pipeline.

---

## Open work

This guide is a draft. Concrete follow-ups before shipping a working MLX ControlNet:

- Port the SD1.5 and SDXL UNets to MLX. Without those the ControlNet residuals have nothing to feed.
- Port `ControlNetModel` to MLX. A good first PR isolates this in `backends/mlx_controlnet.py` (or under a new `backends/mlx/` package) so it is independently reviewable from the UNet port.
- Check in `scripts/convert_controlnet_mlx.py` so the conversion is reproducible from a make target rather than copy-paste.
- Wire `MLXProvider.create_generation_runtime` so it can actually use the converted artifacts (today it raises `NotImplementedError` per [backends/platforms/mlx.py:17](../backends/platforms/mlx.py#L17)).
- Verify int4 quality on a real M-series Mac for at least one canny + SD1.5 path; record latency and quality numbers in this doc.
- Author the V2 MLX ControlNet provider per spec §5–6 once the UNet + ControlNet ports are landing cleanly.
