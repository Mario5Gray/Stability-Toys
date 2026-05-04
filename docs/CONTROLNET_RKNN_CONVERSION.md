# Converting ControlNet Models to RKNN

This guide walks through converting ControlNet checkpoints (canny + depth, SD1.5 + SDXL) into RKNN-compatible models for execution on Rockchip NPUs (RK3588, RK3576). The output of this process is a set of `.rknn` files plus their accompanying I/O metadata that can later be wired into a ControlNet provider for the RKNN backend.

---

## Status and scope

**This guide covers:**

- Sourcing the four target ControlNet checkpoints from Hugging Face.
- Exporting them from PyTorch/diffusers to ONNX.
- Converting ONNX to `.rknn` for RK3588/RK3576 with `rknn-toolkit2`.
- Layout that aligns with the existing `conf/controlnets.yaml` registry.
- Validation checklist before you trust a converted model.

**This guide does not cover:**

- Wiring the RKNN ControlNet provider into the pipeline runtime — that is V2 work per [docs/superpowers/specs/2026-04-18-controlnet-design.md](superpowers/specs/2026-04-18-controlnet-design.md) §V1/V2 split. RKNN ControlNet execution is explicitly out-of-scope for v1.
- Building or operating the host Linux container that runs `rknn-toolkit2`. The toolkit is x86_64 Linux only and lives off the target board.
- End-to-end image generation on the NPU. Once `.rknn` files exist, the V2 RKNN ControlNet provider has to load them, feed conditioning, and patch UNet residuals.

If you only need CUDA inference today, stop reading; the CUDA path does not consume `.rknn` ControlNet files.

---

## Conversion path

Three stages, roughly:

```
HF ControlNet repo  ──►  PyTorch / diffusers ControlNetModel
                        │
                        │  torch.onnx.export  (opset 17, fixed shapes)
                        ▼
                   <model>.onnx
                        │
                        │  rknn-toolkit2 RKNN().load_onnx + .build + .export_rknn
                        ▼
                   <model>.rknn
```

Each `.rknn` file is **one** ControlNet (one control type, one base family). You will produce four files:

| Registry id (already in `conf/controlnets.yaml`) | Source repo | Base family |
|---|---|---|
| `sd15-canny` | `lllyasviel/sd-controlnet-canny` | SD1.5 |
| `sd15-depth` | `lllyasviel/sd-controlnet-depth` (or `lllyasviel/control_v11f1p_sd15_depth`) | SD1.5 |
| `sdxl-canny` | `diffusers/controlnet-canny-sdxl-1.0` | SDXL |
| `sdxl-depth` | `diffusers/controlnet-depth-sdxl-1.0` | SDXL |

The repo names above are the canonical Hugging Face refs; pin a specific revision when you actually convert so the output is reproducible.

---

## Toolchain prerequisites

Run all conversion steps on an **x86_64 Linux host** (Ubuntu 22.04 recommended). The Rockchip board cannot run `rknn-toolkit2` itself — it can only run the runtime side (`rknn-toolkit-lite2` or RKNPU2 C/C++).

### Host packages

```bash
# Python 3.10 or 3.11; rknn-toolkit2 has not validated 3.12 yet — check the
# toolkit's release notes for your version before assuming.
python3 -m venv .venv-rknn
source .venv-rknn/bin/activate

# Pin all of these to versions confirmed by the rknn-toolkit2 release notes
# you are using. The numbers below are illustrative — verify against
# https://github.com/airockchip/rknn-toolkit2/releases for the toolkit
# version you install.
pip install \
  "torch==2.1.*" \
  "diffusers==0.28.*" \
  "transformers==4.41.*" \
  "onnx==1.15.*" \
  "onnxruntime==1.17.*" \
  "numpy<2"

# Install rknn-toolkit2 from its official release (a wheel matching your
# Python version). Do NOT pip install rknn-toolkit2 from PyPI — the
# canonical artifacts are released directly by airockchip.
pip install path/to/rknn_toolkit2-<version>-cp310-cp310-linux_x86_64.whl
```

### Target firmware

The board must run an RKNPU2 driver/firmware whose major version matches the toolkit you converted with. Mismatched runtime/build pairs surface as "RKNN_ERR_LOAD_MODEL" or silent NaN outputs. Document the exact pairing you ship with each `.rknn` file.

---

## Stage 1 — Export ControlNet to ONNX

Each ControlNet has the same I/O shape across all four targets, with the exception that SDXL adds extra conditioning kwargs.

### Inputs

| Name | Shape (SD1.5, 512x512) | Shape (SDXL, 1024x1024) | Dtype |
|---|---|---|---|
| `sample` | (1, 4, 64, 64) | (1, 4, 128, 128) | fp32 |
| `timestep` | (1,) | (1,) | int64 |
| `encoder_hidden_states` | (1, 77, 768) | (1, 77, 2048) | fp32 |
| `controlnet_cond` | (1, 3, 512, 512) | (1, 3, 1024, 1024) | fp32, range [0, 1] |
| `text_embeds` (SDXL only) | — | (1, 1280) | fp32 |
| `time_ids` (SDXL only) | — | (1, 6) | fp32 |

RKNN will not honor dynamic axes; pick one resolution per `.rknn` file and bake it in. If you want both 512 and 768 SD1.5 modes, that is two `.rknn` files.

### Outputs

| Name | Count | Shape (per element, SD1.5 example) |
|---|---|---|
| `down_block_res_samples` | 12 (SD1.5) / 9 (SDXL) | descending feature pyramid, e.g. `(1, 320, 64, 64)` → `(1, 1280, 8, 8)` |
| `mid_block_res_sample` | 1 | smallest feature, e.g. `(1, 1280, 8, 8)` |

These outputs feed the UNet's `down_block_additional_residuals` and `mid_block_additional_residual` arguments at inference time. The V2 RKNN ControlNet provider must collect them from the `.rknn` outputs and pass them into a UNet inference call (which itself either runs on the same NPU as a separate `.rknn`, or on CPU/GPU as a fallback).

### Export script (sketch)

The example below is for SD1.5 canny. Adapt the `repo_id` and resolution constants for the other three.

```python
# scripts/export_controlnet_onnx.py  (illustrative, not yet checked in)
import torch
from diffusers import ControlNetModel

REPO_ID = "lllyasviel/sd-controlnet-canny"
REVISION = "<pin a specific commit sha for reproducibility>"
RESOLUTION = 512
LATENT_RES = RESOLUTION // 8
TEXT_DIM = 768

cn = ControlNetModel.from_pretrained(REPO_ID, revision=REVISION, torch_dtype=torch.float32)
cn.eval()

dummy = {
    "sample": torch.randn(1, 4, LATENT_RES, LATENT_RES),
    "timestep": torch.tensor([1], dtype=torch.int64),
    "encoder_hidden_states": torch.randn(1, 77, TEXT_DIM),
    "controlnet_cond": torch.rand(1, 3, RESOLUTION, RESOLUTION),
}

# diffusers ControlNetModel.forward takes positional args; use a thin wrapper
# so torch.onnx.export sees a flat input list.
class _ExportWrapper(torch.nn.Module):
    def __init__(self, cn): super().__init__(); self.cn = cn
    def forward(self, sample, timestep, encoder_hidden_states, controlnet_cond):
        out = self.cn(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            return_dict=False,
        )
        # out is (down_block_res_samples_tuple, mid_block_res_sample)
        downs, mid = out
        return (*downs, mid)

wrapper = _ExportWrapper(cn).eval()

torch.onnx.export(
    wrapper,
    (dummy["sample"], dummy["timestep"], dummy["encoder_hidden_states"], dummy["controlnet_cond"]),
    "sd15-canny.onnx",
    input_names=["sample", "timestep", "encoder_hidden_states", "controlnet_cond"],
    output_names=[*[f"down_{i}" for i in range(12)], "mid"],
    opset_version=17,
    do_constant_folding=True,
    dynamic_axes=None,  # static shapes are required for RKNN
)
```

For SDXL, extend `_ExportWrapper.forward` to accept `added_cond_kwargs={"text_embeds": ..., "time_ids": ...}` and add those as ONNX inputs. The output count drops from 12 to 9.

### ONNX sanity pass

After export, simplify and validate before handing the file to `rknn-toolkit2`. Optimizers cut graph size by ~30% and remove ops the converter cannot map.

```bash
python -m onnxruntime.tools.optimizer_cli --input sd15-canny.onnx --output sd15-canny.opt.onnx
python -c "import onnx; onnx.checker.check_model('sd15-canny.opt.onnx', full_check=True)"
```

A numerical check against the original PyTorch model is wise — drop in your own sample inputs, run both, and assert per-output `max abs err < 1e-4` (fp32 export should be that tight).

---

## Stage 2 — Convert ONNX to RKNN

```python
# scripts/convert_controlnet_rknn.py  (illustrative)
from rknn.api import RKNN

rknn = RKNN(verbose=True)

rknn.config(
    target_platform="rk3588",          # or rk3576, etc.
    mean_values=[[0.0, 0.0, 0.0]],     # controlnet_cond is already in [0,1]
    std_values=[[1.0, 1.0, 1.0]],
    quantized_dtype="asymmetric_quantized-8",  # int8 — see Stage 3 for calibration
    optimization_level=3,
)

rknn.load_onnx(model="sd15-canny.opt.onnx")
rknn.build(do_quantization=True, dataset="calibration_set.txt")
rknn.export_rknn("sd15-canny.rknn")
rknn.release()
```

Notes:

- `target_platform` must match the SoC you ship to. Mixing rk3588 and rk3576 builds is unsupported.
- `optimization_level=3` is the maximum and trades convert-time for inference speed.
- If int8 quantization degrades quality past acceptable, fall back to `quantized_dtype="float16"` and `do_quantization=False`. SDXL ControlNet at fp16 is large (~1.2 GB) — confirm the board's NPU memory budget can hold it alongside the UNet.

---

## Stage 3 — Calibration data for int8

Quality of int8-quantized ControlNet is highly sensitive to calibration coverage. Capture a representative calibration set from real diffusers runs:

1. Run 20-50 generations through the CUDA path with the **same** ControlNet you are converting.
2. At each step, dump the four input tensors (`sample`, `timestep`, `encoder_hidden_states`, `controlnet_cond`) to disk as `.npy`.
3. Build `calibration_set.txt` listing one `.npy` quad per line, in the order the ONNX model expects them. The exact format `rknn-toolkit2` expects depends on toolkit version — check the docs for `RKNN.build(dataset=...)`.

A calibration set drawn from diverse prompts and control images (varied edge density for canny, varied scene depth for depth) yields markedly better results than synthetic noise.

---

## Validation

Before you ship a converted file:

1. **Numerical sanity:** run the same inputs through PyTorch fp32, ONNX fp32, and the `.rknn` int8/fp16. Compare `mid` and the last few `down_*` outputs. Expect:
   - ONNX vs PyTorch: max abs err < 1e-4
   - RKNN fp16 vs PyTorch: max abs err < 1e-2
   - RKNN int8 vs PyTorch: max abs err < 0.5 (visually acceptable in practice; numerically loose)
2. **End-to-end image:** plumb the `.rknn` outputs into a UNet step (CPU is fine for validation) and render with the same prompt/seed under the CUDA reference. Side-by-side the two outputs — int8 should be recognizably the same composition.
3. **Memory check on board:** load the `.rknn` on the target and call `RKNN().init_runtime()` followed by one inference. Watch `dmesg` for OOM kills; SDXL ControlNet at fp16 may be the difference between "fits" and "swap-thrash."
4. **Latency:** record per-step latency for the converted model on the target. ControlNet runs once per UNet step; if it doubles total step latency, consider the fp16 → int8 trade.

---

## Output layout

The existing registry expects models at `/models/controlnets/<id>/`. RKNN models slot in with a sibling layout that the V2 provider can pick up:

```
/models/controlnets/
  sd15-canny/
    sd15-canny.rknn          # the converted model
    metadata.json            # {"resolution":512, "base":"sd15", "control_type":"canny", "rknn_target":"rk3588", "quant":"int8", "toolkit":"<version>", "source_revision":"<sha>"}
  sd15-depth/
    sd15-depth.rknn
    metadata.json
  sdxl-canny/
    sdxl-canny.rknn
    metadata.json
  sdxl-depth/
    sdxl-depth.rknn
    metadata.json
```

`metadata.json` is informational for v1 — it pins the conversion provenance so future loads can verify compatibility without re-running validation. The V2 RKNN provider will read it to gate compatibility against the active mode and the on-board firmware.

The `path` field in [conf/controlnets.yaml](../conf/controlnets.yaml) already points at `/models/controlnets/<id>` (directory, not file). When the V2 provider lands, it will look inside that directory for `<id>.rknn` plus `metadata.json`. Place the converted files accordingly so no registry edits are needed.

---

## Known gotchas

- **GroupNorm:** RKNN may map `GroupNorm` poorly. If you see large quant error on the residual outputs, check the toolkit's release notes for the GroupNorm op coverage at your toolkit version. `LayerNorm` is generally better-supported.
- **GELU/SiLU:** ControlNet uses SiLU (Swish). Older RKNN toolkits emitted approximations that hurt quality. Confirm your toolkit version maps SiLU to a hardware-accelerated path, not a CPU fallback.
- **Attention:** the cross-attention blocks are the largest contributor to int8 error. If quality is unacceptable, the typical mitigation is per-channel quantization (`asymmetric_quantized-8` with `optimization_level=3` already does this) or quantizing only the projections while leaving softmax in fp16. The latter requires hand-editing the ONNX graph.
- **Scalar timestep:** some toolkit versions choke on scalar (rank-0) inputs. Export `timestep` as `(1,)` not `()`.
- **fp16 controlnet_cond:** the conditioning image must be normalized to `[0, 1]` fp32 *before* it reaches the model. Do not pre-multiply by 2 or shift to `[-1, 1]` — diffusers ControlNet expects `[0, 1]`. Mismatch here silently produces washed-out conditioning, not a hard error.

---

## References

- [Rockchip rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) — host conversion toolchain, release notes, op coverage.
- [Rockchip rknn-toolkit-lite2](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2) — target-side runtime.
- [diffusers ControlNetModel](https://huggingface.co/docs/diffusers/api/models/controlnet) — canonical PyTorch reference for the I/O shape.
- [docs/superpowers/specs/2026-04-18-controlnet-design.md](superpowers/specs/2026-04-18-controlnet-design.md) — ControlNet design (V2 RKNN provider seam is the consumer of the `.rknn` files this guide produces).
- [conf/controlnets.yaml](../conf/controlnets.yaml) — model id ↔ path registry.

---

## Open work

This guide is a draft. Concrete follow-ups before shipping a working RKNN ControlNet on the target:

- Pin and check in `scripts/export_controlnet_onnx.py` and `scripts/convert_controlnet_rknn.py` so the conversion is reproducible from a make target rather than copy-paste.
- Capture an actual calibration set and version it (or version the recipe to regenerate it deterministically).
- Verify int8 quality on a real RK3588 board for at least one canny + SD1.5 path; record latency and quality numbers in this doc.
- Author the V2 RKNN ControlNet provider per spec §5–6, consuming the `.rknn` + `metadata.json` artifacts produced by this pipeline.
