# Z-Image Mode LoRA — Short Design

FP issue: STABL-gftujmyi
Status: approved 2026-11-14 (user supplied the proof LoRA in reply to the short design)
Builds on: STABL-ykowysrg (`docs/superpowers/specs/2026-11-14-hunyuandit-mode-lora-design.md`)
Proven host: enigma (CUDA). Mode `ZImage-Turbo`. LoRA `/models/loras/ZiT/j80z.safetensors`, trigger word `j80z`.

## Intent

The operator wants the HunyuanDiT mode-LoRA behavior on Z-Image Turbo.
A mode `loras:` list must change Z-Image output.
Today the STABL-ykowysrg hook refuses the mode load, because `DiffusersZImageCudaWorker` has no `configure_loras`.

## Goals

- `ZImage-Turbo` and `ZImage-Base` modes apply their `loras:` at mode load.
- The img2img path uses the same LoRA weights.
- Behavior matches HunyuanDiT: strength fixed at load, no per-step cost, a failure fails the mode load.

## Non-goals

- Per-request LoRA or strength.
- GGUF Z-Image transformers. No mode uses them.
- New `st` work. `st modes` already prints the `loras:` line.

## Anti-goals

- A mode that loads with a LoRA that matched no transformer module.
- Adapter layers left in the transformer after the merge.

## Constraints

- diffusers 0.39 `ZImagePipeline` has `ZImageLoraLoaderMixin`: `load_lora_weights`, `set_adapters`, `fuse_lora`, `unload_lora_weights`.
- The loader converts diffusers, kohya (`lora_unet_`), and ComfyUI (`diffusion_model.`) keys. It applies `.alpha` keys when present.
- `peft>=0.10` is in `requirements.txt`.
- The worker admits offload `none` and runtime quantize `none` only. Meta and fp8 weights cannot occur.
- `_ensure_img2img_pipe` builds the img2img pipeline with `from_pipe(self.pipe)`, so it shares the fused transformer.
- j80z: 480 tensors, ComfyUI format, rank 32, bf16, no alpha keys. All 480 convert onto real modules of the 30-layer transformer with matching shapes (checked on meta, 2026-11-14).

## Approaches

| Approach | Result |
|---|---|
| A. diffusers native load + fuse + unload | **Picked.** Least code, all common formats, alpha handled. |
| B. Custom merge like HunyuanDiT | Rejected. Repeats the diffusers converter for three formats. |
| C. Runtime adapters (`set_adapters` per job) | Rejected for this task. Per-request is a separate decision. The operator chose per-mode for HunyuanDiT. |

## Design

`DiffusersZImageCudaWorker.configure_loras(self, specs: Sequence[LoraSpec]) -> None`:

1. No specs: return.
2. Any non-finite strength: raise `ZImageLoraError` before any load.
3. For each spec at index `i`: resolve the path. A directory without a single `.safetensors` file raises. Call `self.pipe.load_lora_weights(path, adapter_name=f"mode_lora_{i}")`.
4. Check that each adapter targets at least one transformer module. Zero targets raises `ZImageLoraError`.
5. `self.pipe.set_adapters(names, adapter_weights=strengths)`.
6. `self.pipe.fuse_lora(components=["transformer"], adapter_names=names, lora_scale=1.0)`.
7. `self.pipe.unload_lora_weights()`. The fused weights stay. The adapter layers go.
8. Log one line per LoRA: path, strength, targeted module count.

A failure at any step raises. The handle then fails the mode load and the governor discards the worker.

`ZImageLoraError(ValueError)` lives in `backends/zimage_cuda_worker.py`.

## Verification

- Unit tests with a tiny real `ZImageTransformer2DModel` in a `ZImagePipeline` with stub components:
  - The fused weight equals `W + strength * (B @ A)` for a ComfyUI-format LoRA without alpha.
  - Two LoRAs add.
  - No adapter layers remain after `configure_loras`.
  - A LoRA with no matching keys refuses.
  - A non-finite strength refuses before any load.
  - The img2img pipeline built after `configure_loras` shares the fused transformer.
- Container suite: no new failures against the 19-ID baseline.
- Enigma live: same prompt and seed, `ZImage-Turbo` plain / LoRA mode / plain. Prompt carries `j80z`. Log line shows the targeted module count.

## Left to the builder

- The exact stub components for the test pipeline.
- How to count targeted modules (PEFT `BaseTunerLayer` instances that hold the adapter name).
