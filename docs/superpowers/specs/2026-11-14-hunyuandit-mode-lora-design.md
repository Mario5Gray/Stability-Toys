# HunyuanDiT Mode LoRA — Design

FP issue: STABL-ykowysrg
Status: approved 2026-11-14
Proven host: enigma (CUDA, RTX 3090, HunyuanDiT v1.2). MPS is secondary.

## Intent

The operator wants to apply a HunyuanDiT LoRA from `st`.
Today a mode `loras:` list in `conf/modes.yml` is metadata only.
`server/mode_config.py` resolves and checks the paths.
`backends/governor.py` records the paths in the model registry and logs "Loading N LoRAs".
Nothing loads the weights into the pipeline.

The operator chose per-mode activation because it has less resistance.
A LoRA is active when its mode is loaded.
The operator activates a LoRA with `st modes switch <mode>`.

## Goals

- A mode `loras:` list changes HunyuanDiT output on CUDA.
- MPS gets the same behavior through worker inheritance.
- `st modes` shows the LoRAs of each mode.
- A mode `loras:` list never fails silently.

## Non-goals

- Per-request LoRA selection or strength.
- Runtime PEFT adapters.
- kohya, ComfyUI, or diffusers-named LoRA formats.
- SD1.5, SDXL, or Z-Image mode LoRAs.
- Conversion of the `HYDiT-ControlNet-v1.2` `.pt` files.
- A commit of the local enigma `conf/modes.yml` edit.
- Trigger phrases. The jade LoRA needs `玉石绘画风格` in the prompt. The operator writes it.

## Anti-goals

- A mode that lists LoRAs and loads without them.
- Base weights that keep a LoRA after a switch to a mode without it.
- A partial merge after a key or shape mismatch.

## Constraints

- diffusers 0.39 has no HunyuanDiT LoRA loader. `HunyuanDiT2DModel` has no adapter API.
- The Tencent files use original hydit key names, with prefix `base_model.model.`.
- The jade and porcelain files have 402 tensors: 40 blocks × 10 keys + 2 pooler keys.
- Both files have `r=64` and `lora_alpha=64`. The scale factor alpha/r is 1.
- Tencent reference code ignores alpha. It applies `lora_scale` only.
- The worker factory does not receive the mode. The handle calls `configure_conditioning(mode.conditioning)` after it builds the worker. Both handles do this (`backends/worker_handle.py:122`, `backends/worker_handle_subprocess.py:263`).
- The CUDA worker can quantize the transformer to fp8 during load. A merge cannot write into quantized weights.
- `governor._load_mode` always tears down the old worker before it builds the new one. A new load reads the weights from disk.
- STABL-eqyjwfdh: `st gen --mode X` unloads the model after the generation. Use `st modes switch X` for proof.

## Approaches

| Approach | Switch cost | Code | Result |
|---|---|---|---|
| A. Merge at mode load (Tencent reference) | Mode reload | Smallest | **Picked by operator** |
| B. Runtime PEFT adapters on diffusers names | None | Larger: key split, injection, per-job scale | Rejected: more resistance |
| C. Merge with saved base weights for undo | Restore copy, about 1.2 GB fp16 | Medium | Rejected: memory cost, and A covers the need |

## Design

### 1. Worker hook

Add an optional worker method `configure_loras(loras)`.
`loras` is a list of `(absolute_path, strength)` pairs.
The handle builds the list from `mode.lora_paths` and `mode.loras[i].strength`.
Both handles call the hook right after `configure_conditioning`.

If the mode has LoRAs and the worker has no `configure_loras`, the handle raises `RuntimeError`.
The governor reports this as a mode load failure.
This matches the current conditioning rule.

Fix the governor log line. It must not say "Loading" before any load occurs.

### 2. HunyuanDiT merge module

New module `backends/hunyuandit_lora.py`.
`DiffusersHunyuanDiTCudaWorker.configure_loras` calls it.
`DiffusersHunyuanDiTMpsWorker` inherits the method.

For each LoRA, in list order:

1. Resolve the path. A directory resolves to `<dir>/adapter_model.safetensors`.
2. Read the tensors. Strip the `base_model.model.` prefix.
3. Validate the full key set and every shape before any write. A missing key, an extra key, or a wrong shape raises an error. No weight changes.
4. Read `adapter_config.json` next to the file when present. Log `r` and `lora_alpha`. Do not apply alpha.
5. Merge into `pipe.transformer` with `torch.no_grad()`, in place:

| LoRA key (per block `i`) | Delta | Target |
|---|---|---|
| `blocks.i.attn1.Wqkv` | `B @ A`, chunk 3 on dim 0 | `attn1.to_q`, `attn1.to_k`, `attn1.to_v` |
| `blocks.i.attn1.out_proj` | `B @ A` | `attn1.to_out.0` |
| `blocks.i.attn2.q_proj` | `B @ A` | `attn2.to_q` |
| `blocks.i.attn2.kv_proj` | `B @ A`, chunk 2 on dim 0 | `attn2.to_k`, `attn2.to_v` |
| `blocks.i.attn2.out_proj` | `B @ A` | `attn2.to_out.0` |
| `pooler.q_proj` | `B @ A` | `time_extra_emb.pooler.q_proj` |

All targets are `.weight`. Compute `B @ A` in fp32 on the weight's device. Add `strength × delta`. Cast the sum to the weight's dtype.

6. Log one line per LoRA: path, strength, r, alpha, and the target count.

Several LoRAs add together.

Strength is fixed at load. To change strength, edit `modes.yml` and reload the mode.

The number of blocks comes from the transformer config. The validation uses that number.

If the transformer is quantized, `configure_loras` raises an error before any read.

### 3. Turning a LoRA off

Define a second mode without `loras:`.
A switch to that mode rebuilds the worker from disk. The base weights are clean.

### 4. ControlNet path

The HunyuanDiT ControlNet pipeline must use the merged `pipe.transformer`.
The plan must verify this in the code and in a test.

### 5. `st modes`

`st modes` prints one extra line for each mode with LoRAs:

```
HunyuanDiT
  model=diffusers/HunyuanDiT-v1.2-Diffusers  size=1024x1024  steps=50  cfg=6.0
  loras: HYDiT-LoRA/jade/adapter_model.safetensors(0.8)
```

The server `/api/modes` response already carries `loras` (`server/mode_config.py:1097`). Regenerate the Go OpenAPI client only if the schema lacks the field.

## Verification

- Unit tests first (TDD), with a tiny fake transformer and a synthetic LoRA file:
  - The merge equals the Tencent reference math.
  - The strength scales the delta.
  - A bad key or shape changes no weight.
  - A directory path resolves to `adapter_model.safetensors`.
  - The handle raises for a worker without the hook.
  - A quantized transformer refuses the merge.
- Go test for the `st modes` LoRA line.
- Live proof on enigma, same prompt and seed:
  - Mode with jade at 0.8 against a mode without LoRAs. The images differ. The jade style shows.
  - The worker log shows the LoRA line.
  - A switch back to the mode without LoRAs gives the base image again.

## Left to the builder

- The exact name and place of the hook in the `PipelineWorker` protocol (`backends/base.py`).
- The form of the `(path, strength)` list across the subprocess spawn boundary. It must pickle.
- The test file names.
