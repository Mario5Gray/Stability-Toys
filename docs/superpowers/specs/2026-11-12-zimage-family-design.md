# Z-Image Family Design

- Date: 2026-11-12
- FP umbrella: `STABL-kbzlfbts`
- Spike: `STABL-ruxdbohc`, script `spikes/zimage_encoder_placement.py`
- Status: revision 3. Revision 2 was approved 2026-11-12. Revision 3 follows the plan review: content-only key order, store lock, strength rule, memory metric. Aligned with the slice 1 plan on three points: worker module, store default root, family check location.
- Plan: `docs/superpowers/plans/2026-11-12-zimage-cuda-baseline.md`
- Revision 2 answers the review of 2026-11-12: store key, written-text gate, runtime admission

## 1. Intent

Add the Tongyi-MAI Z-Image model family to the worker.

The user gave two reasons:

- Z-Image renders written text in images better than the current families.
- Its flow-matching model gives results far above SDXL.

The user set this order:

1. CUDA: txt2img and img2img.
2. CUDA: ControlNet.
3. MPS: repeat items 1 and 2.

RK3588 is out of scope. This spec gives a full design for slice 1 only.

## 2. Goals, non-goals, anti-goals

### Goals for slice 1

- One neutral family, `zimage`, serves Z-Image-Turbo and Z-Image base.
- txt2img and img2img run on CUDA through `st gen --mode <name>`.
- The worker loads Diffusers-layout folders.
- The Qwen text encoder runs as a stored int8 copy, resident on the GPU.
- Certification on enigma covers both downloaded models.

### Non-goals for slice 1

- ControlNet. Slice 2.
- MPS. Slice 3.
- RK3588.
- Transformer-only single-file checkpoints.
- GGUF files, which include `Z-Image-AbliteratedV1.Q8_0.gguf`.
- LoRA and style loading.
- Runtime fp8 quantization of the transformer.
- Encoder swap between GPU and host RAM.
- A prompt-embedding cache.
- Certification of the full-size official fp32 folder. It must load through the same path.
- Z-Image-Edit and Z-Image-Omni-Base. Tongyi-MAI has not released them.

### Anti-goals

- int8 must not reduce written-text accuracy without a measured comparison.
- The worker must not download model files silently.
- The family must not accept a scheduler that is not a flow-match scheduler.
- A front-end change must not come before the CLI path works.

## 3. Verified facts

### Model

- Both checkpoints declare `ZImagePipeline`, `ZImageTransformer2DModel`, `Qwen3Model`, `Qwen2Tokenizer`, `AutoencoderKL`.
- Transformer configs match: dim 3840, 30 layers, 30 heads, 16 latent channels.
- Scheduler is `FlowMatchEulerDiscreteScheduler`. `shift` is 3.0 for Turbo and 6.0 for base.
- Turbo: 9 steps (8 transformer forwards), guidance 0.0.
- Base: 28 to 50 steps, guidance 3.0 to 5.0, negative prompt recommended.

### Environment

- Local env and image `harbor.lan/stability-toys:dev` have `diffusers 0.39.0`, `transformers 4.57.6`, `optimum-quanto 0.2.7`.
- `diffusers 0.39.0` has `ZImagePipeline`, `ZImageImg2ImgPipeline`, `ZImageControlNetPipeline`, `ZImageControlNetModel`.
- No dependency change is necessary for slice 1.
- enigma: one RTX 3090, 23.58 GiB usable. Host RAM 31 GB.
- Models on enigma, each a 20 GB Diffusers folder with a 12.3 GB bf16 transformer:
  - `/media/cold1/ComfyUI/models/repos/Z-Image-Turbo-BF16`
  - `/media/cold1/ComfyUI/models/repos/Z-Image-Base`
- The live server container mounts `/media/cold1/ComfyUI/models` at `/models`, read-write.

### Spike results

Conditions: Turbo, 9 steps, seed 42, one prompt with sign text. Values are peak allocated GiB for the full call.

| Arm | Weights | 512 | 1024 | 1536 |
|---|---|---|---|---|
| Resident bf16 | 19.28 | 19.89 | 21.67 | 22.67 |
| int8 encoder, resident | 15.87 | 16.48 | 18.26 | 21.24 |
| Swap, encoder leaves GPU | 11.74 | 12.34 | 14.12 | 17.10 |

- Speed is equal in all arms: 6.2 to 6.3 s at 1024, 15.3 to 15.9 s at 1536.
- int8 quantize takes 15.2 s. The stored file is 4.11 GiB. Reload takes 4.8 s.
- A reloaded int8 encoder gives bit-equal embeddings to a fresh int8 encoder.
- int8 against bf16 embeddings: cosine 0.99961 mean, 0.99848 minimum.
- Both 1024 images render both sign texts correctly.
- Warm encode takes less than 0.2 s.
- VAE decode drives the peak: +1.9 GiB at 1024, +4.3 GiB at 1536.
- First cold load from `/media/cold1` took 223 s.

Limits of the spike: one prompt, one seed, Turbo only. Host RAM was not measured.

## 4. User decisions

- Slice 1 serves Turbo and base in one family.
- The int8 resident encoder is the default placement.
- The int8 encoder is quantized once and stored.
- The official full-size folder must load, without certification.
- Candidate models are the two folders on enigma.

## 5. Design

The HunyuanDiT family is the template. Each part below names its seam.

### 5.1 Detection

File: `utils/model_detector.py`, Diffusers-folder detector.

- Set `info.transformer_kind = "zimage"` when `transformer/config.json` has `_class_name == "ZImageTransformer2DModel"`.
- No other detector field changes.
- A Z-Image single file stays unidentified in slice 1.

### 5.2 Family profile

File: `backends/family_profiles.py`.

```python
ZIMAGE_PROFILE = FamilyProfile(
    family_id="zimage",
    encoder_roles=("text_encoder",),
    pooled_required=False,
    pooled_projection_role=None,
    control_image_kwarg="control_image",
)
```

- Predicate: `info.transformer_kind == "zimage"`.
- `control_image_kwarg` is not used in slice 1. Slice 2 must verify the value against `ZImageControlNetPipeline`.

### 5.3 Scheduler

File: `backends/scheduler_registry.py`.

- Add id `flow_match_euler` for `FlowMatchEulerDiscreteScheduler`.
- Z-Image modes set `allowed_scheduler_ids: [flow_match_euler]` and `default_scheduler_id: flow_match_euler`.
- `build_scheduler` builds from the baseline config of the checkpoint. The `shift` value of the checkpoint therefore stays.
- A mode does not override `shift`.
- A test must prove that `shift` survives a scheduler rebuild for 3.0 and 6.0.

### 5.4 CUDA worker

File: new module `backends/zimage_cuda_worker.py`. New class `DiffusersZImageCudaWorker(CudaWorkerBase)`. `backends/cuda_worker.py` is not edited.

- Lazy resolvers for `ZImagePipeline` and `ZImageImg2ImgPipeline`, as for HunyuanDiT.
- A dependency preflight raises a typed error with installed versions when a class is absent.
- The worker accepts a Diffusers folder only. Other formats raise a clear `RuntimeError`.
- dtype is bfloat16 for transformer and VAE. The family sets it. A mode does not select fp16.
- `supports_attention_processor_swap = False`. The transformer uses rotary position data.
- `_style_api = "none"`.
- `_quantization_targets` raises. Runtime quantization of the transformer is refused at admission, section 5.5.
- Conditioning is native only. The prompt goes to the pipeline as text.
- txt2img uses `ZImagePipeline`.
- img2img uses `ZImageImg2ImgPipeline.from_pipe(self.pipe, torch_dtype=None)`. `torch_dtype=None` prevents an fp32 recast.
- Step progress and cancel use `callback_on_step_end`, as in the HunyuanDiT worker.
- The worker passes `negative_prompt` only when the mode permits one.
- `cfg_normalization` and `cfg_truncation` keep their pipeline defaults. The request API does not show them.
- Log tag: `zimage-cuda`. A later MPS subclass overrides it.

File: `backends/platforms/cuda_bindings.py`.

```python
"zimage": FamilyPlatformBinding(
    "backends.zimage_cuda_worker.DiffusersZImageCudaWorker",
    ExecutionCapabilities(True, False, False),
),
```

MPS has no `zimage` binding. `create_mps_worker` raises `UnsupportedFamilyError`, which is correct for slice 1.

### 5.5 int8 text encoder

New module: `backends/zimage_text_encoder.py`. It has no server import.

Behavior:

1. Compute a store key for the encoder of the model folder.
2. On a store hit, build the encoder from the stored int8 files. The bf16 encoder is not loaded.
3. On a store miss, load the bf16 encoder, quantize, store, then use the result.
4. Give the encoder to `ZImagePipeline.from_pretrained(..., text_encoder=encoder)`.

Quantize procedure, as proven in the spike:

- `optimum.quanto.quantize(encoder, weights=qint8)`, then move to the GPU, then `freeze(encoder)`.
- Store `model.safetensors`, `quanto_qmap.json`, and the encoder `config.json`.
- Reload: `AutoModel.from_config` under `accelerate.init_empty_weights`, then `quanto.requantize(model, state, qmap, device)`.

Store layout:

- Root: env `ZIMAGE_ENCODER_STORE`. Default is `<parent of the model folder>/.cache/stability-toys/zimage-text-encoder`. The worker gets the model path, not `model_root`.
- One directory for each key.
- Key: SHA-256 of these inputs, in this order:
  - the encoder `config.json` bytes
  - the SHA-256 content digest of each weight shard, ordered by digest value
  - the string `qint8`
  - the quanto version
- The key binds to weight content. Name, size, time, and folder path of a shard are not key inputs. A renamed shard gives the same key.
- The worker reads each shard in full at each load to compute the digest. There is no digest memo in slice 1.
- Measured digest cost on enigma: 8.1 s for the Turbo encoder and 17.6 s for the base encoder.
- The two downloaded encoders have different content digests. Each model therefore gets its own store entry.
- The store entry records its key inputs in `store_manifest.json`. A load compares them with the computed inputs and fails on a difference.
- The manifest also holds shard names and sizes as diagnostic data. Diagnostic data is not a key input.
- A per-key file lock serializes loaders. A loader checks for the entry again after it holds the lock. One key is quantized one time.
- Each writer uses a unique temporary directory. A stale temporary directory from a dead writer is removed under the lock.
- The writer writes to a unique temporary directory, then renames it. A reader never sees a partial store entry.
- A store entry that fails to load is an error. The worker reports it and does not rebuild silently.

Mode field:

- New `ModeConfig` field `text_encoder_quantize`, values `int8` or `none`.
- Family `zimage` uses `int8` when the field is absent.
- `none` loads the bf16 encoder resident. It is the quality reference.
- Mode config validates the value. The worker factory refuses the field for other families, because the family is known only after resolution.

Runtime admission:

The worker checks the resolved runtime settings before it loads a weight file. The settings come from the mode and from env `CUDA_QUANTIZE` and `CUDA_OFFLOAD`.

| Setting | Admitted value | Other values |
|---|---|---|
| offload | `none` | refused |
| runtime quantize of the transformer | `none` | refused |
| `text_encoder_quantize` | `int8`, `none` | refused |
| loader format | Diffusers folder | refused |

- A refusal raises `ZImageUnsupportedRuntimeError`. The message names the setting, its value, and the admitted values.
- The check applies to `text_encoder_quantize: int8` and to `none`.
- Attention slicing and xFormers requests follow the current base-class rule: the worker skips them and logs a warning.
- A refused combination becomes admitted only through a later issue. That issue must add a test and a measured run on enigma.

### 5.6 Modes

File: `conf/modes.yml`. Two modes.

| Field | `ZImage-Turbo` | `ZImage-Base` |
|---|---|---|
| `model` | `repos/Z-Image-Turbo-BF16` | `repos/Z-Image-Base` |
| `default_steps` | 9 | 30 |
| `default_guidance` | 0.0 | 4.0 |
| `default_size` | 1024x1024 | 1024x1024 |
| `resolution_set` | sdxl | sdxl |
| scheduler ids | `flow_match_euler` | `flow_match_euler` |
| `allow_custom_negative_prompt` | false | true |
| negative templates | none | one default template |
| `text_encoder_quantize` | int8 | int8 |
| `controlnet_policy` | absent | absent |

`conf/modes.yml` has user changes that are not committed. The builder adds the two modes without other edits.

### 5.7 Request rules

- A ControlNet attachment for a `zimage` mode is refused by the capability check.
- A scheduler id outside the allowed list is refused by the current policy.
- img2img uses the request `denoise_strength` as the pipeline `strength`. The default 0.75 applies only when the value is absent.
- A strength outside `0 < strength <= 1` is refused. A strength that gives zero effective steps is refused.
- Turbo img2img has few effective steps: 9 steps at strength 0.6 gives 5 forwards. The builder records the effective count in the job log.

## 6. Tests

TDD is mandatory. Unit tests use fakes, as in `tests/test_hunyuandit_worker.py`. No unit test loads real weights.

- Detector: a folder with `ZImageTransformer2DModel` gives `transformer_kind == "zimage"`.
- Family: `resolve_family` returns `zimage` and exactly one match. HunyuanDiT and SD families do not match.
- Scheduler registry: `flow_match_euler` builds, and `shift` survives for 3.0 and 6.0.
- Bindings: CUDA has `zimage` with img2img true and ControlNet false. MPS has no `zimage`.
- Worker: a non-folder path raises. txt2img and img2img call the correct pipeline class with the correct kwargs.
- Worker: Turbo mode sends no negative prompt. Base mode sends it.
- Encoder store: key is stable for equal content.
- Encoder store: key changes when shard content changes and shard name, size, and time stay equal.
- Encoder store: key does not change when only the time of a shard changes.
- Encoder store: a manifest that differs from the computed inputs fails the load.
- Admission: offload `model` and `sequential` are refused before a weight file loads.
- Admission: runtime quantize `fp8` is refused before a weight file loads.
- Admission: env `CUDA_OFFLOAD` and `CUDA_QUANTIZE` are refused the same as mode values.
- Encoder store: a miss quantizes and stores. A hit does not load the bf16 encoder.
- Encoder store: an interrupted write leaves no entry.
- Mode config: `text_encoder_quantize` parses, defaults, and is rejected for other families.

Run tests in the `stability-toys` conda env or the test container.

## 7. Certification on enigma

Certification is CLI-first, through `st gen --mode <name>` against the CUDA server.

For each of `ZImage-Turbo` and `ZImage-Base`:

1. txt2img at 1024x1024 completes and the image is coherent.
2. img2img at 1024x1024 completes and keeps the source composition.
3. Peak allocated memory is recorded for txt2img and img2img. The server process measures it with `torch.cuda.max_memory_allocated()`, with a reset and a synchronize around the generation.
4. Device memory from `nvidia-smi` is recorded as a separate figure. It has no limit.
5. The second server start uses the stored encoder and logs a store hit.

### 7.1 Written-text check

The check runs for both modes, because both default to int8.

The prompts, seeds, settings, and scoring rules below are frozen. A change needs a spec edit before a run.

Settings for each run:

- Size 1024x1024. Steps, guidance, and scheduler are the mode defaults.
- `ZImage-Base` uses its default negative template: `blurry, low quality, watermark`.
- Each prompt runs two times for each mode: `text_encoder_quantize: int8` and `none`.
- Total: 5 prompts x 2 arms x 2 modes = 20 images. Each arm has 6 scored strings.

| Id | Seed | Prompt | Scored strings |
|---|---|---|---|
| P1 | 101 | A weathered wooden sign beside a mountain trail. The sign reads "SUMMIT 2.4 KM" and below it "Keep to the marked path". Morning fog, photorealistic. | `SUMMIT 2.4 KM`, `Keep to the marked path` |
| P2 | 102 | A neon sign in a rainy city window at night. The neon letters read "OPEN LATE". Photorealistic. | `OPEN LATE` |
| P3 | 103 | A hardcover book on a desk. The title on the cover reads "The Quiet Engine". Soft window light, photorealistic. | `The Quiet Engine` |
| P4 | 104 | A chalkboard outside a cafe. Handwritten chalk text reads "Fresh Bread 7 AM". Photorealistic. | `Fresh Bread 7 AM` |
| P5 | 105 | A white paper label on a glass jar. Printed black text reads "Batch 0417". Studio photo. | `Batch 0417` |

Scoring rules:

- One score for each scored string: correct or incorrect.
- Correct means each character is present, in order, with the same letter case. Spaces between words must be visible.
- A missing, added, changed, or duplicated character makes the string incorrect.
- Font, color, size, and line breaks do not affect the score.
- Other text in the image does not affect the score.
- A person reads each full-size image and records the scores before a comparison of arms.
- No re-run with a different seed is permitted.

Pass rule for each mode. Both conditions are necessary:

- Floor: the int8 arm has at least 5 of 6 strings correct.
- Relative: the int8 arm has no more incorrect strings than the bf16 arm.

A mode that fails a condition is not certified with int8. The mode default then needs a new user decision.

### 7.2 Limits

- Turbo int8 at 1024 must peak at or below 19.0 GiB allocated, by the in-process figure. The spike measured 18.26 with the same metric.
- Base has no limit yet. Classifier-free guidance can double the batch. The first base run sets the figure.
- A base run that fails for memory at 1024x1024 blocks base certification.

Evidence goes to the FP issue: commands, pass or fail, peak figures, image paths, commit SHA.

## 8. Slices and FP layout

- Slice 1: this spec. Child issues of `STABL-kbzlfbts`, one for each plan task.
- Slice 2, CUDA ControlNet: its own brainstorm and spec. `ZImageControlNetModel` exists in Diffusers. No control model is selected or tested.
- Slice 3, MPS: its own spec. bfloat16 support and 20 GB memory on the Mac are not verified.

Later issues, not scheduled:

- Transformer-only single-file checkpoints.
- GGUF encoder and transformer files.
- LoRA.
- VAE-decode peak reduction.
- Certification of the official fp32 folder.

## 9. Risks

- int8 text accuracy is proven on one prompt only. Section 7.1 tests five prompts in each mode.
- Base VRAM with guidance is not measured.
- The spike measured a bare pipeline. The server adds its own memory use.
- The live server and other containers share the GPU on enigma.
- The content digest adds 8 to 18 s to each load on enigma, and more from a cold disk.
- A quanto version change invalidates store entries. The key includes the version, so the worker rebuilds.
- The first build of a store entry needs the bf16 encoder in memory: about 8 GiB.
- Cold model load from `/media/cold1` is slow: 223 s measured.

## 10. Calls made by the author

The user did not decide these. Each can change at review.

- Family id is `zimage`. Scheduler id is `flow_match_euler`.
- Single-file loading is out of slice 1. The user had no preference.
- Both models get certification, because both are now downloaded.
- Swap and the embedding cache are dropped from slice 1.
- The store default is under `<parent of the model folder>/.cache`. Env `ZIMAGE_ENCODER_STORE` overrides it.
- The encoder setting is a new mode field, not a reuse of `runtime_quantize`.
- Turbo forbids a custom negative prompt through current mode policy.
- `cfg_normalization` is not a request field.
- Attention-processor swaps are off for this family.
- Base defaults: 30 steps, guidance 4.0.
- The written-text rule: floor of 5 of 6 correct for int8, and no more incorrect strings than bf16.
- The five prompts, their seeds, and the scoring rules in section 7.1.
- The base negative template: `blurry, low quality, watermark`.
- The store key reads each shard in full at each load. A digest memo is a later change.
- Unsupported runtime combinations are refused, not validated, in slice 1.

## 11. Left to the builder

- The seam that sets the worker dtype to bfloat16.
- The latents path: the HunyuanDiT worker has no `run_job_with_latents`. Slice 1 adds none.
- `/status` scope report for the new family: follow the HunyuanDiT pattern.
- Drift anchors for edited files: run `drift refs <path>` before each edit.
