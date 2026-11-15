# MPS Z-Image Turbo and Base Design

FP parent: STABL-zubknmov.
Planning task: STABL-eudrdkpt.
Source baseline: 6c3466050de8432a9ab174bcdacb46bdfe1fb93b.
Scope authorized: user requested both variants and a plan on 2026-11-15.
Status: revised design. Reviewed at 78c7e48. Revisions recorded in ../reports/2026-11-15-mps-zimage-plan-review.md. Implementation has not started.

## Outcome

Add native MPS Z-Image Turbo and Base generation with current CUDA LoRA behavior.
Both variants use the existing zimage family.
Each variant needs separate live acceptance.
ControlNet remains unsupported.

| Behavior | Turbo | Base |
|---|---|---|
| txt2img and plain img2img | Required | Required |
| Default size | 1024x1024 | 1024x1024 |
| Default steps / guidance | 9 / 0.0 | 30 / 4.0 |
| Scheduler | flow_match_euler | flow_match_euler |
| Text encoder | int8 default, none explicit | int8 default, none explicit |
| Mode and request LoRAs | Required | Required |
| Negative prompts | Preserve mode policy | Preserve configured templates and custom prompts |

## Architecture

Reuse DiffusersZImageCudaWorker through a small DiffusersZImageMpsWorker subclass.
The existing constructor already reads its execution_policy before loading weights.
The existing generation loop uses self.device and policy cache release.
Keep the shared constructor, scheduler, conditioning, img2img and LoRA fusion.

Add ZImageMpsTorchPolicy in a separate module.
Extract one protected dtype resolver from MpsTorchPolicy.
Its default implementation retains every existing fp16 rule.
The Z-Image policy returns bf16 and checks a tiny bf16 MPS operation before weight loading.
Keep the existing family rule: CUDA_DTYPE does not select Z-Image precision.
Log ignored non-bf16 overrides through the existing Z-Image constructor.
Do not change environment variables temporarily or mutate global policy objects.

All denoising, text encoding and VAE model execution stays on MPS.
Local model loading and explicit upstream CPU constant construction remain permitted.
Reject PYTORCH_ENABLE_MPS_FALLBACK=1 for the Z-Image policy.
Reject denoiser quantization, offload and xformers through existing shared validation.
Text-encoder int8 remains a separate admitted family setting.

Override job-memory hooks on the MPS worker.
Skip CUDA peak reset/read methods.
Synchronize MPS before reading current and driver allocation.
Log numeric fields as current_allocated_bytes and driver_allocated_bytes.
Report allocator peaks as unavailable.
A failed telemetry read must not fail completed generation.

Add one import-clean zimage binding:
worker_ref = backends.zimage_mps_worker.DiffusersZImageMpsWorker.
execution_capabilities = ExecutionCapabilities(True, False, False).
The binding covers both variants by design.
No filename-based variant detector or new public mode field is required.
Keep the existing generic factory and module-level subprocess entry point.

## Encoder store

Keep bf16 encoder construction for int8 and unquantized paths.
Exercise the real filesystem store with tiny local Qwen3 weights on MPS.
Prove cold build, serialized dtype and post-process-restart hit.
Keep existing content identity, locking and corrupt-entry refusal.
Do not assume separate Turbo/Base encoders share content.
Hash their actual shards and record each resulting key.

The existing store key excludes construction dtype.
This plan introduces no alternate construction dtype.
Any future fp16 construction requires separate store-identity design.
No cache deletion or overwrite of operator-owned entries is authorized.

## LoRA contract

Reuse configure_loras and fuse_diffusers_loras unchanged.
Preserve finite strength, target fit, additive fusion and adapter removal.
Img2img shares the fused transformer.
Request LoRAs remain limited to modes without their own loras list.
Keep path containment, duplicate rejection and existing failure recovery.
A changed ordered LoRA tuple or strength reloads clean model weights.
An identical tuple reuses its loaded authority.
An explicit plain mode removes request LoRAs.
A request without a mode follows existing terminal-authority semantics.

The live proof needs two compatible LoRA files for each variant.
A Turbo LoRA does not establish Base compatibility.
The same file may serve both variants only after separate fit and visual checks.

## Evidence and acceptance

Prior-turn tiny native probes passed on M1 Max with 64 GiB memory.
Runtime: macOS 26.0.1, Torch 2.10.0, Diffusers 0.39.0.
Transformers 4.57.6, PEFT 0.19.1, Quanto 0.2.7, Accelerate 1.14.0.
Tiny bf16 transformer output was finite.
Tiny int8 Qwen in-memory reload output was bit-equal.
Tiny native LoRA fusion changed weights and removed all adapter layers.
Those probes did not establish full-model memory, speed or quality.

Use normal server, HTTP and WebSocket/CLI request paths for acceptance.
Test both inproc and subprocess isolation.
Use one server and one model authority at a time.
Use dedicated configuration, storage, encoder-store and evidence directories.
Never stop the installed server to free a port or memory without operator authorization.

For each variant, prove txt2img, img2img, mode LoRA and request LoRA.
Include two strengths, two LoRAs, zero strength, repeated set and plain restoration.
Test Base negative prompts at guidance 4.0.
Record requests, effective settings, decoded pixels, status, load logs and timings.
Record current/driver MPS memory separately from process RSS.
Report cold build, warm generation and reload durations separately.

Repeat each plain baseline with one fixed seed.
Require exact decoded-pixel repeatability within the acceptance run.
Require the restored plain output to equal the initial plain output.
Stop and report a failed equality check.
A changed tolerance requires operator review before another acceptance claim.
Do not require pixel equality between CUDA and MPS.
Require LoRA output to differ from plain and match the selected active LoRA set.
Pixel differences alone do not prove semantic LoRA quality.
User reviews variant-specific LoRA effect and general image quality.

## Limits and authorization

Current installed configuration points at missing Turbo and ZiT asset paths.
Inventory both model folders and approved LoRAs before live work.
Do not download, copy remote weights, replace models or alter operator configuration implicitly.
Planning authorization does not authorize implementation, real-model runs, restart, deployment, push or merge.
The implementation plan stops for review before code changes.
Full-model execution has a separate operator-go checkpoint.

Existing PNG mode/LoRA recreate work remains STABL-puwnngfo.
Keep that limitation visible.
No new API schema, CLI flag, MLX backend, ControlNet or quantized denoiser is included.

No generation path applies a mode default negative-prompt template at this baseline.
The mode configuration holds the templates.
A worker applies a custom negative prompt only when the mode permits it.
Acceptance records the prompt fields.
Acceptance does not claim template substitution.

Exact pixel equality gate:
Repeat each equality case three times before a product defect claim.
Record every output.
A changed tolerance requires operator review.

agent-policy.rdf was absent from the installed root and source root during scoping.
Recheck before implementation.
No substitute RDF policy was applied.
