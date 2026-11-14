# Z-Image Mode LoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. AGENTS.md forbids subagent-driven development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Z-Image mode `loras:` list loads, fuses, and applies its LoRAs at mode load.

**Architecture:** `DiffusersZImageCudaWorker.configure_loras` uses the diffusers `ZImageLoraLoaderMixin`: load each file as a named adapter, set strengths, fuse into the transformer, unload the adapter layers. The STABL-ykowysrg handle hook already calls `configure_loras` at mode load.

**Tech Stack:** Python 3.12, diffusers 0.39, peft, torch, safetensors, pytest.

**Spec:** `docs/superpowers/specs/2026-11-14-zimage-mode-lora-design.md`

**FP issue:** STABL-gftujmyi. Branch: `feat/stabl-gftujmyi-zimage-mode-lora`.

## Global Constraints

- Python commands run in the `stability-toys` conda env. Multi-file runs use `python -m tests.run <paths> -- -q`.
- Every commit message names STABL-gftujmyi and the exact next step.
- Adapter names: `mode_lora_{i}`, `i` = index in the spec list.
- `fuse_lora(components=["transformer"], adapter_names=names, lora_scale=1.0)`. Strengths go through `set_adapters(names, adapter_weights=strengths)`.
- After `configure_loras`, no PEFT adapter layer remains in the transformer.
- A LoRA that targets zero transformer modules raises `ZImageLoraError`. A non-finite strength raises before any load.
- Do not commit enigma `conf/modes.yml` edits.

## Review Focus

1. An SDXL or HunyuanDiT LoRA in a Z-Image mode. Expect a refusal, not a silent load. Task 1 test `test_foreign_lora_refuses`.
2. A directory path (the HunyuanDiT spec allows directories). Expect the single `.safetensors` file inside to load, and a refusal for zero or several files. Task 1 tests `test_directory_with_one_file_loads` and `test_directory_with_two_files_refuses`.
3. img2img after a LoRA mode load. Expect the fused weights. Task 1 test `test_img2img_pipe_shares_fused_transformer`.
4. Two LoRAs on the same module. Expect the deltas to add. Task 1 test `test_two_loras_add`.
5. A strength of 0. Expect weights unchanged and no error. Task 1 test `test_zero_strength_leaves_weights`.

---

### Task 1: `DiffusersZImageCudaWorker.configure_loras`

**Files:**
- Modify: `backends/zimage_cuda_worker.py` (add `ZImageLoraError` near the other errors at line 43-52, add `configure_loras` to `DiffusersZImageCudaWorker`, replace the comment at line 145)
- Modify: `docs/WORKER_SELECTION.md` step 4 (now HunyuanDiT and Z-Image have `configure_loras`)
- Test: `tests/test_zimage_lora_worker.py`

**Interfaces:**
- Consumes: `backends.mode_loras.LoraSpec(path: str, strength: float)`.
- Produces: `class ZImageLoraError(ValueError)`. `DiffusersZImageCudaWorker.configure_loras(self, specs: Sequence[LoraSpec]) -> None`.

Test fixture (in the test file):
- `_tiny_transformer()` = `ZImageTransformer2DModel(dim=64, n_layers=2, n_refiner_layers=1, n_heads=2, n_kv_heads=2, cap_feat_dim=32, axes_dims=[8, 12, 12], axes_lens=[64, 32, 32])`, seeded with `torch.manual_seed(0)`.
- `_worker()`: `DiffusersZImageCudaWorker.__new__(...)`, `pipe = ZImagePipeline(scheduler=None, vae=None, text_encoder=None, tokenizer=None, transformer=_tiny_transformer())`, `worker_id = 0`, `_img2img_pipe = None`.
- `_comfy_lora(module_names, rank=4, seed=0) -> dict` with keys `diffusion_model.<module>.lora_A.weight` `[rank, in]` and `.lora_B.weight` `[out, rank]`, values `randn * 0.1`. Save with `safetensors.torch.save_file`.
- Target module for single-module tests: `layers.0.attention.to_q`.

- [x] **Step 1: Write the failing tests** in `tests/test_zimage_lora_worker.py`:
  - `test_fused_weight_matches_lora_math` — one LoRA at strength 0.7. `layers.0.attention.to_q.weight` equals `W + 0.7 * (B @ A)` (`atol=1e-5`).
  - `test_two_loras_add` — two LoRAs on the same module, 1.0 and 0.5. The weight equals `W + B1@A1 + 0.5 * B2@A2` (`atol=1e-5`).
  - `test_no_adapter_layers_remain` — after `configure_loras`, no module in the transformer is a `peft.tuners.tuners_utils.BaseTunerLayer`.
  - `test_zero_strength_leaves_weights` — strength 0.0. All transformer weights equal the clone (`atol=0`).
  - `test_foreign_lora_refuses` — a file with only `lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight` and `.lora_up.weight`. `pytest.raises((ZImageLoraError, ValueError))`.
  - `test_non_finite_strength_refuses_before_load` — strength `float("inf")`, path does not exist. Raises `ZImageLoraError` with `"strength"` in the message.
  - `test_directory_with_one_file_loads` — a directory holding one `.safetensors` file. The weight changes.
  - `test_directory_with_two_files_refuses` — raises `ZImageLoraError`.
  - `test_logs_targeted_module_count` — `caplog` has `"strength=0.7 modules=1"`.
  - `test_img2img_pipe_shares_fused_transformer` — patch `backends.zimage_cuda_worker._zimage_img2img_pipeline_cls` to return `ZImageImg2ImgPipeline`. After `configure_loras`, `worker._ensure_img2img_pipe().transformer is worker.pipe.transformer`.

- [x] **Step 2: Run the tests to verify they fail**

Run: `python -m tests.run tests/test_zimage_lora_worker.py -- -q`
Expected: FAIL with `ImportError: cannot import name 'ZImageLoraError'`.

- [x] **Step 3: Implement** `ZImageLoraError` and `configure_loras` per the spec Design steps 1-8. Count targeted modules per adapter before the fuse: `BaseTunerLayer` instances in `self.pipe.transformer` whose `lora_A` holds the adapter name. Log: `"[%s] worker %s merged LoRA %s strength=%s modules=%d"`.

- [x] **Step 4: Update `docs/WORKER_SELECTION.md` step 4.** New text: the HunyuanDiT worker merges Tencent-format LoRAs, and the Z-Image worker loads diffusers, kohya, and ComfyUI LoRAs through diffusers and fuses them. Run `drift check` and confirm the doc stays `ok`. Relink only after a prose review.

- [x] **Step 5: Run the tests to verify they pass**

Run: `python -m tests.run tests/test_zimage_lora_worker.py tests/test_zimage_worker.py tests/test_mode_loras.py -- -q`
Expected: all pass.

- [x] **Step 6: Commit**

```bash
git add backends/zimage_cuda_worker.py tests/test_zimage_lora_worker.py docs/WORKER_SELECTION.md drift.lock
git commit -m "feat(zimage): load + fuse mode LoRAs in configure_loras (STABL-gftujmyi) — next: Task 2 suite + enigma live proof"
```

---

### Task 2: Suite and enigma live proof

Steps 2-6 restart the enigma server. **Wait for the user's go before step 2.**

- [x] **Step 1: Run the CPU test container.** `docker compose -f docker-compose.test.yml build test && docker compose -f docker-compose.test.yml run --rm test`. Expected: no `FAILED` ID outside `docs/superpowers/plans/2026-11-12-zimage-cuda-container/baseline-failures.txt`.

- [x] **Step 2: Push the branch. On enigma, switch to it and add a mode `ZImage-Turbo-j80z`** (copy of `ZImage-Turbo` plus `loras: [{path: ZiT/j80z.safetensors, strength: 1.0}]`). Back up `conf/modes.yml` to `/tmp/stabl-gftujmyi/modes.yml.bak` first. Do not commit.

- [x] **Step 3: Rebuild and restart** with `MODELS_HOST_PATH=/media/cold1/ComfyUI/models FS_HOST_PATH=/media/creative/creative/dreamers WORKFLOW_HOST_PATH=/home/hdd/workspace/Stability-Toys/comfyui-workflows docker compose -f docker-cuda.yml build`, then `up -d` with the same variables. Expected: container healthy, mounts match.

- [ ] **Step 4: Generate A/B/A** with the branch-built `st`, prompt `j80z, portrait photo, soft window light`, `--seed 1234 --steps 9 --cfg 0 --size 1024x1024`. Order: `ZImage-Turbo` → `ZImage-Turbo-j80z` → `ZImage-Turbo`. Use `st modes switch` before each `st gen`. Output under `<output_directory>/stabl-gftujmyi/`. Expected: plain-1 and plain-2 match, plain and j80z differ.

- [x] **Step 5: Check the log.** `docker logs stability-toys 2>&1 | grep "merged LoRA"`. Expected: `merged LoRA /models/loras/ZiT/j80z.safetensors strength=1.0 modules=240` (8 modules × 30 layers).
  - Evidence 2026-11-14 19:57:06: `[zimage-cuda] worker 0 merged LoRA /models/loras/ZiT/j80z.safetensors strength=0.8 modules=240`. The operator deployed the branch (a64f7b5) and set strength 0.8. The operator reported the j80z image a success. A/B/A (step 4) not run yet.

- [ ] **Step 6: Record evidence and restore enigma.** FP comment with counts, SHAs, image paths, and the log line. Restore enigma to `main` and the backed-up `conf/modes.yml` only after the user decides on the PR.
