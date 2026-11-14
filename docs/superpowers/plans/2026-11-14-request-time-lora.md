# Request-Time LoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. AGENTS.md forbids subagent-driven development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A generate request with `loras` runs on a mode without `loras:` plus those LoRAs, with a worker reload only when the LoRA set changes.

**Architecture:** The routes validate request LoRA paths against `lora_root` and pass `tuple[LoraSpec, ...]` to `admit_generation`. The governor identifies an authority by (base mode name, request LoRAs), and builds a derived in-memory mode for a new key. The existing switch path reloads the worker, and the existing `configure_loras` hook merges the LoRAs. SD1.5 and SDXL get `configure_loras` through a diffusers fuse helper that Z-Image also uses.

**Tech Stack:** Python 3.12, FastAPI, pydantic, diffusers 0.39, peft, torch, pytest. Go (cobra) for `st`.

**Spec:** `docs/superpowers/specs/2026-11-14-request-time-lora-design.md`

**FP issue:** STABL-fvwsvjvn. Branch: `feat/stabl-fvwsvjvn-request-loras`.

## Global Constraints

- Python commands run in the `stability-toys` conda env. Multi-file runs use `python -m tests.run <paths> -- -q`.
- Go tests run as `cd cli/go && ST_SERVER= go test ./...`.
- Every commit message names STABL-fvwsvjvn and the exact next step.
- `modes.yml` is never written.
- Request LoRAs run only on a mode with no `loras:`. Otherwise refuse before any switch, with a message that names the mode and says to use a mode without `loras:`.
- Request paths resolve inside `lora_root` after `Path.resolve()` on both sides. The path exists. The strength is finite. No two entries resolve to the same file. A directory selects one weight file first, and every check runs on that resolved file.
- Error contract: a malformed entry gives HTTP 422 or WS `job:error`. A rule failure gives HTTP 400 or WS `job:error`.
- `st gen` inheritance: `--lora` replaces inherited `loras`. `--mode` without `--lora` removes them. Neither keeps them.
- Absent `loras` and `loras: []` mean "no request LoRAs".
- Authority key equality: same base mode name, and equal `tuple[LoraSpec, ...]` (order, resolved path, exact float).
- Run `drift refs <path>` before editing a file. Run `drift check` before each task commit. Relink only after a prose review.
- Do not commit enigma `conf/modes.yml` edits.

## Review Focus

1. A no-mode request with `loras` while a mode with `loras:` is loaded. Expect a refusal and the loaded model untouched. Task 4 test `test_no_mode_request_loras_on_lora_mode_refused`.
2. A request path with `..` that leaves `lora_root`, a symlink inside the root that points outside, or an admitted directory whose weight file is such a symlink. Expect a refusal. Task 3 tests `test_dotdot_escape_refused`, `test_symlink_escape_refused`, `test_directory_symlink_escape_refused`.
3. An in-root HunyuanDiT weight file whose `adapter_config.json` sidecar is a symlink outside the root. Expect a refusal before admission. Task 3 test `test_sidecar_symlink_outside_root_refused`, Task 5 test `test_http_sidecar_escape_is_400_and_not_admitted`.
4. Idle eviction after a request-LoRA job, then a demand reload. Expect the request LoRAs to load again. Task 4 test `test_demand_reload_keeps_request_loras`.
5. A `PUT /api/modes` config reload while request LoRAs are loaded. Expect the reload to keep them. Task 4 test `test_reload_current_mode_keeps_request_loras`.
6. An SDXL kohya LoRA with `lora_te1_`/`lora_te2_` keys. Expect the text encoder parts to fuse and to count toward the fit check. Task 2 test `test_sdxl_text_encoder_2_fuses` and Task 1 test `test_includes_text_encoder_keys_when_fused`.

---

### Task 1: Shared diffusers fuse helper, Z-Image on it

**Files:**
- Create: `backends/diffusers_lora_fuse.py`
- Modify: `backends/zimage_cuda_worker.py` (move `_resolve_zimage_lora_file`, `_count_lora_down_tensors`, `_count_adapter_modules`, and the `configure_loras` body into the helper)
- Test: `tests/test_diffusers_lora_fuse.py`. `tests/test_zimage_lora_worker.py` must pass unchanged.

**Interfaces:**
- Consumes: `backends.mode_loras.LoraSpec(path: str, strength: float)`.
- Produces:
  - `class DiffusersLoraError(ValueError)`.
  - `resolve_lora_file(path: str | Path, error_cls: type[DiffusersLoraError]) -> Path`. A directory must hold exactly one `.safetensors` file.
  - `count_lora_down_tensors(state_dict: Mapping[str, Tensor], *, include_text_encoder: bool) -> int`. Counts keys that end with `.lora_A.weight`, `.lora_down.weight`, or `.lora.down.weight`. Text encoder keys start with `text_encoder` or `lora_te`.
  - `fuse_diffusers_loras(pipe, specs: Sequence[LoraSpec], *, components: Sequence[str], log_tag: str, worker_id: Any, error_cls: type[DiffusersLoraError] = DiffusersLoraError) -> None`. Same steps and log line as the Z-Image `configure_loras` today. `include_text_encoder` is true when a component name starts with `text_encoder`. The module count runs over every named component.
  - `class ZImageLoraError(DiffusersLoraError)` in `backends/zimage_cuda_worker.py`. `configure_loras` calls the helper with `components=("transformer",)` and `error_cls=ZImageLoraError`.

- [x] **Step 1: Write the failing tests** in `tests/test_diffusers_lora_fuse.py`:
  - `test_counts_old_diffusers_dot_keys`: `{"unet.x.lora.down.weight", "unet.x.lora.up.weight"}` counts 1.
  - `test_excludes_text_encoder_keys_when_not_fused`: one `diffusion_model.` A key and one `lora_te1_` down key. `include_text_encoder=False` counts 1.
  - `test_includes_text_encoder_keys_when_fused`: the same dict with `include_text_encoder=True` counts 2.
  - `test_alpha_keys_not_counted`: a `.alpha` key adds 0.
- [x] **Step 2: Run** `python -m tests.run tests/test_diffusers_lora_fuse.py -- -q`. Expected: FAIL with `ModuleNotFoundError: No module named 'backends.diffusers_lora_fuse'`.
- [x] **Step 3: Implement** the helper and move Z-Image onto it.
- [x] **Step 4: Run** `python -m tests.run tests/test_diffusers_lora_fuse.py tests/test_zimage_lora_worker.py tests/test_zimage_worker.py -- -q`. Expected: all pass.
- [x] **Step 5: Commit** `refactor(lora): shared diffusers LoRA fuse helper, Z-Image on it (STABL-fvwsvjvn) — next: Task 2 SD/SDXL configure_loras`.

---

### Task 2: SD1.5 and SDXL `configure_loras`

**Files:**
- Create: `backends/unet_mode_loras.py`
- Modify: `backends/cuda_worker.py` (`DiffusersCudaWorker` at line 960, `DiffusersSDXLCudaWorker` at line 1314 take the mixin first in their bases), `backends/mps_worker.py` (`DiffusersMpsWorker` at line 48 takes the mixin)
- Test: `tests/test_unet_mode_loras.py`

**Interfaces:**
- Consumes: Task 1 `fuse_diffusers_loras`, `DiffusersLoraError`.
- Produces:
  - `class UNetLoraError(DiffusersLoraError)`.
  - `class UNetModeLoraMixin` with `configure_loras(self, specs: Sequence[LoraSpec]) -> None`.
  - Refusals before any read, each with `UNetLoraError`: `self._quantize == "fp8"` or `self._checkpoint_precision == "fp8"` (message has `fp8`), `self._offload == "sequential"` (message has `sequential`), any true value in `self._style_loaded` (message has `style`). Read each attribute with `getattr(..., None)`, because the MPS worker lacks some.
  - Components: `"unet"`, then `"text_encoder"` and `"text_encoder_2"` when the pipe has a non-None attribute with that name.

Test fixtures (verified to construct on diffusers 0.39, 2026-11-14):
- Text encoder: `CLIPTextModel(CLIPTextConfig(hidden_size=32, intermediate_size=37, num_hidden_layers=2, num_attention_heads=4, vocab_size=1000, projection_dim=32, bos_token_id=0, eos_token_id=2, layer_norm_eps=1e-5))`. SDXL `text_encoder_2`: `CLIPTextModelWithProjection` with the same config.
- SD UNet: `UNet2DConditionModel(block_out_channels=(32, 64), layers_per_block=1, sample_size=32, in_channels=4, out_channels=4, down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"), up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), cross_attention_dim=32)`.
- SDXL UNet: the same plus `cross_attention_dim=64, attention_head_dim=(2, 4), use_linear_projection=True, addition_embed_type="text_time", addition_time_embed_dim=8, transformer_layers_per_block=(1, 2), projection_class_embeddings_input_dim=80`.
- Pipelines: `StableDiffusionPipeline(vae=None, text_encoder=te, tokenizer=None, unet=unet, scheduler=None, safety_checker=None, feature_extractor=None, requires_safety_checker=False)` and `StableDiffusionXLPipeline(vae=None, text_encoder=te, text_encoder_2=te2, tokenizer=None, tokenizer_2=None, unet=xunet, scheduler=None)`.
- Worker: `DiffusersCudaWorker.__new__`, then set `pipe`, `worker_id=0`, `_quantize="none"`, `_checkpoint_precision="unknown"`, `_offload="none"`, `_style_loaded={}`, `_img2img_pipe=None`.
- LoRA: diffusers keys `unet.<module>.lora_A.weight` / `.lora_B.weight` and `text_encoder.<module>...`. Modules: `down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q` and `text_model.encoder.layers.0.self_attn.q_proj`.

- [x] **Step 1: Write the failing tests** in `tests/test_unet_mode_loras.py`:
  - `test_sd_unet_and_text_encoder_fuse_math`: strength 0.5. Both weights equal `W + 0.5 * (B @ A)` (`atol=1e-5`). No `BaseTunerLayer` remains in `unet` or `text_encoder`.
  - `test_sdxl_text_encoder_2_fuses`: a LoRA on `text_encoder_2.text_model.encoder.layers.0.self_attn.q_proj` changes that weight by `1.0 * (B @ A)`.
  - `test_fp8_refuses`, `test_sequential_offload_refuses`, `test_style_adapters_loaded_refuses`: raise `UNetLoraError` with `fp8`, `sequential`, `style`.
  - `test_partial_fit_refuses`: one real unet module plus `unet.not_a_module`. Raise `UNetLoraError` with the file name.
  - `test_workers_have_configure_loras`: `DiffusersCudaWorker`, `DiffusersSDXLCudaWorker`, `DiffusersMpsWorker` each have `configure_loras` from `UNetModeLoraMixin`.
  - `test_img2img_shares_fused_unet`: after `configure_loras`, `StableDiffusionImg2ImgPipeline(**worker.pipe.components).unet is worker.pipe.unet`. (The SD workers build img2img with `**self.pipe.components`, `backends/cuda_worker.py:1151` and `:1545`.)
- [x] **Step 2: Run** `python -m tests.run tests/test_unet_mode_loras.py -- -q`. Expected: FAIL with `ModuleNotFoundError: No module named 'backends.unet_mode_loras'`.
- [x] **Step 3: Implement** the mixin and add it to the three workers.
- [x] **Step 4: Run** `python -m tests.run tests/test_unet_mode_loras.py tests/test_mode_loras.py tests/test_zimage_lora_worker.py tests/test_hunyuandit_lora_worker.py -- -q`. Expected: all pass.
- [x] **Step 5: Commit** `feat(lora): SD1.5/SDXL configure_loras via diffusers fuse (STABL-fvwsvjvn) — next: Task 3 request LoRA validation`.

---

### Task 3: Request LoRA validation and derived mode

**Files:**
- Modify: `backends/mode_loras.py`
- Create: `server/request_loras.py`
- Test: `tests/test_request_loras.py`

**Interfaces:**
- Produces in `backends/mode_loras.py`:
  - `class RequestLoraError(ValueError)`.
  - `derive_request_mode(mode, specs: tuple[LoraSpec, ...])`: mutates and returns the given (already deep-copied) mode. With `specs` empty, it returns the mode unchanged. With `specs` set and `mode.loras` non-empty, it raises `RequestLoraError` naming `mode.name`. Otherwise it sets `mode.loras = [LoRAConfig(path=s.path, strength=s.strength) for s in specs]` and `mode.lora_paths = [s.path for s in specs]`.
- Produces in `server/request_loras.py`:
  - `select_weight_file(path: Path) -> Path`: a file returns itself. A directory returns `adapter_model.safetensors` when present, else its only `*.safetensors`. Zero or several candidates raise `RequestLoraError`.
  - `resolve_request_loras(entries: Sequence[Any] | None, lora_root: str) -> tuple[LoraSpec, ...]`. Each entry has `.path` and `.strength`, or keys `path` and `strength` (default 1.0). Order: join with the root, `select_weight_file`, `resolve()`, containment check, sidecar check, duplicate check. All checks use the resolved weight file. Sidecar check: `sidecar = weight.with_name(ADAPTER_CONFIG)` (import `ADAPTER_CONFIG` from `backends.hunyuandit_lora`). When `sidecar.exists()`, `sidecar.resolve()` must be inside the root, else `RequestLoraError` naming the sidecar. A missing sidecar passes. Returns specs with that file path. Raises `RequestLoraError`.

- [x] **Step 1: Write the failing tests** in `tests/test_request_loras.py` (`tmp_path` as `lora_root`):
  - `test_none_and_empty_give_empty_tuple`.
  - `test_relative_path_resolves_under_root`: returns `(LoraSpec(str((root/"a.safetensors").resolve()), 0.8),)`.
  - `test_absolute_path_inside_root_admitted`.
  - `test_absolute_path_outside_root_refused`.
  - `test_dotdot_escape_refused`: `"../x.safetensors"` with the file present outside the root.
  - `test_symlink_escape_refused`: a link inside the root to a file outside.
  - `test_missing_path_refused`.
  - `test_non_finite_strength_refused`: `inf` and `nan`, message has `strength`.
  - `test_duplicate_file_refused`: `a.safetensors` and `./a.safetensors`.
  - `test_directory_symlink_escape_refused`: directory `d/` inside the root holds `w.safetensors`, a symlink to a file outside. Refused.
  - `test_directory_and_its_file_duplicate_refused`: entries `d` and `d/w.safetensors`. Refused.
  - `test_directory_selects_adapter_model`: `d/` holds `adapter_model.safetensors` and `other.safetensors`. The spec path is `d/adapter_model.safetensors` resolved.
  - `test_directory_with_two_candidates_refused`: `d/` holds `a.safetensors` and `b.safetensors`, no `adapter_model.safetensors`.
  - `test_spec_path_is_a_file`: every returned path is a file.
  - `test_sidecar_symlink_outside_root_refused`: `d/adapter_model.safetensors` is a real file in the root. `d/adapter_config.json` is a symlink to a file outside. Refused, and the message names `adapter_config.json`.
  - `test_missing_sidecar_allowed`: a weight file with no `adapter_config.json` passes.
  - `test_in_root_sidecar_allowed`: a real `adapter_config.json` next to the weight passes.
  - `test_default_strength_is_one`.
  - `test_derive_sets_loras_and_paths_in_order`.
  - `test_derive_refuses_mode_with_loras`: message has the mode name and `without loras`.
  - `test_derive_with_no_specs_keeps_mode_loras`.
- [x] **Step 2: Run** `python -m tests.run tests/test_request_loras.py -- -q`. Expected: FAIL with `ModuleNotFoundError: No module named 'server.request_loras'`.
- [x] **Step 3: Implement** both functions.
- [x] **Step 4: Run** `python -m tests.run tests/test_request_loras.py tests/test_mode_loras.py -- -q`. Expected: all pass.
- [x] **Step 5: Commit** `feat(lora): request LoRA validation and derived mode (STABL-fvwsvjvn) — next: Task 4 governor authority key`.

---

### Task 4: Governor authority key

**Files:**
- Modify: `backends/governor.py` (`ActiveModelSnapshot` :67, `ModeSwitchJob` :296, `_resolve_target` :454, `_reserve_authority` :476, `_switch_shortcircuit` :515, `_load_mode` :540 and :600, dispatch fast path :1138, `switch_mode` :1408, `_reserve_and_enqueue_switch` :1422, `admit_generation` :1486, `reload_current_mode` :1511, `reload_if_current`, `unload_current_model`, `_build_runtime_status` :1024)
- Modify: `backends/worker_pool.py` (`admit_generation` :116 and a `get_current_loras` pass-through)
- Test: `tests/test_governor_request_loras.py` (import `_make_multi_mode_config`, `_make_mock_registry`, `_resolve_by_path`, `StubHandle`, `_freeze_dispatch`, `_drain_queue` from `tests.test_governor`)

**Interfaces:**
- Consumes: Task 3 `derive_request_mode`, `RequestLoraError`, `LoraSpec`.
- Produces:
  - `ActiveModelSnapshot.request_loras: tuple[LoraSpec, ...] = ()`. `ModeSwitchJob.request_loras: tuple[LoraSpec, ...] = ()`.
  - `admit_generation(target_mode: Optional[str], request_loras: tuple[LoraSpec, ...] = ()) -> Optional[ActiveModelSnapshot]`, per the spec section 4 table. With no terminal and no `target_mode`, the base is `get_default_mode()`.
  - `switch_mode(mode_name, force=False, request_loras=())`. `st modes switch` keeps calling it with no `request_loras`.
  - `_resolve_target(mode_name, request_loras=())` calls `derive_request_mode` after the deep copy. A `RequestLoraError` propagates before any reservation.
  - `self._current_request_loras: tuple[LoraSpec, ...]`, set with `_current_mode` and cleared with it.
  - `get_current_loras() -> list[dict]`: `[{"path": p, "strength": s}]` from the active snapshot mode `lora_paths` and `loras`. Empty when nothing is loaded.
  - `_build_runtime_status` adds `"current_loras": self.get_current_loras()`.
  - Log on reservation with request LoRAs: `"[Governor] Mode %s with %d request LoRA(s)"`.
  - Leave `vram_bytes=0` in `_reload_from_snapshot`. That site has no measured value (spec section 8 condition not met).

- [x] **Step 1: Write the failing tests** in `tests/test_governor_request_loras.py`. Use `patch("backends.governor.resolve_model", side_effect=_resolve_by_path)`. Specs: `S1 = (LoraSpec("/l/a.safetensors", 0.8),)`, `S2 = (LoraSpec("/l/a.safetensors", 1.0),)`.
  - `test_same_request_loras_do_not_reserve_again`: admit `("mode-a", S1)` two times while dispatch is frozen. The epoch rises one time.
  - `test_different_strength_reserves_new_authority`: `S1` then `S2`. Two reservations, the last has `request_loras == S2`.
  - `test_no_mode_request_keeps_request_loras`: after `("mode-a", S1)`, `admit_generation(None)` returns an authority with `request_loras == S1`.
  - `test_no_mode_with_loras_targets_terminal_base`: terminal `mode-b`, then `admit_generation(None, S1)` gives `mode_name == "mode-b"` and `S1`.
  - `test_no_mode_no_terminal_uses_default_mode`.
  - `test_mode_without_loras_reloads_configured`: terminal `("mode-a", S1)`, then `admit_generation("mode-a")` reserves `("mode-a", ())`.
  - `test_request_loras_on_lora_mode_refused`: `mode-l` has one `LoRAConfig`. `admit_generation("mode-l", S1)` raises `RequestLoraError`. The epoch and `_pending_authorities` do not change.
  - `test_no_mode_request_loras_on_lora_mode_refused`: terminal `mode-l`, `admit_generation(None, S1)` raises `RequestLoraError`. No reservation.
  - `test_derived_mode_holds_request_loras`: the reservation mode has `lora_paths == ["/l/a.safetensors"]` and `loras[0].strength == 0.8`. The mode config's own mode object keeps `loras == []`.
  - `test_switch_mode_targets_plain_key`: terminal `("mode-a", S1)`, then `switch_mode("mode-a")` reserves `("mode-a", ())`.
  - `test_dispatch_fast_path_compares_key`: with a live worker on `("mode-a", ())`, a `ModeSwitchJob` for `("mode-a", S1)` calls `_load_mode`. The result status is not `already_loaded`.
  - `test_reload_current_mode_keeps_request_loras`: after a load of `("mode-a", S1)`, `reload_current_mode()` makes `StubHandle.start_calls[-1]` get a mode with `lora_paths == ["/l/a.safetensors"]`.
  - `test_demand_reload_keeps_request_loras`: after a load of `("mode-a", S1)`, unload the worker as the idle path does, then call `_reload_from_snapshot()`. `start_calls[-1]` mode has the same `lora_paths`.
  - `test_runtime_status_reports_current_loras`: `[{"path": "/l/a.safetensors", "strength": 0.8}]`.
- [x] **Step 2: Run** `python -m tests.run tests/test_governor_request_loras.py -- -q`. Expected: FAIL (`TypeError` on the `request_loras` argument).
- [x] **Step 3: Implement** the key through every listed governor site and the pool pass-throughs.
- [x] **Step 4: Run** `python -m tests.run tests/test_governor_request_loras.py tests/test_governor.py tests/test_governor_metrics.py tests/test_governor_tracing.py tests/test_governor_log_context.py -- -q`. Expected: all pass.
- [x] **Step 5: Commit** `feat(governor): authority key = mode + request LoRAs (STABL-fvwsvjvn) — next: Task 5 HTTP/WS routes`.

---

### Task 5: HTTP and WS routes, status route

**Files:**
- Modify: `server/lcm_sr_server.py` (`RequestLora` model, `GenerateRequest.loras`, `/generate` admission at :619)
- Modify: `server/ws_routes.py` (`_build_generate_request` at :442 passes `loras`, admission at :251)
- Modify: `server/model_routes.py` (`/api/models/status` payload at :167 adds `current_loras` from `runtime.get_current_loras()` when the runtime has it, else `[]`)
- Test: `tests/test_request_lora_routes.py`, `tests/test_ws_build_generate_request.py`

**Interfaces:**
- Consumes: Task 3 `resolve_request_loras`, `RequestLoraError`. Task 4 `admit_generation(target_mode, request_loras)`, `get_current_loras()`.
- Produces: `class RequestLora(BaseModel)` with `path: str`, `strength: float = 1.0`. `GenerateRequest.loras: Optional[List[RequestLora]] = None`.
- Spec section 8 names `st status`. That command does not exist. `st models` prints `/api/models/status` as JSON, so the `current_loras` field reaches the operator with no Go change.
- Both routes call `resolve_request_loras(req.loras, get_mode_config().config.lora_root)` before admission. A `RequestLoraError` from validation or from admission gives HTTP 400 with the message, or WS `job:error` with the message. Admission is not called after a validation error.
- Spec error contract: a malformed entry fails the pydantic schema. HTTP gives 422 (FastAPI default). WS gives `job:error`. Neither calls admission.

- [x] **Step 1: Write the failing tests:**
  - `tests/test_ws_build_generate_request.py::test_loras_pass_through`: params `loras=[{"path": "a.safetensors", "strength": 0.8}]` give `req.loras[0].path == "a.safetensors"` and `.strength == 0.8`.
  - `tests/test_request_lora_routes.py::test_http_outside_root_is_400_and_not_admitted`.
  - `::test_http_valid_loras_reach_admission`: `admit_generation` gets `(mode, (LoraSpec(<resolved>, 0.8),))`.
  - `::test_http_admission_request_lora_error_is_400`.
  - `::test_ws_outside_root_is_job_error_and_not_admitted`.
  - `::test_models_status_has_current_loras`.
  - `::test_http_malformed_entries_are_422_and_not_admitted`: parametrize `{"path": "a", "strength": "abc"}`, `{"strength": 0.5}`, and `loras: "a.safetensors"`.
  - `::test_http_sidecar_escape_is_400_and_not_admitted`: the Task 3 sidecar layout under the patched root. HTTP 400, and `admit_generation` is not called.
  - `::test_ws_malformed_entry_is_job_error_and_not_admitted`: `{"path": "a", "strength": "abc"}`.
  Patch `get_mode_config` to a stub whose `config.lora_root` is `tmp_path`. Use `tests/snapshot_test_helpers.py` `install_mode_backed` for the pool.
- [x] **Step 2: Run** `python -m tests.run tests/test_request_lora_routes.py tests/test_ws_build_generate_request.py -- -q`. Expected: FAIL (no `loras` field).
- [x] **Step 3: Implement** the model, the field, and both admission paths.
- [x] **Step 4: Run** `python -m tests.run tests/test_request_lora_routes.py tests/test_ws_build_generate_request.py tests/test_ws_routes.py tests/test_generaterequest_serialization.py -- -q`. Expected: all pass.
- [x] **Step 5: Regenerate** the Go OpenAPI snapshot only if a Go test compares it to the server schema. Check `cli/go` for the snapshot test before editing.
- [x] **Step 6: Commit** `feat(server): request loras on /generate and WS (STABL-fvwsvjvn) — next: Task 6 st gen --lora + docs`.

---

### Task 6: `st gen --lora`, docs

**Files:**
- Modify: `cli/go/cmd/st/gen.go` (flag next to the `--controlnet` flags at :193)
- Modify: `cli/go/internal/config/precedence.go` (`Flags.Loras`, `applyExplicitFlags`)
- Test: `cli/go/cmd/st/gen_test.go`, `cli/go/internal/config/precedence_test.go`
- Modify: `docs/WORKER_SELECTION.md` step 4

**Interfaces:**
- Produces: `parseLoraFlag(v string) (map[string]any, error)` in `gen.go`. It splits at the last `:` when the suffix parses as a float. Otherwise the whole value is the path and strength is 1.0. An empty path is an error.
- `config.Flags.Loras *[]any`: nil when `--lora` is not given. `gen.go` sets it from the parsed flags.
- `applyExplicitFlags` rules (spec section 9): `Loras != nil` sets `p["loras"] = *Loras`. `Loras == nil` and `Mode != nil` delete `p["loras"]`. Both nil keep the inherited value.
- Help text: `LoRA for this request: path[:strength], repeatable. Needs a mode without loras:. A LoRA for the wrong model family fails the job and unloads the model.`

- [x] **Step 1: Write the failing Go tests:**
  - `TestParseLoraFlag`: `ZiT/j80z.safetensors:0.8` gives (`ZiT/j80z.safetensors`, 0.8). `a.safetensors` gives 1.0. `dir:with:colon` gives path `dir:with:colon`, 1.0. `:0.5` is an error.
  - `TestGenLoraFlagShipsInParams`: two `--lora` flags give two entries in order in the submitted params. No flag and no inherited params give no `loras` key.
  - `precedence_test.go::TestLoraFlagReplacesInheritedLoras`: baked and baseline `loras` both set, `Loras` set. Result is the flag list.
  - `::TestExplicitModeDropsInheritedLoras`: baked `loras` set, `Mode` set, `Loras` nil. Result has no `loras` key.
  - `::TestBaselineLorasDroppedByExplicitMode`: the same with baseline `loras`.
  - `::TestInheritedLorasKeptWithoutModeOrLora`: baked `loras` set, `Mode` and `Loras` nil. The baked list stays.
- [x] **Step 2: Run** `cd cli/go && ST_SERVER= go test ./cmd/st/ ./internal/config/ -run 'TestParseLoraFlag|TestGenLoraFlag|Loras'`. Expected: FAIL (undefined `parseLoraFlag`, unknown field `Loras`).
- [x] **Step 3: Implement** the flag.
- [x] **Step 4: Run** `cd cli/go && ST_SERVER= go test ./...`. Expected: ok for every package.
- [x] **Step 5: Update `docs/WORKER_SELECTION.md` step 4.** Add: SD1.5 and SDXL workers fuse diffusers LoRAs. A request `loras` list loads on a mode without `loras:`, and a changed set reloads the worker. A request LoRA for the wrong family fails the load, and the server has no model until the next valid request. Run `drift check`. Relink only after a prose review.
- [x] **Step 6: Commit** `feat(st): gen --lora; docs for request LoRAs (STABL-fvwsvjvn) — next: Task 7 suite + enigma live proof`.

---

### Task 7: Suite and enigma live proof

Steps 2-6 restart or use the enigma server. **Wait for the user's go before step 2.**

- [x] **Step 1: Run the CPU test container.** `docker compose -f docker-compose.test.yml build test && docker compose -f docker-compose.test.yml run --rm test`. Expected: no `FAILED` ID outside `docs/superpowers/plans/2026-11-12-zimage-cuda-container/baseline-failures.txt`.
- [x] **Step 2: Push the branch. On enigma, check it out and run `docker restart stability-toys-dev`.** The dev compose bind-mounts the source. Expected: container healthy. `/api/modes` lists `ZImage-Turbo` and `ZImage-Turbo-j80z`.
- [x] **Step 3: Generate with the branch-built `st`.** First record `T0=$(ssh enigma.lan date -u +%Y-%m-%dT%H:%M:%SZ)`. Prompt `j80z, portrait photo, soft window light`, `--seed 1234 --steps 9 --cfg 0 --size 1024x1024`, outfiles `run1.png`..`run4.png` under `<output_directory>/stabl-fvwsvjvn/`:
  1. `--mode ZImage-Turbo` (plain).
  2. `--mode ZImage-Turbo --lora ZiT/j80z.safetensors:0.8`.
  3. The same as 2. Expected: no reload (no new `merged LoRA` line).
  4. `--mode ZImage-Turbo` again. Expected: a reload.
  5. `--mode ZImage-Turbo-j80z --lora ZiT/j80z.safetensors`. Expected: refused, and the loaded model stays loaded.
- [x] **Step 4: Compare decoded pixels.** A scratchpad script loads each PNG with PIL as arrays and prints the mean absolute difference (MAD) and `array_equal`. Cast to `int16` or `float32` before the subtraction, because `uint8` subtraction wraps. Expected: run 2 vs run 1 MAD > 5. Run 3 vs run 2 `array_equal` true. Run 4 vs run 1 `array_equal` true. If run 4 vs run 1 is not bit-equal, record the MAD. Stop and report it, and do not claim the plain image came back.
- [x] **Step 5: Check the logs in the experiment window only.** `docker logs --since "$T0" stability-toys-dev 2>&1 | grep -E "merged LoRA|request LoRA|Loading mode"`. Expected: exactly one `merged LoRA /models/loras/ZiT/j80z.safetensors strength=0.8 modules=240`, exactly one `Mode ZImage-Turbo with 1 request LoRA(s)`, and no load between runs 2 and 3.
- [x] **Step 6: SDXL check.** List `/models/loras` on enigma. If an SDXL LoRA and an SDXL mode without `loras:` exist, repeat runs 1-2 and the step 4 run-2-vs-run-1 MAD check with them. Otherwise record "no SDXL LoRA on enigma" in FP.
- [x] **Step 7: Record evidence.** FP comment with counts, SHAs, image paths, MAD values, `T0`, and the log lines. Restore enigma to `main` only after the user decides on the PR.
