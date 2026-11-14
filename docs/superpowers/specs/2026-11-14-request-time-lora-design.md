# Request-Time LoRA — Design

FP issue: STABL-fvwsvjvn
Status: approved 2026-11-14 (revision 3: request LoRAs only on modes without `loras:`)
Builds on: STABL-ykowysrg (HunyuanDiT mode LoRA, PR #161), STABL-gftujmyi (Z-Image mode LoRA, PR #162)
Proven host: enigma (CUDA). Container `stability-toys-dev` runs `docker-compose.dev.yml` with source bind mounts.

## Intent

The operator wants to select LoRAs and strengths in each generate request.
Today a LoRA is active only when a mode in `conf/modes.yml` lists it.
A new strength or a new LoRA needs a `modes.yml` edit and a mode reload.

The feature applies to every family: SD1.5, SDXL, HunyuanDiT, and Z-Image.

## Goals

- A generate request carries a `loras` list. Each entry has `path` and `strength`.
- Request LoRAs run only on a mode that has no `loras:` list. A request with LoRAs for a mode with `loras:` is refused.
- The operator makes a plain mode with no `loras:` for request-time LoRAs.
- If the resulting LoRA set differs from the loaded set, the server reloads the worker, then runs the job.
- If the set is the same, the job runs with no reload.
- The loaded set stays loaded after the job ("what you name is what you get").
- `st gen --lora <path>[:<strength>]` sends the list. The flag is repeatable.

## Non-goals

- A write to `modes.yml` or to any config file.
- An in-place LoRA swap without a reload (approach B). It can come later under the same request field.
- Changes to the old `style_lora` request field or `STYLE_REGISTRY` (`backends/styles.py`).
- New request-LoRA entries in `st modes` output.
- Scheduling that groups jobs by LoRA set.

## Anti-goals

- A request LoRA path that reads a file outside `lora_root`.
- A job that runs on a LoRA set other than the set its request named.
- LoRA weights left in the base weights after a switch to a set without them.
- A config reload (`PUT /api/modes`) that drops the active request LoRAs without notice.

## Constraints

- The HTTP route (`server/lcm_sr_server.py:621`) and the WS route (`server/ws_routes.py:251`) both admit through `admit_generation(req.mode)`.
- The governor identifies an authority by `mode_name` only. Comparisons: `admit_generation` (`backends/governor.py:1505`), `_switch_shortcircuit` (`:525`), the dispatch fast path, and `ModeSwitchJob.target_mode`.
- `_load_mode` tears down the old worker before it builds the new one. A new load reads clean weights from disk.
- `_reload_from_snapshot` rebuilds the worker from `snapshot.mode` after idle eviction.
- Both worker handles call `apply_mode_loras(worker, mode)` with `mode.loras` and `mode.lora_paths` (`backends/mode_loras.py`).
- `configure_loras` exists on the HunyuanDiT CUDA worker (MPS inherits) and on the Z-Image CUDA worker. The SD1.5 and SDXL workers (`DiffusersCudaWorker`, `DiffusersSDXLCudaWorker`, `DiffusersMpsWorker`) do not have it.
- The SDXL CUDA worker admits runtime fp8 quantize and offload modes. A fuse cannot write into fp8 or meta weights.
- The SD and SDXL workers load `STYLE_REGISTRY` adapters at init. The registry is empty today.
- `ModeConfig` crosses the subprocess spawn boundary. It must pickle.
- A Z-Image reload costs about 40 s on enigma.

## Approaches

| Approach | LoRA change costs | Code | Result |
|---|---|---|---|
| A. Derived mode + reload per LoRA-set change | One worker reload | Smallest. Reuses the governor switch path | **Picked by operator** |
| B. In-place swap: restore saved base weights, merge new LoRAs | Seconds | Per-family restore. About 1.3 GB CPU RAM for HunyuanDiT | Rejected for now. Possible later speed-up |
| C. A now, B later | A, then seconds | A now | Not picked. B stays open as a separate issue |

## Design

### 1. Request field

`GenerateRequest` gets `loras: Optional[List[RequestLora]] = None`.
`RequestLora` has `path: str` and `strength: float = 1.0`.
The WS builder `_build_generate_request` reads the same field.

Absent `loras` and `loras: []` mean "no request LoRAs".

### 2. Path and strength validation

Validation runs before admission. A failure returns HTTP 400 or WS `job:error`. No switch is queued.

- A relative path resolves against `lora_root`.
- An absolute path is admitted only when it resolves inside `lora_root`.
- Both sides use `Path.resolve()`, so a symlink cannot escape the root.
- The path must exist as a file or a directory.
- The strength must be finite.
- Two request entries must not resolve to the same file.
- The target base mode (section 4) must have no `loras:` list. Otherwise the request is refused with a message that names the mode and says to use a mode without `loras:`. (Operator decision, 2026-11-14.)

### 3. Authority key

The authority identity becomes the pair (base mode name, request LoRAs).

- `ActiveModelSnapshot` gets `request_loras: tuple[LoraSpec, ...] = ()`. `mode_name` stays the base mode name.
- Each `LoraSpec` holds the resolved absolute path and the strength as a float.
- Two keys are equal when the names are equal and the tuples are equal (order and exact floats).
- Every comparison listed in Constraints compares keys, not names.
- `ModeSwitchJob` carries the key.

### 4. Target selection in `admit_generation`

| `req.mode` | `req.loras` | Target |
|---|---|---|
| absent | absent or empty | The terminal authority, unchanged (today's behavior). Request LoRAs stay. |
| absent | set | (terminal base mode name, request LoRAs) |
| set | absent or empty | (`req.mode`, ()). A loaded request-LoRA set causes a reload to the mode as configured. |
| set | set | (`req.mode`, request LoRAs) |

With no terminal authority and no `req.mode`, a request with `loras` targets the default mode.

`switch_mode(name)` (`st modes switch`) targets (name, ()).

### 5. Derived mode

`_resolve_target(mode_name, request_loras)` deep-copies the mode as today.
With request LoRAs, the base mode has no `loras:` (section 2).
The copy gets the request LoRAs as `mode.loras` (`LoRAConfig` entries) and `mode.lora_paths` (absolute paths), in request order.
The handle hook applies the derived mode with no change.

Request LoRAs never mix with mode LoRAs. (Operator decision, 2026-11-14.)

Idle eviction keeps the snapshot. A demand reload restores the request LoRAs.

### 6. Config reload

`reload_current_mode` and `reload_if_current` re-target the full current key.
The base mode is reread from config. The request LoRAs stay.
This is my call. It follows the anti-goal about config reloads.

### 7. SD1.5 and SDXL workers

Add `configure_loras` to `DiffusersCudaWorker`, `DiffusersSDXLCudaWorker`, and `DiffusersMpsWorker`.
It uses the Z-Image method: diffusers load, `set_adapters`, `fuse_lora`, then removal of the adapter layers.

- Fuse components: `unet` plus each text encoder the pipeline has (`text_encoder`, `text_encoder_2`).
- Refuse fp8 runtime quantize and sequential CPU offload before any load.
- Refuse a non-finite strength, a LoRA that fits no module, and a LoRA that fits only part of the model.
- If any `STYLE_REGISTRY` adapter is loaded, refuse. The two LoRA paths do not combine.

Move the shared diffusers logic into one helper that the Z-Image worker also uses. The Z-Image behavior and tests stay the same.

### 8. Status and logs

- The runtime status shows the base mode name and the full active LoRA list (mode plus request), each with path and strength.
- `st status` prints the LoRA list when it is not empty.
- The governor logs one line per switch: base mode name and request LoRA count.
- Fix the `vram_bytes=0` registration in `_reload_from_snapshot` only if the builder finds a measured value there. Otherwise leave it.

### 9. `st gen --lora`

`--lora <path>[:<strength>]`, repeatable.
The part after the last `:` is the strength when it parses as a float. Otherwise the whole value is the path and the strength is 1.0.
The flag maps to `GenParams.Loras` and the WS `loras` field.

### 10. Failure behavior

A LoRA that fails `configure_loras` fails the mode load. The job gets `ModeLoadFailedError`.
The old worker is already gone, so the server has no model until the next request loads one.
This matches a bad mode today. Validation (section 2) catches path errors before the switch.
A wrong-family LoRA is found only at load.

This is a known limitation. It gets no FP issue. Reasons:
- A bad mode in `modes.yml` has the same behavior today.
- The next request that names a valid set loads a model again.
- A fix needs a per-family fit check before teardown, or a fallback reload of the old authority. Both cost more than the limitation.

The implementation documents the limitation in `docs/WORKER_SELECTION.md` step 4. The `st gen --lora` help text says that a LoRA for the wrong family fails the job and unloads the model.

## Verification

- Unit tests first (TDD):
  - Path validation: relative, absolute inside, absolute outside, symlink escape, missing, non-finite strength.
  - Authority key: equal sets do not reload. A different strength reloads. A no-mode request keeps the set. A request that names a mode and no LoRAs reloads that mode as configured.
  - Idle eviction plus demand reload keeps the request LoRAs.
  - Config reload keeps the request LoRAs.
  - The derived mode holds exactly the request LoRAs, in request order.
  - A request with LoRAs for a mode with `loras:` is refused before any switch. This includes a no-mode request while such a mode is loaded.
  - Two request entries for the same file are refused.
  - SD1.5 and SDXL `configure_loras` with tiny real pipelines: fused weight math, text encoder fuse, refusals.
  - Z-Image tests still pass on the shared helper.
  - Go: `--lora` parse and the WS payload.
- CPU container suite: no new failures against the 19-ID baseline.
- Enigma live:
  - `st gen --mode ZImage-Turbo --lora ZiT/j80z.safetensors:0.8` with the same seed as a plain run. The images differ. The log shows the merge line.
  - A second request with the same LoRA set does not reload.
  - A request for plain `ZImage-Turbo` reloads and matches the plain image.
  - `st gen --mode ZImage-Turbo-j80z --lora ZiT/j80z.safetensors` is refused, and the loaded model stays loaded.
  - SDXL: the same check if enigma has an SDXL LoRA. The builder lists `/models/loras` first.

## Left to the builder

- The exact module and name of the shared diffusers LoRA helper.
- The status field name for the active LoRA list.
- How the dispatch fast path reads the key.
- The tiny SD1.5 and SDXL test pipelines.
