# HunyuanDiT Mode LoRA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. AGENTS.md forbids subagent-driven development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A mode `loras:` list in `conf/modes.yml` merges Tencent HunyuanDiT LoRAs into the transformer at mode load, and `st modes` shows them.

**Architecture:** A pure-torch merge module (`backends/hunyuandit_lora.py`) validates and merges one Tencent PEFT file. A mode helper (`backends/mode_loras.py`) turns a mode into picklable `LoraSpec` values and calls the worker hook `configure_loras`. Both worker handles call the helper after `configure_conditioning`. The HunyuanDiT CUDA worker implements the hook. The MPS worker inherits it.

**Tech Stack:** Python 3.12, torch, safetensors, diffusers 0.39, pytest. Go (cobra) for `st`.

**Spec:** `docs/superpowers/specs/2026-11-14-hunyuandit-mode-lora-design.md`

**FP issue:** STABL-ykowysrg. Branch: `feat/stabl-ykowysrg-hunyuandit-mode-lora`.

## Global Constraints

- Python commands run in the `stability-toys` conda env: `conda activate stability-toys`, then `python -m pytest ...`.
- Go commands run in `cli/go`: `go test ./...`.
- Before you edit a file, run `drift refs <path>`. Before you declare done, run `drift check`.
- Every commit message names STABL-ykowysrg and the exact next step.
- Strength is fixed at mode load. Alpha is not applied.
- Tencent key prefix: `base_model.model.`. A directory path resolves to `<dir>/adapter_model.safetensors`.
- Merge math: `delta = B.float() @ A.float()` on the weight's device. New weight = `(W.float() + strength * delta_chunk).to(W.dtype)`.
- Validate every LoRA of a mode before the first weight write. A failure changes no weight.
- A quantized transformer (`_quantize == "fp8"` or `_checkpoint_precision == "fp8"`) refuses the merge before any file read.
- A mode with LoRAs on a worker without `configure_loras` fails the mode load.
- Do not commit the enigma `conf/modes.yml` edit.
- Do not change the existing per-request `style_lora` path.

## Review Focus

1. A LoRA trained for another family (SDXL file in a HunyuanDiT mode). Expect a clear key-set error and no weight change. Task 1 test `test_validate_rejects_sdxl_style_keys`.
2. A second LoRA in a mode fails validation after the first is valid. Expect no weight change at all. Task 3 test `test_second_bad_lora_changes_no_weight`.
3. A non-finite strength (`.nan`, `.inf` in YAML). Expect a refusal before any write. Task 3 test `test_non_finite_strength_refuses`.
4. Test stubs that are `Mock()` modes. A `Mock` attribute is not a list. Expect no LoRA work. Task 2 test `test_mock_mode_yields_no_specs`.
5. A switch from a LoRA mode to a plain mode. Expect base weights, because the governor rebuilds the worker. Task 5 live step 6.

---

### Task 1: Tencent LoRA merge module

**Files:**
- Create: `backends/hunyuandit_lora.py`
- Create: `tests/hunyuandit_lora_fakes.py`
- Test: `tests/test_hunyuandit_lora.py`

**Interfaces:**
- Produces:
  - `class HunyuanDiTLoraError(ValueError)`
  - `TENCENT_PREFIX = "base_model.model."`
  - `resolve_lora_file(path: str | Path) -> Path`
  - `read_tencent_lora(path: str | Path) -> dict[str, torch.Tensor]` (CPU tensors, prefix stripped)
  - `read_adapter_config(lora_file: Path) -> dict` (sibling `adapter_config.json`, or `{}`)
  - `validate_tencent_lora(transformer: torch.nn.Module, lora: dict[str, torch.Tensor]) -> None`
  - `merge_tencent_lora(transformer: torch.nn.Module, lora: dict[str, torch.Tensor], strength: float) -> int` (calls validate first, returns the number of target weights changed)
- Test helper `tests/hunyuandit_lora_fakes.py`:
  - `make_fake_transformer(num_blocks=2, hidden=8, cross=6, pooler=5, dtype=torch.float32) -> nn.Module` with `blocks[i].attn1.{to_q,to_k,to_v,to_out[0]}`, `blocks[i].attn2.{to_q,to_k,to_v,to_out[0]}`, `time_extra_emb.pooler.q_proj`. All `nn.Linear`. `attn2.to_k` and `attn2.to_v` take `cross` inputs.
  - `make_fake_lora(transformer, rank=2, seed=0, prefix=True) -> dict[str, torch.Tensor]` with the 10 keys per block + 2 pooler keys.
  - `tencent_reference_merge(state_dict, lora, scale, num_layers) -> dict` — the pasted Tencent function, with `num_layers` as a parameter. Copy it verbatim otherwise.

Mapping (module constant). Source keys are `<src>.lora_A.weight` and `<src>.lora_B.weight`. Split the delta with `torch.chunk(delta, len(targets), dim=0)`:

| Source (per block `blocks.i.`) | Targets (`.weight`) |
|---|---|
| `attn1.Wqkv` | `attn1.to_q`, `attn1.to_k`, `attn1.to_v` |
| `attn1.out_proj` | `attn1.to_out.0` |
| `attn2.q_proj` | `attn2.to_q` |
| `attn2.kv_proj` | `attn2.to_k`, `attn2.to_v` |
| `attn2.out_proj` | `attn2.to_out.0` |
| `pooler.q_proj` (once) | `time_extra_emb.pooler.q_proj` |

The block count is `len(transformer.blocks)`. The expected key count is `10 * blocks + 2`. The changed-target count is `7 * blocks + 1`.

Validation rules, each a `HunyuanDiTLoraError` with the key name in the message:
- The key set equals the expected set. Report up to 5 missing and 5 extra keys.
- `A.ndim == 2`, `B.ndim == 2`, `B.shape[1] == A.shape[0]`.
- `A.shape[1]` equals the target `in_features`.
- `B.shape[0]` equals the sum of the target `out_features`. Each target has the same `out_features`.

- [ ] **Step 1: Write the failing tests** in `tests/test_hunyuandit_lora.py`:
  - `test_resolve_lora_file_accepts_dir_and_file` — a dir with `adapter_model.safetensors` resolves to that file. A file path resolves to itself.
  - `test_resolve_lora_file_missing_raises` — `pytest.raises(HunyuanDiTLoraError)`.
  - `test_read_strips_tencent_prefix` — save `make_fake_lora(prefix=True)` with `safetensors.torch.save_file`. Every returned key starts with `blocks.` or `pooler.`.
  - `test_read_adapter_config_optional` — no sibling file gives `{}`. A sibling with `{"r": 64, "lora_alpha": 64}` gives that dict.
  - `test_merge_matches_tencent_reference` — clone the state dict, run `tencent_reference_merge(..., scale=1.0, num_layers=2)`, run `merge_tencent_lora(t, lora, 1.0)`. Every tensor is `torch.allclose(atol=1e-6)`.
  - `test_strength_scales_delta` — with strength 0.5, `W_after - W_before` equals half the strength-1.0 delta (`atol=1e-6`).
  - `test_merge_returns_target_count` — returns `7 * 2 + 1 == 15`.
  - `test_fp16_weight_merge_computes_in_fp32` — a transformer with `dtype=torch.float16`. The result equals `(W.float() + delta).half()` exactly.
  - `test_validate_rejects_missing_key` — drop one key. Raises. All weights equal the clone.
  - `test_validate_rejects_extra_key` — add `blocks.0.mlp.fc1.lora_A.weight`. Raises.
  - `test_validate_rejects_wrong_shape` — replace one `lora_A` with rank-mismatched shape. Raises. No weight changed.
  - `test_validate_rejects_sdxl_style_keys` — a dict with only `lora_unet_down_blocks_0_attentions_0.lora_down.weight`. Raises. The message names a missing `blocks.0.attn1.Wqkv` key.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_hunyuandit_lora.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'backends.hunyuandit_lora'`.

- [ ] **Step 3: Implement `backends/hunyuandit_lora.py`** with the interfaces above. Use `safetensors.torch.load_file(path, device="cpu")`. Use `torch.no_grad()` and `param.data.copy_(...)` for the in-place write. Look up targets with `transformer.get_submodule(name).weight`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_hunyuandit_lora.py -q`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add backends/hunyuandit_lora.py tests/hunyuandit_lora_fakes.py tests/test_hunyuandit_lora.py
git commit -m "feat(hunyuandit): Tencent LoRA validate + merge module (STABL-ykowysrg) — next: Task 2 mode LoRA hook in worker handles"
```

---

### Task 2: Mode LoRA specs and the handle hook

**Files:**
- Create: `backends/mode_loras.py`
- Modify: `backends/worker_handle.py:122-129` (after the conditioning block)
- Modify: `backends/worker_handle_subprocess.py:262-270` (inside `if mode is not None:`, after conditioning)
- Modify: `backends/governor.py:583-584` (log line)
- Modify: `tests/_fault_worker.py` (add a LoRA recording worker)
- Test: `tests/test_mode_loras.py`, `tests/test_worker_handle.py`, `tests/test_subprocess_worker_handle.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class LoraSpec: path: str; strength: float` (picklable, crosses the spawn boundary)
  - `lora_specs_for_mode(mode: Any) -> list[LoraSpec]` — reads `mode.loras` and `mode.lora_paths`. Returns `[]` when `mode.loras` is absent or not a `list`/`tuple`. Raises `ValueError` when the two lists differ in length. Pairs `lora_paths[i]` with `float(loras[i].strength)`.
  - `apply_mode_loras(worker: Any, mode: Any) -> None` — no specs: return. Worker without a callable `configure_loras`: raise `RuntimeError(f"mode lists {n} LoRA(s) but worker {type(worker).__name__} does not support LoRAs")`. Else call `worker.configure_loras(specs)`.
  - Worker hook signature (Task 3 implements it): `configure_loras(self, specs: Sequence[LoraSpec]) -> None`.
  - `tests/_fault_worker.py`: `make_lora_recording_worker(worker_id, resolved, binding)`. Its `configure_loras` stores the specs. Its `run_job` returns `(b"LORAS:" + ";".join(f"{s.path}@{s.strength}").encode(), 0)` in the return shape `RecordingWorker` uses.

- [ ] **Step 1: Write the failing tests**
  - `tests/test_mode_loras.py`:
    - `test_specs_pair_paths_and_strength` — `SimpleNamespace(loras=[SimpleNamespace(path="a.safetensors", strength=0.8)], lora_paths=["/models/loras/a.safetensors"])` gives `[LoraSpec("/models/loras/a.safetensors", 0.8)]`.
    - `test_mock_mode_yields_no_specs` — `lora_specs_for_mode(Mock()) == []`.
    - `test_mode_without_loras_attr_yields_no_specs` — `SimpleNamespace()` gives `[]`.
    - `test_length_mismatch_raises` — 1 LoRA, 0 paths. `pytest.raises(ValueError)`.
    - `test_apply_refuses_worker_without_hook` — `pytest.raises(RuntimeError, match="does not support LoRAs")`.
    - `test_apply_calls_hook_with_specs` — a worker with a recording `configure_loras` receives the spec list.
    - `test_apply_skips_worker_without_hook_when_no_loras` — no error.
    - `test_lora_spec_pickles` — `pickle.loads(pickle.dumps(spec)) == spec`.
  - `tests/test_worker_handle.py`: `test_start_applies_mode_loras` — a factory worker with `configure_loras`, and a `_Mode` class with `conditioning`, `loras`, `lora_paths`. The hook gets one `LoraSpec` after `start`. Also `test_start_refuses_loras_without_hook` — `pytest.raises(RuntimeError, match="does not support LoRAs")`.
  - `tests/test_subprocess_worker_handle.py`: `test_subprocess_handle_applies_mode_loras_in_child` — `SubprocessWorkerHandle("tests._fault_worker.make_lora_recording_worker")` with `SimpleNamespace(model_path=None, conditioning=ConditioningConfig(), loras=[SimpleNamespace(path="j", strength=0.8)], lora_paths=["/m/j"])`. One job returns `b"LORAS:/m/j@0.8"`. Use `loras` entries that pickle (`SimpleNamespace` pickles).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_mode_loras.py tests/test_worker_handle.py tests/test_subprocess_worker_handle.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'backends.mode_loras'`. The handle tests fail on the missing hook call.

- [ ] **Step 3: Implement `backends/mode_loras.py`**, then call `apply_mode_loras(worker, mode)` in both handles right after the conditioning block. In the subprocess child, the call goes inside the existing `try`, so a refusal reaches the parent as a start failure.

- [ ] **Step 4: Fix the governor log line** at `backends/governor.py:583-584`. It runs after `start` succeeds. New text: `"[Governor] Mode %s merged %d LoRA(s) at load"`, with `mode_name` and `len(mode.loras)`. Use lazy `%` arguments.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_mode_loras.py tests/test_worker_handle.py tests/test_subprocess_worker_handle.py tests/test_governor.py tests/test_worker_pool.py -q`
Expected: all pass. The existing `Mock()` mode stubs keep passing, because `lora_specs_for_mode(Mock())` returns `[]`.

- [ ] **Step 6: Commit**

```bash
git add backends/mode_loras.py backends/worker_handle.py backends/worker_handle_subprocess.py backends/governor.py tests/_fault_worker.py tests/test_mode_loras.py tests/test_worker_handle.py tests/test_subprocess_worker_handle.py
git commit -m "feat(workers): apply mode LoRAs through configure_loras hook (STABL-ykowysrg) — next: Task 3 HunyuanDiT worker configure_loras"
```

---

### Task 3: HunyuanDiT worker `configure_loras`

**Files:**
- Modify: `backends/cuda_worker.py` — class `DiffusersHunyuanDiTCudaWorker` (starts at line 1701). Replace the comment at line 1772 with a pointer to `configure_loras`.
- Test: `tests/test_hunyuandit_lora_worker.py`

**Interfaces:**
- Consumes: Task 1 `resolve_lora_file`, `read_tencent_lora`, `read_adapter_config`, `validate_tencent_lora`, `merge_tencent_lora`, `HunyuanDiTLoraError`. Task 2 `LoraSpec`.
- Produces: `DiffusersHunyuanDiTCudaWorker.configure_loras(self, specs: Sequence[LoraSpec]) -> None`. `DiffusersHunyuanDiTMpsWorker` inherits it unchanged.

Order inside `configure_loras`:
1. No specs: return.
2. `self._quantize == "fp8"` or `self._checkpoint_precision == "fp8"`: raise `HunyuanDiTLoraError` that says the transformer is quantized. No file read.
3. Any `not math.isfinite(spec.strength)`: raise `HunyuanDiTLoraError`.
4. For every spec: resolve, read, read config, validate against `self.pipe.transformer`. Keep the results.
5. For every spec, in order: merge. Log one line: `"[%s] worker %s merged LoRA %s strength=%s r=%s alpha=%s targets=%d"` with `self.log_tag`, `self.worker_id`, the file path, strength, `cfg.get("r")`, `cfg.get("lora_alpha")`, and the merge count.

- [ ] **Step 1: Write the failing tests** in `tests/test_hunyuandit_lora_worker.py`. Build the worker with `DiffusersHunyuanDiTCudaWorker.__new__(...)`. Set `pipe = SimpleNamespace(transformer=make_fake_transformer())`, `_quantize = None`, `_checkpoint_precision = "fp16"`, `worker_id = 0`. Write LoRA files with `safetensors.torch.save_file` under `tmp_path`.
  - `test_configure_loras_merges_two_specs_additively` — two LoRAs at 1.0 and 0.5. The final weights equal two sequential `merge_tencent_lora` calls on a clone.
  - `test_configure_loras_accepts_directory_path` — the spec path is the directory.
  - `test_quantized_transformer_refuses_before_read` — `_quantize = "fp8"`, spec path does not exist. Raises `HunyuanDiTLoraError` with `"quantized"` in the message, not a missing-file error.
  - `test_checkpoint_fp8_refuses` — `_checkpoint_precision = "fp8"`. Raises.
  - `test_non_finite_strength_refuses` — strength `float("nan")`. Raises. No weight changed.
  - `test_second_bad_lora_changes_no_weight` — first LoRA valid, second has a missing key. Raises. All weights equal the clone.
  - `test_configure_loras_logs_rank_and_alpha` — with `adapter_config.json` `{"r": 2, "lora_alpha": 2}`, `caplog` has `"r=2 alpha=2 targets=15"`.
  - `test_mps_worker_inherits_configure_loras` — `DiffusersHunyuanDiTMpsWorker.configure_loras is DiffusersHunyuanDiTCudaWorker.configure_loras`.
  - `test_controlnet_pipe_reuses_base_pipe` — patch `backends.cuda_worker._hunyuandit_controlnet_pipeline_cls` with a fake class whose `from_pipe(pipe, **kw)` records `pipe`. `worker._build_controlnet_pipe(object())` passes `worker.pipe`. diffusers `from_pipe` reuses the component objects, so the merged transformer reaches the ControlNet path.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_hunyuandit_lora_worker.py -q`
Expected: FAIL with `AttributeError: ... has no attribute 'configure_loras'`. The MPS and ControlNet tests may already pass.

- [ ] **Step 3: Implement `configure_loras`** in `DiffusersHunyuanDiTCudaWorker` per the order above.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_hunyuandit_lora_worker.py tests/test_hunyuandit_lora.py tests/test_hunyuan_mps_worker.py tests/test_cuda_worker_capabilities.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add backends/cuda_worker.py tests/test_hunyuandit_lora_worker.py
git commit -m "feat(hunyuandit): merge mode LoRAs in configure_loras (STABL-ykowysrg) — next: Task 4 st modes LoRA line"
```

---

### Task 4: `st modes` LoRA line

**Files:**
- Modify: `cli/go/pkg/stclient/http.go:18-29` (type `Mode`) and `:72-100` (decode in `Modes`)
- Modify: `cli/go/cmd/st/modes.go:67-79` (render)
- Test: `cli/go/pkg/stclient/http_test.go`, `cli/go/cmd/st/modes_test.go`

**Interfaces:**
- Produces: `type ModeLora struct { Path string \`json:"path"\`; Strength float64 \`json:"strength"\` }`. `Mode.Loras []ModeLora \`json:"loras,omitempty"\``.
- Render: one line `  loras: <path>(<strength %g>), ...` after the schedulers line. No line when `Loras` is empty.

The server already sends `loras` per mode in `GET /api/modes` (`server/mode_config.py:1095-1102`). No OpenAPI regeneration.

- [ ] **Step 1: Write the failing tests**
  - `http_test.go`: `TestModesDecodesLoras` — body mode `"loras":[{"path":"HYDiT-LoRA/jade/adapter_model.safetensors","strength":0.8,"adapter_name":"lora_adapter_model"}]`. Decoded `Loras[0]` equals `ModeLora{Path: "HYDiT-LoRA/jade/adapter_model.safetensors", Strength: 0.8}`.
  - `modes_test.go`: `TestModesListRendersLoras` — output contains `loras: HYDiT-LoRA/jade/adapter_model.safetensors(0.8)`. A second mode without LoRAs: `strings.Count(out, "loras:") == 1`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd cli/go && go test ./pkg/stclient/ ./cmd/st/ -run 'Loras' -v`
Expected: FAIL. Compile error `unknown field Loras`, or a missing `loras:` line.

- [ ] **Step 3: Implement** the type, the decode field, and the render line.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd cli/go && go test ./...`
Expected: `ok` for every package.

- [ ] **Step 5: Commit**

```bash
git add cli/go/pkg/stclient/http.go cli/go/pkg/stclient/http_test.go cli/go/cmd/st/modes.go cli/go/cmd/st/modes_test.go
git commit -m "feat(st): show mode LoRAs in st modes (STABL-ykowysrg) — next: Task 5 suite + enigma live proof"
```

---

### Task 5: Suite, drift, and enigma live proof

Steps 3 to 8 change the running enigma server. **Wait for the user's go before step 3.**

**Files:**
- Modify: this plan (checkboxes, evidence notes)
- Evidence: images under `<output_directory>/stabl-ykowysrg/` on the host that runs `st`, and their paths in the FP comment

- [ ] **Step 1: Run the CPU test container**

Run: `docker compose -f docker-compose.test.yml build test && docker compose -f docker-compose.test.yml run --rm test > /private/tmp/claude-501/stabl-ykowysrg-suite.log 2>&1; tail -5 /private/tmp/claude-501/stabl-ykowysrg-suite.log`
Expected: the pytest summary shows no new failures against `main`. Record pass and fail counts.

- [ ] **Step 2: Run drift**

Run: `drift check`
Expected: no stale anchors. If stale, update the prose first, then `drift link`.

- [ ] **Step 3: Push the branch and check it out on enigma** (after user go)

```bash
git push -u origin feat/stabl-ykowysrg-hunyuandit-mode-lora
ssh enigma.lan 'cd ~/workspace/Stability-Toys && git fetch && git switch feat/stabl-ykowysrg-hunyuandit-mode-lora && git status --short conf'
```
Expected: `M conf/modes.yml` stays. The branch does not touch `conf/modes.yml`.

- [ ] **Step 4: Add a plain mode on enigma only**

Copy the `HunyuanDiT` block in enigma `conf/modes.yml` to `HunyuanDiT-plain`, without `loras:`. Do not commit.

- [ ] **Step 5: Rebuild and restart the server**

Run: `ssh enigma.lan 'cd ~/workspace/Stability-Toys && docker compose -f docker-cuda.yml build && docker compose -f docker-cuda.yml up -d'`
Expected: container `stability-toys` healthy.

- [ ] **Step 6: Generate the A/B/A images** with one prompt and one seed. `st` must target the enigma server (config `server`, or `--server`). Use `st modes switch`, not `st gen --mode` (STABL-eqyjwfdh). Relative `--outfile` paths join under `output_directory`. Prompt: `玉石绘画风格，一只猫在追蝴蝶`. Seed: `1234`.

```bash
st modes switch HunyuanDiT-plain && st gen '玉石绘画风格，一只猫在追蝴蝶' --seed 1234 --outfile stabl-ykowysrg/plain-1.png
st modes switch HunyuanDiT       && st gen '玉石绘画风格，一只猫在追蝴蝶' --seed 1234 --outfile stabl-ykowysrg/jade-0.8.png
st modes switch HunyuanDiT-plain && st gen '玉石绘画风格，一只猫在追蝴蝶' --seed 1234 --outfile stabl-ykowysrg/plain-2.png
st modes
```
Expected: `jade-0.8.png` differs from `plain-1.png` and shows the jade style. `plain-2.png` matches `plain-1.png` in content (renders may not be bit-identical). `st modes` shows `loras: HYDiT-LoRA/jade/adapter_model.safetensors(0.8)` under `HunyuanDiT`.

- [ ] **Step 7: Check the server log**

Run: `ssh enigma.lan 'docker logs stability-toys 2>&1 | grep -E "merged LoRA|merged [0-9]+ LoRA" | tail -5'`
Expected: `merged LoRA ... strength=0.8 r=64 alpha=64 targets=281` and `Mode HunyuanDiT merged 1 LoRA(s) at load`.

- [ ] **Step 8: Record evidence and close out**

Post one FP comment with the test counts, commit SHAs, image paths and log lines. Link each commit with `fp issue assign STABL-ykowysrg --rev <sha>`. Show the three images to the user. The user decides on the PR and on the enigma branch state.
