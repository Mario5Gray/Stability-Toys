# MPS Z-Image Plan Review

Reviewed revision: 78c7e48f.
Base revision: 6c3466050de8432a9ab174bcdacb46bdfe1fb93b.
Design: docs/superpowers/specs/2026-11-15-mps-zimage-design.md.
Plan: docs/superpowers/plans/2026-11-15-mps-zimage.md.
Parent issue: STABL-zubknmov.
Planning issue: STABL-eudrdkpt.

Status: review record for the plan revision. No implementation started.

## Verdict

Revise before approval.
The architecture is sound.
Two defects break the plan's own GREEN command.
Four factual errors break runbook steps.
No architectural objection.

## Blocking findings

### 1. Task 3 fails the worker-ref test

tests/test_mps_registration.py:152-166 asserts `cls.execution_policy is MPS_TORCH_POLICY` for every family in `WORKER_REF_FAMILIES`.
The guard at tests/test_mps_registration.py:171-175 requires that list to equal `MPS_FAMILY_BINDINGS`.
The plan adds `zimage` to the family set.
The new worker uses `ZIMAGE_MPS_TORCH_POLICY`.
The parametrized case then fails.
Task 3 Step 1 does not instruct a fix.
Fix: change the assertion to a backend identity check, or use a per-family expected policy.

### 2. test_backend_runtimes.py has no zimage assertion

Plan file map line 82 says an existing zimage-no-MPS assertion becomes positive.
Task 3 Step 1 repeats the instruction.
Search of tests/test_backend_runtimes.py finds no zimage text.
The MPS negative case is SDXL, at tests/test_mps_registration.py:88.
Fix: remove the file from Task 3, or name the real edit target.

## Factual findings

### 3. Default negative-template check cannot pass

server/mode_config.py:407 loads `default_negative_prompt_template`.
server/model_routes.py:266 serves it in mode data.
No generation path applies it.
The only template read in backends is the permission check at backends/zimage_cuda_worker.py:270-276.
Runbook Task 4/5 Step 3 says to omit `--negative` and verify the resolved template in PNG metadata.
The template is never resolved.
Fix: remove the check, or record absent substitution as a baseline limitation.

### 4. Acceptance case E has no input

Case E needs a wrong-family LoRA or a known partial-fit file (plan line 729).
The required input list names only two compatible LoRAs per variant.
Fix: add one wrong-family LoRA path to the required inputs.

### 5. Proof filenames are undefined

The comparison snippet at plan lines 781-789 reads `plain-restored.png` and `request-lora-07.png`.
The case table names no output filenames.
Fix: freeze one filename per case.

### 6. Asset inventory is too weak

The preparation code checks only that `model_index.json` exists (plan lines 594-596).
Family resolution needs `transformer_kind == "zimage"` (backends/family_profiles.py:125-133).
A folder can exist and still fail family resolution.
Fix: assert family resolution in the inventory step.

## Minor findings

- The policy test table row for a non-bf16 `CUDA_DTYPE` tests constructor behavior. That code is backends/zimage_cuda_worker.py:126-136. Move the row to the worker tests.
- The worker test does not assert the policy identity. Add `worker.execution_policy is ZIMAGE_MPS_TORCH_POLICY`.
- The new policy imports the private name `_bool_env`. Define the helper locally, or make it public.
- The tiny Qwen3 snippet uses `torch_dtype=`. The installed transformers emits a deprecation warning. `dtype=` works on transformers 4.57.6.
- Task 2 lists Task 1 as a dependency. Encoder-store parity does not need the policy work. Remove the edge to allow parallel work.
- Task 6 must compare exact drift pairs. The 48 pairs are recorded in 2026-11-15-mps-zimage-drift-baseline.txt.

## Risk

Exact pixel equality gates the following cases:

- P1 equals P2.
- P3 equals P1.
- Z equals P1.
- RZ equals P1.

No repo acceptance spike gates on bit equality for MPS.
This is the likely acceptance blocker.
Add a repeat procedure before a product defect claim.
Run each equality case three times and record every result.

## Verified claims

- Commit 78c7e48 has parent 6c34660 and changes only the two documents.
- `git diff --check 6c34660..78c7e48` is clean.
- The plan has 29 fenced blocks: 12 Python and 17 Bash.
- All 12 Python blocks compile. All 17 Bash blocks pass `bash -n`.
- `drift check` reports 48 stale pairs.
- Runtime versions match the design: torch 2.10.0, transformers 4.57.6, diffusers 0.39.0, quanto 0.2.7.
- The FP tree has the parent and six child issues with the ledger IDs and titles.
- The worker uses its execution policy before weight load.
- The only CUDA calls in the Z-Image worker are the peak hooks at backends/zimage_cuda_worker.py:222-231.
- The `_log_job` signature and call site match the planned override.
- The Z-Image worker refuses ControlNet in `run_job` at backends/zimage_cuda_worker.py:312-313.
- `ExecutionCapabilities(True, False, False)` means img2img yes, ControlNet no, combined no (backends/platforms/base.py:23-26). It matches the CUDA binding.
- The encoder store key excludes construction dtype and device. The module header states this.
- Store hit and miss logs exist (backends/zimage_text_encoder.py:198, 213). Restart-hit proof is possible.
- Mode defaults match the design: Turbo 9 steps at 0.0, Base 30 steps at 4.0, 1024x1024, flow_match_euler, int8 (conf/modes.yml:261-287).
- Every CLI flag in the runbook exists.
- Every environment variable in the runbook exists.
- The `/generate` route exists at server/lcm_sr_server.py:618.
- `/api/models/status` exposes `current_loras` and the capability fields, and has no authority epoch.
- The `server/run.py` port is hardcoded to 4200.
- The Turbo and Base assets are absent at the inspected native paths.
- `agent-policy.rdf` is absent from the installed root and the source root.
- docs/WORKER_SELECTION.md:77 still says zimage has no MPS binding.

## Resolved item

The plan sets `FP_AGENT_NAME='sigma'` for implementation commands.
This Claude session is theta.
sigma is a separate agent.
The plan now uses `FP_AGENT_NAME='theta'`.
