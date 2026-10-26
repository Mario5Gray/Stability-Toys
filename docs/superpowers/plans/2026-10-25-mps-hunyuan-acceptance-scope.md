# MPS HunyuanDiT Acceptance Client Repair Plan

> **For agentic workers:** Use `superpowers:executing-plans` to execute this plan task by task. Do not dispatch subagents. The issue is STABL-dvozpehl. This document is a plan; no implementation or new live acceptance is claimed.

**Goal:** Make the four HunyuanDiT MPS acceptance clients report real admission and current platform refusals without treating widened scope as a regression.

**Architecture:** A small pure helper decides whether a negative control applies to the active HunyuanDiT MPS cell. The clients read `/api/models/status` after their first successful HunyuanDiT render, require `current_mode == "HunyuanDiT"`, and use the reported ControlNet scope and combined-operation capability. Applicable controls require an exact platform `job:error`; inapplicable controls remain visible as `not_applicable` and never count as passes. Historical records in `docs/MPS_BACKEND.md` keep their original commit attribution.

**Tech stack:** Python 3.12, WebSocket acceptance clients, pytest through `python -m tests.run`.

**Baseline:** Plan against local `origin/main` at `741a6f0`. Recheck the implementation base before coding. The current MPS HunyuanDiT cell admits Canny, Depth and Pose; `max_attachments=1`; `supports_img2img_and_controlnet=false` (`backends/platforms/mps_bindings.py`). `/api/models/status` describes the active snapshot's family (`server/model_routes.py`). The checked-in HunyuanDiT mode also limits attachments to one (`conf/modes.yml`), so a platform two-attachment refusal requires a host mode config that permits two.

**Evidence checkpoint before edits:** Run `FP_AGENT_NAME='sigma' fp context STABL-dvozpehl`, `git rev-parse HEAD`, and `git status --short`. Inspect the historical Canny record at `58084f6`, Depth widening at `ef13200`, and Pose widening at `3542bd6` with `git show --stat <commit>` and the matching sections of `docs/MPS_BACKEND.md`. Expected: the two type refusals belong to older cell scopes; the current binding lists Canny, Depth, and Pose. If the implementation base differs, recheck the file map and tests before following the steps below.

## Scope and evidence rules

- This repairs acceptance clients. It does not widen the MPS worker's capabilities or change admission behavior.
- Canny's old Depth refusal and Depth's old Pose refusal are historical measurements. Keep the old documents tied to their recorded commits; do not restate those measurements as current.
- A client must fail if the status request fails, the payload is malformed, `backend != "mps"`, or `current_mode != MODE`. Do not turn missing information into `not_applicable`.
- `not_applicable` means the active cell does not exclude the control. It is neither a pass nor evidence that the whole request would pass mode policy. The required render and artifact checks remain required.
- The Pose `--map` file is a locally produced map. Its shape check cannot prove the server map bytes. Label it `local_map_is_pose_map` in the report and documentation. STABL-wlimlymz owns retrieval of the server artifact.
- A live M1 Max rerun can guard against a future change in request ordering or status timing. It is optional for this client repair; offline tests and exact source inspection are the completion evidence.

## File map

| File | Responsibility |
| --- | --- |
| `spikes/mps_hunyuan_scope.py` | Pure status validation, applicability, and expected platform message construction. No network or MPS import. |
| `spikes/mps_hunyuan_canny_acceptance.py` | Correct Depth fixture, live status read, conditional controls, admission check. |
| `spikes/mps_hunyuan_depth_acceptance.py` | Correct Pose fixture, live status read, conditional controls, admission check. |
| `spikes/mps_hunyuan_pose_acceptance.py` | Live status read, conditional attachment and combined controls, admission check, honest local-map label. |
| `spikes/mps_hunyuan_img2img_acceptance.py` | Replace hardcoded admission success with observed render results. |
| `tests/test_mps_hunyuan_acceptance_clients.py` | Pure applicability, message, fixture, and admission-result tests. No server or MPS host. |
| `docs/MPS_BACKEND.md` | Explain current client semantics without rewriting historical result rows. |

## Task 1: Pure current-scope decision and message builder

**Files:** Create `spikes/mps_hunyuan_scope.py`; create `tests/test_mps_hunyuan_acceptance_clients.py`.

1. Write failing tests for `validate_status(status, mode)`, `evaluate(scope, capability, control)`, and `expected_message(scope, control)`. Use status fixtures with `backend="mps"`, `current_mode="HunyuanDiT"`, `capabilities.controlnet_scope={"allowed_control_types":["canny","depth","pose"],"max_attachments":1}`, and `supports_img2img_and_controlnet=False`.
2. Cover `current_mode="lcm-general"`, a missing scope, missing combined flag, wrong backend, and malformed scope with explicit errors. In particular, `lcm-general` must fail before any control can become `not_applicable`. A present `allowed_control_types: null` or `max_attachments: null` means no platform narrowing and is valid.
3. Cover these decisions:

   | Control | Canny-only, max 1, combined false | Canny/Depth/Pose, max 1, combined false | Canny/Depth/Pose, max 2, combined true |
   | --- | --- | --- | --- |
   | `depth` | `applies` | `not_applicable` | `not_applicable` |
   | `pose` | `applies` | `not_applicable` | `not_applicable` |
   | `two_attachments` | `applies` | `applies` | `not_applicable` |
   | `combined_pose` | `applies` | `applies` | `not_applicable` |

4. Implement `evaluate(scope, capability, control) -> Literal["applies", "not_applicable"]`. `depth` and `pose` read ordered `allowed_control_types`; `two_attachments` reads `max_attachments`; `combined_pose` reads `supports_img2img_and_controlnet`. Do not infer combined capability from ControlNet scope. Reject unknown controls and missing fields.
5. Implement `expected_message` using the live ordered `allowed_control_types` for `allowed: ...`, the live `max_attachments` for the attachment refusal, and the existing family capability refusal for `combined_pose`. Do not keep a copied `allowed:` literal in a client.
6. Run `python -m tests.run tests/test_mps_hunyuan_acceptance_clients.py -- -q --no-cov`. Expected: the new tests fail before implementation and pass afterward. Commit the helper and tests as one focused change, with `STABL-dvozpehl` in the message.

**Decision contract:** The helper has no server imports. It accepts an already validated status capability payload and returns only the two strings above. Callers decide whether to submit a control request. A platform refusal counts only when the WebSocket terminal type is `job:error` and its error text equals the generated expected message. A timeout, mode-policy refusal, or generic `error` frame fails the check.

## Task 2: Repair fixtures before making refusals conditional

**Files:** Modify `spikes/mps_hunyuan_canny_acceptance.py` and `spikes/mps_hunyuan_depth_acceptance.py`; extend `tests/test_mps_hunyuan_acceptance_clients.py`.

1. Write tests that call Canny `attachment(source_ref="src", control_type="depth")` and Depth `attachment(source_ref="src", control_type="pose")`. Assert `control_type`, `model_id`, and `preprocess.id` agree. Assert the Canny and Depth defaults retain their current thresholds and options.
2. Run the focused file and confirm the fixture assertions fail on the existing clients. Import clients by file path or make `spikes` importable without changing their direct `python spikes/...py` entry points.
3. In Canny, select `hunyuandit-canny` plus Canny preprocess options for `canny`, and `hunyuandit-depth` plus Depth preprocess options for `depth`. In Depth, select `hunyuandit-depth` plus Depth options for `depth`, and `hunyuandit-pose` plus Pose options for `pose`. Reject an unknown type. Keep the positive render fixture byte-for-byte equivalent.
4. Run `python -m tests.run tests/test_mps_hunyuan_acceptance_clients.py -- -q --no-cov`. Expected: fixture tests pass. Commit the two clients and focused tests with `STABL-dvozpehl`.

**Reason:** The reported fixture problem is listed under STABL-wzaicgbg as an unverified finding, not finding 4 of that issue. This task verifies it, then fixes it here because dynamic applicability would otherwise expose the mismatched request. Cross-reference the issue in the completion report; do not claim its other findings are resolved.

## Task 3: Make Canny, Depth, and Pose controls follow the active cell

**Files:** Modify `spikes/mps_hunyuan_canny_acceptance.py`, `spikes/mps_hunyuan_depth_acceptance.py`, `spikes/mps_hunyuan_pose_acceptance.py`; extend `tests/test_mps_hunyuan_acceptance_clients.py`.

1. Add a focused orchestration test with fake `render` and `get_json` calls. The first successful HunyuanDiT render must precede the decisive status read. A startup `lcm-general` status must cause failure. An active HunyuanDiT status must be recorded in the JSON report.
2. Move the status read from the start of each client to immediately after its first successful HunyuanDiT render. Use `validate_status` before reading the scope or combined flag. If the read fails or validation fails, print the report with the failure and exit nonzero.
3. For each negative control, call `evaluate`. If `not_applicable`, do not submit a job; record `{ "decision": "not_applicable", "reason": "active platform cell does not exclude this control" }`. If `applies`, submit the job and require the exact generated platform error. Record the active status, expected text, terminal type, actual text, and `refused_by_the_platform` boolean. Keep a check only for a submitted refusal; calculate overall success from required positive checks and all submitted refusal checks.
4. Add tests for Canny on current three-type scope, Depth on current three-type scope, Pose on current `max_attachments=1` and combined=false, and a simulated widened future cell. Verify `not_applicable` remains visible and is not counted as a pass. Verify a mode-policy message, wrong platform message, and timeout each fail an applicable control.
5. Run `python -m tests.run tests/test_mps_hunyuan_acceptance_clients.py -- -q --no-cov`. Expected: all focused tests pass. Commit the three clients and focused tests with `STABL-dvozpehl`.

## Task 4: Replace all four admission tautologies

**Files:** Modify the Canny, Depth, Pose, and img2img acceptance clients; extend `tests/test_mps_hunyuan_acceptance_clients.py`.

1. Write tests that pass a set of successful result records with `ok=True`, `backend="mps"`, and `seed=4242`; then change each field in turn. Assert only the fully matching set is admitted. Include the three img2img runs and all positive ControlNet runs.
2. Run the focused tests and confirm the existing `sha256 is not None` and hardcoded `True` checks fail these cases.
3. Preserve `ok` in each saved render record. Calculate `admitted` from every required positive render's `ok`, `backend`, and `seed`, with the expected count and labels. `render()` already sets `ok=True` only after a `job:complete` with an output; no PNG hash is an admission signal.
4. Rename Pose's map check and report field to `local_map_is_pose_map`; keep its local-input guard and numeric shape measurements. Change its description and `docs/MPS_BACKEND.md` wording so the check claims only the operator-supplied map's shape. Do not claim equality with the server's control map.
5. Run `python -m tests.run tests/test_mps_hunyuan_acceptance_clients.py -- -q --no-cov`. Expected: all focused tests pass. Commit the four clients, tests, and scoped documentation with `STABL-dvozpehl`.

## Task 5: Verify and report the boundary

**Files:** `docs/MPS_BACKEND.md` only if Task 4 did not already cover all wording; no production files.

1. Run `python -m tests.run tests/test_mps_hunyuan_acceptance_clients.py -- -q --no-cov`. Run `python -m ruff check spikes/mps_hunyuan_scope.py spikes/mps_hunyuan_canny_acceptance.py spikes/mps_hunyuan_depth_acceptance.py spikes/mps_hunyuan_pose_acceptance.py spikes/mps_hunyuan_img2img_acceptance.py tests/test_mps_hunyuan_acceptance_clients.py`. If `docs/MPS_BACKEND.md` changes, run `drift refs docs/MPS_BACKEND.md` and `drift check --changed docs/MPS_BACKEND.md`.
2. Inspect `git diff --check`, the named-file diff, and `git status --short`. Preserve unrelated dirty files. Stage only plan-owned paths.
3. Report the old historical commit-scoped measurements separately from new offline tests. State that offline tests prove client selection, messages, and reporting; they do not prove live MPS rendering or server map-byte identity.
4. If a configured M1 Max server is already available, optionally rerun the three ControlNet clients and img2img client with `--out` prefixes under `/tmp`. Record the server commit, active status payload, mode config attachment limit, reports, and any failures. This is a guard against future reordering, not a required gate for this repair. If the host mode permits only one attachment, do not claim the two-attachment response tested the platform limit.

**Completion condition:** Focused tests cover every applicability branch, exact error matching, wrong active family, coherent fixtures, four real admission checks, and the honest local-map label. The four clients produce no false failure solely because the MPS ControlNet scope widened. No implementation step may turn a skipped control into a pass.
