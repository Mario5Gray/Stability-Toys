# img2img + ControlNet — Follow-ups (Parallel Group C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Project policy forbids subagent-driven development — execute inline.

**Goal:** Prove the two existing metadata/provenance surfaces survive the new combined execution path, and document + contract-test the CLI/WS surface for combined requests — both verification/documentation work, no new product surfaces.

**Architecture:** No new code paths. Task 1 adds regression coverage against code that already exists unconditionally on the combined path (the `pnginfo.add_text("controlnet", ...)` call in `backends/cuda_worker.py` sits after the img2img/txt2img branch split, so it already fires for combined requests once Group B lands — this task proves that, it doesn't build it). Task 2 documents and contract-tests a CLI capability that already exists client-side (`st gen` can already send `--init-image` and `--controlnet` together; only the server rejects the combination today) — once Group B lands, the combination just works end-to-end.

**Tech Stack:** Python/pytest (metadata), Go (`cli/go`) + Python/pytest (WS contract).

**Prerequisite:** Both tasks depend on `docs/superpowers/plans/2026-07-08-img2img-controlnet-pipeline-wiring.md` (Group B) being merged — there is nothing to verify or document until combined execution actually runs.

**FP tree:** `STABL-ztaxgbhv` (parent). This plan covers `STABL-pligndni` (metadata) and `STABL-anhahetw` (CLI/WS contract).

## Global Constraints

- Verification-only: no new provenance schema fields, no asset-store changes, no frontend/gallery work (FP guardrail on `STABL-pligndni`).
- CLI/WS scope only: HTTP `/generate` remains txt2img-only for this phase — do not add `init_image_ref` to `GenerateRequest` or touch `CudaGenerationRuntime` (FP guardrail on `STABL-anhahetw`). Do not regenerate the OpenAPI snapshot (`cli/go/internal/openapi/openapi.gen.go`) — the WS `job:submit` contract this task documents is not part of the OpenAPI-described HTTP surface.
- Run Python tests with the Miniforge base env active. Run Go tests from `cli/go`: `go test ./... -v`.

## File Structure

- **Test:** `tests/test_worker_controlnet_metadata.py` (add combined-path PNG-chunk test; add the `StableDiffusionControlNetImg2ImgPipeline` fake this file needs, mirroring Group B's addition to `tests/test_cuda_worker_controlnet.py`).
- **Test:** `tests/test_controlnet_success_contract.py` (add a note-only combined-path variant of the WS `job:complete` artifact test, to document that this surface is provably independent of `init_image`).
- **Test:** `tests/test_ws_build_generate_request.py` (combined `init_image_ref` + `controlnets` passthrough test).
- **Test:** `cli/go/cmd/st/gen_test.go` (combined `--init-image` + `--controlnet` params test).
- **Modify:** `cli/go/USAGE.md` (document the combination under the existing `## ControlNet` section).

---

### Task 1: Verify metadata/provenance stamping on the combined path

**Files:**
- Modify: `tests/test_worker_controlnet_metadata.py`

**Interfaces:**
- Consumes: `DiffusersCudaWorker.run_job` (Group B), `ControlNetBinding` (`server/controlnet_execution.py:18-26`), the existing `_binding`/`_make_req`/`_make_worker`/`_fake_cache`/`_stamped_png` fixtures already in this file.

- [ ] **Step 1: Register the combined-pipeline fake this file needs**

`tests/test_worker_controlnet_metadata.py` stubs the `diffusers` module tree independently from `tests/test_cuda_worker_controlnet.py` (separate `sys.modules.setdefault` block at the top of each file) — Group B's fake for `StableDiffusionControlNetImg2ImgPipeline` living in the other test file does not cover this one. Add it here too. In `tests/test_worker_controlnet_metadata.py`, after:

```python
class _FakeStableDiffusionXLControlNetPipeline(_FakePipelineBase):
    @classmethod
    def from_pipe(cls, pipe, controlnet):
        return cls()
```

add:

```python
class _FakeStableDiffusionControlNetImg2ImgPipeline(_FakePipelineBase):
    @classmethod
    def from_pipe(cls, pipe, controlnet):
        return cls()
```

and after:

```python
sys.modules["diffusers"].StableDiffusionXLControlNetPipeline = _FakeStableDiffusionXLControlNetPipeline
```

add:

```python
sys.modules["diffusers"].StableDiffusionControlNetImg2ImgPipeline = _FakeStableDiffusionControlNetImg2ImgPipeline
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_worker_controlnet_metadata.py`:

```python
def test_run_job_writes_controlnet_chunk_on_combined_img2img_path(tmp_path: Path):
    worker = _make_worker()
    req = _make_req()
    control_png = _stamped_png(tmp_path)
    job = SimpleNamespace(
        req=req,
        init_image=_bare_png(),
        controlnet_bindings=[_binding("cn-1", "canny", "sdxl-canny", control_png, 0.8)],
    )
    fake_generator = MagicMock()
    fake_generator.manual_seed.return_value = fake_generator
    cache = _fake_cache()

    with patch("backends.cuda_worker.torch.Generator", return_value=fake_generator), \
         patch("backends.cuda_worker.torch.inference_mode") as mock_inference, \
         patch("backends.cuda_worker.torch.cuda.empty_cache"), \
         patch("backends.cuda_worker.PngImagePlugin.PngInfo") as mock_pnginfo, \
         patch("backends.controlnet_cache.get_controlnet_cache", return_value=cache):
        mock_inference.return_value.__enter__.return_value = None
        mock_inference.return_value.__exit__.return_value = None

        worker.run_job(job)

    add_text_calls = mock_pnginfo.return_value.add_text.call_args_list
    assert add_text_calls[0].args[0] == "lcm"
    assert add_text_calls[1].args[0] == "controlnet"
    payload = json.loads(add_text_calls[1].args[1])
    assert payload[0]["attachment_id"] == "cn-1"
    assert payload[0]["source"]["tool"] == "canny_map"
```

Note this test does **not** mock `_decode_control_image` (unlike `test_run_job_writes_controlnet_chunk_when_bindings_present`) — both `control_png` and `_bare_png()` are real 8x8 PNGs, so the real aspect-ratio validator and real `_decode_control_image`/`Image.open(...).resize(...)` calls run unmocked, exercising the actual combined code path end-to-end rather than just its metadata side-effect.

- [ ] **Step 3: Run to verify failure**

Run: `pytest tests/test_worker_controlnet_metadata.py -k combined_img2img -v`
Expected: FAIL if run before Group B lands (`NotImplementedError`); if Group B is already merged, this step should already PASS — in that case skip straight to Step 4 confirming green, since the "verify existing behavior" nature of this task means there may be nothing left to fix.

- [ ] **Step 4: Run full file to verify no regressions**

Run: `pytest tests/test_worker_controlnet_metadata.py -v`
Expected: PASS — all existing tests plus the new one. If Step 3 already passed, this step is the actual verification deliverable for `STABL-pligndni`: proof the combined path stamps metadata correctly with zero code changes needed.

- [ ] **Step 5: Document why the response-artifact surface needs no combined-path test**

`tests/test_controlnet_success_contract.py::test_ws_job_complete_includes_controlnet_artifacts` exercises `ws_routes._finish_generate`, which builds the `job:complete.controlnet_artifacts` frame purely from `req._controlnet_artifacts` (set during preprocessing) — it never reads `job.init_image`. This surface is structurally independent of img2img; no combined-path variant can exercise a code path the existing test doesn't already cover. Add a one-line comment recording this so a future reader doesn't wonder why no combined-path artifact-header test exists. In `tests/test_controlnet_success_contract.py`, above `def test_ws_job_complete_includes_controlnet_artifacts():`, add:

```python
# No combined-path (init_image + controlnets) variant needed: this frame is built
# from req._controlnet_artifacts alone (see server/ws_routes.py _finish_generate),
# which never reads job.init_image — img2img presence cannot affect this assertion.
```

- [ ] **Step 6: Commit**

```bash
git add tests/test_worker_controlnet_metadata.py tests/test_controlnet_success_contract.py
git commit -m "test(controlnet): verify PNG provenance stamping survives combined img2img+controlnet path (STABL-pligndni)"
```

---

### Task 2: Document + contract-test CLI/WS surface for combined requests

**Files:**
- Modify: `tests/test_ws_build_generate_request.py`
- Modify: `cli/go/cmd/st/gen_test.go`
- Modify: `cli/go/USAGE.md`

**Interfaces:**
- Consumes: `_build_generate_request` (`server/ws_routes.py`, unchanged by this task — already threads both `init_image` params and `controlnets` independently, per Group A's Task 1 which validates rejection *before* this function's output is used, not inside it), `buildGenParams` (`cli/go/cmd/st/gen.go`, unchanged — already threads `InitImage`/`--controlnet` independently).

- [ ] **Step 1: Write the WS-side contract test**

Append to `tests/test_ws_build_generate_request.py`:

```python
def test_build_generate_request_passes_through_init_image_ref_and_controlnets_together():
    params = {
        "prompt": "a cat",
        "init_image_ref": "abc123",
        "denoise_strength": 0.6,
        "controlnets": [
            {
                "attachment_id": "cn_1",
                "control_type": "canny",
                "map_asset_ref": "asset_a",
            }
        ],
    }
    req = _build_generate_request(params)
    assert req.denoise_strength == 0.6
    assert req.controlnets is not None
    assert len(req.controlnets) == 1
    # init_image_ref itself is WS-only and never reaches GenerateRequest (see
    # server/ws_routes.py handle_job_submit, which reads params.get("init_image_ref")
    # directly) — asserting its absence here documents that seam rather than
    # exercising a bug.
    assert not hasattr(req, "init_image_ref")
```

- [ ] **Step 2: Run to verify it already passes**

Run: `pytest tests/test_ws_build_generate_request.py -v`
Expected: PASS immediately — `_build_generate_request` already threads `denoise_strength` and `controlnets` independently today; this test documents/pins the existing passthrough contract rather than fixing a bug. (Group A's `reject_combined_img2img_controlnet` guard lives in `handle_job_submit`, one layer above this function, so it doesn't affect this unit test.)

- [ ] **Step 3: Write the CLI-side contract test**

Append to `cli/go/cmd/st/gen_test.go`:

```go
// TestBuildGenParamsCombinedInitImageAndControlnet pins that st gen can already
// build a request carrying both init_image_ref and controlnets — the CLI has never
// needed a change for the combined case; only the server rejected it (STABL-ztaxgbhv).
func TestBuildGenParamsCombinedInitImageAndControlnet(t *testing.T) {
	cn := `{"attachment_id":"a1","control_type":"canny","map_asset_ref":"fileref:M1"}`
	args := genArgs{Prompt: "an owl", InitImage: "fileref:R1", Controlnets: []string{cn}}
	p, err := buildGenParams(nil, args)
	if err != nil {
		t.Fatal(err)
	}
	if p["init_image_ref"] != "R1" {
		t.Fatalf("init_image_ref not threaded: %+v", p)
	}
	list, ok := p["controlnets"].([]any)
	if !ok || len(list) != 1 {
		t.Fatalf("controlnets not threaded: %+v", p["controlnets"])
	}
}
```

- [ ] **Step 4: Run to verify it already passes**

Run: `cd cli/go && go test ./cmd/st/... -run TestBuildGenParamsCombinedInitImageAndControlnet -v`
Expected: PASS immediately, for the same reason as Step 2 — `buildGenParams` threads `InitImage` and `Controlnets` independently already.

- [ ] **Step 5: Document the combination in USAGE.md**

In `cli/go/USAGE.md`, in the `## ControlNet` section, after the existing "From a JSON file" / preset examples, add:

```markdown
### Combined with img2img

`--init-image` and `--controlnet`/`--controlnet-file` can be given together on the
CLI — the request carries both `init_image_ref` and `controlnets`:

\`\`\`bash
st gen "an owl in watercolor style" \
  --init-image ./sketch.png \
  --controlnet "{\"attachment_id\":\"cn-1\",\"control_type\":\"canny\",\"map_asset_ref\":\"$MAP_REF\"}"
\`\`\`

Server-side support for this combination ships as part of `STABL-ztaxgbhv`. Before
that lands, the server rejects it fail-fast with a `job:error` naming "img2img"
rather than silently ignoring one half of the request.
```

- [ ] **Step 6: Run full suites for touched files**

Run: `pytest tests/test_ws_build_generate_request.py -v` and `cd cli/go && go test ./cmd/st/... -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/test_ws_build_generate_request.py cli/go/cmd/st/gen_test.go cli/go/USAGE.md
git commit -m "docs(cli): document and contract-test combined --init-image + --controlnet CLI/WS surface (STABL-anhahetw)"
```

---

## Self-Review

**FP issue coverage:**
- `STABL-pligndni` (metadata verification) → Task 1. ✓
- `STABL-anhahetw` (CLI/WS contract + docs) → Task 2. ✓

**Guardrail coverage:** Task 1 adds zero new provenance/schema/asset-store surfaces — the entire task is a regression test against code Group B already wrote for other reasons, plus a documentation comment explaining why one surface needs no test. Task 2 touches no HTTP code, adds no OpenAPI regeneration, and its "new" tests both pass without any implementation change — they pin an already-correct CLI/WS contract, matching the `STABL-anhahetw` guardrail exactly ("document/contract-test CLI+WS only... HTTP remains txt2img-only").

**Placeholder scan:** no TBD/TODO; Step 3 of Task 1 explicitly branches on "if this already passes, skip to Step 4" rather than hiding an ambiguous no-op step.

**Type consistency:** `_build_generate_request(params: dict)` (Task 2 Step 1) and `buildGenParams(cfg, args genArgs)` (Task 2 Step 3) are both used with their existing, unmodified signatures — this plan calls them, it doesn't change them.
