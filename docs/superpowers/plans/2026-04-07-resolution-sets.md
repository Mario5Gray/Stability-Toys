# Resolution Sets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add config-owned resolution sets, expose them through mode metadata, enforce them on both HTTP and WebSocket generation paths, and render mode-specific size choices with aspect ratios in the UI.

**Architecture:** Keep resolution policy in `conf/modes.yml` and `server/mode_config.py`, expose resolved entries via `/api/modes`, and enforce size validity through one shared backend helper used by both `/generate` and `ws_routes.py`. On the frontend, reuse the existing mode-default application path in `useGenerationParams` and extend the size formatter rather than adding a parallel formatting system.

**Tech Stack:** Python, FastAPI, Pydantic, YAML config loading, React, Vitest, Testing Library

---

## File Structure

- Modify: `server/mode_config.py`
  Add `resolution_set` and resolved `resolution_options` to mode config, parse top-level `resolution_sets`, and validate that `default` exists and every mode's `default_size` belongs to its resolved set.
- Create: `server/generation_constraints.py`
  Shared helper for applying mode size defaults and validating resolved size policy for both HTTP and WS generation paths.
- Modify: `server/lcm_sr_server.py`
  Replace inline size default logic with the shared constraint helper and reject invalid sizes before queue submission.
- Modify: `server/ws_routes.py`
  Apply the same mode-aware size finalization and validation before `GenerationJob` submission.
- Modify: `server/model_routes.py`
  Continue returning `config.to_dict()`, but tests must verify the response now carries `resolution_set` and `resolution_options`.
- Modify: `conf/modes.yml`
  Add `resolution_sets.default` and curated `resolution_sets.sdxl`, then assign `resolution_set` on relevant modes.
- Modify: `conf/modes.yaml.example`
  Mirror the new config contract in the example file.
- Modify: `lcm-sr-ui/src/utils/helpers.js`
  Extend `formatSizeDisplay()` to accept an optional aspect ratio and render `1024×1024 • 1:1`.
- Modify: `lcm-sr-ui/src/utils/generationControls.js`
  Extend `applyModeControlDefaultsToDraft()` so mode switches also reset draft size to `default_size`.
- Modify: `lcm-sr-ui/src/hooks/useGenerationParams.js`
  Flow the size reset through the existing `applyModeControlDefaults()` path instead of introducing a second reset mechanism.
- Modify: `lcm-sr-ui/src/components/options/OptionsPanel.jsx`
  Replace `SIZE_OPTIONS` with `modeState.activeMode?.resolution_options`, and constrain the select viewport to five visible rows.
- Modify: `tests/test_mode_config.py`
  Add parsing and validation coverage for `resolution_sets`.
- Modify: `tests/test_model_routes.py`
  Verify `/api/modes` includes `resolution_set` and `resolution_options`.
- Modify: `tests/test_ws_routes.py`
  Add WebSocket rejection coverage for invalid sizes in mode-system generation.
- Modify: `lcm-sr-ui/src/components/options/OptionsPanel.test.jsx`
  Add size label and dropdown viewport coverage.
- Modify: `lcm-sr-ui/src/hooks/useModeConfig.test.jsx`
  Adjust mocked mode payloads to include resolution metadata where needed.
- Modify: `lcm-sr-ui/src/hooks/useGenerationParams.test.jsx`
  Add mode-switch reset coverage for `default_size`.

## Notes Before Implementation

- `resolutions_sdxl.csv` is seed input only. Do not parse it at runtime.
- `compat_endpoints.py` is out of scope for this pass. Do not add enforcement there.
- Validation in `/generate` belongs after mode defaults finalize `req.size`, before queue submission.
- Validation in `ws_routes.py` must happen before `GenerationJob` submission so the frontend path cannot bypass policy.
- The stricter contract is intentional: `resolution_sets.default` is required once this feature ships. Do not synthesize it from `SIZE_OPTIONS`.

### Task 1: Add Config Support For Resolution Sets

**Files:**
- Modify: `server/mode_config.py`
- Test: `tests/test_mode_config.py`

- [ ] **Step 1: Write the failing mode-config tests**

Add tests that cover successful parsing, unknown set rejection, missing `default` rejection, and `default_size` membership enforcement.

```python
def test_mode_config_parses_resolution_sets_and_mode_selector(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl
resolution_sets:
  default:
    - size: 512x512
      aspect_ratio: "1:1"
  sdxl:
    - size: 1024x1024
      aspect_ratio: "1:1"
    - size: 896x1152
      aspect_ratio: "7:9"
modes:
  sdxl:
    model: checkpoints/sdxl/model.safetensors
    resolution_set: sdxl
    default_size: 1024x1024
""".strip()
    )

    from server.mode_config import ModeConfigManager

    manager = ModeConfigManager(str(tmp_path))
    mode = manager.get_mode("sdxl")

    assert mode.resolution_set == "sdxl"
    assert mode.resolution_options == [
        {"size": "1024x1024", "aspect_ratio": "1:1"},
        {"size": "896x1152", "aspect_ratio": "7:9"},
    ]
```

```python
def test_mode_config_requires_default_resolution_set(tmp_path):
    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
default_mode: base
resolution_sets:
  sdxl:
    - size: 1024x1024
      aspect_ratio: "1:1"
modes:
  base:
    model: checkpoints/base/model.safetensors
    default_size: 512x512
""".strip()
    )

    from server.mode_config import ModeConfigManager

    with pytest.raises(ValueError, match="resolution_sets.default"):
        ModeConfigManager(str(tmp_path))
```

- [ ] **Step 2: Run the focused mode-config tests and confirm failure**

Run: `pytest tests/test_mode_config.py -k "resolution_set or resolution_sets" -q`

Expected: FAIL with missing `resolution_set` / `resolution_options` fields or missing validation logic.

- [ ] **Step 3: Implement parsing, resolution, and validation in `server/mode_config.py`**

Introduce a small dataclass for a resolution entry, add config fields to `ModeConfig`, parse top-level `resolution_sets`, resolve the effective set per mode, and validate the contract during `_load_config()`.

```python
@dataclass
class ResolutionOption:
    size: str
    aspect_ratio: str


@dataclass
class ModeConfig:
    ...
    resolution_set: Optional[str] = None
    resolution_options: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ModesYAML:
    ...
    resolution_sets: Dict[str, List[ResolutionOption]]
```

```python
raw_resolution_sets = data.get("resolution_sets")
if not isinstance(raw_resolution_sets, dict) or "default" not in raw_resolution_sets:
    raise ValueError("modes.yml missing required resolution_sets.default")

resolution_sets: Dict[str, List[ResolutionOption]] = {}
for set_name, entries in raw_resolution_sets.items():
    resolution_sets[set_name] = [
        ResolutionOption(
            size=str(entry["size"]),
            aspect_ratio=str(entry["aspect_ratio"]),
        )
        for entry in (entries or [])
    ]
```

```python
resolved_set_name = mode_data.get("resolution_set") or "default"
if resolved_set_name not in resolution_sets:
    raise ValueError(f"Mode '{mode_name}' references unknown resolution_set '{resolved_set_name}'")

resolved_entries = resolution_sets[resolved_set_name]
allowed_sizes = {entry.size for entry in resolved_entries}
default_size = mode_data.get("default_size", "512x512")
if default_size not in allowed_sizes:
    raise ValueError(
        f"Mode '{mode_name}' default_size '{default_size}' is not present in resolution_set '{resolved_set_name}'"
    )
```

- [ ] **Step 4: Serialize the new fields through `to_dict()`**

Make sure `resolution_set` and `resolution_options` survive round-trips so `/api/modes` can expose them without special-casing routes.

```python
"resolution_set": mode.resolution_set,
"resolution_options": list(mode.resolution_options),
```

- [ ] **Step 5: Run the mode-config suite**

Run: `pytest tests/test_mode_config.py -q`

Expected: PASS for the new resolution tests and the existing mode-config coverage.

- [ ] **Step 6: Commit**

```bash
git add server/mode_config.py tests/test_mode_config.py
git commit -m "feat: add config-owned resolution sets"
```

### Task 2: Seed Resolution Sets And Expose Them Through Mode Metadata

**Files:**
- Modify: `conf/modes.yml`
- Modify: `conf/modes.yaml.example`
- Modify: `tests/test_model_routes.py`

- [ ] **Step 1: Add failing route-serialization coverage**

Extend `/api/modes` serialization coverage to assert the resolved set name and entries are present.

```python
async def test_list_modes_includes_resolution_metadata():
    config = Mock()
    config.to_dict.return_value = {
        "default_mode": "SDXL",
        "modes": {
            "SDXL": {
                "model": "checkpoints/sdxl/model.safetensors",
                "default_size": "1024x1024",
                "resolution_set": "sdxl",
                "resolution_options": [
                    {"size": "1024x1024", "aspect_ratio": "1:1"},
                    {"size": "896x1152", "aspect_ratio": "7:9"},
                ],
            }
        },
    }

    with patch("server.model_routes.get_mode_config", return_value=config):
        data = await model_routes.list_modes()

    assert data["modes"]["SDXL"]["resolution_set"] == "sdxl"
    assert data["modes"]["SDXL"]["resolution_options"][0]["aspect_ratio"] == "1:1"
```

- [ ] **Step 2: Run the model-routes test and confirm failure**

Run: `pytest tests/test_model_routes.py -k resolution -q`

Expected: FAIL until the mocked payloads and assertions line up with the new config structure.

- [ ] **Step 3: Seed `conf/modes.yml` with `default` and curated `sdxl` entries**

Use `resolutions_sdxl.csv` only as a human curation source. Normalize the chosen entries directly into YAML.

```yaml
resolution_sets:
  default:
    - size: "512x512"
      aspect_ratio: "1:1"
    - size: "768x768"
      aspect_ratio: "1:1"
  sdxl:
    - size: "1024x1024"
      aspect_ratio: "1:1"
    - size: "896x1152"
      aspect_ratio: "7:9"
    - size: "1152x896"
      aspect_ratio: "9:7"
    - size: "832x1216"
      aspect_ratio: "13:19"
    - size: "1216x832"
      aspect_ratio: "19:13"
```

Add `resolution_set: sdxl` to the SDXL mode and keep its `default_size` in the curated set.

- [ ] **Step 4: Mirror the new contract in `conf/modes.yaml.example`**

```yaml
resolution_sets:
  default:
    - size: "512x512"
      aspect_ratio: "1:1"
```

This file must demonstrate the required `resolution_sets.default` contract so new configs do not start invalid.

- [ ] **Step 5: Run the route-serialization suite**

Run: `pytest tests/test_model_routes.py -q`

Expected: PASS, with `/api/modes` returning the new resolution metadata through `config.to_dict()`.

- [ ] **Step 6: Commit**

```bash
git add conf/modes.yml conf/modes.yaml.example tests/test_model_routes.py
git commit -m "feat: seed resolution sets in mode config"
```

### Task 3: Enforce Resolution Sets On HTTP And WebSocket Generation Paths

**Files:**
- Create: `server/generation_constraints.py`
- Modify: `server/lcm_sr_server.py`
- Modify: `server/ws_routes.py`
- Test: `tests/test_ws_routes.py`

- [ ] **Step 1: Write the failing shared-validation and WebSocket tests**

Add one direct helper test or one focused WS integration test that proves invalid size rejection happens before job submission.

```python
def test_finalize_mode_generate_request_replaces_env_default_and_accepts_mode_default():
    from types import SimpleNamespace
    from server.generation_constraints import finalize_mode_generate_request

    req = SimpleNamespace(size="512x512")
    mode = SimpleNamespace(
        name="SDXL",
        default_size="1024x1024",
        resolution_options=[{"size": "1024x1024", "aspect_ratio": "1:1"}],
    )

    finalize_mode_generate_request(req, mode, env_default_size="512x512")

    assert req.size == "1024x1024"
```

```python
def test_generate_mode_system_rejects_invalid_size_before_submit():
    app.state.use_mode_system = True
    pool = MagicMock()
    pool.get_current_mode.return_value = "SDXL"
    pool.submit_job.return_value = MagicMock()
    app.state.worker_pool = pool

    fake_lcm_module = types.ModuleType("server.lcm_sr_server")

    class _FakeGenerateRequest:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    fake_lcm_module.GenerateRequest = _FakeGenerateRequest
    fake_lcm_module._store_image_blob = lambda *args, **kwargs: None
    fake_lcm_module.get_mode_config = lambda: SimpleNamespace(
        get_mode=lambda name: SimpleNamespace(
            name=name,
            default_size="1024x1024",
            resolution_options=[{"size": "1024x1024", "aspect_ratio": "1:1"}],
        )
    )
```

The assertion should verify WS sends an error and `pool.submit_job` is never called for `size="768x768"`.

- [ ] **Step 2: Run the focused WS test and confirm failure**

Run: `pytest tests/test_ws_routes.py -k "invalid_size or resolution" -q`

Expected: FAIL because the mode-system WS path currently builds a request and submits it without any size-policy check.

- [ ] **Step 3: Implement a shared helper in `server/generation_constraints.py`**

Do not duplicate policy between HTTP and WS. Create one helper that applies the mode default when the request still uses the environment default, then validates the finalized size against the mode's resolved options.

```python
def finalize_mode_generate_request(req, mode, *, env_default_size: str) -> None:
    if req.size == env_default_size:
        req.size = mode.default_size

    allowed_sizes = {
        str(entry["size"])
        for entry in (mode.resolution_options or [])
    }
    if req.size not in allowed_sizes:
        raise ValueError(f"size '{req.size}' is not allowed for mode '{mode.name}'")
```

- [ ] **Step 4: Wire the helper into `/generate`**

In `server/lcm_sr_server.py`, replace the inline `req.size = mode.default_size` branch with the helper, and convert `ValueError` into `HTTPException(status_code=400, ...)`.

```python
from server.generation_constraints import finalize_mode_generate_request

...
if current_mode:
    mode = mode_config.get_mode(current_mode)
    try:
        finalize_mode_generate_request(
            req,
            mode,
            env_default_size=os.environ.get("DEFAULT_SIZE", "512x512"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
```

Keep this block before `GenerationJob(req=req)` so invalid requests never hit the queue.

- [ ] **Step 5: Wire the helper into `server/ws_routes.py`**

After `_build_generate_request(params)` and before `GenerationJob(req=req, ...)`, load the current mode from `get_mode_config()`, finalize `req.size`, and return a WS error instead of queuing the job when invalid.

```python
from server.generation_constraints import finalize_mode_generate_request
from server.mode_config import get_mode_config

...
current_mode = state.worker_pool.get_current_mode()
if current_mode:
    mode = get_mode_config().get_mode(current_mode)
    try:
        finalize_mode_generate_request(
            req,
            mode,
            env_default_size=os.environ.get("DEFAULT_SIZE", "512x512"),
        )
    except ValueError as e:
        await hub.send(client_id, _error(str(e), corr_id))
        return
```

- [ ] **Step 6: Run backend verification**

Run: `pytest tests/test_ws_routes.py -q`

Expected: PASS, including the new invalid-size rejection path.

- [ ] **Step 7: Commit**

```bash
git add server/generation_constraints.py server/lcm_sr_server.py server/ws_routes.py tests/test_ws_routes.py
git commit -m "feat: enforce resolution sets on generate paths"
```

### Task 4: Render Mode-Aware Size Options And Reset Size On Mode Switch

**Files:**
- Modify: `lcm-sr-ui/src/utils/helpers.js`
- Modify: `lcm-sr-ui/src/utils/generationControls.js`
- Modify: `lcm-sr-ui/src/hooks/useGenerationParams.js`
- Modify: `lcm-sr-ui/src/components/options/OptionsPanel.jsx`
- Test: `lcm-sr-ui/src/components/options/OptionsPanel.test.jsx`
- Test: `lcm-sr-ui/src/hooks/useGenerationParams.test.jsx`
- Test: `lcm-sr-ui/src/hooks/useModeConfig.test.jsx`

- [ ] **Step 1: Add failing frontend tests**

Cover the formatter, the five-row viewport, and the mode-switch size reset.

```jsx
it('renders size labels with aspect ratio from the active mode', async () => {
  const params = makeParams();
  renderOptionsPanel(
    makeModeState('SDXL', {
      default_size: '1024x1024',
      resolution_options: [
        { size: '1024x1024', aspect_ratio: '1:1' },
        { size: '896x1152', aspect_ratio: '7:9' },
      ],
    }),
    params
  );

  fireEvent.click(screen.getByRole('combobox', { name: /size/i }));
  expect(screen.getByText('1024×1024 • 1:1')).toBeInTheDocument();
  expect(screen.getByText('896×1152 • 7:9')).toBeInTheDocument();
});
```

```jsx
it('applyModeControlDefaults resets draft size to the mode default', () => {
  const { result } = renderHook(() => useGenerationParams(null, null, vi.fn(), null));

  act(() => {
    result.current.setSizeDirect('768x768');
    result.current.applyModeControlDefaults({
      default_size: '1024x1024',
      resolution_options: [{ size: '1024x1024', aspect_ratio: '1:1' }],
    });
  });

  expect(result.current.draft.size).toBe('1024x1024');
});
```

- [ ] **Step 2: Run the focused frontend tests and confirm failure**

Run: `cd lcm-sr-ui && yarn vitest run src/components/options/OptionsPanel.test.jsx src/hooks/useGenerationParams.test.jsx src/hooks/useModeConfig.test.jsx`

Expected: FAIL because the UI still renders `SIZE_OPTIONS` and `applyModeControlDefaults()` does not touch size.

- [ ] **Step 3: Extend the existing formatter and mode-default helper**

Do not add a second size formatter. Extend the existing helper and the existing draft-default path.

```javascript
export function formatSizeDisplay(size, aspectRatio = null) {
  const formatted = String(size).replace(/x/i, "×");
  return aspectRatio ? `${formatted} • ${aspectRatio}` : formatted;
}
```

```javascript
export function applyModeControlDefaultsToDraft(draft, mode) {
  const next = { ...draft };
  ...
  if (mode?.default_size) {
    next.size = mode.default_size;
  }
  return next;
}
```

- [ ] **Step 4: Flow the reset through `useGenerationParams()`**

Keep the reset in the current `App.jsx -> params.applyModeControlDefaults(activeMode)` path.

```javascript
const applyModeControlDefaults = useCallback(
  (mode) => {
    const next = applyModeControlDefaultsToDraft(
      {
        size,
        negativePrompt,
        schedulerId,
      },
      mode
    );
    setSize(next.size || DEFAULT_SIZE);
    setNegativePrompt(next.negativePrompt || '');
    setSchedulerId(next.schedulerId || null);
  },
  [size, negativePrompt, schedulerId]
);
```

- [ ] **Step 5: Replace `SIZE_OPTIONS` in `OptionsPanel.jsx` with mode entries**

Derive the list from `activeMode.resolution_options`, fall back to an empty list if missing, and constrain the select content height to five visible rows.

```javascript
const sizeOptions = Array.isArray(activeMode?.resolution_options)
  ? activeMode.resolution_options
  : [];
```

```jsx
<SelectContent className={`${CSS_CLASSES.SELECT_CONTENT} max-h-60 overflow-y-auto`}>
  {sizeOptions.map((option) => (
    <SelectItem
      key={option.size}
      className={CSS_CLASSES.SELECT_ITEM}
      value={option.size}
    >
      {formatSizeDisplay(option.size, option.aspect_ratio)}
    </SelectItem>
  ))}
</SelectContent>
```

This is the user-visible behavior change: mode switches now reset size to `default_size`.

- [ ] **Step 6: Run frontend verification**

Run: `cd lcm-sr-ui && yarn vitest run src/components/options/OptionsPanel.test.jsx src/hooks/useGenerationParams.test.jsx src/hooks/useModeConfig.test.jsx`

Expected: PASS with labels rendered as `resolution • aspect ratio`, compact dropdown viewport, and size reset on mode switch.

- [ ] **Step 7: Commit**

```bash
git add lcm-sr-ui/src/utils/helpers.js lcm-sr-ui/src/utils/generationControls.js lcm-sr-ui/src/hooks/useGenerationParams.js lcm-sr-ui/src/components/options/OptionsPanel.jsx lcm-sr-ui/src/components/options/OptionsPanel.test.jsx lcm-sr-ui/src/hooks/useGenerationParams.test.jsx lcm-sr-ui/src/hooks/useModeConfig.test.jsx
git commit -m "feat: add mode-aware resolution selector"
```

### Task 5: Final Verification

**Files:**
- Verify only

- [ ] **Step 1: Run backend verification**

Run: `pytest tests/test_mode_config.py tests/test_model_routes.py tests/test_ws_routes.py -q`

Expected: PASS

- [ ] **Step 2: Run frontend verification**

Run: `cd lcm-sr-ui && yarn vitest run src/components/options/OptionsPanel.test.jsx src/hooks/useGenerationParams.test.jsx src/hooks/useModeConfig.test.jsx`

Expected: PASS

- [ ] **Step 3: Perform manual validation**

Check these in a running app:

```text
1. Switch into SDXL and confirm size resets to the SDXL default.
2. Open the Size selector and confirm entries render like "1024×1024 • 1:1".
3. Confirm only five rows are visible before internal scrolling.
4. Submit an unsupported size through HTTP and confirm 400.
5. Submit an unsupported size through WS and confirm a job error / no backend queue submission.
```

- [ ] **Step 4: Final commit if manual validation changes anything**

```bash
git status --short
git add -A
git commit -m "test: finish resolution set rollout"
```

## Self-Review

- Spec coverage:
  - Config-owned sets: Task 1
  - Seed SDXL/default sets from YAML, not runtime CSV: Task 2
  - `/api/modes` exposure: Task 2
  - Backend enforcement: Task 3
  - WS enforcement called out by Theta: Task 3
  - UI labels, compact viewport, and mode-switch reset: Task 4
  - Manual validation and regression coverage: Task 5
- Placeholder scan:
  - No `TODO`, `TBD`, or “implement later” markers remain.
- Type consistency:
  - Uses `resolution_set`, `resolution_options`, and `default_size` consistently across config, backend, and UI tasks.
