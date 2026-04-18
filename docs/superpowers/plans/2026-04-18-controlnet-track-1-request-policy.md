# ControlNet Track 1 — Request Contract, Mode Policy, Backend Enforcement

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `controlnets` to the generation request, `controlnet_policy` to mode config and `/api/modes`, and a backend enforcement layer that validates/normalizes attachments and stubs provider dispatch so the feature can merge while Track 2 and Track 3 are built.

**Architecture:** Pydantic models for the request side in a new `server/controlnet_models.py`. Dataclasses for the policy side added to `server/mode_config.py`. Enforcement in a new `server/controlnet_constraints.py` invoked from both HTTP `/generate` and the WebSocket job-submit path, mirroring the existing `finalize_mode_generate_request` seam. Provider dispatch is a stub that raises a specific error when a validated ControlNet request arrives — Track 3 replaces it with real execution.

**Tech Stack:** Python 3, FastAPI, Pydantic v2, PyYAML, pytest, pytest-asyncio. No new runtime dependencies.

**FP parent issue:** STABL-iajgqfqp (under top-level STABL-utbuhifx).

**Spec:** [docs/superpowers/specs/2026-04-18-controlnet-design.md](../specs/2026-04-18-controlnet-design.md) (sections 1, 2, 7 and portions of 9 touching request/enforcement).

---

## File Structure

### New files

| Path | Responsibility |
| ---- | -------------- |
| `server/controlnet_models.py` | Pydantic models for request-side attachments: `ControlNetAttachment`, `ControlNetPreprocessRequest`. Field-level validation (ranges, regex) only. No policy awareness. |
| `server/controlnet_constraints.py` | `enforce_controlnet_policy(req, mode)` — policy-aware validation + defaulting. Raises `ValueError` with spec-defined attachment-invalid classes. Invoked from HTTP and WS after `finalize_mode_generate_request`. |
| `tests/test_controlnet_models.py` | Pydantic field validation tests. |
| `tests/test_controlnet_constraints.py` | Policy enforcement tests (one per invalid class, plus valid-path and defaulting). |

### Modified files

| Path | What changes |
| ---- | ------------ |
| `server/mode_config.py` | Add `ControlNetPolicy` + `ControlNetControlTypePolicy` dataclasses. Add field to `ModeConfig`. Parse `controlnet_policy` YAML block in `_load_config`. Serialize in `to_dict`. |
| `server/model_routes.py` | Include `controlnet_policy` in `/api/modes` response. |
| `server/lcm_sr_server.py` | Add `controlnets: Optional[List[ControlNetAttachment]] = None` to `GenerateRequest`. Invoke `enforce_controlnet_policy` after `finalize_mode_generate_request`. Reject at the stub dispatch before worker submission. |
| `server/ws_routes.py` | Pass `controlnets` through `_build_generate_request`. Invoke `enforce_controlnet_policy` after `finalize_mode_generate_request`. |
| `tests/test_mode_config.py` | Add round-trip tests for `controlnet_policy` YAML → dataclass → `to_dict`. |
| `tests/test_model_routes.py` | Add `/api/modes` serialization test for `controlnet_policy`. |

### Deliberately out of scope for this track

- `conf/controlnets.yaml` model registry (Track 3)
- Preprocessor protocol or implementations (Track 2)
- Asset layer extensions (Track 2)
- Response-side `controlnet_artifacts` serialization (Track 2)
- Frontend UI (Track 3)

---

## Task 1: Pydantic request models

**Files:**
- Create: `server/controlnet_models.py`
- Test: `tests/test_controlnet_models.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_controlnet_models.py
import pytest
from pydantic import ValidationError

from server.controlnet_models import ControlNetAttachment, ControlNetPreprocessRequest


def test_attachment_accepts_map_ref_path():
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        map_asset_ref="asset_abc",
        strength=0.8,
        start_percent=0.0,
        end_percent=1.0,
    )
    assert att.map_asset_ref == "asset_abc"
    assert att.source_asset_ref is None
    assert att.preprocess is None


def test_attachment_accepts_source_plus_preprocess_path():
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="depth",
        source_asset_ref="asset_src",
        preprocess=ControlNetPreprocessRequest(id="depth", options={}),
    )
    assert att.source_asset_ref == "asset_src"
    assert att.preprocess.id == "depth"


def test_attachment_rejects_neither_source_nor_map():
    with pytest.raises(ValidationError, match="map_asset_ref or source_asset_ref"):
        ControlNetAttachment(attachment_id="cn_1", control_type="canny")


def test_attachment_rejects_both_source_and_map():
    with pytest.raises(ValidationError, match="exactly one of"):
        ControlNetAttachment(
            attachment_id="cn_1",
            control_type="canny",
            map_asset_ref="asset_a",
            source_asset_ref="asset_b",
            preprocess=ControlNetPreprocessRequest(id="canny"),
        )


def test_attachment_rejects_source_without_preprocess():
    with pytest.raises(ValidationError, match="preprocess"):
        ControlNetAttachment(
            attachment_id="cn_1",
            control_type="canny",
            source_asset_ref="asset_a",
        )


def test_attachment_rejects_strength_out_of_range():
    with pytest.raises(ValidationError):
        ControlNetAttachment(
            attachment_id="cn_1",
            control_type="canny",
            map_asset_ref="asset_a",
            strength=-0.1,
        )


def test_attachment_rejects_inverted_percent_range():
    with pytest.raises(ValidationError, match="start_percent"):
        ControlNetAttachment(
            attachment_id="cn_1",
            control_type="canny",
            map_asset_ref="asset_a",
            start_percent=0.8,
            end_percent=0.2,
        )


def test_attachment_rejects_blank_attachment_id():
    with pytest.raises(ValidationError):
        ControlNetAttachment(
            attachment_id="",
            control_type="canny",
            map_asset_ref="asset_a",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_controlnet_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'server.controlnet_models'`

- [ ] **Step 3: Create the models module**

```python
# server/controlnet_models.py
"""
Pydantic request models for ControlNet attachments.

These models enforce field-level invariants (types, ranges, exclusive-or input
paths). Policy-aware validation (allowed control_types for a mode, allowed
model_ids, etc.) lives in server/controlnet_constraints.py and runs after
the request is parsed.
"""
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class ControlNetPreprocessRequest(BaseModel):
    id: str = Field(..., min_length=1, description="Preprocessor id, e.g. 'canny' or 'depth'.")
    options: Dict[str, Any] = Field(default_factory=dict)


class ControlNetAttachment(BaseModel):
    attachment_id: str = Field(..., min_length=1, description="Client-generated unique id within the request.")
    control_type: str = Field(..., min_length=1, description="Canonical control type, e.g. 'canny'.")
    model_id: Optional[str] = Field(default=None, description="Optional; mode policy may supply a default.")
    map_asset_ref: Optional[str] = Field(default=None, description="Pre-derived control-map asset ref.")
    source_asset_ref: Optional[str] = Field(default=None, description="Source image asset ref for preprocessing.")
    preprocess: Optional[ControlNetPreprocessRequest] = Field(default=None)
    strength: float = Field(default=1.0, ge=0.0, le=2.0)
    start_percent: float = Field(default=0.0, ge=0.0, le=1.0)
    end_percent: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_input_path(self) -> "ControlNetAttachment":
        has_map = self.map_asset_ref is not None
        has_source = self.source_asset_ref is not None
        if not has_map and not has_source:
            raise ValueError("attachment must supply map_asset_ref or source_asset_ref")
        if has_map and has_source:
            raise ValueError("attachment must supply exactly one of map_asset_ref or source_asset_ref")
        if has_source and self.preprocess is None:
            raise ValueError("source_asset_ref requires a preprocess block")
        if has_map and self.preprocess is not None:
            raise ValueError("map_asset_ref is incompatible with a preprocess block")
        if self.start_percent > self.end_percent:
            raise ValueError("start_percent must be <= end_percent")
        return self
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_controlnet_models.py -v`
Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add server/controlnet_models.py tests/test_controlnet_models.py
git commit -m "feat(controlnet): request pydantic models with field validation (STABL-iajgqfqp)"
```

---

## Task 2: Add `controlnets` to `GenerateRequest`

**Files:**
- Modify: `server/lcm_sr_server.py` (around line 118, the `GenerateRequest` class)
- Test: extend `tests/test_controlnet_models.py`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_controlnet_models.py
def test_generate_request_accepts_controlnets_list():
    from server.lcm_sr_server import GenerateRequest

    req = GenerateRequest(
        prompt="a cat",
        controlnets=[
            {
                "attachment_id": "cn_1",
                "control_type": "canny",
                "map_asset_ref": "asset_a",
            }
        ],
    )
    assert req.controlnets is not None
    assert len(req.controlnets) == 1
    assert req.controlnets[0].attachment_id == "cn_1"


def test_generate_request_controlnets_defaults_to_none():
    from server.lcm_sr_server import GenerateRequest

    req = GenerateRequest(prompt="a cat")
    assert req.controlnets is None
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_controlnet_models.py::test_generate_request_accepts_controlnets_list -v`
Expected: FAIL — pydantic complains about extra field `controlnets`.

- [ ] **Step 3: Add field to `GenerateRequest`**

In `server/lcm_sr_server.py`, add import at the top of the request-schema section (after the existing pydantic import around line 58):

```python
from server.controlnet_models import ControlNetAttachment
```

Then, inside the `class GenerateRequest(BaseModel):` body (after line 144, `denoise_strength`), append:

```python
    controlnets: Optional[List[ControlNetAttachment]] = Field(
        default=None,
        description="Optional list of ControlNet attachments; validated against the active mode's controlnet_policy.",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_controlnet_models.py -v`
Expected: all tests PASS (including the 2 new ones).

- [ ] **Step 5: Commit**

```bash
git add server/lcm_sr_server.py tests/test_controlnet_models.py
git commit -m "feat(controlnet): wire controlnets field into GenerateRequest (STABL-iajgqfqp)"
```

---

## Task 3: Pass `controlnets` through WS `_build_generate_request`

**Files:**
- Modify: `server/ws_routes.py` (the `_build_generate_request` function around line 277)
- Test: `tests/test_ws_build_generate_request.py` (new)

- [ ] **Step 1: Write failing test**

```python
# tests/test_ws_build_generate_request.py
from server.ws_routes import _build_generate_request


def test_build_generate_request_passes_through_controlnets():
    params = {
        "prompt": "a cat",
        "controlnets": [
            {
                "attachment_id": "cn_1",
                "control_type": "canny",
                "map_asset_ref": "asset_a",
            }
        ],
    }
    req = _build_generate_request(params)
    assert req.controlnets is not None
    assert len(req.controlnets) == 1
    assert req.controlnets[0].control_type == "canny"


def test_build_generate_request_default_controlnets_is_none():
    req = _build_generate_request({"prompt": "a cat"})
    assert req.controlnets is None
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_ws_build_generate_request.py -v`
Expected: FAIL — `controlnets` is not forwarded; `req.controlnets` is `None` on the first test.

- [ ] **Step 3: Forward the field in `_build_generate_request`**

In `server/ws_routes.py`, modify `_build_generate_request` (around line 277). Inside the `GenerateRequest(...)` constructor call, add one argument before the closing `)`:

```python
        controlnets=params.get("controlnets"),
```

Full function after edit:

```python
def _build_generate_request(params: dict):
    from server.lcm_sr_server import GenerateRequest

    return GenerateRequest(
        prompt=params.get("prompt", ""),
        negative_prompt=params.get("negative_prompt"),
        scheduler_id=params.get("scheduler_id"),
        size=params.get("size", os.environ.get("DEFAULT_SIZE", "512x512")),
        num_inference_steps=params.get(
            "num_inference_steps",
            params.get("steps", int(os.environ.get("DEFAULT_STEPS", "4"))),
        ),
        guidance_scale=params.get(
            "guidance_scale",
            params.get("cfg", float(os.environ.get("DEFAULT_GUIDANCE", "1.0"))),
        ),
        seed=params.get("seed"),
        superres=params.get("superres", False),
        superres_magnitude=params.get("superres_magnitude", 2),
        denoise_strength=params.get("denoise_strength", 0.75),
        controlnets=params.get("controlnets"),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ws_build_generate_request.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add server/ws_routes.py tests/test_ws_build_generate_request.py
git commit -m "feat(controlnet): forward controlnets through ws request builder (STABL-iajgqfqp)"
```

---

## Task 4: Add `ControlNetPolicy` dataclasses to `mode_config.py`

**Files:**
- Modify: `server/mode_config.py` (dataclass section, around lines 20–88)

- [ ] **Step 1: Write failing test**

Append to `tests/test_mode_config.py`:

```python
def test_controlnet_policy_dataclass_defaults():
    from server.mode_config import ControlNetPolicy, ControlNetControlTypePolicy

    policy = ControlNetPolicy()
    assert policy.enabled is False
    assert policy.max_attachments == 0
    assert policy.allow_reuse_emitted_maps is False
    assert policy.allowed_control_types == {}

    type_policy = ControlNetControlTypePolicy(default_model_id="sdxl-canny")
    assert type_policy.allowed_model_ids == []
    assert type_policy.allow_preprocess is True
    assert type_policy.default_strength == 1.0
    assert type_policy.min_strength == 0.0
    assert type_policy.max_strength == 2.0
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_mode_config.py::test_controlnet_policy_dataclass_defaults -v`
Expected: FAIL — `ImportError: cannot import name 'ControlNetPolicy'`.

- [ ] **Step 3: Add dataclasses**

In `server/mode_config.py`, after the existing `ChatConnectionConfig` dataclass (around line 49), add:

```python
@dataclass
class ControlNetControlTypePolicy:
    """Per-control-type policy within a mode's controlnet_policy."""
    default_model_id: Optional[str] = None
    allowed_model_ids: List[str] = field(default_factory=list)
    allow_preprocess: bool = True
    default_strength: float = 1.0
    min_strength: float = 0.0
    max_strength: float = 2.0


@dataclass
class ControlNetPolicy:
    """Mode-owned ControlNet policy.

    When `enabled` is False, any request carrying `controlnets` is rejected.
    `allowed_control_types` maps canonical control-type names (e.g. 'canny')
    to their per-type policy. Absent control types are forbidden.
    """
    enabled: bool = False
    max_attachments: int = 0
    allow_reuse_emitted_maps: bool = False
    allowed_control_types: Dict[str, ControlNetControlTypePolicy] = field(default_factory=dict)
```

Also add a `controlnet_policy` field to `ModeConfig` (inside the existing `@dataclass class ModeConfig:` around line 53, after `metadata: Dict[str, Any] = field(default_factory=dict)` — before the "Resolved absolute paths" comment):

```python
    controlnet_policy: ControlNetPolicy = field(default_factory=ControlNetPolicy)
```

(No forward-reference string is needed because the new dataclasses are defined above `ModeConfig` in the file.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mode_config.py::test_controlnet_policy_dataclass_defaults -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/mode_config.py tests/test_mode_config.py
git commit -m "feat(controlnet): mode_config dataclasses for controlnet_policy (STABL-iajgqfqp)"
```

---

## Task 5: Parse `controlnet_policy` YAML in `_load_config`

**Files:**
- Modify: `server/mode_config.py` (`_load_config`, mode loop around lines 200–290, and `_parse_controlnet_policy` helper)

- [ ] **Step 1: Write failing test**

Append to `tests/test_mode_config.py`:

```python
def test_mode_config_parses_controlnet_policy(tmp_path):
    from server.mode_config import ModeConfigManager

    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl-cn
resolution_sets:
  default:
    - size: 1024x1024
      aspect_ratio: "1:1"
modes:
  sdxl-cn:
    model: checkpoints/sdxl.safetensors
    default_size: 1024x1024
    controlnet_policy:
      enabled: true
      max_attachments: 3
      allow_reuse_emitted_maps: true
      allowed_control_types:
        canny:
          default_model_id: sdxl-canny
          allowed_model_ids: [sdxl-canny]
          allow_preprocess: true
          default_strength: 0.8
          min_strength: 0.0
          max_strength: 1.5
        depth:
          default_model_id: sdxl-depth
          allowed_model_ids: [sdxl-depth]
"""
    )
    mgr = ModeConfigManager(config_path=str(tmp_path))
    mode = mgr.config.modes["sdxl-cn"]
    policy = mode.controlnet_policy
    assert policy.enabled is True
    assert policy.max_attachments == 3
    assert policy.allow_reuse_emitted_maps is True
    assert set(policy.allowed_control_types) == {"canny", "depth"}
    canny = policy.allowed_control_types["canny"]
    assert canny.default_model_id == "sdxl-canny"
    assert canny.allowed_model_ids == ["sdxl-canny"]
    assert canny.max_strength == 1.5
    depth = policy.allowed_control_types["depth"]
    assert depth.allow_preprocess is True  # default
    assert depth.default_strength == 1.0  # default


def test_mode_config_absent_controlnet_policy_is_disabled(tmp_path):
    from server.mode_config import ModeConfigManager

    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl-plain
resolution_sets:
  default:
    - size: 1024x1024
      aspect_ratio: "1:1"
modes:
  sdxl-plain:
    model: checkpoints/sdxl.safetensors
    default_size: 1024x1024
"""
    )
    mgr = ModeConfigManager(config_path=str(tmp_path))
    mode = mgr.config.modes["sdxl-plain"]
    assert mode.controlnet_policy.enabled is False
    assert mode.controlnet_policy.allowed_control_types == {}


def test_mode_config_rejects_controlnet_policy_with_unknown_keys(tmp_path):
    from server.mode_config import ModeConfigManager

    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl-bad
resolution_sets:
  default:
    - size: 1024x1024
      aspect_ratio: "1:1"
modes:
  sdxl-bad:
    model: checkpoints/sdxl.safetensors
    default_size: 1024x1024
    controlnet_policy:
      enabled: true
      bogus_field: 1
"""
    )
    import pytest
    with pytest.raises(ValueError, match="controlnet_policy"):
        ModeConfigManager(config_path=str(tmp_path))
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_mode_config.py -k controlnet_policy -v`
Expected: FAIL — `controlnet_policy` is not parsed; fields stay at default.

- [ ] **Step 3: Add parser helper and wire it into mode loading**

In `server/mode_config.py`, add a helper method on `ModeConfigManager` (place it near the existing `_parse_chat_connection_config` around line 344):

```python
    _ALLOWED_CONTROLNET_POLICY_KEYS = {
        "enabled",
        "max_attachments",
        "allow_reuse_emitted_maps",
        "allowed_control_types",
    }

    _ALLOWED_CONTROLNET_TYPE_KEYS = {
        "default_model_id",
        "allowed_model_ids",
        "allow_preprocess",
        "default_strength",
        "min_strength",
        "max_strength",
    }

    def _parse_controlnet_policy(self, mode_name: str, raw: Any) -> ControlNetPolicy:
        if raw is None:
            return ControlNetPolicy()
        if not isinstance(raw, dict):
            raise ValueError(f"Mode '{mode_name}' controlnet_policy must be a mapping")
        unknown = set(raw.keys()) - self._ALLOWED_CONTROLNET_POLICY_KEYS
        if unknown:
            raise ValueError(
                f"Mode '{mode_name}' controlnet_policy has unknown keys: {sorted(unknown)}"
            )

        allowed_types_raw = raw.get("allowed_control_types") or {}
        if not isinstance(allowed_types_raw, dict):
            raise ValueError(
                f"Mode '{mode_name}' controlnet_policy.allowed_control_types must be a mapping"
            )
        allowed_types: Dict[str, ControlNetControlTypePolicy] = {}
        for type_name, type_raw in allowed_types_raw.items():
            if type_raw is None:
                type_raw = {}
            if not isinstance(type_raw, dict):
                raise ValueError(
                    f"Mode '{mode_name}' controlnet_policy.allowed_control_types.{type_name} must be a mapping"
                )
            unknown_type = set(type_raw.keys()) - self._ALLOWED_CONTROLNET_TYPE_KEYS
            if unknown_type:
                raise ValueError(
                    f"Mode '{mode_name}' controlnet_policy.allowed_control_types.{type_name} has unknown keys: {sorted(unknown_type)}"
                )
            allowed_ids = type_raw.get("allowed_model_ids") or []
            if not isinstance(allowed_ids, list) or not all(isinstance(x, str) for x in allowed_ids):
                raise ValueError(
                    f"Mode '{mode_name}' controlnet_policy.allowed_control_types.{type_name}.allowed_model_ids must be a list of strings"
                )
            allowed_types[type_name] = ControlNetControlTypePolicy(
                default_model_id=type_raw.get("default_model_id"),
                allowed_model_ids=list(allowed_ids),
                allow_preprocess=bool(type_raw.get("allow_preprocess", True)),
                default_strength=float(type_raw.get("default_strength", 1.0)),
                min_strength=float(type_raw.get("min_strength", 0.0)),
                max_strength=float(type_raw.get("max_strength", 2.0)),
            )

        return ControlNetPolicy(
            enabled=bool(raw.get("enabled", False)),
            max_attachments=int(raw.get("max_attachments", 0)),
            allow_reuse_emitted_maps=bool(raw.get("allow_reuse_emitted_maps", False)),
            allowed_control_types=allowed_types,
        )
```

Then, inside the mode-construction loop in `_load_config` (around line 252 where `ModeConfig(...)` is built), add the field. Find the `ModeConfig(...)` constructor call and append:

```python
                controlnet_policy=self._parse_controlnet_policy(mode_name, mode_data.get("controlnet_policy")),
```

(Add the argument before the closing `)` of the `ModeConfig(...)` call.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mode_config.py -k controlnet_policy -v`
Expected: all 3 new tests PASS.

- [ ] **Step 5: Commit**

```bash
git add server/mode_config.py tests/test_mode_config.py
git commit -m "feat(controlnet): parse controlnet_policy from modes.yml (STABL-iajgqfqp)"
```

---

## Task 6: Serialize `controlnet_policy` in `ModeConfigManager.to_dict`

**Files:**
- Modify: `server/mode_config.py` (`to_dict`, around line 568)

- [ ] **Step 1: Write failing test**

Append to `tests/test_mode_config.py`:

```python
def test_to_dict_serializes_controlnet_policy(tmp_path):
    from server.mode_config import ModeConfigManager

    cfg = tmp_path / "modes.yml"
    cfg.write_text(
        """
model_root: /models
lora_root: /models/loras
default_mode: sdxl-cn
resolution_sets:
  default:
    - size: 1024x1024
      aspect_ratio: "1:1"
modes:
  sdxl-cn:
    model: checkpoints/sdxl.safetensors
    default_size: 1024x1024
    controlnet_policy:
      enabled: true
      max_attachments: 2
      allow_reuse_emitted_maps: true
      allowed_control_types:
        canny:
          default_model_id: sdxl-canny
          allowed_model_ids: [sdxl-canny]
"""
    )
    mgr = ModeConfigManager(config_path=str(tmp_path))
    data = mgr.to_dict()
    policy = data["modes"]["sdxl-cn"]["controlnet_policy"]
    assert policy["enabled"] is True
    assert policy["max_attachments"] == 2
    assert policy["allow_reuse_emitted_maps"] is True
    canny = policy["allowed_control_types"]["canny"]
    assert canny["default_model_id"] == "sdxl-canny"
    assert canny["allowed_model_ids"] == ["sdxl-canny"]
    assert canny["allow_preprocess"] is True
    assert canny["default_strength"] == 1.0
    assert canny["min_strength"] == 0.0
    assert canny["max_strength"] == 2.0
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_mode_config.py::test_to_dict_serializes_controlnet_policy -v`
Expected: FAIL — `KeyError: 'controlnet_policy'`.

- [ ] **Step 3: Add serialization**

In `server/mode_config.py`, edit `to_dict`. Inside the mode dict comprehension (around line 582), add the serialized policy as a new key alongside existing fields. After the `"metadata": mode.metadata,` line, add:

```python
                    "controlnet_policy": {
                        "enabled": mode.controlnet_policy.enabled,
                        "max_attachments": mode.controlnet_policy.max_attachments,
                        "allow_reuse_emitted_maps": mode.controlnet_policy.allow_reuse_emitted_maps,
                        "allowed_control_types": {
                            type_name: {
                                "default_model_id": type_policy.default_model_id,
                                "allowed_model_ids": list(type_policy.allowed_model_ids),
                                "allow_preprocess": type_policy.allow_preprocess,
                                "default_strength": type_policy.default_strength,
                                "min_strength": type_policy.min_strength,
                                "max_strength": type_policy.max_strength,
                            }
                            for type_name, type_policy in mode.controlnet_policy.allowed_control_types.items()
                        },
                    },
```

- [ ] **Step 4: Run test to verify pass**

Run: `pytest tests/test_mode_config.py::test_to_dict_serializes_controlnet_policy -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/mode_config.py tests/test_mode_config.py
git commit -m "feat(controlnet): serialize controlnet_policy in ModeConfigManager.to_dict (STABL-iajgqfqp)"
```

---

## Task 7: Expose `controlnet_policy` in `/api/modes`

**Files:**
- Modify: `server/model_routes.py` (the mode dict comprehension in `list_modes`, around line 154)
- Test: `tests/test_model_routes.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_model_routes.py`:

```python
async def test_list_modes_includes_controlnet_policy():
    from unittest.mock import Mock, patch
    from server import model_routes

    config = Mock()
    config.to_dict.return_value = {
        "default_mode": "sdxl-cn",
        "chat": {},
        "resolution_sets": {"default": [{"size": "1024x1024", "aspect_ratio": "1:1"}]},
        "modes": {
            "sdxl-cn": {
                "model": "checkpoints/sdxl.safetensors",
                "loras": [],
                "default_size": "1024x1024",
                "default_steps": 20,
                "default_guidance": 7.0,
                "resolution_set": "default",
                "resolution_options": [{"size": "1024x1024", "aspect_ratio": "1:1"}],
                "negative_prompt_templates": {},
                "default_negative_prompt_template": None,
                "allow_custom_negative_prompt": False,
                "allowed_scheduler_ids": None,
                "default_scheduler_id": None,
                "controlnet_policy": {
                    "enabled": True,
                    "max_attachments": 2,
                    "allow_reuse_emitted_maps": True,
                    "allowed_control_types": {
                        "canny": {
                            "default_model_id": "sdxl-canny",
                            "allowed_model_ids": ["sdxl-canny"],
                            "allow_preprocess": True,
                            "default_strength": 0.8,
                            "min_strength": 0.0,
                            "max_strength": 2.0,
                        }
                    },
                },
            },
        },
    }

    with patch("server.model_routes.get_mode_config", return_value=config):
        data = await model_routes.list_modes()

    policy = data["modes"]["sdxl-cn"]["controlnet_policy"]
    assert policy["enabled"] is True
    assert policy["allowed_control_types"]["canny"]["default_model_id"] == "sdxl-canny"


async def test_list_modes_controlnet_policy_defaults_when_absent():
    from unittest.mock import Mock, patch
    from server import model_routes

    config = Mock()
    config.to_dict.return_value = {
        "default_mode": "sd15",
        "chat": {},
        "resolution_sets": {"default": [{"size": "512x512", "aspect_ratio": "1:1"}]},
        "modes": {
            "sd15": {
                "model": "checkpoints/sd15.safetensors",
                "loras": [],
                "default_size": "512x512",
                "default_steps": 20,
                "default_guidance": 7.0,
                "resolution_set": "default",
                "resolution_options": [{"size": "512x512", "aspect_ratio": "1:1"}],
                "negative_prompt_templates": {},
                "default_negative_prompt_template": None,
                "allow_custom_negative_prompt": False,
                "allowed_scheduler_ids": None,
                "default_scheduler_id": None,
            },
        },
    }

    with patch("server.model_routes.get_mode_config", return_value=config):
        data = await model_routes.list_modes()

    policy = data["modes"]["sd15"]["controlnet_policy"]
    assert policy["enabled"] is False
    assert policy["allowed_control_types"] == {}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_model_routes.py -k controlnet_policy -v`
Expected: FAIL — `controlnet_policy` key missing from response.

- [ ] **Step 3: Add field to `/api/modes` response**

In `server/model_routes.py`, edit `list_modes` (around line 154). Inside the mode dict comprehension, after the `"default_scheduler_id"` entry, add:

```python
                "controlnet_policy": mode_data.get("controlnet_policy") or {
                    "enabled": False,
                    "max_attachments": 0,
                    "allow_reuse_emitted_maps": False,
                    "allowed_control_types": {},
                },
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model_routes.py -k controlnet_policy -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add server/model_routes.py tests/test_model_routes.py
git commit -m "feat(controlnet): expose controlnet_policy in /api/modes (STABL-iajgqfqp)"
```

---

## Task 8: Backend enforcement — `enforce_controlnet_policy`

**Files:**
- Create: `server/controlnet_constraints.py`
- Create: `tests/test_controlnet_constraints.py`

This task enforces the spec's "Attachment-invalid classes" and applies mode defaults.

- [ ] **Step 1: Write failing tests — one per invalid class plus valid + defaulting**

```python
# tests/test_controlnet_constraints.py
import pytest

from server.controlnet_models import ControlNetAttachment, ControlNetPreprocessRequest
from server.controlnet_constraints import enforce_controlnet_policy
from server.mode_config import (
    ControlNetControlTypePolicy,
    ControlNetPolicy,
    ModeConfig,
)


def _make_mode(policy: ControlNetPolicy) -> ModeConfig:
    return ModeConfig(
        name="m",
        model="model.safetensors",
        default_size="1024x1024",
        resolution_options=[{"size": "1024x1024", "aspect_ratio": "1:1"}],
        controlnet_policy=policy,
    )


def _req(controlnets):
    class R:
        pass
    r = R()
    r.controlnets = controlnets
    return r


def _canny_policy() -> ControlNetPolicy:
    return ControlNetPolicy(
        enabled=True,
        max_attachments=2,
        allow_reuse_emitted_maps=True,
        allowed_control_types={
            "canny": ControlNetControlTypePolicy(
                default_model_id="sdxl-canny",
                allowed_model_ids=["sdxl-canny"],
                allow_preprocess=True,
                default_strength=0.8,
                min_strength=0.0,
                max_strength=1.5,
            )
        },
    )


def test_none_controlnets_is_noop():
    enforce_controlnet_policy(_req(None), _make_mode(ControlNetPolicy()))


def test_empty_list_is_noop():
    enforce_controlnet_policy(_req([]), _make_mode(ControlNetPolicy()))


def test_rejects_when_policy_disabled():
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        map_asset_ref="asset_a",
    )
    with pytest.raises(ValueError, match="does not enable ControlNet"):
        enforce_controlnet_policy(_req([att]), _make_mode(ControlNetPolicy()))


def test_rejects_unknown_control_type():
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="pose",
        map_asset_ref="asset_a",
    )
    with pytest.raises(ValueError, match="control_type 'pose'"):
        enforce_controlnet_policy(_req([att]), _make_mode(_canny_policy()))


def test_rejects_model_id_not_in_allowed_list():
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        map_asset_ref="asset_a",
        model_id="rogue-canny",
    )
    with pytest.raises(ValueError, match="model_id 'rogue-canny'"):
        enforce_controlnet_policy(_req([att]), _make_mode(_canny_policy()))


def test_applies_default_model_id_when_omitted():
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        map_asset_ref="asset_a",
    )
    req = _req([att])
    enforce_controlnet_policy(req, _make_mode(_canny_policy()))
    assert req.controlnets[0].model_id == "sdxl-canny"


def test_rejects_strength_outside_policy_bounds():
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        map_asset_ref="asset_a",
        strength=1.8,  # pydantic allows up to 2.0, policy caps at 1.5
    )
    with pytest.raises(ValueError, match="strength"):
        enforce_controlnet_policy(_req([att]), _make_mode(_canny_policy()))


def test_rejects_exceeding_max_attachments():
    atts = [
        ControlNetAttachment(
            attachment_id=f"cn_{i}",
            control_type="canny",
            map_asset_ref=f"asset_{i}",
        )
        for i in range(3)
    ]
    with pytest.raises(ValueError, match="max_attachments"):
        enforce_controlnet_policy(_req(atts), _make_mode(_canny_policy()))


def test_rejects_duplicate_attachment_id():
    atts = [
        ControlNetAttachment(
            attachment_id="cn_dup",
            control_type="canny",
            map_asset_ref="asset_a",
        ),
        ControlNetAttachment(
            attachment_id="cn_dup",
            control_type="canny",
            map_asset_ref="asset_b",
        ),
    ]
    with pytest.raises(ValueError, match="duplicate attachment_id"):
        enforce_controlnet_policy(_req(atts), _make_mode(_canny_policy()))


def test_rejects_preprocess_when_type_policy_forbids():
    policy = _canny_policy()
    policy.allowed_control_types["canny"].allow_preprocess = False
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        source_asset_ref="asset_src",
        preprocess=ControlNetPreprocessRequest(id="canny"),
    )
    with pytest.raises(ValueError, match="preprocessing not allowed"):
        enforce_controlnet_policy(_req([att]), _make_mode(policy))


def test_valid_attachment_passes_through_unchanged():
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        map_asset_ref="asset_a",
        model_id="sdxl-canny",
        strength=1.0,
        start_percent=0.0,
        end_percent=0.75,
    )
    req = _req([att])
    enforce_controlnet_policy(req, _make_mode(_canny_policy()))
    # model_id unchanged; start/end preserved; no exception
    assert req.controlnets[0].model_id == "sdxl-canny"
    assert req.controlnets[0].end_percent == 0.75
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_controlnet_constraints.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.controlnet_constraints'`.

- [ ] **Step 3: Implement the enforcer**

```python
# server/controlnet_constraints.py
"""
Policy-aware validation and normalization for ControlNet attachments.

Runs after Pydantic parsing (which handles field-level invariants) and after
finalize_mode_generate_request (which handles size/steps/guidance defaulting).

Raises ValueError on any invalid attachment. Callers on the HTTP path surface
this as 400; callers on the WS path surface it as a pre_submit_job_error.
"""
from typing import Any


def enforce_controlnet_policy(req: Any, mode: Any) -> None:
    attachments = getattr(req, "controlnets", None)
    if not attachments:
        return

    policy = getattr(mode, "controlnet_policy", None)
    if policy is None or not policy.enabled:
        raise ValueError(f"mode '{mode.name}' does not enable ControlNet")

    if len(attachments) > policy.max_attachments:
        raise ValueError(
            f"request has {len(attachments)} ControlNet attachments; "
            f"mode '{mode.name}' allows max_attachments={policy.max_attachments}"
        )

    seen_ids: set[str] = set()
    for attachment in attachments:
        if attachment.attachment_id in seen_ids:
            raise ValueError(f"duplicate attachment_id '{attachment.attachment_id}' in request")
        seen_ids.add(attachment.attachment_id)

        type_policy = policy.allowed_control_types.get(attachment.control_type)
        if type_policy is None:
            raise ValueError(
                f"control_type '{attachment.control_type}' not allowed for mode '{mode.name}'"
            )

        if attachment.preprocess is not None and not type_policy.allow_preprocess:
            raise ValueError(
                f"preprocessing not allowed for control_type '{attachment.control_type}' "
                f"in mode '{mode.name}'"
            )

        if attachment.model_id is None:
            if type_policy.default_model_id is None:
                raise ValueError(
                    f"model_id required for control_type '{attachment.control_type}' "
                    f"in mode '{mode.name}' (no default configured)"
                )
            attachment.model_id = type_policy.default_model_id
        elif attachment.model_id not in type_policy.allowed_model_ids:
            raise ValueError(
                f"model_id '{attachment.model_id}' not allowed for control_type "
                f"'{attachment.control_type}' in mode '{mode.name}'"
            )

        if not (type_policy.min_strength <= attachment.strength <= type_policy.max_strength):
            raise ValueError(
                f"strength {attachment.strength} outside policy range "
                f"[{type_policy.min_strength}, {type_policy.max_strength}] for "
                f"control_type '{attachment.control_type}' in mode '{mode.name}'"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_controlnet_constraints.py -v`
Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add server/controlnet_constraints.py tests/test_controlnet_constraints.py
git commit -m "feat(controlnet): policy enforcement layer with attachment-invalid classes (STABL-iajgqfqp)"
```

---

## Task 9: Wire enforcement into HTTP `/generate` and WS job-submit

**Files:**
- Modify: `server/lcm_sr_server.py` (around line 520, after `finalize_mode_generate_request` in the `/generate` handler)
- Modify: `server/ws_routes.py` (around line 148, after `finalize_mode_generate_request` in `_handle_job_submit`)

- [ ] **Step 1: Write failing integration tests**

```python
# tests/test_controlnet_dispatch.py
import pytest
from unittest.mock import Mock


def test_ws_build_rejects_disabled_mode_controlnet():
    """WS path: invalid controlnets surface via pre_submit_job_error."""
    from server.controlnet_constraints import enforce_controlnet_policy
    from server.controlnet_models import ControlNetAttachment
    from server.mode_config import ControlNetPolicy, ModeConfig

    class R:
        pass
    req = R()
    req.controlnets = [
        ControlNetAttachment(
            attachment_id="cn_1",
            control_type="canny",
            map_asset_ref="asset_a",
        )
    ]
    mode = ModeConfig(
        name="m",
        model="x.safetensors",
        default_size="1024x1024",
        resolution_options=[{"size": "1024x1024", "aspect_ratio": "1:1"}],
        controlnet_policy=ControlNetPolicy(enabled=False),
    )
    with pytest.raises(ValueError, match="does not enable ControlNet"):
        enforce_controlnet_policy(req, mode)
```

(The dispatch wiring in this task is integration glue; the core behavior is already covered by tests in Task 8. This test exists to lock the combined boundary.)

- [ ] **Step 2: Run test to verify pass (sanity — should already pass against Task 8's module)**

Run: `pytest tests/test_controlnet_dispatch.py -v`
Expected: PASS.

- [ ] **Step 3: Wire enforcement in HTTP `/generate`**

In `server/lcm_sr_server.py`, find the `/generate` handler block (around line 514 where `current_mode = runtime.get_current_mode() ...`). The existing code is:

```python
        try:
            finalize_mode_generate_request(
                req,
                mode,
                env_default_size=os.environ.get("DEFAULT_SIZE", "512x512"),
                env_default_steps=int(os.environ.get("DEFAULT_STEPS", "4")),
                env_default_guidance=float(os.environ.get("DEFAULT_GUIDANCE", "1.0")),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
```

After the closing of that `except` block (so, as a sibling try/except, not nested), append:

```python
        try:
            from server.controlnet_constraints import enforce_controlnet_policy
            enforce_controlnet_policy(req, mode)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
```

- [ ] **Step 4: Wire enforcement in WS job-submit**

In `server/ws_routes.py`, find the block around line 142. Immediately after the `finalize_mode_generate_request(...)` call inside the `try:`, add:

```python
                from server.controlnet_constraints import enforce_controlnet_policy
                enforce_controlnet_policy(req, mode)
```

The existing `except Exception as e: pre_submit_job_error = str(e)` already captures `ValueError`, so no further wiring is needed. The WS path will surface the error as a job-submit ack error rather than an HTTP 400.

- [ ] **Step 5: Manual sanity — run the full test suite**

Run: `pytest tests/test_controlnet_models.py tests/test_controlnet_constraints.py tests/test_controlnet_dispatch.py tests/test_mode_config.py tests/test_model_routes.py tests/test_ws_build_generate_request.py -v`
Expected: all prior tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add server/lcm_sr_server.py server/ws_routes.py tests/test_controlnet_dispatch.py
git commit -m "feat(controlnet): enforce policy on HTTP /generate and WS job-submit (STABL-iajgqfqp)"
```

---

## Task 10: Stub provider dispatch

When a `controlnets` request passes all validation in Tracks 1–2, Track 3 will hand it off to the CUDA provider. Until Track 3 lands, we must reject such requests with a specific error so nothing silently runs a ControlNet-less generation pretending to have honored the request.

**Files:**
- Modify: `server/controlnet_constraints.py` (add `ensure_controlnet_dispatch_supported`)
- Modify: `server/lcm_sr_server.py` and `server/ws_routes.py` (invoke the stub after enforcement)
- Test: extend `tests/test_controlnet_constraints.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_controlnet_constraints.py`:

```python
def test_dispatch_stub_rejects_validated_controlnets():
    from server.controlnet_constraints import ensure_controlnet_dispatch_supported

    class R:
        pass
    r = R()
    r.controlnets = [object()]  # presence is all that matters for the stub
    with pytest.raises(NotImplementedError, match="ControlNet provider not yet implemented"):
        ensure_controlnet_dispatch_supported(r)


def test_dispatch_stub_noop_when_no_controlnets():
    from server.controlnet_constraints import ensure_controlnet_dispatch_supported

    class R:
        pass
    r = R()
    r.controlnets = None
    ensure_controlnet_dispatch_supported(r)  # no exception
    r.controlnets = []
    ensure_controlnet_dispatch_supported(r)  # no exception
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_controlnet_constraints.py::test_dispatch_stub_rejects_validated_controlnets -v`
Expected: FAIL — `ImportError: cannot import name 'ensure_controlnet_dispatch_supported'`.

- [ ] **Step 3: Add the stub**

Append to `server/controlnet_constraints.py`:

```python
def ensure_controlnet_dispatch_supported(req: Any) -> None:
    """
    Stub that rejects any request with a validated `controlnets` list.

    Replaced in Track 3 with a real provider dispatch. Keeping this stub
    explicit (rather than silently dropping) ensures Track 1 can merge while
    execution and preprocessing are still unimplemented.
    """
    attachments = getattr(req, "controlnets", None)
    if attachments:
        raise NotImplementedError(
            "ControlNet provider not yet implemented on this backend "
            "(Track 3 delivers execution)"
        )
```

- [ ] **Step 4: Invoke the stub in HTTP and WS paths**

In `server/lcm_sr_server.py`, immediately after the `enforce_controlnet_policy(...)` call added in Task 9, append:

```python
            try:
                from server.controlnet_constraints import ensure_controlnet_dispatch_supported
                ensure_controlnet_dispatch_supported(req)
            except NotImplementedError as e:
                raise HTTPException(status_code=501, detail=str(e))
```

In `server/ws_routes.py`, immediately after the `enforce_controlnet_policy(...)` call added in Task 9, append:

```python
                from server.controlnet_constraints import ensure_controlnet_dispatch_supported
                ensure_controlnet_dispatch_supported(req)
```

(The surrounding `except Exception as e:` already captures `NotImplementedError`.)

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_controlnet_constraints.py tests/test_controlnet_dispatch.py -v`
Expected: all PASS including the 2 new stub tests.

- [ ] **Step 6: Commit**

```bash
git add server/controlnet_constraints.py server/lcm_sr_server.py server/ws_routes.py tests/test_controlnet_constraints.py
git commit -m "feat(controlnet): stub provider dispatch with 501/NotImplementedError (STABL-iajgqfqp)"
```

---

## Task 11: Full-suite verification + FP wrap

**Files:** none modified; verification and issue hygiene.

- [ ] **Step 1: Run the full relevant test suite**

Run:
```bash
pytest \
  tests/test_controlnet_models.py \
  tests/test_controlnet_constraints.py \
  tests/test_controlnet_dispatch.py \
  tests/test_ws_build_generate_request.py \
  tests/test_mode_config.py \
  tests/test_model_routes.py \
  -v
```
Expected: all PASS, no new failures.

- [ ] **Step 2: Run the existing broader suite to catch regressions**

Run: `pytest tests/ -x --ignore=tests/test_sdxl_worker.py --ignore=tests/test_cuda_worker_base.py --ignore=tests/test_cuda_worker_capabilities.py -v`

(The CUDA/SDXL worker suites require a GPU; skip them in environments without one. If running on a CUDA host, drop the `--ignore` flags.)

Expected: no regressions introduced by this track.

- [ ] **Step 3: FP wrap**

```bash
fp issue update --status done STABL-iajgqfqp
fp comment STABL-iajgqfqp "Track 1 complete. Request contract, mode policy, /api/modes serialization, and enforcement layer landed. Dispatch stub rejects controlnets with 501 pending Track 3. All controlnet-specific tests pass; broader suite shows no regressions."
```

- [ ] **Step 4: Update FP top parent**

```bash
fp comment STABL-utbuhifx "Track 1 (STABL-iajgqfqp) merged. Track 2 (STABL-nsrpodvu) ready to start — plan authoring queued."
```

---

## Testing Notes For The Implementer

- **Pydantic v2 vs v1.** This repo uses Pydantic v2 (`model_validator(mode="after")`). If you see v1-style `@validator` in the codebase, mirror v2 in new code and leave existing v1 untouched.
- **Use `model_construct` sparingly.** The tests in Task 1 rely on full validation; do not bypass it with `model_construct` unless you are intentionally testing the raw path.
- **Dataclass `field(default_factory=...)`.** Needed for any mutable default on `ControlNetPolicy`/`ControlNetControlTypePolicy` (lists, dicts). The plan already uses it; do not collapse to `= []` or `= {}`.
- **Avoid touching `_validate_paths`.** Track 1 adds no new filesystem paths; `controlnets.yaml` resolution is Track 3.
- **Import locations matter.** The plan imports `enforce_controlnet_policy` and `ensure_controlnet_dispatch_supported` inside the HTTP handler and WS handler local scopes to avoid pulling `server.controlnet_*` into module load during FastAPI app startup. Follow the same pattern if you refactor.
- **WS error surface.** The WS path captures `ValueError` and `NotImplementedError` through the existing broad `except Exception` around `finalize_mode_generate_request`. The resulting `pre_submit_job_error` becomes an error ack. No new error framing is required in Track 1.
