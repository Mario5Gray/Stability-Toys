# ControlNet Track 3 Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Track 2 ControlNet execution stub with real CUDA backend execution that supports both preprocess-driven and direct `map_asset_ref` requests, validates model compatibility against a global local-path registry, preserves multi-attachment order, reuses loaded ControlNet models in-process, and emits `controlnet_artifacts` on successful HTTP and WS responses.

**Architecture:** Keep Track 2 request policy and preprocessing intact, then add a backend-only Track 3 layer composed of a global `conf/controlnets.yaml` registry, a request-time binding resolver, a process-local ControlNet model cache, and CUDA worker integration. HTTP `/generate` keeps its binary image body and exposes success artifacts through an HTTP header; WS `job:complete` exposes the same artifacts as a top-level array.

**Tech Stack:** FastAPI, Pydantic v2, PyYAML, PIL, Diffusers, Torch CUDA, existing worker-pool runtime, pytest with mocked Diffusers/Torch seams.

---

## File Structure

### Create

- `conf/controlnets.yaml`
- `server/controlnet_registry.py`
- `server/controlnet_execution.py`
- `backends/controlnet_cache.py`
- `tests/test_controlnet_registry.py`
- `tests/test_controlnet_execution.py`
- `tests/test_controlnet_cache.py`
- `tests/test_cuda_worker_controlnet.py`
- `tests/test_controlnet_success_contract.py`
- `docs/TESTING_CONTROLNET_TRACK3.md`

### Modify

- `server/controlnet_constraints.py`
- `server/lcm_sr_server.py`
- `server/ws_routes.py`
- `backends/platforms/base.py`
- `backends/platforms/cuda.py`
- `backends/platforms/cpu.py`
- `backends/platforms/rknn.py`
- `backends/worker_pool.py`
- `backends/cuda_worker.py`
- `tests/test_ws_routes.py`
- `tests/test_controlnet_http_contract.py`
- `tests/test_worker_pool.py`

### Responsibility Split

- `server/controlnet_registry.py`
  Parses and validates `conf/controlnets.yaml`, exposes strict/lazy validation, and resolves `model_id` to a local on-disk ControlNet model spec.
- `server/controlnet_execution.py`
  Builds ordered runtime bindings from a validated request plus active mode and `AssetStore`.
- `backends/controlnet_cache.py`
  Owns process-local reuse of loaded Diffusers `ControlNetModel` objects with pin/unpin and bounded eviction.
- `backends/cuda_worker.py`
  Applies ordered ControlNet bindings to the active Diffusers pipeline for both SD1.5-style and SDXL workers.
- `server/lcm_sr_server.py` and `server/ws_routes.py`
  Keep Track 2 preprocessing, stop calling the unconditional stub on the CUDA/mode-system path, and emit success-path artifacts.

## Contract Choices Locked By This Plan

- The backend registry lives in `conf/controlnets.yaml` and supports only local paths.
- Compatibility is fail-fast: one bad attachment rejects the whole request before worker execution.
- Execution remains list-oriented even if implementation needs internal simplification.
- HTTP success metadata rides in an `X-ControlNet-Artifacts` JSON header because `/generate` still returns raw image bytes.
- WS success metadata uses a top-level `controlnet_artifacts` array on `job:complete`.
- Cache is process-local only. Restart-persistent cache is out of scope.

### Task 1: Add The ControlNet Registry And Validation Policy

**Files:**
- Create: `conf/controlnets.yaml`
- Create: `server/controlnet_registry.py`
- Test: `tests/test_controlnet_registry.py`

- [ ] **Step 1: Write the failing registry tests**

```python
def test_registry_loads_local_controlnet_specs(tmp_path):
    config_path = tmp_path / "controlnets.yaml"
    model_dir = tmp_path / "models" / "sdxl-canny"
    model_dir.mkdir(parents=True)
    config_path.write_text(
        "models:\n"
        "  sdxl-canny:\n"
        f"    path: {model_dir}\n"
        "    control_types: [canny]\n"
        "    compatible_with: [sdxl]\n",
        encoding="utf-8",
    )

    from server.controlnet_registry import load_controlnet_registry

    registry = load_controlnet_registry(config_path=str(config_path), validation_mode="strict")
    spec = registry.get_required("sdxl-canny")
    assert spec.model_id == "sdxl-canny"
    assert spec.control_types == ("canny",)
    assert spec.compatible_with == ("sdxl",)


def test_registry_rejects_missing_local_path_in_strict_mode(tmp_path):
    config_path = tmp_path / "controlnets.yaml"
    config_path.write_text(
        "models:\n"
        "  sdxl-depth:\n"
        "    path: /does/not/exist\n"
        "    control_types: [depth]\n"
        "    compatible_with: [sdxl]\n",
        encoding="utf-8",
    )

    from server.controlnet_registry import load_controlnet_registry

    with pytest.raises(ValueError, match="does not exist"):
        load_controlnet_registry(config_path=str(config_path), validation_mode="strict")
```

- [ ] **Step 2: Run the registry tests to verify they fail**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_controlnet_registry.py -q`

Expected: `ERROR` because `server.controlnet_registry` does not exist yet.

- [ ] **Step 3: Write the minimal registry implementation**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml


@dataclass(frozen=True)
class ControlNetModelSpec:
    model_id: str
    path: str
    control_types: tuple[str, ...]
    compatible_with: tuple[str, ...]


class ControlNetRegistry:
    def __init__(self, specs: Dict[str, ControlNetModelSpec], validation_mode: str) -> None:
        self._specs = specs
        self.validation_mode = validation_mode

    def get(self, model_id: str) -> Optional[ControlNetModelSpec]:
        return self._specs.get(model_id)

    def get_required(self, model_id: str) -> ControlNetModelSpec:
        spec = self.get(model_id)
        if spec is None:
            raise ValueError(f"unknown ControlNet model_id '{model_id}'")
        if self.validation_mode == "lazy":
            _validate_local_path(spec)
        return spec


def _validate_local_path(spec: ControlNetModelSpec) -> None:
    if not Path(spec.path).exists():
        raise ValueError(f"ControlNet model path does not exist: {spec.path}")


def load_controlnet_registry(*, config_path: str = "conf/controlnets.yaml", validation_mode: str = "strict") -> ControlNetRegistry:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    models = raw.get("models") or {}
    specs: Dict[str, ControlNetModelSpec] = {}
    for model_id, data in models.items():
        spec = ControlNetModelSpec(
            model_id=model_id,
            path=str(data["path"]),
            control_types=tuple(data["control_types"]),
            compatible_with=tuple(data["compatible_with"]),
        )
        if validation_mode == "strict":
            _validate_local_path(spec)
        specs[model_id] = spec
    return ControlNetRegistry(specs=specs, validation_mode=validation_mode)
```

- [ ] **Step 4: Add singleton access and validation-mode env support**

```python
_registry_singleton: Optional[ControlNetRegistry] = None


def get_controlnet_registry() -> ControlNetRegistry:
    global _registry_singleton
    if _registry_singleton is None:
        import os

        _registry_singleton = load_controlnet_registry(
            config_path=os.environ.get("CONTROLNET_REGISTRY_PATH", "conf/controlnets.yaml"),
            validation_mode=os.environ.get("CONTROLNET_REGISTRY_VALIDATION", "strict").strip().lower(),
        )
    return _registry_singleton


def reset_controlnet_registry() -> None:
    global _registry_singleton
    _registry_singleton = None
```

- [ ] **Step 5: Re-run the registry tests**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_controlnet_registry.py -q`

Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add conf/controlnets.yaml server/controlnet_registry.py tests/test_controlnet_registry.py
git commit -m "feat(controlnet): add local-path controlnet registry and validation policy"
```

### Task 2: Resolve Runtime Bindings And Fail Fast On Incompatibility

**Files:**
- Create: `server/controlnet_execution.py`
- Modify: `server/controlnet_constraints.py`
- Test: `tests/test_controlnet_execution.py`

- [ ] **Step 1: Write failing binding-resolution tests**

```python
def test_resolve_controlnet_bindings_rejects_wrong_family(tmp_path):
    from server.asset_store import AssetStore
    from server.controlnet_models import ControlNetAttachment
    from server.controlnet_execution import resolve_controlnet_bindings

    store = AssetStore()
    ref = store.insert("control_map", b"png-bytes")
    req = type("Req", (), {
        "controlnets": [ControlNetAttachment(
            attachment_id="cn_1",
            control_type="canny",
            model_id="sd15-canny",
            map_asset_ref=ref,
        )]
    })()
    mode = type("Mode", (), {"name": "sdxl-mode", "model_path": "/tmp/sdxl.safetensors"})()

    with pytest.raises(ValueError, match="incompatible with active mode family"):
        resolve_controlnet_bindings(req, mode=mode, store=store, active_family="sdxl")


def test_resolve_controlnet_bindings_preserves_request_order(tmp_path):
    from server.asset_store import AssetStore
    from server.controlnet_models import ControlNetAttachment
    from server.controlnet_execution import resolve_controlnet_bindings

    store = AssetStore()
    ref1 = store.insert("control_map", b"first-map")
    ref2 = store.insert("control_map", b"second-map")
    req = type("Req", (), {
        "controlnets": [
            ControlNetAttachment(attachment_id="cn_1", control_type="canny", model_id="sdxl-canny", map_asset_ref=ref1),
            ControlNetAttachment(attachment_id="cn_2", control_type="depth", model_id="sdxl-depth", map_asset_ref=ref2),
        ]
    })()
    mode = type("Mode", (), {"name": "sdxl-mode", "model_path": "/tmp/sdxl.safetensors"})()

    bindings = resolve_controlnet_bindings(req, mode=mode, store=store, active_family="sdxl")
    assert [binding.attachment_id for binding in bindings] == ["cn_1", "cn_2"]
    assert bindings[0].control_image_bytes == b"first-map"
    assert bindings[1].control_image_bytes == b"second-map"
```

- [ ] **Step 2: Run the binding tests to verify they fail**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_controlnet_execution.py -q`

Expected: `ERROR` because `server.controlnet_execution` does not exist yet.

- [ ] **Step 3: Write the runtime binding dataclasses and resolver**

```python
from __future__ import annotations

from dataclasses import dataclass

from server.asset_store import AssetStore
from server.controlnet_registry import get_controlnet_registry


@dataclass(frozen=True)
class ControlNetBinding:
    attachment_id: str
    control_type: str
    model_id: str
    model_path: str
    control_image_bytes: bytes
    strength: float
    start_percent: float
    end_percent: float


def resolve_controlnet_bindings(req, *, mode, store: AssetStore, active_family: str) -> list[ControlNetBinding]:
    attachments = getattr(req, "controlnets", None) or []
    if not attachments:
        return []

    registry = get_controlnet_registry()
    bindings: list[ControlNetBinding] = []
    for attachment in attachments:
        spec = registry.get_required(attachment.model_id)
        if attachment.control_type not in spec.control_types:
            raise ValueError(
                f"model_id '{attachment.model_id}' does not support control_type '{attachment.control_type}'"
            )
        if active_family not in spec.compatible_with:
            raise ValueError(
                f"model_id '{attachment.model_id}' is incompatible with active mode family '{active_family}'"
            )
        entry = store.resolve(attachment.map_asset_ref)
        bindings.append(
            ControlNetBinding(
                attachment_id=attachment.attachment_id,
                control_type=attachment.control_type,
                model_id=attachment.model_id,
                model_path=spec.path,
                control_image_bytes=entry.data,
                strength=float(attachment.strength),
                start_percent=float(attachment.start_percent),
                end_percent=float(attachment.end_percent),
            )
        )
    return bindings
```

- [ ] **Step 4: Replace the unconditional dispatch stub with backend-aware gating**

```python
def ensure_controlnet_dispatch_supported(req: Any, *, supports_controlnet: bool) -> None:
    attachments = getattr(req, "controlnets", None) or []
    if attachments and not supports_controlnet:
        raise NotImplementedError(
            "ControlNet provider not yet implemented on this backend "
            "(Track 3 delivers execution only on CUDA mode-system)"
        )
```

- [ ] **Step 5: Add family detection helper for request-time validation**

```python
def active_model_family_from_variant(variant: str) -> str:
    if variant.startswith("sdxl"):
        return "sdxl"
    if variant.startswith("sd1") or variant.startswith("sd2"):
        return "sd15"
    raise ValueError(f"unsupported active model family for ControlNet: {variant}")
```

- [ ] **Step 6: Re-run the binding tests**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_controlnet_execution.py -q`

Expected: `2 passed`

- [ ] **Step 7: Commit**

```bash
git add server/controlnet_execution.py server/controlnet_constraints.py tests/test_controlnet_execution.py
git commit -m "feat(controlnet): resolve ordered runtime bindings with fail-fast compatibility"
```

### Task 3: Add A Process-Local ControlNet Model Cache

**Files:**
- Create: `backends/controlnet_cache.py`
- Test: `tests/test_controlnet_cache.py`

- [ ] **Step 1: Write failing cache tests**

```python
def test_controlnet_cache_reuses_loaded_model_for_same_model_id():
    from backends.controlnet_cache import ControlNetModelCache

    calls = []

    def loader(path: str):
        calls.append(path)
        return {"path": path}

    cache = ControlNetModelCache(max_entries=2)
    first = cache.acquire("sdxl-canny", "/models/sdxl-canny", loader=loader)
    cache.release("sdxl-canny")
    second = cache.acquire("sdxl-canny", "/models/sdxl-canny", loader=loader)

    assert first is second
    assert calls == ["/models/sdxl-canny"]


def test_controlnet_cache_does_not_evict_pinned_entries():
    from backends.controlnet_cache import ControlNetModelCache

    cache = ControlNetModelCache(max_entries=1)
    cache.acquire("sdxl-canny", "/models/sdxl-canny", loader=lambda path: {"path": path})
    cache.acquire("sdxl-depth", "/models/sdxl-depth", loader=lambda path: {"path": path})
    assert "sdxl-canny" in cache.snapshot()["entries"]
```

- [ ] **Step 2: Run the cache tests to verify they fail**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_controlnet_cache.py -q`

Expected: `ERROR` because `backends.controlnet_cache` does not exist yet.

- [ ] **Step 3: Implement the cache with pin/unpin and LRU eviction**

```python
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable


@dataclass
class CacheEntry:
    model_id: str
    model_path: str
    model: Any
    pin_count: int = 0


class ControlNetModelCache:
    def __init__(self, max_entries: int = 4) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()

    def acquire(self, model_id: str, model_path: str, *, loader: Callable[[str], Any]) -> Any:
        with self._lock:
            entry = self._entries.get(model_id)
            if entry is None:
                entry = CacheEntry(model_id=model_id, model_path=model_path, model=loader(model_path))
                self._entries[model_id] = entry
            else:
                self._entries.move_to_end(model_id)
            entry.pin_count += 1
            self._evict_if_needed()
            return entry.model

    def release(self, model_id: str) -> None:
        with self._lock:
            entry = self._entries[model_id]
            entry.pin_count -= 1

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self._max_entries:
            victim_id, victim = next(iter(self._entries.items()))
            if victim.pin_count > 0:
                break
            self._entries.pop(victim_id)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {"entries": list(self._entries.keys())}
```

- [ ] **Step 4: Add singleton access for worker code**

```python
_CACHE: ControlNetModelCache | None = None


def get_controlnet_cache() -> ControlNetModelCache:
    global _CACHE
    if _CACHE is None:
        _CACHE = ControlNetModelCache()
    return _CACHE


def reset_controlnet_cache() -> None:
    global _CACHE
    _CACHE = None
```

- [ ] **Step 5: Re-run the cache tests**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_controlnet_cache.py -q`

Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add backends/controlnet_cache.py tests/test_controlnet_cache.py
git commit -m "feat(controlnet): add process-local controlnet model cache"
```

### Task 4: Wire ControlNet Bindings Into The CUDA Runtime And Worker Pool

**Files:**
- Modify: `backends/platforms/base.py`
- Modify: `backends/platforms/cuda.py`
- Modify: `backends/platforms/cpu.py`
- Modify: `backends/platforms/rknn.py`
- Modify: `backends/worker_pool.py`
- Test: `tests/test_worker_pool.py`

- [ ] **Step 1: Write failing runtime tests for CUDA-only binding resolution**

```python
def test_cuda_runtime_attaches_controlnet_bindings_before_queueing(mock_mode_config, mock_registry):
    from backends.platforms.cuda import CudaGenerationRuntime

    pool = Mock()
    pool.get_current_mode.return_value = "sdxl-general"
    pool.submit_job.return_value = Future()
    req = SimpleNamespace(controlnets=[SimpleNamespace(attachment_id="cn_1")])

    with patch("backends.platforms.cuda.resolve_controlnet_bindings", return_value=["binding"]) as resolve:
        runtime = CudaGenerationRuntime(pool=pool)
        runtime.submit_generate(req)

    queued_job = pool.submit_job.call_args[0][0]
    assert queued_job.controlnet_bindings == ["binding"]
    resolve.assert_called_once()
```

- [ ] **Step 2: Run the worker/runtime tests to verify they fail**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_worker_pool.py -k controlnet -q`

Expected: `FAIL` because `GenerationJob` does not carry `controlnet_bindings` and CUDA runtime does not resolve them.

- [ ] **Step 3: Extend backend capabilities and generation job metadata**

```python
@dataclass(frozen=True)
class BackendCapabilities:
    supports_generation: bool
    supports_modes: bool
    supports_superres: bool
    supports_model_registry_stats: bool
    supports_img2img: bool
    supports_controlnet: bool = False


@dataclass
class GenerationJob(Job):
    req: Any
    init_image: Optional[bytes] = None
    controlnet_bindings: list[Any] = field(default_factory=list)
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
```

- [ ] **Step 4: Resolve bindings in the CUDA runtime before queueing**

```python
from server.asset_store import get_store
from server.controlnet_execution import resolve_controlnet_bindings
from server.controlnet_registry import active_model_family_from_variant
from server.mode_config import get_mode_config
from utils.model_detector import detect_model


def submit_generate(self, req: Any, *, timeout_s: float = 0.25):
    from backends.worker_pool import GenerationJob

    bindings = []
    if getattr(req, "controlnets", None):
        mode_name = self._pool.get_current_mode()
        mode = get_mode_config().get_mode(mode_name)
        family = active_model_family_from_variant(detect_model(mode.model_path).variant.value)
        bindings = resolve_controlnet_bindings(req, mode=mode, store=get_store(), active_family=family)

    job = GenerationJob(req=req, controlnet_bindings=bindings)
    return self._pool.submit_job(job)
```

- [ ] **Step 5: Mark only CUDA as supporting ControlNet**

```python
class CUDAProvider:
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(True, True, True, True, True, True)


class CPUProvider:
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(False, True, False, False, False, False)


class RKNNProvider:
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(True, False, True, False, False, False)
```

- [ ] **Step 6: Re-run the worker/runtime tests**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_worker_pool.py -k controlnet -q`

Expected: targeted ControlNet runtime tests pass.

- [ ] **Step 7: Commit**

```bash
git add backends/platforms/base.py backends/platforms/cuda.py backends/platforms/cpu.py backends/platforms/rknn.py backends/worker_pool.py tests/test_worker_pool.py
git commit -m "feat(controlnet): resolve cuda controlnet bindings before worker execution"
```

### Task 5: Execute Ordered ControlNet Bindings In CUDA Workers

**Files:**
- Modify: `backends/cuda_worker.py`
- Modify: `backends/worker_pool.py`
- Create: `tests/test_cuda_worker_controlnet.py`

- [ ] **Step 1: Write failing CUDA worker tests with stubbed Diffusers controlnets**

```python
def test_sdxl_worker_passes_controlnet_lists_in_request_order():
    from backends.cuda_worker import DiffusersSDXLCudaWorker

    worker = _make_stub_sdxl_worker()
    job = SimpleNamespace(
        req=_fake_req(),
        init_image=None,
        controlnet_bindings=[
            SimpleNamespace(model_id="sdxl-canny", model_path="/models/canny", control_image_bytes=b"a", strength=0.4, start_percent=0.0, end_percent=0.8),
            SimpleNamespace(model_id="sdxl-depth", model_path="/models/depth", control_image_bytes=b"b", strength=0.9, start_percent=0.1, end_percent=1.0),
        ],
    )

    worker.run_job(job)

    kwargs = worker.pipe.call_args.kwargs
    assert kwargs["controlnet_conditioning_scale"] == [0.4, 0.9]
    assert kwargs["control_guidance_start"] == [0.0, 0.1]
    assert kwargs["control_guidance_end"] == [0.8, 1.0]
```

- [ ] **Step 2: Run the CUDA worker tests to verify they fail**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_cuda_worker_controlnet.py -q`

Expected: `ERROR` because the worker does not yet understand `controlnet_bindings`.

- [ ] **Step 3: Add control-image decode helpers and cache-backed ControlNet loading**

```python
def _decode_control_image(data: bytes, size: tuple[int, int]) -> Image.Image:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    if image.size != size:
        image = image.resize(size)
    return image


def _load_controlnet_model(self, binding):
    from backends.controlnet_cache import get_controlnet_cache
    from diffusers import ControlNetModel

    cache = get_controlnet_cache()
    return cache.acquire(
        binding.model_id,
        binding.model_path,
        loader=lambda path: ControlNetModel.from_pretrained(path, torch_dtype=self.dtype, local_files_only=True),
    )
```

- [ ] **Step 4: Thread ordered ControlNet kwargs into both SD1.5 and SDXL `run_job()`**

```python
bindings = getattr(job, "controlnet_bindings", []) or []
pipe = self.pipe
call_kwargs = {
    "prompt": req.prompt,
    "negative_prompt": getattr(req, "negative_prompt", None),
    "width": width,
    "height": height,
    "num_inference_steps": int(req.num_inference_steps),
    "guidance_scale": float(req.guidance_scale),
    "generator": gen,
}
loaded_ids: list[str] = []
if bindings:
    size = (width, height)
    controlnets = []
    images = []
    scales = []
    starts = []
    ends = []
    for binding in bindings:
        controlnets.append(self._load_controlnet_model(binding))
        loaded_ids.append(binding.model_id)
        images.append(_decode_control_image(binding.control_image_bytes, size))
        scales.append(binding.strength)
        starts.append(binding.start_percent)
        ends.append(binding.end_percent)
    call_kwargs["controlnet"] = controlnets[0] if len(controlnets) == 1 else controlnets
    call_kwargs["image"] = images[0] if len(images) == 1 else images
    call_kwargs["controlnet_conditioning_scale"] = scales[0] if len(scales) == 1 else scales
    call_kwargs["control_guidance_start"] = starts[0] if len(starts) == 1 else starts
    call_kwargs["control_guidance_end"] = ends[0] if len(ends) == 1 else ends

try:
    out = pipe(**call_kwargs)
finally:
    from backends.controlnet_cache import get_controlnet_cache

    cache = get_controlnet_cache()
    for model_id in loaded_ids:
        cache.release(model_id)
```

- [ ] **Step 5: Re-run the CUDA worker tests**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_cuda_worker_controlnet.py -q`

Expected: targeted CUDA worker tests pass for single and multi-attachment inputs.

- [ ] **Step 6: Commit**

```bash
git add backends/cuda_worker.py tests/test_cuda_worker_controlnet.py
git commit -m "feat(controlnet): execute ordered controlnet bindings in cuda workers"
```

### Task 6: Complete HTTP And WS Success-Path Transport

**Files:**
- Modify: `server/lcm_sr_server.py`
- Modify: `server/ws_routes.py`
- Modify: `tests/test_controlnet_http_contract.py`
- Modify: `tests/test_ws_routes.py`
- Create: `tests/test_controlnet_success_contract.py`

- [ ] **Step 1: Write failing success-path contract tests**

```python
def test_http_generate_success_exposes_controlnet_artifacts_header():
    resp = client.post("/generate", json=_valid_controlnet_request())
    assert resp.status_code == 200
    assert json.loads(resp.headers["X-ControlNet-Artifacts"])[0]["attachment_id"] == "cn_1"


async def test_ws_job_complete_includes_controlnet_artifacts():
    complete = await _submit_successful_ws_generate(_valid_controlnet_request())
    assert complete["type"] == "job:complete"
    assert complete["controlnet_artifacts"][0]["attachment_id"] == "cn_1"
```

- [ ] **Step 2: Run the success-path tests to verify they fail**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_controlnet_success_contract.py tests/test_controlnet_http_contract.py tests/test_ws_routes.py -k controlnet -q`

Expected: `FAIL` because current success responses do not include artifacts and CUDA success is not yet wired.

- [ ] **Step 3: Stop calling the unconditional stub on supported CUDA mode-system paths**

```python
provider = get_backend_provider()
supports_controlnet = provider.capabilities().supports_controlnet and current_mode is not None
ensure_controlnet_dispatch_supported(req, supports_controlnet=supports_controlnet)
```

- [ ] **Step 4: Emit artifacts on HTTP success and WS completion**

```python
# server/lcm_sr_server.py
headers = {
    "Cache-Control": "no-store",
    "X-Seed": str(seed),
    "X-SuperRes": "1" if did_superres else "0",
}
if emitted_artifacts:
    headers["X-ControlNet-Artifacts"] = json.dumps([artifact.model_dump() for artifact in emitted_artifacts])


# server/ws_routes.py
await hub.send(client_id, {
    "type": "job:complete",
    "jobId": job_id,
    "outputs": outputs,
    "meta": {"seed": int(seed), "backend": os.environ.get("BACKEND", ""), "sr": did_sr},
    "controlnet_artifacts": [artifact.model_dump() for artifact in getattr(req, "_controlnet_artifacts", [])],
})
```

- [ ] **Step 5: Persist emitted artifacts on the request object immediately after preprocessing**

```python
emitted_artifacts = preprocess_controlnet_attachments(req, get_store())
setattr(req, "_controlnet_artifacts", emitted_artifacts)
```

- [ ] **Step 6: Re-run the HTTP and WS contract tests**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_controlnet_success_contract.py tests/test_controlnet_http_contract.py tests/test_ws_routes.py -k controlnet -q`

Expected: all targeted ControlNet HTTP/WS contract tests pass.

- [ ] **Step 7: Commit**

```bash
git add server/lcm_sr_server.py server/ws_routes.py tests/test_controlnet_http_contract.py tests/test_ws_routes.py tests/test_controlnet_success_contract.py
git commit -m "feat(controlnet): emit controlnet artifacts on successful http and ws responses"
```

### Task 7: Add Startup Validation And Real-CUDA Validation Checklist

**Files:**
- Modify: `server/lcm_sr_server.py`
- Create: `docs/TESTING_CONTROLNET_TRACK3.md`
- Test: `tests/test_controlnet_registry.py`

- [ ] **Step 1: Write the failing startup-validation test**

```python
def test_strict_registry_validation_runs_at_startup(monkeypatch, tmp_path):
    bad_config = tmp_path / "controlnets.yaml"
    bad_config.write_text(
        "models:\n"
        "  broken:\n"
        "    path: /does/not/exist\n"
        "    control_types: [canny]\n"
        "    compatible_with: [sdxl]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CONTROLNET_REGISTRY_PATH", str(bad_config))
    monkeypatch.setenv("CONTROLNET_REGISTRY_VALIDATION", "strict")

    from server.controlnet_registry import reset_controlnet_registry
    from server.lcm_sr_server import _validate_controlnet_registry_for_startup

    reset_controlnet_registry()
    with pytest.raises(ValueError, match="does not exist"):
        _validate_controlnet_registry_for_startup()
```

- [ ] **Step 2: Run the startup-validation test to verify it fails**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_controlnet_registry.py -k startup -q`

Expected: `FAIL` because startup validation helper does not exist yet.

- [ ] **Step 3: Add startup validation hook in `lifespan()`**

```python
def _validate_controlnet_registry_for_startup() -> None:
    from server.controlnet_registry import get_controlnet_registry

    get_controlnet_registry()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("CONTROLNET_REGISTRY_VALIDATION", "strict").strip().lower() == "strict":
        _validate_controlnet_registry_for_startup()
    yield
```

- [ ] **Step 4: Write the manual CUDA validation checklist**

```markdown
# Track 3 ControlNet CUDA Validation

1. Run one successful `canny` request from `source_asset_ref`.
2. Run one successful `depth` request from `source_asset_ref`.
3. Reuse the emitted artifact with `map_asset_ref`.
4. Run a two-attachment request and confirm both bindings are applied in order.
5. Submit an incompatible `model_id` and confirm fail-fast rejection before generation.
6. Repeat requests and observe cache reuse without OOM.
7. Confirm HTTP `X-ControlNet-Artifacts` header on success.
8. Confirm WS `job:complete.controlnet_artifacts` on success.
```

- [ ] **Step 5: Run the focused startup test and the full Track 3 automated slice**

Run: `source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_controlnet_registry.py tests/test_controlnet_execution.py tests/test_controlnet_cache.py tests/test_cuda_worker_controlnet.py tests/test_controlnet_success_contract.py tests/test_controlnet_http_contract.py tests/test_ws_routes.py tests/test_worker_pool.py -q`

Expected: full Track 3 automated slice passes.

- [ ] **Step 6: Commit**

```bash
git add server/lcm_sr_server.py docs/TESTING_CONTROLNET_TRACK3.md tests/test_controlnet_registry.py
git commit -m "docs(controlnet): add track 3 cuda validation gate and startup checks"
```

## Self-Review

- Spec coverage:
  - registry + strict/lazy validation: Tasks 1 and 7
  - fail-fast compatibility and both input paths: Task 2
  - process-local cache: Task 3
  - ordered multi-attachment CUDA execution: Tasks 4 and 5
  - HTTP and WS success artifacts: Task 6
  - manual GPU validation gate: Task 7
- Placeholder scan:
  - no `TBD`, `TODO`, or unnamed files remain.
  - every task names concrete files and commands.
- Type consistency:
  - runtime binding type is `ControlNetBinding`
  - generation jobs carry `controlnet_bindings`
  - HTTP success metadata uses `X-ControlNet-Artifacts`
  - WS success metadata uses top-level `controlnet_artifacts`
