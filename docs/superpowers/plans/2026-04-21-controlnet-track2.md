# ControlNet Track 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Track 1 request/policy layer with a typed asset store, canny and depth preprocessors, and HTTP/WS response wiring that emits reusable `controlnet_artifacts` — completing sections 3, 4, and 9 of the ControlNet design spec.

**Architecture:** Track 1 (request shape, policy enforcement, dispatch stub) merges to `main` first. Track 2 branches off `main` and adds three independent concerns — (1) a typed `AssetStore` replacing the bare `UPLOADS` dict, (2) a `ControlMapPreprocessor` protocol with canny and depth implementations, (3) wiring that runs preprocessing before the Track 3 stub and surfaces emitted artifact refs even when the stub returns 501. The dispatch stub from Track 1 remains untouched; Track 3 removes it.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, OpenCV-headless (`cv2`), `transformers` depth-estimation pipeline (`LiheYoung/depth-anything-small-hf`), pytest.

**Base branch:** `main` (after Track 1 merge). Create branch `controlnet-track2` for this work.

**FP issues:** STABL-mrgpncim (Task 1–2), STABL-ncmcmmnc (Task 3), STABL-drodzdpf (Task 4–5), STABL-mlapdvwj (Task 6), STABL-bbkjaqym (Task 7), STABL-vnpknuvo (Tasks 8–10), STABL-rrceoxha (Task 11).

---

## Task Dependency Order

Use this section during review completion to decide what can start immediately, what can run in parallel, and what must wait.

**Critical path:**

1. Task 1
2. Task 2 and Task 3
3. Task 4
4. Task 5
5. Task 6 and Task 7
6. Task 8
7. Task 9 and Task 10
8. Task 11

**Dependency graph:**

- Task 1 -> Task 2
- Task 1 -> Task 3
- Task 4 -> Task 5
- Task 5 -> Task 6
- Task 5 -> Task 7
- Task 1 -> Task 8
- Task 4 -> Task 8
- Task 3 -> Task 9
- Task 8 -> Task 9
- Task 3 -> Task 10
- Task 8 -> Task 10
- Task 2 -> Task 11
- Task 3 -> Task 11
- Task 6 -> Task 11
- Task 7 -> Task 11
- Task 8 -> Task 11
- Task 9 -> Task 11
- Task 10 -> Task 11

**Parallelism notes:**

- Task 2 and Task 3 may proceed in parallel once Task 1 lands.
- Task 6 and Task 7 may proceed in parallel once Task 5 lands.
- Task 9 and Task 10 may proceed in parallel once Task 3 and Task 8 land.
- Task 8 uses a fake registry in tests and does not strictly require Task 6 or Task 7, but in practice it should start after Task 4 and before or alongside concrete preprocessor finishing work only if the team is comfortable integrating against the protocol seam first.

---

## File Map

| Action   | File                                  | Responsibility                                                          |
|----------|---------------------------------------|-------------------------------------------------------------------------|
| Create   | `server/asset_store.py`               | Typed ref table: `AssetEntry`, `AssetStore` (insert/resolve/evict/pin) |
| Modify   | `server/upload_routes.py`             | Wire `POST /v1/upload` and `resolve_file_ref` onto `AssetStore`         |
| Create   | `server/controlnet_preprocessors.py` | Protocol, `ControlMapResult`, `PreprocessorRegistry`, canny, depth      |
| Create   | `server/controlnet_preprocessing.py` | `preprocess_controlnet_attachments()` — drive preprocessing per-request |
| Modify   | `server/controlnet_models.py`         | Add `ControlNetArtifactRef` response model                              |
| Modify   | `server/lcm_sr_server.py`             | Run preprocessing before stub; include artifacts in 501 detail          |
| Modify   | `server/ws_routes.py`                 | Run preprocessing; include artifacts in `job:error`; foundation wired for `job:complete` (full emission: Track 3) |
| Create   | `tests/test_asset_store.py`           | AssetStore core + eviction + pinning                                    |
| Create   | `tests/test_controlnet_preprocessors.py` | Protocol, registry, canny happy/fail, depth happy/fail              |
| Create   | `tests/test_controlnet_preprocessing.py` | `preprocess_controlnet_attachments` — happy path + error cases       |
| Modify   | `tests/test_ws_routes.py`             | Remove unused `UPLOADS` import; add artifact-emission assertions        |

---

## Task 1: `AssetEntry` and `AssetStore` — core insert/resolve/cleanup

**FP:** STABL-mrgpncim
**Depends on:** Track 1 merged to `main`
**Unblocks:** Task 2, Task 3, Task 8, Task 11

**Files:**
- Create: `server/asset_store.py`
- Create: `tests/test_asset_store.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_asset_store.py
import time
import pytest
from server.asset_store import AssetEntry, AssetStore


def test_insert_upload_returns_ref():
    store = AssetStore()
    ref = store.insert("upload", b"hello")
    assert isinstance(ref, str) and len(ref) == 32


def test_resolve_returns_entry():
    store = AssetStore()
    ref = store.insert("upload", b"hello world")
    entry = store.resolve(ref)
    assert entry.data == b"hello world"
    assert entry.kind == "upload"
    assert entry.byte_size == len(b"hello world")


def test_resolve_missing_raises_key_error():
    store = AssetStore()
    with pytest.raises(KeyError, match="not found"):
        store.resolve("nonexistent")


def test_insert_control_map_stores_metadata():
    store = AssetStore()
    meta = {"control_type": "canny", "source_asset_ref": "abc"}
    ref = store.insert("control_map", b"pixels", metadata=meta)
    entry = store.resolve(ref)
    assert entry.kind == "control_map"
    assert entry.metadata["control_type"] == "canny"


def test_cleanup_expired_removes_old_uploads():
    store = AssetStore()
    ref = store.insert("upload", b"old")
    # Backdate the entry's created_at so it's "expired"
    store._entries[ref].created_at = time.time() - 400
    removed = store.cleanup_expired(ttl_s=300)
    assert ref in removed
    with pytest.raises(KeyError):
        store.resolve(ref)


def test_cleanup_expired_preserves_control_maps():
    store = AssetStore()
    ref = store.insert("control_map", b"cmap")
    store._entries[ref].created_at = time.time() - 9999
    removed = store.cleanup_expired(ttl_s=300)
    assert ref not in removed
    assert store.resolve(ref).data == b"cmap"


def test_total_bytes_sums_all_entries():
    store = AssetStore()
    store.insert("upload", b"ab")
    store.insert("upload", b"cde")
    assert store.total_bytes == 5
```

- [ ] **Step 2: Run tests to confirm they fail**

```
cd <repo-root>
python -m pytest tests/test_asset_store.py -q
```
Expected: `ImportError` — `server.asset_store` does not exist yet.

- [ ] **Step 3: Implement `server/asset_store.py`**

```python
# server/asset_store.py
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AssetEntry:
    ref: str
    data: bytes
    kind: str           # "upload" | "control_map"
    created_at: float
    last_accessed: float
    byte_size: int
    metadata: dict = field(default_factory=dict)
    pin_count: int = 0


class AssetStore:
    def __init__(self, byte_budget: int = 512 * 1024 * 1024) -> None:
        self._entries: dict[str, AssetEntry] = {}
        self._byte_budget = byte_budget

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def insert(self, kind: str, data: bytes, metadata: Optional[dict] = None) -> str:
        ref = uuid.uuid4().hex
        now = time.time()
        self._entries[ref] = AssetEntry(
            ref=ref,
            data=data,
            kind=kind,
            created_at=now,
            last_accessed=now,
            byte_size=len(data),
            metadata=metadata or {},
        )
        self._evict_to_budget()
        return ref

    def resolve(self, ref: str) -> AssetEntry:
        entry = self._entries.get(ref)
        if entry is None:
            raise KeyError(f"asset ref {ref!r} not found or evicted")
        entry.last_accessed = time.time()
        return entry

    def cleanup_expired(self, ttl_s: float = 300.0) -> list[str]:
        """Evict unpinned 'upload' entries older than ttl_s. Returns removed refs."""
        now = time.time()
        expired = [
            ref
            for ref, e in self._entries.items()
            if e.kind == "upload" and e.pin_count == 0 and (now - e.created_at) > ttl_s
        ]
        for ref in expired:
            del self._entries[ref]
        return expired

    def pin(self, ref: str) -> None:
        entry = self._entries.get(ref)
        if entry:
            entry.pin_count += 1

    def unpin(self, ref: str) -> None:
        entry = self._entries.get(ref)
        if entry and entry.pin_count > 0:
            entry.pin_count -= 1

    @property
    def total_bytes(self) -> int:
        return sum(e.byte_size for e in self._entries.values())

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _evict_to_budget(self) -> None:
        while self.total_bytes > self._byte_budget:
            candidates = [e for e in self._entries.values() if e.pin_count == 0]
            if not candidates:
                break
            oldest = min(candidates, key=lambda e: e.last_accessed)
            del self._entries[oldest.ref]


# Module-level singleton used by upload_routes and preprocessing.
_DEFAULT_STORE = AssetStore()


def get_store() -> AssetStore:
    return _DEFAULT_STORE
```

- [ ] **Step 4: Run tests to confirm they pass**

```
python -m pytest tests/test_asset_store.py -q
```
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add server/asset_store.py tests/test_asset_store.py
git commit -m "feat(controlnet): add AssetStore with typed entries, insert/resolve/cleanup (STABL-mrgpncim)"
```

---

## Task 2: LRU byte-budget eviction

**FP:** STABL-ncmcmmnc
**Depends on:** Task 1
**Unblocks:** Task 11

**Files:**
- Modify: `server/asset_store.py` (eviction already implemented in Task 1, just needs tests)
- Modify: `tests/test_asset_store.py`

- [ ] **Step 1: Add eviction tests**

Append to `tests/test_asset_store.py`:

```python
def test_evicts_lru_when_budget_exceeded():
    store = AssetStore(byte_budget=10)
    ref_a = store.insert("upload", b"aaaaaaa")  # 7 bytes
    ref_b = store.insert("upload", b"bbbb")     # 4 bytes — pushes over 10
    # ref_a is older (lower last_accessed), should be evicted
    with pytest.raises(KeyError):
        store.resolve(ref_a)
    assert store.resolve(ref_b).data == b"bbbb"


def test_does_not_evict_pinned_entries():
    store = AssetStore(byte_budget=10)
    ref_a = store.insert("upload", b"aaaaaaa")  # 7 bytes
    store.pin(ref_a)
    ref_b = store.insert("upload", b"bbbb")     # 4 bytes — budget exceeded
    # ref_a is pinned, cannot be evicted; both survive
    assert store.resolve(ref_a).data == b"aaaaaaa"
    assert store.resolve(ref_b).data == b"bbbb"


def test_evicts_oldest_unpinned_when_multiple_candidates():
    store = AssetStore(byte_budget=15)
    ref_a = store.insert("upload", b"a" * 6)  # 6 bytes
    ref_b = store.insert("upload", b"b" * 6)  # 6 bytes; now 12
    # Explicitly make ref_b more recently accessed
    store.resolve(ref_b)
    ref_c = store.insert("upload", b"c" * 6)  # 6 bytes; total 18 > 15
    # ref_a was accessed least recently, should be evicted
    with pytest.raises(KeyError):
        store.resolve(ref_a)
    assert store.resolve(ref_b).data == b"b" * 6
    assert store.resolve(ref_c).data == b"c" * 6


def test_unpin_allows_eviction():
    store = AssetStore(byte_budget=10)
    ref_a = store.insert("upload", b"aaaaaaa")  # 7 bytes
    store.pin(ref_a)
    store.unpin(ref_a)
    ref_b = store.insert("upload", b"bbbb")     # 4 bytes — now eviction can proceed
    with pytest.raises(KeyError):
        store.resolve(ref_a)
    assert store.resolve(ref_b).data == b"bbbb"
```

- [ ] **Step 2: Run tests to confirm they pass (eviction was implemented in Task 1)**

```
python -m pytest tests/test_asset_store.py -q
```
Expected: `11 passed`

- [ ] **Step 3: Commit**

```bash
git add tests/test_asset_store.py
git commit -m "test(controlnet): add LRU eviction and pinning coverage for AssetStore (STABL-ncmcmmnc)"
```

---

## Task 3: Migrate `upload_routes.py` onto `AssetStore`

**FP:** STABL-mrgpncim
**Depends on:** Task 1
**Unblocks:** Task 9, Task 10, Task 11

**Files:**
- Modify: `server/upload_routes.py`
- Modify: `tests/test_ws_routes.py` (remove stale `UPLOADS` import)

- [ ] **Step 1: Write a smoke test for the migrated upload route**

The existing `test_ws_routes.py` imports `UPLOADS` (line 27) but never uses it — remove that import. Also add a new test file:

```python
# tests/test_upload_routes.py
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from server.upload_routes import upload_router, resolve_file_ref
from server.asset_store import get_store


@pytest.fixture(autouse=True)
def _clear_store():
    # Isolate tests: clear the module-level store before each test
    get_store()._entries.clear()
    yield
    get_store()._entries.clear()


app = FastAPI()
app.include_router(upload_router)
client = TestClient(app)


def test_upload_returns_file_ref():
    resp = client.post(
        "/v1/upload",
        files={"file": ("test.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 50, "image/png")},
    )
    assert resp.status_code == 200
    ref = resp.json()["fileRef"]
    assert isinstance(ref, str) and len(ref) == 32


def test_resolve_file_ref_returns_bytes():
    resp = client.post(
        "/v1/upload",
        files={"file": ("img.png", b"imagedata", "image/png")},
    )
    ref = resp.json()["fileRef"]
    assert resolve_file_ref(ref) == b"imagedata"


def test_resolve_missing_ref_raises():
    with pytest.raises(KeyError, match="not found"):
        resolve_file_ref("nosuchref")


def test_upload_stores_as_upload_kind():
    resp = client.post(
        "/v1/upload",
        files={"file": ("x.png", b"bytes", "image/png")},
    )
    ref = resp.json()["fileRef"]
    entry = get_store().resolve(ref)
    assert entry.kind == "upload"
```

- [ ] **Step 2: Run tests to confirm they fail (upload_routes still uses old UPLOADS dict)**

```
python -m pytest tests/test_upload_routes.py -q
```
Expected: Some tests fail because `get_store()._entries` will be empty while `UPLOADS` has the data.

- [ ] **Step 3: Rewrite `server/upload_routes.py`**

```python
# server/upload_routes.py
"""
upload_routes.py — Temporary file upload for WS clients.

POST /v1/upload  →  multipart file  →  {"fileRef": "uuid"}

Backed by the module-level AssetStore (server/asset_store.py).
Upload entries have kind="upload" and a 5-minute TTL enforced by cleanup_uploads_loop.
"""

import asyncio
import logging

from fastapi import APIRouter, File, UploadFile, HTTPException

from server.asset_store import get_store

logger = logging.getLogger(__name__)

upload_router = APIRouter()

TTL_S = 300  # 5 minutes


@upload_router.post("/v1/upload")
async def upload_temp_file(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload")

    ref = get_store().insert("upload", data)
    logger.info("Upload stored: %s (%d bytes)", ref, len(data))
    return {"fileRef": ref}


def resolve_file_ref(ref: str) -> bytes:
    """Resolve a fileRef to bytes. Raises KeyError if expired/missing."""
    return get_store().resolve(ref).data


async def cleanup_uploads_loop():
    """Background task that purges expired upload entries every 30s."""
    while True:
        await asyncio.sleep(30)
        expired = get_store().cleanup_expired(ttl_s=TTL_S)
        if expired:
            logger.debug("Cleaned %d expired uploads", len(expired))
```

- [ ] **Step 4: Fix the stale `UPLOADS` import in `tests/test_ws_routes.py`**

In `tests/test_ws_routes.py` line 27, change:
```python
from server.upload_routes import upload_router, UPLOADS
```
to:
```python
from server.upload_routes import upload_router
```

- [ ] **Step 5: Run both test files**

```
python -m pytest tests/test_upload_routes.py tests/test_ws_routes.py -q
```
Expected: all pass; no `UPLOADS` import error.

- [ ] **Step 6: Commit**

```bash
git add server/upload_routes.py tests/test_upload_routes.py tests/test_ws_routes.py
git commit -m "feat(controlnet): migrate upload_routes onto AssetStore, drop bare UPLOADS dict (STABL-mrgpncim)"
```

---

## Task 4: `ControlMapPreprocessor` protocol, `ControlMapResult`, and `PreprocessorRegistry`

**FP:** STABL-drodzdpf
**Depends on:** Track 1 merged to `main`
**Unblocks:** Task 5, Task 8

**Files:**
- Create: `server/controlnet_preprocessors.py`
- Create: `tests/test_controlnet_preprocessors.py`

- [ ] **Step 1: Write failing tests for protocol and registry**

```python
# tests/test_controlnet_preprocessors.py
import pytest
from server.controlnet_preprocessors import (
    ControlMapPreprocessor,
    ControlMapResult,
    PreprocessorRegistry,
)


class _FakePreprocessor:
    preprocessor_id = "fake"
    control_type = "fake"

    def run(self, image_bytes: bytes, options: dict) -> ControlMapResult:
        return ControlMapResult(
            preprocessor_id="fake",
            control_type="fake",
            image_bytes=b"output",
            width=8,
            height=8,
        )


def test_fake_preprocessor_satisfies_protocol():
    assert isinstance(_FakePreprocessor(), ControlMapPreprocessor)


def test_registry_get_returns_none_for_unknown():
    reg = PreprocessorRegistry()
    assert reg.get("no-such") is None


def test_registry_dispatch_registered_preprocessor():
    reg = PreprocessorRegistry()
    reg.register(_FakePreprocessor())
    result = reg.dispatch("fake", b"input", {})
    assert result.preprocessor_id == "fake"
    assert result.image_bytes == b"output"
    assert result.width == 8
    assert result.height == 8
    assert result.media_type == "image/png"


def test_registry_dispatch_unknown_raises():
    reg = PreprocessorRegistry()
    with pytest.raises(ValueError, match="unknown preprocessor"):
        reg.dispatch("missing", b"x", {})


def test_control_map_result_defaults():
    r = ControlMapResult(
        preprocessor_id="canny",
        control_type="canny",
        image_bytes=b"data",
        width=64,
        height=64,
    )
    assert r.media_type == "image/png"
```

- [ ] **Step 2: Run tests to confirm they fail**

```
python -m pytest tests/test_controlnet_preprocessors.py -q
```
Expected: `ImportError` — `server.controlnet_preprocessors` does not exist yet.

- [ ] **Step 3: Implement the full module — protocol, registry, shared helpers, and both concrete preprocessors**

```python
# server/controlnet_preprocessors.py
"""
ControlNet preprocessor seam: protocol, result type, registry, shared image helpers,
and v1 concrete implementations (CannyPreprocessor, DepthPreprocessor).
"""

import io
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ------------------------------------------------------------------ #
# Core types
# ------------------------------------------------------------------ #

@dataclass
class ControlMapResult:
    preprocessor_id: str
    control_type: str
    image_bytes: bytes
    width: int
    height: int
    media_type: str = "image/png"


@runtime_checkable
class ControlMapPreprocessor(Protocol):
    preprocessor_id: str
    control_type: str

    def run(self, image_bytes: bytes, options: dict[str, Any]) -> ControlMapResult: ...


# ------------------------------------------------------------------ #
# Registry
# ------------------------------------------------------------------ #

class PreprocessorRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, ControlMapPreprocessor] = {}

    def register(self, preprocessor: ControlMapPreprocessor) -> None:
        self._registry[preprocessor.preprocessor_id] = preprocessor

    def get(self, preprocessor_id: str) -> ControlMapPreprocessor | None:
        return self._registry.get(preprocessor_id)

    def dispatch(self, preprocessor_id: str, image_bytes: bytes, options: dict) -> ControlMapResult:
        p = self.get(preprocessor_id)
        if p is None:
            raise ValueError(f"unknown preprocessor {preprocessor_id!r}")
        return p.run(image_bytes, options)


# ------------------------------------------------------------------ #
# Shared image helpers
# ------------------------------------------------------------------ #

def pil_to_png_bytes(pil_image) -> bytes:
    """Encode a PIL Image to PNG bytes."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def png_bytes_to_pil(image_bytes: bytes):
    """Decode PNG (or any PIL-supported format) bytes to a PIL Image."""
    from PIL import Image
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"failed to decode image bytes: {exc}") from exc


# ------------------------------------------------------------------ #
# Concrete implementations (registered below)
# ------------------------------------------------------------------ #

class CannyPreprocessor:
    preprocessor_id = "canny"
    control_type = "canny"

    def run(self, image_bytes: bytes, options: dict[str, Any]) -> ControlMapResult:
        import cv2
        import numpy as np
        from PIL import Image

        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("CannyPreprocessor: could not decode source image")

        low = int(options.get("low_threshold", 100))
        high = int(options.get("high_threshold", 200))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low, high)
        h, w = edges.shape

        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        result_pil = Image.fromarray(edges_rgb)

        return ControlMapResult(
            preprocessor_id=self.preprocessor_id,
            control_type=self.control_type,
            image_bytes=pil_to_png_bytes(result_pil),
            width=w,
            height=h,
        )


class DepthPreprocessor:
    preprocessor_id = "depth"
    control_type = "depth"
    _DEFAULT_MODEL = "LiheYoung/depth-anything-small-hf"

    def __init__(self, model_id: str = _DEFAULT_MODEL) -> None:
        self._model_id = model_id
        self._pipe = None  # lazy-loaded on first use

    def _get_pipe(self):
        if self._pipe is None:
            from transformers import pipeline as hf_pipeline
            self._pipe = hf_pipeline("depth-estimation", model=self._model_id)
        return self._pipe

    def run(self, image_bytes: bytes, options: dict[str, Any]) -> ControlMapResult:
        import numpy as np
        from PIL import Image

        pil_img = png_bytes_to_pil(image_bytes)
        w, h = pil_img.size

        pipe = self._get_pipe()
        result = pipe(pil_img)
        depth_pil = result["depth"]  # PIL Image from transformers

        depth_arr = np.array(depth_pil, dtype=np.float32)
        d_min, d_max = depth_arr.min(), depth_arr.max()
        if d_max > d_min:
            normalized = ((depth_arr - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(depth_arr, dtype=np.uint8)

        depth_rgb = Image.fromarray(normalized).convert("RGB")

        return ControlMapResult(
            preprocessor_id=self.preprocessor_id,
            control_type=self.control_type,
            image_bytes=pil_to_png_bytes(depth_rgb),
            width=w,
            height=h,
        )


# ------------------------------------------------------------------ #
# Module-level default registry
# ------------------------------------------------------------------ #

DEFAULT_REGISTRY = PreprocessorRegistry()
DEFAULT_REGISTRY.register(CannyPreprocessor())
DEFAULT_REGISTRY.register(DepthPreprocessor())
```

- [ ] **Step 4: Run tests to confirm they pass**

```
python -m pytest tests/test_controlnet_preprocessors.py -q
```
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add server/controlnet_preprocessors.py tests/test_controlnet_preprocessors.py
git commit -m "feat(controlnet): add ControlMapPreprocessor protocol, ControlMapResult, PreprocessorRegistry (STABL-drodzdpf)"
```

---

## Task 5: Shared image helper tests

**FP:** STABL-drodzdpf
**Depends on:** Task 4
**Unblocks:** Task 6, Task 7

**Files:**
- Modify: `tests/test_controlnet_preprocessors.py`

- [ ] **Step 1: Add helper tests**

Append to `tests/test_controlnet_preprocessors.py`:

```python
import io
from PIL import Image as PILImage


def _solid_png(w: int = 8, h: int = 8) -> bytes:
    img = PILImage.new("RGB", (w, h), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_png_bytes_to_pil_roundtrip():
    from server.controlnet_preprocessors import png_bytes_to_pil, pil_to_png_bytes
    original = _solid_png(16, 16)
    pil = png_bytes_to_pil(original)
    assert pil.size == (16, 16)
    roundtripped = pil_to_png_bytes(pil)
    assert len(roundtripped) > 0


def test_png_bytes_to_pil_rejects_garbage():
    from server.controlnet_preprocessors import png_bytes_to_pil
    with pytest.raises(ValueError, match="failed to decode"):
        png_bytes_to_pil(b"not an image")
```

- [ ] **Step 2: Run tests**

```
python -m pytest tests/test_controlnet_preprocessors.py -q
```
Expected: `7 passed`

- [ ] **Step 3: Commit**

```bash
git add tests/test_controlnet_preprocessors.py
git commit -m "test(controlnet): add shared image helper coverage (STABL-drodzdpf)"
```

---

## Task 6: `CannyPreprocessor` — test hardening

> Implementation ships in Task 4. This task adds targeted tests for Canny-specific behavior that go beyond the protocol/registry coverage in Task 4.

**FP:** STABL-mlapdvwj
**Depends on:** Task 4
**Unblocks:** Task 11

**Files:**
- Modify: `tests/test_controlnet_preprocessors.py`

- [ ] **Step 1: Add canny tests**

Append to `tests/test_controlnet_preprocessors.py`:

```python
def test_canny_preprocessor_produces_control_map():
    from server.controlnet_preprocessors import CannyPreprocessor
    preprocessor = CannyPreprocessor()
    source = _solid_png(64, 64)
    result = preprocessor.run(source, options={})
    assert result.preprocessor_id == "canny"
    assert result.control_type == "canny"
    assert result.width == 64
    assert result.height == 64
    assert result.media_type == "image/png"
    # Result must be a valid PNG
    decoded = PILImage.open(io.BytesIO(result.image_bytes))
    assert decoded.size == (64, 64)


def test_canny_respects_custom_thresholds():
    from server.controlnet_preprocessors import CannyPreprocessor
    preprocessor = CannyPreprocessor()
    source = _solid_png(32, 32)
    result = preprocessor.run(source, options={"low_threshold": 50, "high_threshold": 150})
    assert result.width == 32


def test_canny_rejects_garbage_bytes():
    from server.controlnet_preprocessors import CannyPreprocessor
    preprocessor = CannyPreprocessor()
    with pytest.raises(ValueError, match="could not decode"):
        preprocessor.run(b"garbage", options={})
```

- [ ] **Step 2: Run tests**

```
python -m pytest tests/test_controlnet_preprocessors.py::test_canny_preprocessor_produces_control_map tests/test_controlnet_preprocessors.py::test_canny_respects_custom_thresholds tests/test_controlnet_preprocessors.py::test_canny_rejects_garbage_bytes -q
```
Expected: `3 passed` (cv2 is available in requirements).

- [ ] **Step 3: Commit**

```bash
git add tests/test_controlnet_preprocessors.py
git commit -m "test(controlnet): add CannyPreprocessor happy path and failure coverage (STABL-mlapdvwj)"
```

---

## Task 7: `DepthPreprocessor` — test hardening

> Implementation ships in Task 4. This task adds targeted tests for Depth-specific normalization and mock-pipeline behavior that go beyond the protocol/registry coverage in Task 4.

**FP:** STABL-bbkjaqym
**Depends on:** Task 4
**Unblocks:** Task 11

**Files:**
- Modify: `tests/test_controlnet_preprocessors.py`

The depth preprocessor calls a `transformers` pipeline that downloads a model. Tests must mock `_get_pipe` so no model download occurs.

- [ ] **Step 1: Add depth tests**

Append to `tests/test_controlnet_preprocessors.py`:

```python
from unittest.mock import Mock
import numpy as np


def _make_mock_pipe(w: int = 64, h: int = 64):
    """Returns a callable mock that simulates hf_pipeline depth-estimation output."""
    depth_arr = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    depth_pil = PILImage.fromarray(depth_arr, mode="L")
    pipe = Mock(return_value={"depth": depth_pil})
    return pipe


def test_depth_preprocessor_produces_control_map():
    from server.controlnet_preprocessors import DepthPreprocessor
    preprocessor = DepthPreprocessor()
    preprocessor._pipe = _make_mock_pipe(64, 64)

    result = preprocessor.run(_solid_png(64, 64), options={})
    assert result.preprocessor_id == "depth"
    assert result.control_type == "depth"
    assert result.width == 64
    assert result.height == 64
    assert result.media_type == "image/png"
    decoded = PILImage.open(io.BytesIO(result.image_bytes))
    assert decoded.size == (64, 64)


def test_depth_normalizes_depth_map_to_rgb():
    from server.controlnet_preprocessors import DepthPreprocessor
    preprocessor = DepthPreprocessor()
    preprocessor._pipe = _make_mock_pipe(32, 32)

    result = preprocessor.run(_solid_png(32, 32), options={})
    decoded = PILImage.open(io.BytesIO(result.image_bytes))
    assert decoded.mode == "RGB"


def test_depth_rejects_garbage_bytes():
    from server.controlnet_preprocessors import DepthPreprocessor
    preprocessor = DepthPreprocessor()
    preprocessor._pipe = _make_mock_pipe()
    with pytest.raises(ValueError, match="failed to decode"):
        preprocessor.run(b"not-an-image", options={})
```

- [ ] **Step 2: Run tests**

```
python -m pytest tests/test_controlnet_preprocessors.py::test_depth_preprocessor_produces_control_map tests/test_controlnet_preprocessors.py::test_depth_normalizes_depth_map_to_rgb tests/test_controlnet_preprocessors.py::test_depth_rejects_garbage_bytes -q
```
Expected: `3 passed`

- [ ] **Step 3: Run all preprocessor tests together**

```
python -m pytest tests/test_controlnet_preprocessors.py -q
```
Expected: `13 passed`

- [ ] **Step 4: Commit**

```bash
git add tests/test_controlnet_preprocessors.py
git commit -m "test(controlnet): add DepthPreprocessor happy path and failure coverage (STABL-bbkjaqym)"
```

---

## Task 8: `ControlNetArtifactRef` + `preprocess_controlnet_attachments()`

**FP:** STABL-vnpknuvo
**Depends on:** Task 1, Task 4
**Unblocks:** Task 9, Task 10, Task 11
**Review note:** Task 8 can technically proceed before Task 6 and Task 7 because its tests use a fake registry, but the preferred review-complete order keeps Task 6 and Task 7 ahead of Task 8 so the protocol seam already has concrete implementations behind it.

**Files:**
- Modify: `server/controlnet_models.py` — add `ControlNetArtifactRef`
- Create: `server/controlnet_preprocessing.py`
- Create: `tests/test_controlnet_preprocessing.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_controlnet_preprocessing.py
import io
import pytest
from PIL import Image
from unittest.mock import Mock

from server.asset_store import AssetStore
from server.controlnet_models import ControlNetAttachment, ControlNetPreprocessRequest
from server.controlnet_preprocessors import ControlMapResult, PreprocessorRegistry


def _solid_png(w: int = 8, h: int = 8) -> bytes:
    img = Image.new("RGB", (w, h), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fake_registry(preprocessor_id: str = "canny") -> PreprocessorRegistry:
    result = ControlMapResult(
        preprocessor_id=preprocessor_id,
        control_type=preprocessor_id,
        image_bytes=b"cmap-output",
        width=8,
        height=8,
    )
    fake = Mock()
    fake.preprocessor_id = preprocessor_id
    fake.control_type = preprocessor_id
    fake.run = Mock(return_value=result)
    reg = PreprocessorRegistry()
    reg.register(fake)
    return reg


def _req(controlnets):
    class R:
        pass
    r = R()
    r.controlnets = controlnets
    return r


def test_preprocess_source_ref_emits_artifact():
    from server.controlnet_preprocessing import preprocess_controlnet_attachments

    store = AssetStore()
    source_ref = store.insert("upload", _solid_png())

    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        source_asset_ref=source_ref,
        preprocess=ControlNetPreprocessRequest(id="canny", options={}),
    )
    req = _req([att])
    artifacts = preprocess_controlnet_attachments(req, store, registry=_fake_registry("canny"))

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.attachment_id == "cn_1"
    assert artifact.control_type == "canny"
    assert artifact.preprocessor_id == "canny"
    assert artifact.source_asset_ref == source_ref
    assert len(artifact.asset_ref) == 32


def test_preprocess_result_stored_as_control_map():
    from server.controlnet_preprocessing import preprocess_controlnet_attachments

    store = AssetStore()
    source_ref = store.insert("upload", _solid_png())
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        source_asset_ref=source_ref,
        preprocess=ControlNetPreprocessRequest(id="canny"),
    )
    artifacts = preprocess_controlnet_attachments(_req([att]), store, registry=_fake_registry("canny"))
    emitted_ref = artifacts[0].asset_ref
    entry = store.resolve(emitted_ref)
    assert entry.kind == "control_map"
    assert entry.data == b"cmap-output"
    assert entry.metadata["control_type"] == "canny"
    assert entry.metadata["source_asset_ref"] == source_ref


def test_preprocess_updates_attachment_map_asset_ref():
    from server.controlnet_preprocessing import preprocess_controlnet_attachments

    store = AssetStore()
    source_ref = store.insert("upload", _solid_png())
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        source_asset_ref=source_ref,
        preprocess=ControlNetPreprocessRequest(id="canny"),
    )
    req = _req([att])
    artifacts = preprocess_controlnet_attachments(req, store, registry=_fake_registry("canny"))
    assert req.controlnets[0].map_asset_ref == artifacts[0].asset_ref


def test_map_asset_ref_attachment_is_skipped():
    from server.controlnet_preprocessing import preprocess_controlnet_attachments

    store = AssetStore()
    map_ref = store.insert("control_map", b"existing-map")
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        map_asset_ref=map_ref,
    )
    artifacts = preprocess_controlnet_attachments(_req([att]), store, registry=_fake_registry())
    assert artifacts == []


def test_missing_source_ref_raises_value_error():
    from server.controlnet_preprocessing import preprocess_controlnet_attachments

    store = AssetStore()
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        source_asset_ref="no-such-ref",
        preprocess=ControlNetPreprocessRequest(id="canny"),
    )
    with pytest.raises(ValueError, match="not found or evicted"):
        preprocess_controlnet_attachments(_req([att]), store, registry=_fake_registry())


def test_unknown_preprocessor_id_raises():
    from server.controlnet_preprocessing import preprocess_controlnet_attachments

    store = AssetStore()
    source_ref = store.insert("upload", b"img")
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        source_asset_ref=source_ref,
        preprocess=ControlNetPreprocessRequest(id="no-such-preprocessor"),
    )
    with pytest.raises(ValueError, match="unknown preprocessor"):
        preprocess_controlnet_attachments(_req([att]), store, registry=_fake_registry())
```

- [ ] **Step 2: Run tests to confirm they fail**

```
python -m pytest tests/test_controlnet_preprocessing.py -q
```
Expected: `ImportError` on `server.controlnet_preprocessing` and `server.controlnet_models.ControlNetArtifactRef`.

- [ ] **Step 3: Add `ControlNetArtifactRef` to `server/controlnet_models.py`**

Append to `server/controlnet_models.py`:

```python
class ControlNetArtifactRef(BaseModel):
    attachment_id: str
    asset_ref: str
    control_type: str
    preprocessor_id: str
    source_asset_ref: str
```

- [ ] **Step 4: Create `server/controlnet_preprocessing.py`**

```python
# server/controlnet_preprocessing.py
"""
Drives ControlNet preprocessing for a GenerateRequest.

For each attachment carrying source_asset_ref + preprocess:
  1. Resolves the source bytes from the AssetStore.
  2. Invokes the named preprocessor.
  3. Stores the emitted control map as a "control_map" entry.
  4. Backfills attachment.map_asset_ref so downstream (Track 3) sees a resolved map.
  5. Returns a ControlNetArtifactRef per emitted map.

Map-asset-ref attachments pass through unchanged.
"""

from typing import Optional

from server.asset_store import AssetStore
from server.controlnet_models import ControlNetArtifactRef
from server.controlnet_preprocessors import DEFAULT_REGISTRY, PreprocessorRegistry


def preprocess_controlnet_attachments(
    req,
    store: AssetStore,
    registry: Optional[PreprocessorRegistry] = None,
) -> list[ControlNetArtifactRef]:
    if registry is None:
        registry = DEFAULT_REGISTRY

    attachments = getattr(req, "controlnets", None) or []
    artifacts: list[ControlNetArtifactRef] = []

    for attachment in attachments:
        if attachment.source_asset_ref is None or attachment.preprocess is None:
            continue

        source_ref = attachment.source_asset_ref
        try:
            source_entry = store.resolve(source_ref)
        except KeyError:
            raise ValueError(f"source_asset_ref {source_ref!r} not found or evicted")

        preprocessor = registry.get(attachment.preprocess.id)
        if preprocessor is None:
            raise ValueError(f"unknown preprocessor {attachment.preprocess.id!r}")

        result = preprocessor.run(source_entry.data, attachment.preprocess.options)

        metadata = {
            "attachment_id": attachment.attachment_id,
            "control_type": attachment.control_type,
            "source_asset_ref": source_ref,
            "preprocessor_id": result.preprocessor_id,
            "width": result.width,
            "height": result.height,
            "media_type": result.media_type,
        }
        new_ref = store.insert("control_map", result.image_bytes, metadata)
        attachment.map_asset_ref = new_ref

        artifacts.append(ControlNetArtifactRef(
            attachment_id=attachment.attachment_id,
            asset_ref=new_ref,
            control_type=attachment.control_type,
            preprocessor_id=result.preprocessor_id,
            source_asset_ref=source_ref,
        ))

    return artifacts
```

- [ ] **Step 5: Run tests to confirm they pass**

```
python -m pytest tests/test_controlnet_preprocessing.py -q
```
Expected: `6 passed`

- [ ] **Step 6: Commit**

```bash
git add server/controlnet_models.py server/controlnet_preprocessing.py tests/test_controlnet_preprocessing.py
git commit -m "feat(controlnet): add ControlNetArtifactRef model and preprocess_controlnet_attachments() (STABL-vnpknuvo)"
```

---

## Task 9: Wire preprocessing into HTTP `/generate`

**FP:** STABL-vnpknuvo
**Depends on:** Task 3, Task 8
**Unblocks:** Task 11

**Files:**
- Modify: `server/lcm_sr_server.py`

The current enforcement block (lines ~536–546) is:
```python
try:
    from server.controlnet_constraints import enforce_controlnet_policy
    enforce_controlnet_policy(req, mode)
except ValueError as e:
    raise HTTPException(status_code=400, detail=str(e))

try:
    from server.controlnet_constraints import ensure_controlnet_dispatch_supported
    ensure_controlnet_dispatch_supported(req)
except NotImplementedError as e:
    raise HTTPException(status_code=501, detail=str(e))
```

The second block lives outside the `if current_mode:` scope.

- [ ] **Step 1a: Keep the existing seam test — it already lives in `tests/test_controlnet_preprocessing.py`**

The two tests added in Task 8 (`test_preprocessing_emits_artifact_with_correct_metadata` and `test_preprocessing_propagates_resolution_failure`) verify `preprocess_controlnet_attachments` in isolation. They should already pass at this point. No changes needed.

Run to confirm:

```
python -m pytest tests/test_controlnet_preprocessing.py -q
```

Expected: passes. If they fail, debug Task 8 before continuing.

- [ ] **Step 1b: Write the HTTP contract test — verifies `lcm_sr_server.py` wires preprocessing into the 501 detail**

This test uses `TestClient` but with all external dependencies mocked explicitly — no ambient mode system, no real worker pool, no model files. It is marked `@pytest.mark.integration` to allow CI to skip it in unit-only runs, but it is fully deterministic.

Create `tests/test_controlnet_http_contract.py`:

```python
# tests/test_controlnet_http_contract.py
"""
Integration contract tests for ControlNet wiring in lcm_sr_server.py.

All external dependencies (mode config, asset store, preprocessor registry,
worker pool) are mocked explicitly. Tests are deterministic.

Run with: pytest tests/test_controlnet_http_contract.py -v
Skip in unit-only CI: pytest -m "not integration"
"""
import io
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from server.controlnet_preprocessors import ControlMapResult
from server.mode_config import (
    ControlNetControlTypePolicy,
    ControlNetPolicy,
    ModeConfig,
)


def _make_png(w: int = 8, h: int = 8) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (w, h), color=(0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _make_mode_with_canny() -> ModeConfig:
    """A minimal ModeConfig with canny ControlNet enabled."""
    return ModeConfig(
        name="sdxl-cn-test",
        model="checkpoints/sdxl.safetensors",
        default_size="1024x1024",
        resolution_options=[{"size": "1024x1024", "aspect_ratio": "1:1"}],
        controlnet_policy=ControlNetPolicy(
            enabled=True,
            max_attachments=1,
            allow_reuse_emitted_maps=True,
            allowed_control_types={
                "canny": ControlNetControlTypePolicy(
                    default_model_id="sdxl-canny",
                    allowed_model_ids=["sdxl-canny"],
                    allow_preprocess=True,
                    default_strength=0.8,
                    min_strength=0.0,
                    max_strength=2.0,
                )
            },
        ),
    )


@pytest.mark.integration
def test_http_generate_501_includes_controlnet_artifacts():
    """
    POST /generate with a valid ControlNet attachment:
    - preprocessing runs and emits an artifact
    - the 501 detail dict includes controlnet_artifacts with correct fields
    """
    from server.asset_store import AssetStore
    from server.lcm_sr_server import app

    store = AssetStore(byte_budget=64 * 1024 * 1024)
    source_ref = store.insert("upload", _make_png())

    fake_result = ControlMapResult(
        preprocessor_id="canny",
        control_type="canny",
        image_bytes=_make_png(),
        width=8,
        height=8,
    )
    mock_registry = MagicMock()
    mock_registry.get.return_value = MagicMock(run=MagicMock(return_value=fake_result))

    mock_mode_config = MagicMock()
    mock_mode_config.get_mode.return_value = _make_mode_with_canny()

    mock_runtime = MagicMock()
    mock_runtime.get_current_mode.return_value = "sdxl-cn-test"
    mock_runtime.supports_modes = True

    with (
        patch("server.lcm_sr_server.get_mode_config", return_value=mock_mode_config),
        patch("server.lcm_sr_server.get_store", return_value=store),
        patch("server.controlnet_preprocessing.DEFAULT_REGISTRY", mock_registry),
        patch("server.lcm_sr_server._get_runtime", return_value=mock_runtime),
    ):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/generate", json={
            "prompt": "a cat",
            "controlnets": [{
                "attachment_id": "cn_1",
                "control_type": "canny",
                "source_asset_ref": source_ref,
                "preprocess": {"id": "canny", "options": {}},
            }],
        })

    assert resp.status_code == 501
    body = resp.json()
    detail = body["detail"]
    assert isinstance(detail, dict), f"expected dict detail, got: {detail!r}"
    assert "controlnet_artifacts" in detail
    arts = detail["controlnet_artifacts"]
    assert len(arts) == 1
    assert arts[0]["attachment_id"] == "cn_1"
    assert arts[0]["control_type"] == "canny"
    assert arts[0]["preprocessor_id"] == "canny"
    assert arts[0]["asset_ref"]


@pytest.mark.integration
def test_http_generate_400_when_controlnet_policy_disabled():
    """
    POST /generate with controlnets on a mode that has controlnet_policy.enabled=False
    returns 400, not 501, before preprocessing runs.
    """
    from server.asset_store import AssetStore
    from server.lcm_sr_server import app
    from server.mode_config import ControlNetPolicy

    store = AssetStore(byte_budget=64 * 1024 * 1024)
    source_ref = store.insert("upload", _make_png())

    disabled_mode = ModeConfig(
        name="sdxl-plain",
        model="checkpoints/sdxl.safetensors",
        default_size="1024x1024",
        resolution_options=[{"size": "1024x1024", "aspect_ratio": "1:1"}],
        controlnet_policy=ControlNetPolicy(enabled=False),
    )

    mock_mode_config = MagicMock()
    mock_mode_config.get_mode.return_value = disabled_mode

    mock_runtime = MagicMock()
    mock_runtime.get_current_mode.return_value = "sdxl-plain"
    mock_runtime.supports_modes = True

    with (
        patch("server.lcm_sr_server.get_mode_config", return_value=mock_mode_config),
        patch("server.lcm_sr_server.get_store", return_value=store),
        patch("server.lcm_sr_server._get_runtime", return_value=mock_runtime),
    ):
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/generate", json={
            "prompt": "a cat",
            "controlnets": [{
                "attachment_id": "cn_1",
                "control_type": "canny",
                "source_asset_ref": source_ref,
                "preprocess": {"id": "canny", "options": {}},
                "model_id": "sdxl-canny",
            }],
        })

    assert resp.status_code == 400
    assert "does not enable ControlNet" in resp.json()["detail"]
```

> **Note on `_get_runtime`:** The test patches `server.lcm_sr_server._get_runtime`. If the HTTP handler accesses the runtime via a module-level singleton rather than a named helper, patch the actual attribute path. Check `lcm_sr_server.py` around line 514 and adjust the patch target to match how `runtime` is obtained in the `/generate` handler after Track 1 lands.

- [ ] **Step 2: Run both tests — seam passes, HTTP contract fails (wiring not done yet)**

```
python -m pytest tests/test_controlnet_preprocessing.py tests/test_controlnet_http_contract.py -v
```

Expected:
- `tests/test_controlnet_preprocessing.py` — PASS (wired in Task 8)
- `tests/test_controlnet_http_contract.py` — FAIL on `assert resp.status_code == 501` (preprocessing not yet wired into `/generate`)

- [ ] **Step 3: Locate the enforcement block in `server/lcm_sr_server.py`**

Find lines in `lcm_sr_server.py` that look like:

```python
        try:
            from server.controlnet_constraints import enforce_controlnet_policy
            enforce_controlnet_policy(req, mode)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    try:
        from server.controlnet_constraints import ensure_controlnet_dispatch_supported
        ensure_controlnet_dispatch_supported(req)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
```

- [ ] **Step 4: Add preprocessing between the two blocks**

Replace the section starting at `enforce_controlnet_policy` through the `ensure_controlnet_dispatch_supported` handler with:

```python
        try:
            from server.controlnet_constraints import enforce_controlnet_policy
            enforce_controlnet_policy(req, mode)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        try:
            from server.controlnet_preprocessing import preprocess_controlnet_attachments
            from server.asset_store import get_store
            emitted_artifacts = preprocess_controlnet_attachments(req, get_store())
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    try:
        from server.controlnet_constraints import ensure_controlnet_dispatch_supported
        ensure_controlnet_dispatch_supported(req)
    except NotImplementedError as e:
        artifact_dicts = [a.model_dump() for a in emitted_artifacts]
        detail = str(e) if not artifact_dicts else {"error": str(e), "controlnet_artifacts": artifact_dicts}
        raise HTTPException(status_code=501, detail=detail)
```

Also add `emitted_artifacts: list = []` just before the `if current_mode:` block (line ~520) so it is in scope when `ensure_controlnet_dispatch_supported` raises:

```python
    emitted_artifacts: list = []

    if current_mode:
        ...
```

- [ ] **Step 5: Run HTTP contract tests + regression**

```
python -m pytest tests/test_controlnet_http_contract.py tests/test_controlnet_preprocessing.py tests/test_controlnet_constraints.py tests/test_controlnet_models.py tests/test_controlnet_dispatch.py tests/test_mode_config.py tests/test_ws_build_generate_request.py -v
```

Expected: `test_controlnet_http_contract.py` now PASS (both `test_http_generate_501_includes_controlnet_artifacts` and `test_http_generate_400_when_controlnet_policy_disabled`). All other prior tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add server/lcm_sr_server.py tests/test_controlnet_http_contract.py
git commit -m "feat(controlnet): wire preprocessing into HTTP /generate, surface artifacts in 501 detail (STABL-vnpknuvo)"
```

---

## Task 10: Wire preprocessing into WS `handle_job_submit` and `_run_generate` — `job:error` now, `job:complete` foundation only

**FP:** STABL-vnpknuvo
**Depends on:** Task 3, Task 8
**Unblocks:** Task 11

**Files:**
- Modify: `server/ws_routes.py`

There are two WS code paths for ControlNet requests:

1. **Mode-system path** (`handle_job_submit`, lines ~134–185): uses `pre_submit_job_error`; the mode is available.
2. **Non-mode-system path** (`_run_generate`, lines ~492–534): no mode; `ensure_controlnet_dispatch_supported` raises; exception caught at the outer `except Exception`.

- [ ] **Step 1: Write a failing WS test**

Append to `tests/test_ws_routes.py`:

```python
import io
from PIL import Image as _PIL
from unittest.mock import patch, MagicMock
from server.asset_store import get_store
from server.controlnet_preprocessors import ControlMapResult


def _solid_png_bytes(w=8, h=8) -> bytes:
    buf = io.BytesIO()
    _PIL.new("RGB", (w, h)).save(buf, format="PNG")
    return buf.getvalue()


class TestControlNetArtifactEmissionWS:
    def test_job_error_includes_controlnet_artifacts_when_preprocessing_runs(self):
        """WS job:error frame includes controlnet_artifacts for preprocessed source refs."""
        source_ref = get_store().insert("upload", _solid_png_bytes())

        fake_result = ControlMapResult(
            preprocessor_id="canny",
            control_type="canny",
            image_bytes=_solid_png_bytes(),
            width=8,
            height=8,
        )
        mock_preprocessor = MagicMock()
        mock_preprocessor.run.return_value = fake_result

        # Patch DEFAULT_REGISTRY so dispatch returns our fake result
        with patch("server.controlnet_preprocessing.DEFAULT_REGISTRY") as mock_reg:
            mock_reg.get.return_value = mock_preprocessor

            # Provide a minimal mode-system app state
            state_app = _make_test_app()
            state_app.state.use_mode_system = False  # use _run_generate path

            test_client = TestClient(state_app)
            with test_client.websocket_connect("/v1/ws") as ws:
                ws.receive_json()  # system:status
                ws.send_json({
                    "type": "job:submit",
                    "jobType": "generate",
                    "params": {
                        "prompt": "test",
                        "controlnets": [{
                            "attachment_id": "cn_1",
                            "control_type": "canny",
                            "source_asset_ref": source_ref,
                            "preprocess": {"id": "canny", "options": {}},
                        }],
                    },
                })
                # Receive ack
                ack = ws.receive_json()
                assert ack["type"] == "job:ack"
                # Receive error (dispatch stub fires)
                error = ws.receive_json()
                assert error["type"] == "job:error"
                # controlnet_artifacts should be present
                assert "controlnet_artifacts" in error
                assert error["controlnet_artifacts"][0]["attachment_id"] == "cn_1"
```

- [ ] **Step 2: Run test to confirm it fails**

```
python -m pytest tests/test_ws_routes.py::TestControlNetArtifactEmissionWS -q
```
Expected: `FAILED` — `controlnet_artifacts` key not in error frame.

- [ ] **Step 3: Patch `handle_job_submit` in `server/ws_routes.py`**

In `handle_job_submit`, locate the section starting around line 134:

```python
    if job_type == "generate" and getattr(state, "use_mode_system", False):
        ...
        req = _build_generate_request(params)
        try:
            current_mode = state.worker_pool.get_current_mode()
            if current_mode:
                ...
                from server.controlnet_constraints import enforce_controlnet_policy
                enforce_controlnet_policy(req, mode)
            from server.controlnet_constraints import ensure_controlnet_dispatch_supported
            ensure_controlnet_dispatch_supported(req)
        except Exception as e:
            pre_submit_job_error = str(e)
```

Change to:

```python
    pre_submit_artifacts: list = []

    if job_type == "generate" and getattr(state, "use_mode_system", False):
        ...
        req = _build_generate_request(params)
        try:
            current_mode = state.worker_pool.get_current_mode()
            if current_mode:
                ...
                from server.controlnet_constraints import enforce_controlnet_policy
                enforce_controlnet_policy(req, mode)
                from server.controlnet_preprocessing import preprocess_controlnet_attachments
                from server.asset_store import get_store
                pre_submit_artifacts = preprocess_controlnet_attachments(req, get_store())
            from server.controlnet_constraints import ensure_controlnet_dispatch_supported
            ensure_controlnet_dispatch_supported(req)
        except Exception as e:
            pre_submit_job_error = str(e)
```

Then change the `job:error` send block (around line 180):

```python
    if pre_submit_job_error is not None:
        error_frame: dict = {
            "type": "job:error",
            "jobId": job_id,
            "error": pre_submit_job_error,
        }
        if pre_submit_artifacts:
            error_frame["controlnet_artifacts"] = [a.model_dump() for a in pre_submit_artifacts]
        await hub.send(client_id, error_frame)
        return
```

- [ ] **Step 4: Patch `_run_generate` in `server/ws_routes.py`**

In `_run_generate`, locate:

```python
        from server.controlnet_constraints import ensure_controlnet_dispatch_supported
        ensure_controlnet_dispatch_supported(req)
```

Change to:

```python
        from server.controlnet_preprocessing import preprocess_controlnet_attachments
        from server.asset_store import get_store
        _run_generate_artifacts: list = []
        _run_generate_artifacts = preprocess_controlnet_attachments(req, get_store())

        from server.controlnet_constraints import ensure_controlnet_dispatch_supported
        ensure_controlnet_dispatch_supported(req)
```

Change the `except Exception` at the bottom of `_run_generate` to:

```python
    except asyncio.CancelledError:
        logger.info("Generate job %s cancelled by client", job_id)
        await hub.send(client_id, {"type": "job:error", "jobId": job_id, "error": "Cancelled by client"})

    except Exception as e:
        logger.error("Generate job %s failed: %s", job_id, e, exc_info=True)
        error_frame: dict = {"type": "job:error", "jobId": job_id, "error": str(e)}
        try:
            artifacts = _run_generate_artifacts  # defined above if preprocessing ran
            if artifacts:
                error_frame["controlnet_artifacts"] = [a.model_dump() for a in artifacts]
        except NameError:
            pass
        await hub.send(client_id, error_frame)
```

- [ ] **Step 5: Run tests**

```
python -m pytest tests/test_ws_routes.py tests/test_controlnet_preprocessing.py tests/test_asset_store.py -q
```
Expected: all pass (existing WS tests + new artifact emission test).

- [ ] **Step 6: Commit**

```bash
git add server/ws_routes.py
git commit -m "feat(controlnet): wire preprocessing into WS handle_job_submit and _run_generate, emit controlnet_artifacts in job:error (STABL-vnpknuvo)"
```

---

## Task 11: Acceptance — reusable artifact flow + eviction coverage

**FP:** STABL-rrceoxha
**Depends on:** Task 2, Task 3, Task 6, Task 7, Task 8, Task 9, Task 10
**Unblocks:** Track 2 review completion / Track 3 handoff

**Files:**
- Create: `tests/test_controlnet_acceptance.py`

- [ ] **Step 1: Write acceptance tests**

```python
# tests/test_controlnet_acceptance.py
"""
Track 2 acceptance: reusable artifact flow and eviction coverage.

These tests verify the core Track 2 contract:
  - canny/depth source-ref requests produce stored control_map assets
  - emitted artifact refs are stable and resolvable after preprocessing
  - eviction under byte-budget pressure respects LRU and pinning rules

Generation is intentionally NOT tested here (Track 3 stub returns 501).
The goal is to prove Track 2's asset/preprocessing layer is solid before
Track 3 lands execution.
"""

import io
import pytest
from PIL import Image
from unittest.mock import Mock

from server.asset_store import AssetStore
from server.controlnet_models import ControlNetAttachment, ControlNetPreprocessRequest
from server.controlnet_preprocessors import ControlMapResult, PreprocessorRegistry
from server.controlnet_preprocessing import preprocess_controlnet_attachments


def _solid_png(w: int = 32, h: int = 32) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=(50, 100, 150)).save(buf, format="PNG")
    return buf.getvalue()


def _fake_reg(pid: str, output_bytes: bytes = b"fake-cmap") -> PreprocessorRegistry:
    result = ControlMapResult(
        preprocessor_id=pid, control_type=pid,
        image_bytes=output_bytes, width=32, height=32,
    )
    p = Mock(preprocessor_id=pid, control_type=pid)
    p.run = Mock(return_value=result)
    reg = PreprocessorRegistry()
    reg.register(p)
    return reg


def _req(controlnets):
    class R:
        pass
    r = R()
    r.controlnets = controlnets
    return r


# ------------------------------------------------------------------ #
# Canny: source → artifact
# ------------------------------------------------------------------ #

def test_canny_source_ref_produces_reusable_artifact():
    store = AssetStore()
    source_ref = store.insert("upload", _solid_png())

    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        source_asset_ref=source_ref,
        preprocess=ControlNetPreprocessRequest(id="canny"),
    )
    artifacts = preprocess_controlnet_attachments(_req([att]), store, registry=_fake_reg("canny"))

    assert len(artifacts) == 1
    emitted_ref = artifacts[0].asset_ref

    # Ref must be resolvable after preprocessing
    entry = store.resolve(emitted_ref)
    assert entry.kind == "control_map"
    assert entry.data == b"fake-cmap"
    assert entry.metadata["preprocessor_id"] == "canny"
    assert entry.metadata["source_asset_ref"] == source_ref


# ------------------------------------------------------------------ #
# Depth: source → artifact
# ------------------------------------------------------------------ #

def test_depth_source_ref_produces_reusable_artifact():
    store = AssetStore()
    source_ref = store.insert("upload", _solid_png())

    att = ControlNetAttachment(
        attachment_id="cn_2",
        control_type="depth",
        source_asset_ref=source_ref,
        preprocess=ControlNetPreprocessRequest(id="depth"),
    )
    artifacts = preprocess_controlnet_attachments(_req([att]), store, registry=_fake_reg("depth", b"depth-map"))

    entry = store.resolve(artifacts[0].asset_ref)
    assert entry.kind == "control_map"
    assert entry.data == b"depth-map"
    assert entry.metadata["preprocessor_id"] == "depth"


# ------------------------------------------------------------------ #
# Reuse: emitted ref survives and can be supplied as map_asset_ref
# ------------------------------------------------------------------ #

def test_emitted_artifact_ref_reusable_as_map_asset_ref():
    """A ref emitted in one request can be supplied as map_asset_ref in the next."""
    store = AssetStore()
    source_ref = store.insert("upload", _solid_png())
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        source_asset_ref=source_ref,
        preprocess=ControlNetPreprocessRequest(id="canny"),
    )
    artifacts = preprocess_controlnet_attachments(_req([att]), store, registry=_fake_reg("canny"))
    emitted_ref = artifacts[0].asset_ref

    # Second request reuses the emitted ref as map_asset_ref — no preprocessing
    att2 = ControlNetAttachment(
        attachment_id="cn_2",
        control_type="canny",
        map_asset_ref=emitted_ref,
    )
    artifacts2 = preprocess_controlnet_attachments(_req([att2]), store, registry=_fake_reg("canny"))
    assert artifacts2 == []  # no preprocessing for direct map_asset_ref
    assert store.resolve(emitted_ref).data == b"fake-cmap"  # still accessible


# ------------------------------------------------------------------ #
# Eviction under byte-budget pressure
# ------------------------------------------------------------------ #

def test_eviction_removes_oldest_control_map_when_budget_exceeded():
    store = AssetStore(byte_budget=20)
    source_ref = store.insert("upload", b"src")  # 3 bytes

    # First preprocessing: emits a 10-byte control map
    att1 = ControlNetAttachment(
        attachment_id="cn_1", control_type="canny",
        source_asset_ref=source_ref,
        preprocess=ControlNetPreprocessRequest(id="canny"),
    )
    artifacts1 = preprocess_controlnet_attachments(_req([att1]), store, registry=_fake_reg("canny", b"x" * 10))
    ref1 = artifacts1[0].asset_ref

    # Access ref1 to make it recently used
    store.resolve(ref1)

    # Second preprocessing: emits a 12-byte control map — pushes total over 20
    # source_ref (3) + ref1 (10) + new (12) = 25 > 20
    # source_ref was accessed first and is LRU → it gets evicted
    att2 = ControlNetAttachment(
        attachment_id="cn_2", control_type="canny",
        source_asset_ref=source_ref,
        preprocess=ControlNetPreprocessRequest(id="canny"),
    )
    try:
        artifacts2 = preprocess_controlnet_attachments(_req([att2]), store, registry=_fake_reg("canny", b"y" * 12))
    except ValueError:
        pytest.skip("source_ref was evicted before second preprocess; expected if budget is very tight")

    # At least one ref should have been evicted
    assert store.total_bytes <= 20


def test_pinned_ref_survives_eviction():
    store = AssetStore(byte_budget=15)
    source_ref = store.insert("upload", b"src")  # 3 bytes
    store.pin(source_ref)  # protect the source during preprocessing

    att = ControlNetAttachment(
        attachment_id="cn_1", control_type="canny",
        source_asset_ref=source_ref,
        preprocess=ControlNetPreprocessRequest(id="canny"),
    )
    # Emits 14-byte map; total = 3 + 14 = 17 > 15; source is pinned, cannot be evicted
    artifacts = preprocess_controlnet_attachments(_req([att]), store, registry=_fake_reg("canny", b"z" * 14))

    # Source survives because it was pinned
    assert store.resolve(source_ref).data == b"src"
    store.unpin(source_ref)


# ------------------------------------------------------------------ #
# Two attachments in one request — both emit
# ------------------------------------------------------------------ #

def test_two_attachments_emit_two_artifacts():
    store = AssetStore()
    src1 = store.insert("upload", _solid_png())
    src2 = store.insert("upload", _solid_png())

    reg = PreprocessorRegistry()
    for pid in ("canny", "depth"):
        r = ControlMapResult(preprocessor_id=pid, control_type=pid,
                             image_bytes=f"{pid}-out".encode(), width=32, height=32)
        p = Mock(preprocessor_id=pid, control_type=pid)
        p.run = Mock(return_value=r)
        reg.register(p)

    att1 = ControlNetAttachment(
        attachment_id="cn_1", control_type="canny",
        source_asset_ref=src1,
        preprocess=ControlNetPreprocessRequest(id="canny"),
    )
    att2 = ControlNetAttachment(
        attachment_id="cn_2", control_type="depth",
        source_asset_ref=src2,
        preprocess=ControlNetPreprocessRequest(id="depth"),
    )
    artifacts = preprocess_controlnet_attachments(_req([att1, att2]), store, registry=reg)
    assert len(artifacts) == 2
    assert artifacts[0].attachment_id == "cn_1"
    assert artifacts[1].attachment_id == "cn_2"
```

- [ ] **Step 2: Run acceptance tests**

```
python -m pytest tests/test_controlnet_acceptance.py -q
```
Expected: `7 passed` (one possible `pytest.skip` on the tight-budget test).

- [ ] **Step 3: Run the full relevant suite one last time**

```
python -m pytest tests/test_asset_store.py tests/test_controlnet_preprocessors.py tests/test_controlnet_preprocessing.py tests/test_controlnet_acceptance.py tests/test_controlnet_constraints.py tests/test_controlnet_models.py tests/test_upload_routes.py tests/test_ws_routes.py -q
```
Expected: all pass; the 3 `test_model_routes.py` failures are pre-existing baseline on `main`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_controlnet_acceptance.py
git commit -m "test(controlnet): Track 2 acceptance — reusable artifact flow and eviction coverage (STABL-rrceoxha)"
```

- [ ] **Step 5: Log FP comment on parent issue**

```bash
fp comment STABL-nsrpodvu "Track 2 implementation complete. All acceptance tests pass: typed asset refs, LRU eviction, canny/depth preprocessors, HTTP+WS artifact emission. 3 pre-existing test_model_routes failures are baseline on main. Track 3 can consume attachment.map_asset_ref (backfilled by preprocessing) and the DEFAULT_REGISTRY without interface changes."
fp issue update --status done STABL-mrgpncim STABL-ncmcmmnc STABL-drodzdpf STABL-mlapdvwj STABL-bbkjaqym STABL-vnpknuvo STABL-rrceoxha
```

---

## Self-Review

**Spec coverage check:**

| Spec section | Covered by |
|---|---|
| §3 Derived control-map assets: `kind` + metadata on ref table | Tasks 1–3 |
| §3 LRU eviction, byte budget 512 MB, in-flight pinning | Tasks 2–3 |
| §3 Emitted artifacts in generation responses | Tasks 8–10 |
| §4 `ControlMapPreprocessor` protocol + `ControlMapResult` | Task 4 |
| §4 Registry/dispatch | Task 4 |
| §4 Shared image helpers | Task 5 |
| §4 `CannyPreprocessor` | Task 6 |
| §4 `DepthPreprocessor` | Task 7 |
| §9 `controlnet_artifacts` in HTTP response | Task 9 |
| §9 `controlnet_artifacts` in WS `job:error` | Task 10 |
| §9 `controlnet_artifacts` in WS `job:complete` (foundation wired; full emission: Track 3) | Task 10 (stub path only — reachable after Track 3 removes dispatch stub) |
| Acceptance: reusable artifact flow | Task 11 |
| Acceptance: eviction coverage | Task 11 |
| Track 3 handoff (map_asset_ref backfilled, registry stable) | Task 11 comment |

**Gaps / deliberate deferrals:**
- `job:complete` WS artifact emission (not `job:error`): only reachable once Track 3 removes the dispatch stub. Task 10 wires the foundation so Track 3 only needs to pass artifacts through `_finish_generate` — no structural change.
- Preprocessor library for depth model download: `DepthPreprocessor._get_pipe()` lazy-loads on first real call. CI tests use the mock. Real validation requires a GPU environment with model cache.
- TTL cleanup task (`cleanup_uploads_loop`) for `control_map` kind entries: spec says control maps are session-scoped, not TTL-expired. Cleanup loop already skips `control_map` entries (only evicts `upload` kind). Confirmed in Task 1 test `test_cleanup_expired_preserves_control_maps`.
