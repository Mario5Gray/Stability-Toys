# ControlNet Metadata Transfer Implementation Plan

> **For agentic workers:** This plan is executed **inline** via
> superpowers:executing-plans. AGENTS.md forbids sub-agent-driven development
> in this repo — do NOT dispatch subagents per task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Control-map tools stamp their own provenance into emitted PNGs, and
the render worker transfers that provenance (plus the ControlNet generation
params actually used) into a `controlnet` chunk on the generation PNG.

**Architecture:** A shared `cn_metadata` helper writes a `controlnet_map` PNG
tEXt chunk from each of the three map tools. At render time the worker reads
that chunk off the raw map bytes it already holds
(`ControlNetBinding.control_image_bytes`) and writes a combined `controlnet`
chunk alongside the existing `lcm` chunk. No new plumbing through the job or
the frozen `ControlNetBinding`.

**Tech Stack:** Python 3.10+, Pillow (`PngImagePlugin` tEXt chunks), pytest.
Spec: `docs/superpowers/specs/2026-07-04-controlnet-metadata-transfer-design.md`.

## Global Constraints

- Control-map chunk key: `controlnet_map` (single PNG tEXt chunk). Verbatim.
- Generation chunk key: `controlnet` (parallel to existing `lcm`). Verbatim.
- Payload schema version: `1` (integer).
- `scripts/` ships a **flat module list** (`py-modules`), not a package; any
  new shared module must be added to `py-modules` or installed entry points
  break with `ModuleNotFoundError`.
- Scripts import the helper as a top-level module: `import cn_metadata`.
- Do NOT modify the frozen `ControlNetBinding` dataclass or the
  `X-ControlNet-Artifacts` / `job:complete.controlnet_artifacts` response.
- Do NOT modify `server/controlnet_preprocessors.py` (server preprocess path
  intentionally records `source: null`).
- `depth_map.py --colorize` `_color` preview is NOT stamped.
- Tests must be fully offline — no model downloads, no network, no GPU.
- `device` in metadata = the requested `--device` CLI arg (pose accepts but
  does not apply it; still recorded).

---

## File Structure

- Create `scripts/cn_metadata.py` — payload builder + PNG writer (Task 1).
- Modify `scripts/pyproject.toml` — add `cn_metadata` to `py-modules` (Task 1).
- Create `tests/test_cn_metadata.py` — helper unit tests + packaging (Task 1).
- Modify `scripts/canny_map.py` — stamp via helper (Task 2).
- Modify `tests/test_canny_map.py` — assert chunk + packaging (Task 2).
- Modify `scripts/depth_map.py`, `scripts/pose_map.py` — stamp via helper (Task 3).
- Create `tests/test_depth_pose_metadata.py` — in-process monkeypatch tests (Task 3).
- Create `server/controlnet_metadata.py` — `read_control_map_metadata` (Task 4).
- Create `tests/test_controlnet_metadata.py` — reader tests (Task 4).
- Modify `backends/cuda_worker.py` — `_controlnet_metadata` + both `run_job`
  chunk sites (Task 5).
- Create `tests/test_worker_controlnet_metadata.py` — worker assembly test (Task 5).

---

### Task 1: Shared `cn_metadata` helper + packaging

**Files:**
- Create: `scripts/cn_metadata.py`
- Modify: `scripts/pyproject.toml:42`
- Test: `tests/test_cn_metadata.py`

**Interfaces:**
- Produces:
  - `build_map_metadata(*, tool: str, control_type: str, source_size: tuple[int, int], params: dict[str, Any]) -> dict[str, Any]`
  - `save_with_metadata(image: PIL.Image.Image, destination: pathlib.Path, payload: dict[str, Any]) -> None`
  - Module constants `CHUNK_KEY = "controlnet_map"`, `SCHEMA_VERSION = 1`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cn_metadata.py`:

```python
import json
import sys
import tomllib
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cn_metadata import build_map_metadata, save_with_metadata, CHUNK_KEY, SCHEMA_VERSION


def test_build_map_metadata_common_and_params():
    payload = build_map_metadata(
        tool="canny_map",
        control_type="canny",
        source_size=(320, 240),
        params={"low_threshold": 50, "high_threshold": 150},
    )
    assert payload["tool"] == "canny_map"
    assert payload["version"] == SCHEMA_VERSION
    assert payload["control_type"] == "canny"
    assert payload["source_width"] == 320
    assert payload["source_height"] == 240
    assert isinstance(payload["created_at"], str) and payload["created_at"]
    assert payload["low_threshold"] == 50
    assert payload["high_threshold"] == 150


def test_save_with_metadata_roundtrip(tmp_path):
    img = Image.new("L", (16, 12), 200)
    dest = tmp_path / "map.png"
    payload = build_map_metadata(
        tool="depth_map",
        control_type="depth",
        source_size=(16, 12),
        params={"model": "depth-anything"},
    )
    save_with_metadata(img, dest, payload)

    out = Image.open(dest)
    assert out.format == "PNG"
    assert json.loads(out.text[CHUNK_KEY]) == payload


def test_pyproject_lists_cn_metadata_module():
    data = tomllib.loads((ROOT / "scripts" / "pyproject.toml").read_text())
    assert "cn_metadata" in data["tool"]["setuptools"]["py-modules"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cn_metadata.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cn_metadata'`.

- [ ] **Step 3: Create the helper**

Create `scripts/cn_metadata.py`:

```python
"""Embed ControlNet provenance metadata into emitted control-map PNGs.

Shared by the standalone map tools (canny_map, depth_map, pose_map). Metadata
is written as a single PNG tEXt chunk keyed "controlnet_map" holding a JSON
payload describing the tool, its parameters, and the source dimensions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, PngImagePlugin

CHUNK_KEY = "controlnet_map"
SCHEMA_VERSION = 1


def build_map_metadata(
    *,
    tool: str,
    control_type: str,
    source_size: tuple[int, int],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the controlnet_map payload: common fields plus tool params."""
    width, height = source_size
    payload: dict[str, Any] = {
        "tool": tool,
        "version": SCHEMA_VERSION,
        "control_type": control_type,
        "source_width": int(width),
        "source_height": int(height),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(params)
    return payload


def save_with_metadata(
    image: Image.Image, destination: Path, payload: dict[str, Any]
) -> None:
    """Save a PIL image to PNG with the controlnet_map metadata chunk."""
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text(CHUNK_KEY, json.dumps(payload))
    image.save(destination, format="PNG", pnginfo=pnginfo)
```

- [ ] **Step 4: Add the module to packaging**

In `scripts/pyproject.toml`, change line 42 from:

```toml
py-modules = ["depth_map", "pose_map", "canny_map"]
```

to:

```toml
py-modules = ["depth_map", "pose_map", "canny_map", "cn_metadata"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_cn_metadata.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/cn_metadata.py scripts/pyproject.toml tests/test_cn_metadata.py
git commit -m "feat(controlnet): add cn_metadata helper for control-map provenance"
```

---

### Task 2: canny_map stamps metadata

**Files:**
- Modify: `scripts/canny_map.py:105`
- Test: `tests/test_canny_map.py`

**Interfaces:**
- Consumes: `cn_metadata.build_map_metadata`, `cn_metadata.save_with_metadata`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_canny_map.py` (add `import json` near the top imports if
not present):

```python
def test_canny_map_embeds_metadata(tmp_path: Path):
    source = tmp_path / "source.png"
    dest = tmp_path / "edges.png"
    _write_fixture(source, size=(96, 64))

    result = _run_script(source, dest, "--low-threshold", "50", "--high-threshold", "150")

    assert result.returncode == 0, result.stderr
    out = Image.open(dest)
    meta = json.loads(out.text["controlnet_map"])
    assert meta["tool"] == "canny_map"
    assert meta["control_type"] == "canny"
    assert meta["low_threshold"] == 50
    assert meta["high_threshold"] == 150
    assert meta["blur"] == 0
    assert meta["invert"] is False
    assert (meta["source_width"], meta["source_height"]) == (96, 64)
```

Also add a packaging assertion to the existing
`test_pyproject_exposes_canny_install_surface`:

```python
    assert "cn_metadata" in data["tool"]["setuptools"]["py-modules"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_canny_map.py::test_canny_map_embeds_metadata -v`
Expected: FAIL — `KeyError: 'controlnet_map'` (chunk not written yet).

- [ ] **Step 3: Wire the script**

In `scripts/canny_map.py`, add to the top imports:

```python
from cn_metadata import build_map_metadata, save_with_metadata
```

Replace `result.save(args.destination)` (line 105) with:

```python
    payload = build_map_metadata(
        tool="canny_map",
        control_type="canny",
        source_size=img.size,
        params={
            "low_threshold": args.low_threshold,
            "high_threshold": args.high_threshold,
            "blur": args.blur,
            "invert": args.invert,
            "max_res": args.max_res,
        },
    )
    save_with_metadata(result, args.destination, payload)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_canny_map.py -v`
Expected: PASS (all existing tests plus the new one; `out.mode == "L"` still holds).

- [ ] **Step 5: Commit**

```bash
git add scripts/canny_map.py tests/test_canny_map.py
git commit -m "feat(controlnet): stamp controlnet_map metadata from canny_map"
```

---

### Task 3: depth_map and pose_map stamp metadata

**Files:**
- Modify: `scripts/depth_map.py:137`
- Modify: `scripts/pose_map.py:131`
- Test: `tests/test_depth_pose_metadata.py`

**Interfaces:**
- Consumes: `cn_metadata.build_map_metadata`, `cn_metadata.save_with_metadata`.
- Test seams: monkeypatch `depth_map.depth_anything` and `pose_map.dwpose`
  (the default-model functions) to return a dummy `PIL.Image` — no models run.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_depth_pose_metadata.py`:

```python
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import depth_map
import pose_map


def _fixture(path: Path, size=(32, 24)):
    Image.new("RGB", size, "gray").save(path)


def test_depth_map_embeds_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(
        depth_map, "depth_anything",
        lambda img, size, device: Image.new("L", img.size, 128),
    )
    src = tmp_path / "src.png"
    dest = tmp_path / "depth.png"
    _fixture(src)
    monkeypatch.setattr(
        sys, "argv",
        ["depth_map", str(src), str(dest), "--model", "depth-anything", "--size", "small"],
    )
    depth_map.main()

    meta = json.loads(Image.open(dest).text["controlnet_map"])
    assert meta["tool"] == "depth_map"
    assert meta["control_type"] == "depth"
    assert meta["model"] == "depth-anything"
    assert meta["size"] == "small"
    assert meta["device"] == "cpu"
    assert (meta["source_width"], meta["source_height"]) == (32, 24)


def test_pose_map_embeds_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pose_map, "dwpose",
        lambda img: Image.new("RGB", img.size, "black"),
    )
    src = tmp_path / "src.png"
    dest = tmp_path / "pose.png"
    _fixture(src)
    monkeypatch.setattr(sys, "argv", ["pose_map", str(src), str(dest)])
    pose_map.main()

    meta = json.loads(Image.open(dest).text["controlnet_map"])
    assert meta["tool"] == "pose_map"
    assert meta["control_type"] == "pose"
    assert meta["model"] == "dwpose"
    assert meta["parts"] == ["body", "face", "hands"]
    assert (meta["source_width"], meta["source_height"]) == (32, 24)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_depth_pose_metadata.py -v`
Expected: FAIL — `KeyError: 'controlnet_map'`.

- [ ] **Step 3: Wire depth_map**

In `scripts/depth_map.py`, add to the imports:

```python
from cn_metadata import build_map_metadata, save_with_metadata
```

Replace `depth.save(args.destination)` (line 137) with:

```python
    payload = build_map_metadata(
        tool="depth_map",
        control_type="depth",
        source_size=img.size,
        params={
            "model": args.model,
            "size": args.size,
            "device": args.device,
            "invert": args.invert,
            "max_res": args.max_res,
        },
    )
    save_with_metadata(depth, args.destination, payload)
```

Leave the `--colorize` `_color` save path (the `colorize(depth).save(...)`
block) unchanged — it is a visualization, not a control map.

- [ ] **Step 4: Wire pose_map**

In `scripts/pose_map.py`, add to the imports:

```python
from cn_metadata import build_map_metadata, save_with_metadata
```

Replace `result.save(args.destination)` (line 131) with:

```python
    payload = build_map_metadata(
        tool="pose_map",
        control_type="pose",
        source_size=img.size,
        params={
            "model": args.model,
            "parts": sorted(parts),
            "device": args.device,
            "max_res": args.max_res,
        },
    )
    save_with_metadata(result, args.destination, payload)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_depth_pose_metadata.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/depth_map.py scripts/pose_map.py tests/test_depth_pose_metadata.py
git commit -m "feat(controlnet): stamp controlnet_map metadata from depth_map and pose_map"
```

---

### Task 4: Reader `read_control_map_metadata`

**Files:**
- Create: `server/controlnet_metadata.py`
- Test: `tests/test_controlnet_metadata.py`

**Interfaces:**
- Produces: `read_control_map_metadata(png_bytes: bytes) -> dict[str, Any] | None`.
  Returns the parsed `controlnet_map` payload, or `None` for absent chunk,
  malformed JSON, non-dict JSON, or undecodable bytes. Never raises.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_controlnet_metadata.py`:

```python
import io
import json
import sys
from pathlib import Path

from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parents[1]

from server.controlnet_metadata import read_control_map_metadata


def _png_with_chunk(value: str) -> bytes:
    info = PngImagePlugin.PngInfo()
    info.add_text("controlnet_map", value)
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def _png_no_chunk() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_valid_chunk_returns_dict():
    payload = {"tool": "canny_map", "control_type": "canny", "version": 1}
    out = read_control_map_metadata(_png_with_chunk(json.dumps(payload)))
    assert out == payload


def test_absent_chunk_returns_none():
    assert read_control_map_metadata(_png_no_chunk()) is None


def test_malformed_json_returns_none():
    assert read_control_map_metadata(_png_with_chunk("{not valid json")) is None


def test_non_dict_json_returns_none():
    assert read_control_map_metadata(_png_with_chunk("[1, 2, 3]")) is None


def test_non_png_bytes_returns_none():
    assert read_control_map_metadata(b"this is not a png") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_controlnet_metadata.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server.controlnet_metadata'`.

- [ ] **Step 3: Create the reader**

Create `server/controlnet_metadata.py`:

```python
"""Read ControlNet provenance metadata off a control-map PNG.

Consumes the "controlnet_map" tEXt chunk written by the standalone map tools
(see scripts/cn_metadata.py). Tolerant of missing/malformed data: any failure
returns None so the render path never raises on provenance.
"""

from __future__ import annotations

import io
import json
from typing import Any

from PIL import Image

CHUNK_KEY = "controlnet_map"


def read_control_map_metadata(png_bytes: bytes) -> dict[str, Any] | None:
    """Return the parsed controlnet_map payload, or None if absent/unreadable."""
    try:
        with Image.open(io.BytesIO(png_bytes)) as img:
            raw = getattr(img, "text", {}).get(CHUNK_KEY)
    except Exception:
        return None
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_controlnet_metadata.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add server/controlnet_metadata.py tests/test_controlnet_metadata.py
git commit -m "feat(controlnet): add read_control_map_metadata reader"
```

---

### Task 5: Worker writes the `controlnet` chunk

**Files:**
- Modify: `backends/cuda_worker.py` (`CudaWorkerBase`; both `run_job` chunk sites at `:566` and `:898`)
- Test: `tests/test_worker_controlnet_metadata.py`

**Interfaces:**
- Consumes: `server.controlnet_metadata.read_control_map_metadata`;
  `ControlNetBinding` fields (`attachment_id`, `control_type`, `model_id`,
  `strength`, `start_percent`, `end_percent`, `control_image_bytes`).
- Produces: `CudaWorkerBase._controlnet_metadata(bindings: list) -> list[dict]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_worker_controlnet_metadata.py`:

```python
import io
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cn_metadata import build_map_metadata, save_with_metadata
from server.controlnet_execution import ControlNetBinding
from backends.cuda_worker import CudaWorkerBase


def _stamped_png(tmp_path) -> bytes:
    payload = build_map_metadata(
        tool="canny_map", control_type="canny",
        source_size=(8, 8), params={"low_threshold": 100, "high_threshold": 200},
    )
    dest = tmp_path / "m.png"
    save_with_metadata(Image.new("RGB", (8, 8), "white"), dest, payload)
    return dest.read_bytes()


def _bare_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    return buf.getvalue()


def _binding(attachment_id, control_type, model_id, image_bytes, strength):
    return ControlNetBinding(
        attachment_id=attachment_id, control_type=control_type,
        model_id=model_id, model_path="/x", control_image_bytes=image_bytes,
        strength=strength, start_percent=0.0, end_percent=0.7,
    )


def test_controlnet_metadata_source_populated_and_null(tmp_path):
    worker = CudaWorkerBase.__new__(CudaWorkerBase)  # no GPU/env init
    bindings = [
        _binding("cn-1", "canny", "sdxl-canny", _stamped_png(tmp_path), 0.8),
        _binding("cn-2", "depth", "sdxl-depth", _bare_png(), 1.0),
    ]
    out = worker._controlnet_metadata(bindings)

    assert len(out) == 2
    assert out[0]["attachment_id"] == "cn-1"
    assert out[0]["control_type"] == "canny"
    assert out[0]["generation"] == {
        "model_id": "sdxl-canny", "strength": 0.8,
        "start_percent": 0.0, "end_percent": 0.7,
    }
    assert out[0]["source"]["tool"] == "canny_map"
    assert out[1]["source"] is None


def test_controlnet_metadata_empty_bindings():
    worker = CudaWorkerBase.__new__(CudaWorkerBase)
    assert worker._controlnet_metadata([]) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_worker_controlnet_metadata.py -v`
Expected: FAIL — `AttributeError: 'CudaWorkerBase' object has no attribute '_controlnet_metadata'`.

- [ ] **Step 3: Add the method to `CudaWorkerBase`**

In `backends/cuda_worker.py`, add this method to `CudaWorkerBase` (e.g. right
after `__init__`):

```python
    def _controlnet_metadata(self, bindings: list[Any]) -> list[dict[str, Any]]:
        """Per-attachment ControlNet provenance for the generation PNG.

        source: embedded controlnet_map payload from the map PNG, or None.
        generation: the ControlNet params actually used for this render.
        """
        from server.controlnet_metadata import read_control_map_metadata

        entries: list[dict[str, Any]] = []
        for binding in bindings:
            entries.append({
                "attachment_id": binding.attachment_id,
                "control_type": binding.control_type,
                "generation": {
                    "model_id": binding.model_id,
                    "strength": binding.strength,
                    "start_percent": binding.start_percent,
                    "end_percent": binding.end_percent,
                },
                "source": read_control_map_metadata(binding.control_image_bytes),
            })
        return entries
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_worker_controlnet_metadata.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Wire both `run_job` chunk sites**

In `backends/cuda_worker.py`, at the `DiffusersCudaWorker.run_job` chunk site
(after the `pnginfo.add_text("lcm", ...)` block near line 566, before
`buf = io.BytesIO()`), add:

```python
            if bindings:
                pnginfo.add_text("controlnet", json.dumps(self._controlnet_metadata(bindings)))
```

Apply the identical addition at the `DiffusersSDXLCudaWorker.run_job` chunk
site (after the `pnginfo.add_text("lcm", ...)` block near line 898, before
`buf = io.BytesIO()`). `bindings` is already in local scope in both methods.

- [ ] **Step 6: Verify existing worker tests still pass**

Run: `python -m pytest tests/test_worker_controlnet_metadata.py -v`
Expected: PASS. (The guarded `add_text` on the GPU render path is validated by
the `if bindings:` guard plus the `_controlnet_metadata` unit tests; the full
render path requires CUDA and is exercised in the CUDA integration path, not
in offline CI.)

- [ ] **Step 7: Commit**

```bash
git add backends/cuda_worker.py tests/test_worker_controlnet_metadata.py
git commit -m "feat(controlnet): write controlnet provenance chunk into generation PNG"
```

---

## Self-Review

**Spec coverage:**
- Part 1 (scripts embed metadata) → Tasks 1–3. ✓
- Packaging + `py-modules` + test contract → Task 1 (pyproject) + Task 2 (assertion). ✓
- `--colorize` exclusion → Task 3 Step 3 leaves the `_color` save untouched. ✓
- `device` = requested CLI arg → Task 3 records `args.device` for depth and pose. ✓
- Part 2 reader (tolerant) → Task 4. ✓
- Part 2 worker assembly + both chunk sites + empty-bindings guard → Task 5. ✓
- Non-goals (no `ControlNetBinding` change, no preprocessor change, no response
  surface change) → honored; no task touches them. ✓
- Offline tests → cn_metadata unit tests, canny subprocess, depth/pose
  monkeypatch, reader synthetic PNGs, worker `__new__` + fake bindings. ✓

**Type consistency:** `build_map_metadata` / `save_with_metadata` /
`read_control_map_metadata` / `_controlnet_metadata` signatures are identical
across the tasks that define and consume them. Chunk keys `controlnet_map` and
`controlnet` are used verbatim throughout.

**Placeholder scan:** No TBD/TODO; every code and test step shows complete code.
