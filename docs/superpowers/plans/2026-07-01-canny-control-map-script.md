# Canny Control Map Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local OpenCV-backed canny control-map utility that operators can install, run, and feed into existing ControlNet upload and generation flows.

**Architecture:** Keep this feature entirely in the local helper-script layer. First extend `scripts/pyproject.toml` so the canny helper installs cleanly and exposes `st-canny-map`, then add `scripts/canny_map.py` plus CLI-level tests that exercise output shape and validation, and finally update `scripts/USAGE.md` to document install paths and operator usage.

**Tech Stack:** Python 3.10+, pytest, Pillow, NumPy, OpenCV (`opencv-python-headless`), setuptools optional extras.

---

## File Structure

- Modify: `scripts/pyproject.toml`
  Purpose: add the `canny` optional dependency group, update the aggregate `all` extra, register `st-canny-map`, and include `canny_map` in setuptools module discovery.
- Create: `scripts/canny_map.py`
  Purpose: implement local canny-map generation with `source`/`destination`, `--low-threshold`, `--high-threshold`, `--blur`, `--max-res`, and `--invert`.
- Create: `tests/test_canny_map.py`
  Purpose: verify packaging metadata, missing-path handling, `L`-mode output, invert behavior, resize behavior, blur behavior, and invalid blur rejection.
- Modify: `scripts/USAGE.md`
  Purpose: document installation, direct and console-script invocation, parameter table, examples, and the handoff into `st gen --control-image canny:...`.

### Task 1: Add Installable Packaging Surface for the Canny Helper

**Files:**
- Modify: `scripts/pyproject.toml`
- Create: `tests/test_canny_map.py`

- [ ] **Step 1: Write the failing packaging test**

```python
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_PYPROJECT = ROOT / "scripts" / "pyproject.toml"


def test_pyproject_exposes_canny_install_surface():
    data = tomllib.loads(SCRIPTS_PYPROJECT.read_text())
    project = data["project"]

    assert project["optional-dependencies"]["canny"] == [
        "opencv-python-headless>=4.5",
    ]
    assert project["optional-dependencies"]["all"] == [
        "st-controlnet-helpers[depth,pose,canny]",
    ]
    assert project["scripts"]["st-canny-map"] == "canny_map:main"
    assert "canny_map" in data["tool"]["setuptools"]["py-modules"]
```

- [ ] **Step 2: Run the packaging test to verify it fails**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_canny_map.py::test_pyproject_exposes_canny_install_surface -q
```

Expected: FAIL because `scripts/pyproject.toml` does not yet define the `canny` extra, the `st-canny-map` console entry point, or the `canny_map` module registration.

- [ ] **Step 3: Update `scripts/pyproject.toml`**

```toml
[project.optional-dependencies]
depth = [
    "transformers>=4.35",
    "controlnet-aux>=0.0.7",
    "matplotlib",           # only needed for --colorize
]
pose = [
    "controlnet-aux>=0.0.7",
    "mediapipe==0.10.14",
]
canny = [
    "opencv-python-headless>=4.5",
]
all = [
    "st-controlnet-helpers[depth,pose,canny]",
]

[project.scripts]
st-depth-map = "depth_map:main"
st-pose-map  = "pose_map:main"
st-canny-map = "canny_map:main"

[tool.setuptools]
py-modules = ["depth_map", "pose_map", "canny_map"]
```

- [ ] **Step 4: Install the canny extra into the active base environment**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pip install -e "./scripts[canny]"
```

Expected: pip installs the editable helper package and pulls in `opencv-python-headless`, making the later CLI tests runnable without ad hoc dependency setup.

- [ ] **Step 5: Re-run the packaging test**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_canny_map.py::test_pyproject_exposes_canny_install_surface -q
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/pyproject.toml tests/test_canny_map.py
git commit -m "build(scripts): add canny helper packaging surface"
```

### Task 2: Implement the OpenCV Canny Script and CLI Behavior Tests

**Files:**
- Create: `scripts/canny_map.py`
- Modify: `tests/test_canny_map.py`

- [ ] **Step 1: Add the failing CLI behavior tests**

Append these tests to `tests/test_canny_map.py` below the packaging test:

```python
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image, ImageDraw


SCRIPT = ROOT / "scripts" / "canny_map.py"


def _run_script(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(str(arg) for arg in args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def _write_fixture(path: Path, size: tuple[int, int] = (96, 64)) -> None:
    img = Image.new("RGB", size, "black")
    draw = ImageDraw.Draw(img)
    draw.rectangle((12, 12, size[0] - 12, size[1] - 12), outline="white", width=4)
    draw.line((0, size[1] // 2, size[0], size[1] // 2), fill="white", width=2)
    img.save(path)


def test_missing_source_path_reports_error(tmp_path: Path):
    missing = tmp_path / "missing.png"
    dest = tmp_path / "edges.png"

    result = _run_script(missing, dest)

    assert result.returncode != 0
    assert f"error: source not found: {missing}" in result.stderr


def test_generates_l_mode_edge_map(tmp_path: Path):
    source = tmp_path / "source.png"
    dest = tmp_path / "edges.png"
    _write_fixture(source)

    result = _run_script(source, dest)

    assert result.returncode == 0, result.stderr
    out = Image.open(dest)
    assert out.mode == "L"
    arr = np.array(out)
    assert arr.max() == 255
    assert arr.min() == 0


def test_invert_flips_edge_polarity(tmp_path: Path):
    source = tmp_path / "source.png"
    base_dest = tmp_path / "edges.png"
    inv_dest = tmp_path / "edges_invert.png"
    _write_fixture(source)

    base = _run_script(source, base_dest)
    inverted = _run_script(source, inv_dest, "--invert")

    assert base.returncode == 0, base.stderr
    assert inverted.returncode == 0, inverted.stderr
    base_arr = np.array(Image.open(base_dest))
    inv_arr = np.array(Image.open(inv_dest))
    assert np.array_equal(inv_arr, 255 - base_arr)


def test_max_res_caps_longest_edge(tmp_path: Path):
    source = tmp_path / "source.png"
    dest = tmp_path / "edges.png"
    _write_fixture(source, size=(400, 200))

    result = _run_script(source, dest, "--max-res", "100")

    assert result.returncode == 0, result.stderr
    out = Image.open(dest)
    assert out.size == (100, 50)


def test_valid_blur_value_exercises_blur_path(tmp_path: Path):
    source = tmp_path / "source.png"
    dest = tmp_path / "edges.png"
    _write_fixture(source)

    result = _run_script(source, dest, "--blur", "5")

    assert result.returncode == 0, result.stderr
    assert dest.exists()


@pytest.mark.parametrize("blur", ["2", "-1"])
def test_invalid_blur_values_fail_fast(tmp_path: Path, blur: str):
    source = tmp_path / "source.png"
    dest = tmp_path / "edges.png"
    _write_fixture(source)

    result = _run_script(source, dest, "--blur", blur)

    assert result.returncode != 0
    assert "--blur must be 0 or a positive odd integer" in result.stderr
```

- [ ] **Step 2: Run the CLI tests to verify they fail**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_canny_map.py -q
```

Expected: FAIL because `scripts/canny_map.py` does not exist yet.

- [ ] **Step 3: Implement `scripts/canny_map.py`**

```python
#!/usr/bin/env python3
"""Generate a canny edge map from an image for use with ControlNet."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image(path: Path, max_res: int | None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max_res:
        w, h = img.size
        scale = max_res / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return img


def validate_blur(value: int) -> int:
    if value == 0:
        return value
    if value < 0 or value % 2 == 0:
        raise argparse.ArgumentTypeError("--blur must be 0 or a positive odd integer")
    return value


def canny_edges(
    img: Image.Image,
    *,
    low_threshold: int,
    high_threshold: int,
    blur: int,
    invert: bool,
) -> Image.Image:
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    if blur > 0:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    if invert:
        edges = 255 - edges
    return Image.fromarray(edges, mode="L")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a canny edge map for ControlNet.")
    parser.add_argument("source", type=Path, help="Input image path")
    parser.add_argument("destination", type=Path, help="Output canny map path (PNG recommended)")
    parser.add_argument(
        "--low-threshold",
        type=int,
        default=100,
        help="Low hysteresis threshold for Canny (default: 100)",
    )
    parser.add_argument(
        "--high-threshold",
        type=int,
        default=200,
        help="High hysteresis threshold for Canny (default: 200)",
    )
    parser.add_argument(
        "--blur",
        type=validate_blur,
        default=0,
        help="Optional Gaussian blur kernel size; use 0 to disable blur (default: 0)",
    )
    parser.add_argument(
        "--max-res",
        type=int,
        default=None,
        metavar="PX",
        help="Cap longest edge before processing (e.g. 1024)",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert edge polarity after Canny",
    )
    args = parser.parse_args()

    if not args.source.exists():
        print(f"error: source not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    args.destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading  {args.source}")
    img = load_image(args.source, args.max_res)
    print(
        "running  canny"
        f" (low={args.low_threshold}, high={args.high_threshold}, blur={args.blur})"
    )

    result = canny_edges(
        img,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        blur=args.blur,
        invert=args.invert,
    )
    result.save(args.destination)
    print(f"saved    {args.destination}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full canny test file**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_canny_map.py -q
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/canny_map.py tests/test_canny_map.py
git commit -m "feat(scripts): add canny control map utility"
```

### Task 3: Document the Canny Helper for Operators

**Files:**
- Modify: `scripts/USAGE.md`

- [ ] **Step 1: Update the install section to include canny**

Replace the opening install snippet with:

````md
```bash
# installs st-depth-map, st-pose-map, and st-canny-map onto PATH
make install-controlnet-scripts
make install-controlnet-scripts EXTRAS=depth
make install-controlnet-scripts EXTRAS=pose
make install-controlnet-scripts EXTRAS=canny

# or directly with pip
pip install "./scripts[all]"
```
````

Also update the “After install both forms are equivalent” example so it shows the new console command:

````md
```bash
st-canny-map photo.jpg canny.png
python scripts/canny_map.py photo.jpg canny.png
```
````

- [ ] **Step 2: Add a `canny_map.py` section**

Insert a new section after `pose_map.py` with this content:

````md
## canny_map.py

Generate an 8-bit grayscale canny edge map from an image.

**Install deps**
```bash
pip install opencv-python-headless pillow numpy
```

**Parameters**

| Argument | Default | Description |
|---|---|---|
| `source` | — | Input image path |
| `destination` | — | Output canny map path |
| `--low-threshold` | `100` | Low hysteresis threshold |
| `--high-threshold` | `200` | High hysteresis threshold |
| `--blur` | `0` | Gaussian blur kernel size; `0` disables blur |
| `--max-res` | none | Cap longest edge in pixels before processing |
| `--invert` | off | Flip polarity after edge detection |

**Examples**

```bash
# Default settings
python scripts/canny_map.py photo.jpg canny.png

# Softer edges after a light blur
python scripts/canny_map.py photo.jpg canny.png \
  --low-threshold 75 --high-threshold 180 --blur 5

# Resize first, then invert
python scripts/canny_map.py photo.jpg canny.png \
  --max-res 1024 --invert
```

Output: grayscale PNG in mode `L` where edge pixels are white and the background
is black (unless `--invert` is set).
````

- [ ] **Step 3: Add the ControlNet handoff example**

Make sure the “Feeding maps to `st gen`” section contains an explicit canny example like this:

````md
```bash
python scripts/canny_map.py photo.jpg canny.png
st gen "city street, cinematic lighting" \
  --control-image canny:./canny.png
```
````

Leave the existing explanation of the `<type>:` prefix intact; it already documents why `canny:` becomes the validated `control_type`.

- [ ] **Step 4: Run doc-adjacent verification**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_canny_map.py -q
```

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && st-canny-map --help
```

Expected:

- pytest still passes
- `st-canny-map --help` prints the CLI usage including `--low-threshold`, `--high-threshold`, `--blur`, `--max-res`, and `--invert`

- [ ] **Step 5: Commit**

```bash
git add scripts/USAGE.md
git commit -m "docs(scripts): document canny control map workflow"
```

## Self-Review

- Spec coverage:
  - packaging surface and `all` extra update: Task 1
  - local OpenCV script with no `torch` / no `--device`: Task 2
  - blur validation, `L` output, invert and resize behavior: Task 2
  - install docs and `st gen --control-image canny:...` operator handoff: Task 3
- Placeholder scan:
  - no `TODO`, `TBD`, or “similar to previous task” shortcuts remain
- Type consistency:
  - `st-canny-map`, `canny_map.py`, `--blur`, `--max-res`, and `canny:` are used consistently across packaging, script, tests, and docs

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-01-canny-control-map-script.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
