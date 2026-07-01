# Canny Control Map Script Design

## Goal

Add a local utility script that generates a reusable canny edge map image for
ControlNet workflows without adding new server behavior or model dependencies.

## Scope

In scope:

- add `scripts/canny_map.py`
- integrate it into `scripts/pyproject.toml`
- produce a local grayscale canny control-map image from an input image
- document how operators run it and feed the result into `st upload` or
  `st gen --control-image`
- add script-level tests for basic behavior

Out of scope:

- backend preprocessing changes
- new ControlNet server control types
- frontend/UI changes
- model downloads or annotator-based preprocessing

## Existing Context

The repo already ships `scripts/depth_map.py` as a local preprocessing utility
for ControlNet depth maps. The new canny tool should be a sibling script with a
matching command-line shape and similar operator ergonomics.

Unlike `depth_map.py` and `pose_map.py`, this script does not need `torch` or
the float8 compatibility shim. It should remain the lightest script in this
family: PIL + NumPy + OpenCV only.

The current operator flow already supports reusable control maps:

1. generate a local map image
2. upload it with `st upload canny:<path>`
3. attach it with `st gen --control-image canny:<path>` or a raw
   `--controlnet` payload

This means the requested utility belongs in `scripts/`, not in backend request
handling.

## Design

### New script

Create `scripts/canny_map.py`.

Responsibilities:

- load a local input image
- optionally downscale before processing
- convert to grayscale
- optionally apply Gaussian blur
- run OpenCV Canny edge detection
- optionally invert the final map
- save a single-channel output image suitable for ControlNet upload/reuse

The script remains fully local and deterministic. It should not talk to the
server and should not require ControlNet model assets.

The script intentionally does not expose a `--device` flag. Existing sibling
scripts need it because they run model-backed preprocessing. This script uses
OpenCV locally on CPU and should keep that simpler surface.

### CLI contract

Command shape mirrors `scripts/depth_map.py`:

```bash
python scripts/canny_map.py source.jpg canny.png
```

Arguments:

- positional `source`: input image path
- positional `destination`: output image path

Flags:

- `--low-threshold` integer, default `100`
- `--high-threshold` integer, default `200`
- `--blur` integer kernel size, default `0`
- `--max-res` integer longest-edge cap before processing, default `None`
- `--invert` boolean flag

Behavior notes:

- `--blur 0` means no blur
- non-zero blur must be a positive odd integer `>= 1`
- reject even blur values and negative values with a clear error
- output should be saved in PIL mode `L` (8-bit grayscale), which matches the
  expected single-channel canny control-map shape
- progress output should stay simple and script-like, matching the tone of
  `scripts/depth_map.py`

### Processing pipeline

Pipeline:

1. open image and convert to RGB
2. if `--max-res` is set and the longest edge exceeds it, downscale with
   Lanczos
3. convert to grayscale
4. if `--blur > 0`, apply Gaussian blur with the requested kernel size
5. run `cv2.Canny(gray, low_threshold, high_threshold)`
6. if `--invert`, invert the final edge map
7. write the result to `destination` as mode `L`

This keeps the script dependency-light and aligned with the user’s request to
prefer OpenCV over annotator-backed tooling.

### Packaging

Update `scripts/pyproject.toml` so the new utility is installable through the
same helper package as the existing scripts.

Required changes:

- add a `canny` optional dependency group
- register `st-canny-map = "canny_map:main"` under `[project.scripts]`
- include `canny_map` in `[tool.setuptools].py-modules`

Expected shape:

```toml
[project.optional-dependencies]
canny = [
    "opencv-python-headless>=4.5",
]
```

```toml
[project.scripts]
st-depth-map = "depth_map:main"
st-pose-map = "pose_map:main"
st-canny-map = "canny_map:main"
```

```toml
[tool.setuptools]
py-modules = ["depth_map", "pose_map", "canny_map"]
```

If the helper package exposes a convenience aggregate extra, include `canny`
there as well so the documented install path remains coherent.

## Testing

Add `tests/test_canny_map.py`.

Test surface:

- missing source path exits non-zero
- basic invocation creates an output file
- output image is mode `L` and non-empty on a deterministic fixture
- `--invert` changes output polarity
- `--max-res` exercises the resize path
- valid odd blur value (for example `--blur 5`) exercises the blur path
- invalid blur kernel values fail with a clear error

Tests should verify behavior and file outputs, not subjective edge quality.
Avoid brittle golden-image assertions.

## Documentation

Update `scripts/USAGE.md`:

- add a `canny_map.py` section
- document the install path through `scripts/pyproject.toml`
- document the required OpenCV dependency surface
- provide basic examples
- show the operator handoff into ControlNet:

```bash
python scripts/canny_map.py photo.jpg canny.png
st gen "..." --control-image canny:./canny.png
```

If script dependency installation is already centralized in repo tooling, hook
the doc into that path. Otherwise document the needed Python package explicitly.

## Files

Create:

- `scripts/canny_map.py`
- `tests/test_canny_map.py`

Modify:

- `scripts/pyproject.toml`
- `scripts/USAGE.md`

## Risks and Constraints

- OpenCV must be available in the operator environment; this is the only new
  runtime dependency for the script itself
- threshold defaults may not fit every image, so the CLI must expose both
  thresholds directly
- output polarity can vary by operator preference, so `--invert` is required
  rather than hardcoding one convention

## Acceptance

This design is complete when:

- a local operator can generate a canny control map from an input image
- the output can be reused in existing ControlNet upload/generation flows
- no backend or worker changes are required
