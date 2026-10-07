# Scripts

Utility scripts for ControlNet preprocessor image generation.

---

## Install

Run the scripts directly with `python scripts/<name>.py`, or install them as
console commands:

```bash
# installs st-depth-map, st-pose-map, st-canny-map, st-vector-map, and st-resize-for-model onto PATH
make install-controlnet-scripts            # all extras (depth + pose + canny)
make install-controlnet-scripts EXTRAS=depth   # depth backends only
make install-controlnet-scripts EXTRAS=pose    # pose backends only
make install-controlnet-scripts EXTRAS=canny   # canny backends only
make install-controlnet-scripts EXTRAS=vector  # st-vector-map with VTracer (not part of all)

# or directly with pip
pip install "./scripts[all]"
```

After install both forms are equivalent:

```bash
st-canny-map photo.jpg canny.png             # console script
python scripts/canny_map.py photo.jpg canny.png   # direct
```

> Only the `depth` and `pose` extras require `torch`. The `canny` and `vector`
> extras install without it. `st-depth-map` and `st-pose-map` print an install
> hint when `torch` is missing.
>
> On macOS/Apple Silicon, install `torch` via conda **first**
> (`conda install pytorch -c pytorch`). The conda build satisfies the
> `torch>=2.1` pin, so pip does not replace it.
>
> The `vector` extra pins `vtracer==0.6.15`. The `all` extra does not include it.

`make install` installs both the `st` CLI and these scripts in one shot.

---

## st-vector-map

Trace PNG or JPEG masks and edge maps into SVG with physical dimensions.
Use PNG for lossless mask pixels. Select exactly one physical dimension.
Supported pixel modes: `1`, `L`, `LA`, `P`, `RGB`, and `RGBA`.
Convert other modes, including 16-bit grayscale and CMYK, to supported 8-bit input before tracing.
Use RGBA PNG when conversion must preserve alpha.

```bash
st-vector-map mask.png silhouette.svg --input-kind mask --width-mm 100
st-vector-map edges.png detail.svg --input-kind edges --width-mm 100 \
  --line-width-mm 0.8 --include-mask allowed.png --exclude-mask removed.png --json
```

Default material has luminance >= 128. `--invert` reverses material selection.
`--alpha` selects alpha >= 128 instead. Inputs with unused alpha produce warnings.
Palette transparency supports alpha selection. Inputs without alpha reject `--alpha`.

EXIF orientation applies before thresholding and dimension checks.
External masks must match oriented source dimensions. Source inversion and alpha options do not change their luminance selection.
`--include-mask` limits material. `--exclude-mask` removes material and always wins.
Both constraints apply before and after edge expansion.
`--mask` remains reserved for future image-mode silhouette selection.

`--max-res PX` limits longest processing side without upscaling.
Binary masks use nearest-neighbour resize. Shorter side rounds half upward, with minimum one pixel.
Selected physical dimension stays exact. Other dimension follows processed aspect ratio with one uniform scale.
All layers keep complete canvas and origin. No layer crop or centering occurs.

`--line-width-mm` requires edge mode. It specifies nominal width for an isolated one-pixel line.
Let `p = requested_mm / mm_per_px` and `r = max(0, ceil((p - 1) / 2 - 1e-9))`.
Tolerance prevents floating-point noise from adding two pixels at exact odd widths.
Square dilation uses side `2*r + 1`. Existing wide bands expand further. Diagonal widths remain approximate.
For example, 3.1 requested pixels becomes five pixels.
Stderr reports requested and nominal achieved millimetres, kernel size, radius, expansion, dimensions, and scale.

Warnings identify requested widths below four processing pixels and possible thin material or background gaps.
Feature diagnostics check 4x4 square coverage. Outside canvas counts as background.
Curved or diagonal boundaries can produce false positives. Diagnostics never repair geometry.
Both width and feature warnings remain when both conditions apply.

`--json` emits one result object. Warnings appear in `diagnostics` and on stderr.
Successful conversion with warnings returns exit 0. Invalid settings return 2. Processing failures return 1.
Failure publishes no SVG. `--overwrite` never permits replacing source, recipe, or constraint inputs.

Recipe fields match option names with underscores: `include_mask`, `exclude_mask`, `line_width_mm`, and `max_res`.
Recipes require `schema_version: 1`. Recipe paths resolve relative to recipe directory. Explicit CLI options override recipe values.
Image mode, preview, and artifact bundles remain separate sprint tasks.

---

## st-resize-for-model / resize-for-model.sh

Resize a flat directory of images into model-friendly generation buckets using
ImageMagick. The script inspects each image aspect ratio, chooses the nearest
target shape, and writes the resized image to an output directory with the same
basename.

**Install deps**
```bash
brew install imagemagick
# or use your platform package manager for ImageMagick 7+
```

**Parameters**

| Argument | Default | Description |
|---|---|---|
| `IMAGE_DIR` | — | Directory containing input images |
| `--out DIR` | — | Output directory |
| `--profile` | `sd15` | `sd15` buckets: `512x512`, `768x512`, `512x768`, `768x768`; `sdxl` buckets: `1024x1024`, `1152x896`, `896x1152` |
| `--mode` | `crop` | `crop` fills the target and center-crops; `pad` preserves the full image and pads with black |

**Examples**

```bash
# SD1.5 buckets, crop-to-fill by default
st-resize-for-model ./raw-images --profile sd15 --out ./resized-sd15

# SDXL buckets, preserve full images with black padding
st-resize-for-model ./raw-images --profile sdxl --mode pad --out ./resized-sdxl

# Direct source-tree form also works without installation
scripts/resize-for-model.sh ./raw-images --profile sd15 --out ./resized-sd15
```

Supported input extensions are `png`, `jpg`, `jpeg`, `webp`, `bmp`, `tif`,
`tiff`, and `heic`. The scan is non-recursive.

---

## depth_map.py

Generate a grayscale depth map from an image.

**Install deps**
```bash
pip install transformers torch pillow numpy matplotlib
pip install controlnet-aux  # required for --model midas only
```

**Parameters**

| Argument | Default | Description |
|---|---|---|
| `source` | — | Input image path |
| `destination` | — | Output depth map path |
| `--model` | `depth-anything` | `depth-anything` / `midas` / `zoe` |
| `--size` | `small` | `small` / `base` / `large` (depth-anything only) |
| `--device` | `cpu` | `cpu` / `cuda` / `mps` |
| `--max-res` | none | Cap longest edge in pixels before inference |
| `--invert` | off | Flip polarity — white=far instead of white=near |
| `--colorize` | off | Also save a jet-colormap visualization alongside grayscale |

**Examples**

```bash
# Quickest — small Depth-Anything model on CPU
python scripts/depth_map.py photo.jpg depth.png

# Large model on GPU with colorized preview
python scripts/depth_map.py photo.jpg depth.png \
  --model depth-anything --size large \
  --device cuda --colorize

# MiDaS, cap at 768px longest edge, invert polarity
python scripts/depth_map.py photo.jpg depth.png \
  --model midas --max-res 768 --invert

# ZoeDepth on Apple Silicon
python scripts/depth_map.py photo.jpg depth.png \
  --model zoe --device mps
```

Output: grayscale PNG where **white = near, black = far** (unless `--invert`).
When `--colorize` is set, a second file is saved with `_color` appended to the stem.

---

## pose_map.py

Generate a skeleton pose map from an image.

**Install deps**
```bash
pip install controlnet-aux   # openpose + dwpose
pip install mediapipe==0.10.14. # mediapipe only
```

**Parameters**

| Argument | Default | Description |
|---|---|---|
| `source` | — | Input image path |
| `destination` | — | Output pose map path |
| `--model` | `dwpose` | `openpose` / `dwpose` / `mediapipe` |
| `--parts` | `body,face,hands` | Comma-separated parts (openpose only) |
| `--device` | `cpu` | `cpu` / `cuda` / `mps` |
| `--max-res` | none | Cap longest edge in pixels before inference |
| `--show-keypoints` | off | Draw keypoint dots only, no limb connections (mediapipe only) |
| `--overlay` | off | Draw skeleton on original image instead of black background (mediapipe only) |

**Examples**

```bash
# DWPose (default, recommended)
python scripts/pose_map.py photo.jpg pose.png

# OpenPose — body + hands only, no face
python scripts/pose_map.py photo.jpg pose.png \
  --model openpose --parts body,hands

# OpenPose — all parts, GPU, cap resolution
python scripts/pose_map.py photo.jpg pose.png \
  --model openpose --parts body,face,hands \
  --device cuda --max-res 768

# MediaPipe — full skeleton overlay
python scripts/pose_map.py photo.jpg pose.png --model mediapipe

# MediaPipe — keypoints only on black canvas
python scripts/pose_map.py photo.jpg pose.png \
  --model mediapipe --show-keypoints
```

Output: RGB PNG with colored keypoints and limb connections on a **black background** — correct for ControlNet conditioning.
`--overlay` (mediapipe) draws on the original image instead, for visualization only.
`--show-keypoints` (mediapipe) outputs white dots on black with no connections.

---

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

---

## vector_map.py

Trace a binary mask into an SVG with a physical size in millimetres.
The command uses VTracer 0.6.15 in polygon mode. Install the `vector` extra.

> **Status: walking skeleton (S2.2).** Only `--input-kind mask` converts.
> Edge mode, image mode, `--max-res`, `--line-width-mm`, `--mask`, `--alpha`,
> `--preview`, alpha-bearing input, and EXIF orientation other than 1 fail with
> exit code 2. The message names the task that adds the feature.

**Parameters**

| Argument | Default | Description |
|---|---|---|
| `source` | — | Mask image. Optional with `--recipe` when the recipe sets `input` |
| `destination` | — | Output SVG path |
| `--input-kind` | — | `mask`. Required on the command line or in the recipe |
| `--width-mm` / `--height-mm` | — | Physical size of the complete canvas. Give exactly one |
| `--invert` / `--no-invert` | off | Treat dark pixels as material |
| `--recipe` | none | `schema_version: 1` JSON recipe |
| `--overwrite` | off | Replace an existing destination |
| `--json` | off | Print one result object on stdout |

Material is luminance >= 128 (white). The SVG root states the size in `mm`
with a pixel `viewBox`. OpenSCAD imports this form at the stated size.

**Recipe**

Settings apply in this order. A later value wins:

1. Built-in defaults.
2. The recipe.
3. Explicit command-line options.

A command-line `--width-mm` replaces a recipe `height_mm`, and conversely.
Recipe paths are relative to the recipe file. Unknown fields fail.
The `vtracer` object accepts `mode` (`polygon` only) and `filter_speckle` (0 to 128).

```json
{"schema_version": 1, "input": "mask.png", "input_kind": "mask",
 "width_mm": 100, "vtracer": {"filter_speckle": 4}}
```

**Exit codes**

| Code | Meaning |
|---|---|
| 0 | The SVG was written |
| 2 | Arguments or configuration are invalid, or the feature is not available yet |
| 1 | Processing or I/O failed, for example an unreadable image or an empty trace |

**Examples**

```bash
# 100 mm wide silhouette from a white-on-black mask
st-vector-map mask.png silhouette.svg --input-kind mask --width-mm 100

# Source and settings from a recipe, machine-readable result
st-vector-map silhouette.svg --recipe recipe.json --json
```

---

## Model comparison

### Depth

| Model | Speed | Quality | cpu | cuda | mps | Notes |
| --- | --- | --- | :---: | :---: | :---: | --- |
| `depth-anything` small | Fast | Good | ✅ | ✅ | ✅ | Best default choice |
| `depth-anything` large | Slow | Best | ✅ | ✅ | ✅ | Use when detail matters |
| `midas` | Fast | Good | ✅ | ✅ | ✅ | Older; reliable fallback |
| `zoe` | Medium | Good | ✅ | ✅ | ⚠️ | May fall back to CPU for unsupported ops |

### Pose

| Model | Quality | cpu | cuda | mps | Notes |
| --- | --- | :---: | :---: | :---: | --- |
| `dwpose` | Best | ✅ | ✅ | ⚠️ | `controlnet_aux` may ignore device hint; runs CPU in practice |
| `openpose` | Good | ✅ | ✅ | ⚠️ | Same device caveat as dwpose |
| `mediapipe` | Fine | ✅ | ✅ | ✅ | Doesn't use PyTorch; MPS irrelevant but fully native on Mac |

### Device notes

**cuda** — NVIDIA GPU. Fastest for all models.

**mps** — Apple Silicon (M1/M2/M3/M4). Requires macOS 12.3+. The default
`pip install torch` on macOS ships the MPS-capable build — no extra flags needed.
Gives a real speedup for depth models. Pose models (`dwpose`, `openpose`) via
`controlnet_aux` don't reliably respect the device hint and typically run on CPU
regardless.

**cpu** — universal fallback. Use when no GPU is available or a model doesn't
support the target device.

```bash
# Apple Silicon — depth (MPS speedup)
python scripts/depth_map.py photo.jpg depth.png --device mps

# Apple Silicon — pose (cpu is effectively the same)
python scripts/pose_map.py photo.jpg pose.png --device cpu
```

---

## Running in Docker

The `controlnet-tools` image stage extends the server image with all script
dependencies pre-installed. Use it when you want a self-contained environment
without touching your local Python setup.

### Build

```bash
docker build --target controlnet-tools -t st-controlnet-tools .
```

The stage inherits torch (and CUDA if built with `--build-arg BACKEND=cuda`)
from the server base, then adds `transformers`, `controlnet-aux`, `mediapipe`,
`matplotlib`, and `opencv-python-headless`.

### Run interactively

```bash
# Mount a local folder as /images — read inputs and write outputs there
docker run --rm -it -v $(pwd)/images:/images st-controlnet-tools
```

Inside the container the working directory is `/app/scripts`, so the scripts
are on the path directly:

```bash
# Depth map
python depth_map.py /images/input.png /images/depth.png --model depth-anything

# Pose map
python pose_map.py /images/input.png /images/pose.png

# Canny edge map
python canny_map.py /images/input.png /images/canny.png

# With CUDA (requires --build-arg BACKEND=cuda at build time)
python depth_map.py /images/input.png /images/depth.png --device cuda
```

### One-shot (non-interactive)

```bash
docker run --rm \
  -v $(pwd)/images:/images \
  st-controlnet-tools \
  python depth_map.py /images/input.png /images/depth.png --model depth-anything --size large
```

---

## Feeding maps to `st gen`

Once you have a depth/pose/canny map, attach it to a generation. The CLI
offers a one-step shorthand and a manual two-step path.

### One step: `--control-image`

`st gen --control-image <type>:<path>` uploads the map and attaches it in a
single command:

```bash
# depth map produced above, applied as a ControlNet
st gen "A majestic girl holding a crystal orb in each hand" \
  --seed 69823301 \
  --control-image depth:./depth.png
```

```bash
# canny edge map produced above, applied as a ControlNet
python scripts/canny_map.py photo.jpg canny.png
st gen "city street, cinematic lighting" \
  --control-image canny:./canny.png
```

The CLI uploads the file, then injects a ControlNet attachment of the form
`{attachment_id, control_type, map_asset_ref}` into the request. `attachment_id`
is auto-generated (`ctrl-0`, `ctrl-1`, …).

The flag is **repeatable** — stack multiple control types:

```bash
st gen "..." \
  --control-image depth:./depth.png \
  --control-image canny:./edges.png
```

### The `<type>:` prefix (bucket / control_type)

The prefix before the colon serves two roles:

| Role | Where it goes | Effect |
| --- | --- | --- |
| Upload bucket | `type` form field on `POST /v1/upload` | **Intent label only** — the server currently ignores it for routing |
| `control_type` | the ControlNet attachment | **Meaningful** — validated against the model's declared `control_types` and the mode policy |

So `depth:` and `canny:` matter because they become the attachment's
`control_type`, which the server checks against the mode's
`allowed_control_types` (see `conf/modes.yml`) and the model registry
(`conf/controlnets.yaml`). A `control_type` the active mode doesn't permit is
rejected before generation.

> The type prefix is **required** for `--control-image`. Omitting it errors with
> `missing control_type prefix (use type:<path>, e.g. depth:./map.png)`.

### Conditioning strength: `--control-strength`

Controls how strongly the map steers the result (`0.0`–`2.0`). Applies to every
`--control-image` attachment in the same command:

```bash
st gen "..." --control-image depth:./depth.png --control-strength 0.65
```

- **Unset** → the attachment omits `strength`, so the server applies the mode
  policy's `default_strength` (typically `1.0`).
- An explicit `--control-strength 0` is honored as zero (not treated as unset).
- Higher = the structure of the map dominates; lower = the prompt has more
  freedom.

`--control-strength` only affects `--control-image` attachments. Raw
`--controlnet` JSON and `--controlnet-file` entries carry their own `strength`
field and are left untouched.

### Manual two-step (raw JSON)

For full control over attachment fields, upload and attach separately:

```bash
# 1. upload, capture the fileref
ref=$(st upload depth:./depth.png --json | jq -r .fileRef)

# 2. hand-write the attachment and pass it through
echo '{"attachment_id":"a1","control_type":"depth","map_asset_ref":"'$ref'","strength":0.8}' > cn.json
st gen "..." --controlnet-file ./cn.json
```

`--controlnet '<json>'` (repeatable, inline) and config presets
(`--controlnet @name`) are the other two ways to supply attachments. All three
merge with `--control-image` entries into a single `controlnets` list.

### Requirements

ControlNet execution runs **only on the CUDA mode-system backend**. A CPU/RKNN
backend reports `ControlNet provider not yet implemented on this backend`. The
active mode must also enable ControlNet in its `controlnet_policy` block.
