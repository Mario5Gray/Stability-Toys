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
make install-controlnet-scripts EXTRAS=vector  # st-vector-map with VTracer and resvg-py (not part of all)

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

Trace PNG or JPEG masks, edge maps, and image silhouettes into SVG with physical dimensions.
Use PNG for lossless mask pixels. Select exactly one physical dimension.
Supported pixel modes: `1`, `L`, `LA`, `P`, `RGB`, and `RGBA`.
Convert other modes, including 16-bit grayscale and CMYK, to supported 8-bit input before tracing.
Use RGBA PNG when conversion must preserve alpha.

```bash
st-vector-map mask.png silhouette.svg --input-kind mask --width-mm 100
st-vector-map edges.png detail.svg --input-kind edges --width-mm 100 \
  --line-width-mm 0.8 --include-mask allowed.png --exclude-mask removed.png --json
st-vector-map photo.png silhouette.svg --input-kind image --mask reviewed-mask.png --width-mm 100
st-vector-map logo.png silhouette.svg --input-kind image --threshold 128 --invert --width-mm 100
```

Default material has luminance >= 128. `--invert` reverses material selection.
`--alpha` selects alpha >= 128 instead. Inputs with unused alpha produce warnings.
Palette transparency supports alpha selection. Inputs without alpha reject `--alpha`.

EXIF orientation applies before thresholding and dimension checks.
External masks must match oriented source dimensions. Source inversion and alpha options do not change their luminance selection.
`--include-mask` limits material. `--exclude-mask` removes material and always wins.
Both constraints apply before and after edge expansion.

Image mode requires exactly one silhouette method. Without a method, the command exits 2.
`--mask PATH` selects white pixels (luminance >= 128) of PATH. The source sets the canvas only.
`--alpha` selects source alpha >= 128.
`--threshold N` selects source luminance >= N. N is an integer from 0 to 255.
`--invert` reverses every image method.
Selection happens at oriented source resolution, before the common binary resize.
The mask file must match oriented source dimensions. Mask-file alpha is ignored with an `alpha_ignored` warning.
With `--mask`, source alpha selects nothing and gives no warning. With `--threshold`, source alpha gives the warning.
Recipe fields `mask`, `alpha`, and `threshold` form one group. A later source that selects a method replaces the earlier method.
For example, `--threshold 200` replaces recipe `alpha: true`. `--no-alpha` clears inherited alpha only.
Two methods from one source exit 2. Mask and edge modes reject `--mask` and `--threshold` with exit 2.
An empty silhouette after constraints exits 1 before tracing.
The manifest records the method under `preparation.silhouette` and names the layer `silhouette`.

**Layered image mode**

`--layers ROLES` selects image-mode roles. ROLES is a comma-separated list of `silhouette`, `structure`, and `detail`.
The list must include `silhouette`. Name each role once. Output order is always silhouette, structure, detail.
An empty, unknown, duplicate, or silhouette-free list exits 2. Mask and edge modes reject `--layers` and role settings with exit 2.
Without `--layers`, image mode writes one silhouette SVG as before. `--layers silhouette` writes a layered bundle with one layer.

```bash
st-vector-map photo.png relief.svg --input-kind image --mask reviewed-mask.png --width-mm 100 \
  --layers silhouette,structure,detail --structure-width-mm 0.8 --detail-width-mm 0.5 --preview
st-vector-map photo.png relief.svg --input-kind image --mask reviewed-mask.png --width-mm 100 \
  --layers silhouette,structure --structure-map structure.png
```

By default, Canny on the source makes structure and detail. Structure uses 100/200 with blur 3. Detail uses 50/100 with blur 3.
`--structure-map PATH` and `--detail-map PATH` use white pixels of PATH instead of Canny.
A CLI map replaces the recipe Canny settings of that role.
`--structure-width-mm` and `--detail-width-mm` widen that role. A CLI width replaces the recipe width.
A role setting for a role outside the resolved selection exits 2. This includes the default run without `--layers`.
The command never accepts a setting that it would ignore.

Recipe `layers` is a JSON list of role names. Recipe `structure` and `detail` objects accept these fields only:

| Field | Value |
|---|---|
| `source` | `canny` (default) or `map` |
| `path` | Map path. Required for `map`. Rejected for `canny` |
| `canny` | Object with `low_threshold`, `high_threshold`, and `blur`. Rejected for `map` |
| `width_mm` | Positive nominal line width |
| `gap_close_mm` | Positive width of the widest gap to close. Recipe only, no CLI option |
| `include_mask`, `exclude_mask` | Role constraint paths |

An omitted Canny value takes the role default. An omitted detail `blur` stays 3, the same as structure.
Detail must use the structure blur. A different blur moves outlines about 1 px and leaves slivers after structure removal.
`gap_close_mm` is the widest gap to close, not a kernel width. The radius is `r = max(1, ceil(gap_px / 2))`.
Row and column kernels are clamped to the canvas. The manifest records requested and achieved gap widths.
Unknown fields, wrong types, booleans as numbers, and invalid thresholds or blur exit 2. Nonfinite or nonpositive lengths also exit 2.
Role paths resolve relative to the recipe directory. Every role map and role constraint is a hashed input. Each path is read once.

Every role uses the silhouette canvas and origin. Structure and detail are clipped to the silhouette. Structure is removed from detail.
Each nonempty selected role becomes `<stem>.layers/<role>.svg`. These separate normalized SVGs are the authoritative SCAD inputs.
`<stem>.svg` holds the same layers as `<g id="silhouette">`, `<g id="structure">`, and `<g id="detail">` groups. Use it to view and edit.
All files state the same millimetre root size and pixel `viewBox`. Each group copies its layer paths without change.
An empty optional role stays selected. It gets a `role_empty` warning, a manifest entry with `svg: null`, and no SVG file.
An empty silhouette exits 1.
`--json` success adds `artifacts.layers`. It maps each role to the path of a file from this run only.
`counts.layers` counts published layers. `counts.paths` sums their paths.

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
Processing failure publishes no successful bundle. Publication failure can leave files without a completion manifest.
`--overwrite` never permits replacing source, recipe, or constraint inputs.

**Artifact bundle**

Conversion publishes `<stem>.svg` and `<stem>.vector.json` for requested `<stem>.svg` destination.
Manifest records consumed input hashes, installed package versions, resolved settings, physical canvas, counts, warnings, and output hashes.
Standalone layer has `id: standalone` and `height_mm: null`. Conversion assigns no relief height.
Manifest contains no timestamp or self-hash. A bundle is complete only when every listed artifact hash matches.
Source, recipe, and constraint hashes describe exact bytes consumed during parsing and preparation.
`--json` success includes `artifacts.svg` and `artifacts.manifest`.

All requested output collisions fail before preparation. Default publication uses hard links and refuses competing files atomically.
Filesystems without hard-link support return a clear error. Explicit `--overwrite` permits replacement of owned output names with `os.replace`.
No automatic overwrite fallback occurs. Input-alias checks still apply.
Publication stages files on destination filesystem, invalidates old manifest, replaces outputs, then publishes new manifest last.
Several file publications do not form an atomic transaction. Unrelated files remain unchanged.
If staging cleanup fails after manifest publication, publisher removes new manifest before reporting failure.

A layered run owns all three `<stem>.layers/<role>.svg` names, selected or not. Without `--overwrite`, any existing name exits 2.
`<stem>.layers` must be a directory. A symlink fails.
With `--overwrite`, publication invalidates the old manifest and replaces files. Then it removes owned files that this run did not publish.
The new manifest publishes last. Unrelated files and the `.layers` directory remain.
A later standalone run does not own `.layers/`, so an old layer directory can remain.
The current manifest identifies the current bundle. Use only files that it lists with matching hashes.
A layered manifest lists every selected role under `layers` with `id`, `height_mm: null`, `mode: null`, `svg`, and `counts`.
`roles` records each optional role request with resolved Canny values, widths, expansion, and gap closing.
`counts` sums the layer counts and adds `layers` and `combined_bytes`. Standalone manifests keep the two-field layer entry.

`--debug-bundle` adds `<stem>.debug/mask.png` and `<stem>.debug/recipe.json`.
Successful manifest lists both hashes. No other debug filenames belong to bundle.
Symlinked debug directories fail. Overwrite preserves unrelated files within ordinary debug directory.
Replay recipe traces final prepared mask without repeating inversion, constraints, resizing, or edge expansion.

A layered `--debug-bundle` adds `<stem>.debug/<role>.png` for each selected role and `<stem>.debug/recipe.json`.
A layered run owns all three role PNG names. Overwrite removes unselected role PNGs. It does not own `mask.png`.
The replay recipe uses `silhouette.png` as source and silhouette mask, and each role PNG as a map.
Replay traces the saved masks. It does not repeat Canny, constraints, gap closing, or widening.

```bash
st-vector-map edges.png detail.svg --input-kind edges --width-mm 100 --line-width-mm 0.8 --debug-bundle
st-vector-map replay.svg --recipe detail.debug/recipe.json
```

Tracing failure retains prepared debug files when their publication succeeds. Failed JSON result reports `artifacts.debug`.
Failure diagnostic directs retry with `--overwrite`. Retry without overwrite refuses retained debug files.
Before retained debug files replace old artifacts, publisher invalidates old completion manifest.
If debug publication fails, diagnostic preserves both original processing error and debug publication error.

**Preview**

`--preview` adds `<stem>.preview.png` to the bundle. The manifest lists its hash.
The image has three labeled panels at processing resolution, one image pixel per processing pixel:

- Prepared material mask. Material is black.
- Rendered vector output. resvg-py 0.5.0 renders the published SVG with default anti-aliasing.
- Vector overlay on the oriented source, resized to the processing canvas with nearest-neighbour.

The overlay legend shows mask and vector overlap, vector-only material, and mask-only material.
Compare the first two panels to see what fitting changed. The mask panel alone is not proof of vector output.
Edge mode uses the supplied edge image as source. The preview has no access to an earlier Canny input photograph.

A layered preview has one column for each selected role. The first row shows the prepared mask.
The second row shows the render of the published `<stem>.layers/<role>.svg`.
An empty role gets a gray vector panel labelled `<Role> vector: empty, no SVG`. Nothing is rendered for it.
The third row shows the published roles on the source in role colours. Silhouette is sky blue, structure vermillion, and detail bluish green.
Later roles draw over earlier roles. The legend lists the published roles. Manifest `preview` adds `layers` and `legend`.

The renderer gets the published paths under a pixel root size, so one `viewBox` unit is one pixel.
Do not render the millimetre root with `dpi = 25.4 / mm_per_px`. resvg-py 0.5.0 converts millimetres in float32.
At 25.4/96 mm/px, 256 px becomes 255.99998 px, and edges move.

Labels use a Pillow embedded font. The command loads no system font.
With FreeType on Pillow 10.1 or later, labels use embedded Aileron at 12 px. Otherwise they use the embedded bitmap font.
Manifest `preview` records renderer version, rendering settings, resolution, Pillow version, font class, and FreeType version.
FreeType version is `null` when Pillow has no FreeType. Without `--preview`, manifest `preview` is `null`.
Same inputs, settings, dependency versions, and build give the same bytes. Bytes across platforms can differ.

`--preview` without resvg-py fails with exit 1 before preparation. Renderer failure publishes no SVG, preview, or manifest.
With `--debug-bundle`, renderer failure retains debug files like tracing failure.
Conversion without `--preview` never imports resvg-py. `--json` success adds `artifacts.preview`.

**SVG output and inspection**

The SVG root states the canvas in `mm` with a pixel `viewBox="0 0 W H"`. OpenSCAD imports this form at the stated size.
Normalization changes only the root size. Paths, fill, transforms, and element order stay as VTracer wrote them.
Lengths are plain decimals, for example `100mm` or `0.0000002mm`. A positive size never becomes `0mm`.

Before publication, the command checks VTracer output against the pinned polygon dialect:

- One `svg` root in the SVG namespace, with `g` and `path` elements only. Paths are leaf elements.
- Root attributes `version`, `width`, `height`, and `viewBox` only. `g` and `path` accept `id`, `transform`, `fill`, and `fill-rule`. `path` also accepts `d`.
- Black fill (`black`, `#000`, `#000000`) and `nonzero` fill rule only.
- One `translate(x)` or `translate(x,y)` transform per element. Nested translations add.
- Path data uses absolute `Mx,y`, `Lx,y`, and `Z` tokens only, separated by spaces. Each subpath closes with an explicit `Z`.
- Coordinates are finite ASCII decimal numbers. NaN, infinity, overflow, and underscores fail.
- Raw `width` and `height` equal the processing size in pixels. A raw `viewBox` must be `0 0 W H`.
- A raw `viewBox` accepts spaces or commas between numbers. Empty comma fields fail.
- Group nesting above 256 levels fails with a clear error before XML serialization.

DTDs, processing instructions, images, `use`, `defs`, scripts, masks, clipping, styles, strokes, and opacity fail.
The command never removes unsupported content and never repairs geometry. It reports incompatible upstream output with exit 1.
A pass does not prove that paths have no self-intersections.

VTracer can return closed subpaths with one or two distinct points, for example `M3,4 Z`.
Inside a path with a valid polygon, these subpaths stay unchanged. They add no area.
Each layer with such subpaths gets one `degenerate_subpaths` warning with the count.
A path must contain at least one polygon with three non-collinear points.
When a path contains only points or lines, the command fails. The message reads "VTracer returned N paths containing only points or lines."
A higher `--max-res` can help only when processing resolution can increase. A higher `filter_speckle` removes small islands on purpose.
Measured examples, not guarantees: on 64x64 random-noise masks, speckle 4 cleared these paths and speckle 2 did not.
Speckle 1 gave the same output as speckle 0.

**Limits**

| Option / recipe field | Default | Measures |
|---|---:|---|
| `--max-svg-bytes` / `max_svg_bytes` | 20971520 | UTF-8 bytes. Checked on raw output before XML parsing, and again on normalized output |
| `--max-paths` / `max_paths` | 10000 | `path` elements per layer |
| `--max-path-commands` / `max_path_commands` | 1000000 | `M`, `L`, and `Z` commands per layer, including degenerate subpaths |

Limits are wrapper policy. They are not measured VTracer or OpenSCAD capacities. They never go to VTracer.
They cannot stop VTracer from allocating its own output.
Values must be positive integers. Zero, negative, boolean, decimal, and null values return exit 2.
A count over its limit returns exit 1. The message gives the count and the limit.
A count found while the check stops early is a lower bound, for example `path count at least 101 exceeds limit 100`.
The message suggests a lower `--max-res` or a different recipe `vtracer.filter_speckle` (0..128). Both can change geometry.
The command never changes these settings or retries.

Recipe fields match option names with underscores: `mask`, `threshold`, `include_mask`, `exclude_mask`, `line_width_mm`, `max_res`, `max_svg_bytes`, `max_paths`, and `max_path_commands`.
Layered image mode adds `layers`, `structure`, and `detail`. See layered image mode above.
Recipes require `schema_version: 1`. Recipe paths resolve relative to recipe directory. Explicit CLI options override recipe values.

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

> Historical S2.2 interface summary below. Current mask/edge behavior and artifact bundle contract appear under [st-vector-map](#st-vector-map).

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
