# Scripts

Utility scripts for ControlNet preprocessor image generation.

---

## Install

Run the scripts directly with `python scripts/<name>.py`, or install them as
console commands:

```bash
# installs st-depth-map, st-pose-map, and st-canny-map onto PATH
make install-controlnet-scripts            # all extras (depth + pose + canny)
make install-controlnet-scripts EXTRAS=depth   # depth backends only
make install-controlnet-scripts EXTRAS=pose    # pose backends only
make install-controlnet-scripts EXTRAS=canny   # canny backends only

# or directly with pip
pip install "./scripts[all]"
```

After install both forms are equivalent:

```bash
st-canny-map photo.jpg canny.png             # console script
python scripts/canny_map.py photo.jpg canny.png   # direct
```

> On macOS/Apple Silicon, install `torch` via conda **first**
> (`conda install pytorch -c pytorch`); the extras pull only the non-torch deps.

`make install` installs both the `st` CLI and these scripts in one shot.

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
