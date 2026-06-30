# Scripts

Utility scripts for ControlNet preprocessor image generation.

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
pip install mediapipe        # mediapipe only
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

Output: RGB PNG with colored keypoints and limb connections on a black background.
`--show-keypoints` (mediapipe) outputs white dots on black with no connections.

---

## Model comparison

### Depth

| Model | Speed | Quality | Notes |
|---|---|---|---|
| `depth-anything` small | Fast | Good | Best default choice |
| `depth-anything` large | Slow | Best | Use when detail matters |
| `midas` | Fast | Good | Older; reliable fallback |
| `zoe` | Medium | Good | Better metric (real-world scale) depth |

### Pose

| Model | Quality | Notes |
|---|---|---|
| `dwpose` | Best | Recommended; more accurate than OpenPose |
| `openpose` | Good | Classic ControlNet preprocessor; supports face+hands |
| `mediapipe` | Fine | No model download; fast; 33-keypoint body only |
