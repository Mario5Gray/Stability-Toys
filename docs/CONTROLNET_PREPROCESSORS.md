# ControlNet Preprocessors

The server builds control maps from source images. The seam lives in
`server/controlnet_preprocessors.py`. A request names a preprocessor in its
`preprocess` block. `preprocess_controlnet_attachments()` resolves the source
asset, dispatches the preprocessor, and stores the map in the `control_map`
bucket.

## Registered preprocessors

| id | control type | runtime requirement | output |
|----|--------------|---------------------|--------|
| `canny` | `canny` | `opencv-python-headless` | white edges on black |
| `depth` | `depth` | `transformers` | grayscale depth as RGB |
| `pose` | `pose` | `controlnet-aux==0.0.10` | OpenPose skeleton on black |

`preprocess.id` must match a registered id. An unknown id raises
`ValueError("unknown preprocessor ...")`.

## Pose preprocessor

### Map format

The map follows the format of the HunyuanDiT Pose ControlNet checkpoint
`HunyuanDiT-v1.1-ControlNet-Pose`. That checkpoint ships its own sample map as
`pose.jpg`. The format is:

- a black background
- a colored 18-keypoint body skeleton with colored joint dots
- white face keypoint dots
- no hands

`controlnet_aux.OpenposeDetector` produces this format. `DWposeDetector` needs
`mmdet` and `mmpose`, so the server does not use it. MediaPipe draws a different
style, so the server does not use it either.

### Options

| option | values | default | meaning |
|--------|--------|---------|---------|
| `model` | `openpose` | `openpose` | only supported model |
| `parts` | `body`, `face` | `body,face` | skeleton parts to draw |
| `device` | `cpu` | `cpu` | execution device |
| `max_res` | 64 to 4096 | 1024 | longest edge before inference |

An unsupported `model`, an unsupported `parts` value other than `body` or
`face`, or a `device` other than `cpu` raises `ValueError`.
`hands` is not a supported part. A null `parts` value uses the default. Any
`parts` value that is neither a string nor a list of strings raises
`ValueError("parts must be a string or a list of strings ...")`.

### Execution device

Pose detection runs on CPU only. `OpenposeDetector` accepts a `.to(device)`
call, but no accelerator path is implemented or verified. This class passes no
device to the detector. A non-cpu `device` option raises
`ValueError("pose preprocessing runs on cpu only. ...")` instead of running on
the wrong device.

Every produced map records its execution device. The metadata carries
`model`, `parts`, `device`, and `max_res`. The metadata reaches two places:

- the stored `control_map` asset metadata, beside `width`, `height` and
  `preprocessor_id`
- the `metadata` field of the `ControlNetArtifactRef` the request returns

## Reserved metadata

`preprocess_controlnet_attachments()` owns seven provenance keys:
`attachment_id`, `control_type`, `source_asset_ref`, `preprocessor_id`,
`width`, `height`, and `media_type`. A preprocessor may add its own metadata
keys. A preprocessor that returns one of those seven keys raises
`ValueError("preprocessor ... may not set reserved metadata keys: ...")`
instead of overwriting provenance.

### Runtime requirement

The server image installs `controlnet-aux==0.0.10` from `requirements.txt`. The
import is lazy. If the package is absent, the first pose request raises
`RuntimeError` that names the pinned install command:
`pip install 'controlnet-aux==0.0.10'`.

### Model weights

`OpenposeDetector.from_pretrained("lllyasviel/Annotators")` downloads three
checkpoints on first use. They total about 490 MB and land in the Hugging Face
cache (`HF_HOME`). The hand checkpoint downloads even though the server draws no
hands. Pre-seed that cache to avoid the download at request time. The download
happens once per cache.

### Scope

This preprocessor produces a pose map. It does not run HunyuanDiT Pose
ControlNet inference, and it does not prove MPS acceptance. Backend execution
stays in STABL-poidjrjv. Live HunyuanDiT Pose ControlNet acceptance stays in
STABL-ziorpomq.

## Command-line parity

`st-pose-map` in `scripts/pose_map.py` builds pose maps outside the server. It
supports three models. Its `--device` flag accepts only `cpu`, because that
script also passes no device to its detectors. A non-cpu value exits 1 before
any detection runs. `--model dwpose` needs `mmdet` and `mmpose` and prints an
install hint when they are missing.
