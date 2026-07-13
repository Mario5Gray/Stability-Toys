#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/resize-for-model.sh IMAGE_DIR --out OUT_DIR [--profile sd15|sdxl] [--mode crop|pad]

Resize each supported image in IMAGE_DIR to the nearest model-friendly aspect bucket.

Options:
  --profile sd15   Use SD 1.5 buckets: 512x512, 768x512, 512x768, 768x768 (default)
  --profile sdxl   Use SDXL buckets: 1024x1024, 1152x896, 896x1152
  --mode crop      Crop-to-fill with centered extent (default)
  --mode pad       Preserve the full image and pad with black
  --out DIR        Output directory
  -h, --help       Show this help

Supported image extensions: png, jpg, jpeg, webp, bmp, tif, tiff, heic
EOF
}

die() {
  echo "$*" >&2
  exit 2
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

input_dir=$1
shift

profile=sd15
mode=crop
out_dir=

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || die "--profile requires a value"
      profile=$2
      shift 2
      ;;
    --mode)
      [[ $# -ge 2 ]] || die "--mode requires a value"
      mode=$2
      shift 2
      ;;
    --out)
      [[ $# -ge 2 ]] || die "--out requires a value"
      out_dir=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ -d "$input_dir" ]] || die "image directory does not exist: $input_dir"
[[ -n "$out_dir" ]] || die "--out is required"

case "$profile" in
  sd15)
    targets=(512x512 768x512 512x768 768x768)
    ;;
  sdxl)
    targets=(1024x1024 1152x896 896x1152)
    ;;
  *)
    die "profile must be one of: sd15, sdxl"
    ;;
esac

case "$mode" in
  crop|pad)
    ;;
  *)
    die "mode must be one of: crop, pad"
    ;;
esac

magick_bin=${MAGICK_BIN:-magick}
command -v "$magick_bin" >/dev/null 2>&1 || die "magick not found on PATH; install ImageMagick or set MAGICK_BIN"

mkdir -p "$out_dir"

is_supported_image() {
  local file=$1
  local ext
  ext=$(printf '%s' "${file##*.}" | tr '[:upper:]' '[:lower:]')
  case "$ext" in
    png|jpg|jpeg|webp|bmp|tif|tiff|heic)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

choose_target() {
  local image_w=$1
  local image_h=$2
  local best=
  local best_diff=
  local best_area=0
  local target target_w target_h diff area better

  for target in "${targets[@]}"; do
    target_w=${target%x*}
    target_h=${target#*x}
    area=$((target_w * target_h))
    diff=$(awk -v iw="$image_w" -v ih="$image_h" -v tw="$target_w" -v th="$target_h" 'BEGIN {
      d = (iw / ih) - (tw / th)
      if (d < 0) d = -d
      printf "%.12f", d
    }')

    if [[ -z "$best" ]]; then
      better=1
    else
      better=$(awk -v d="$diff" -v bd="$best_diff" -v area="$area" -v ba="$best_area" 'BEGIN {
        if (d < bd) print 1
        else if (d == bd && area > ba) print 1
        else print 0
      }')
    fi

    if [[ "$better" == 1 ]]; then
      best=$target
      best_diff=$diff
      best_area=$area
    fi
  done

  printf '%s\n' "$best"
}

found=0
while IFS= read -r -d '' image; do
  is_supported_image "$image" || continue
  found=1

  dimensions=$("$magick_bin" identify -format '%w %h' "$image")
  read -r image_w image_h <<<"$dimensions"
  [[ -n "${image_w:-}" && -n "${image_h:-}" ]] || die "could not read dimensions for: $image"

  target=$(choose_target "$image_w" "$image_h")
  output="$out_dir/$(basename "$image")"

  if [[ "$mode" == crop ]]; then
    "$magick_bin" "$image" -auto-orient -resize "${target}^" -gravity center -extent "$target" "$output"
  else
    "$magick_bin" "$image" -auto-orient -resize "$target" -background black -gravity center -extent "$target" "$output"
  fi

  printf '%s -> %s (%s, %s)\n' "$image" "$output" "$target" "$mode"
done < <(find "$input_dir" -maxdepth 1 -type f -print0)

if [[ "$found" -eq 0 ]]; then
  echo "no supported images found in: $input_dir" >&2
  exit 1
fi
