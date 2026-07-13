#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/bin" "$tmp/in" "$tmp/out"
touch "$tmp/in/landscape.png" "$tmp/in/portrait.jpg" "$tmp/in/square.webp"

cat >"$tmp/bin/magick" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "identify" ]]; then
  file="${@: -1}"
  case "$(basename "$file")" in
    landscape.png) printf "1600 900" ;;
    portrait.jpg) printf "900 1600" ;;
    square.webp) printf "1000 1000" ;;
    *) printf "512 512" ;;
  esac
  exit 0
fi

printf '%q ' "$@" >>"$MAGICK_LOG"
printf '\n' >>"$MAGICK_LOG"
touch "${@: -1}"
SH
chmod +x "$tmp/bin/magick"

export PATH="$tmp/bin:$PATH"
export MAGICK_LOG="$tmp/magick.log"

"$repo_root/scripts/resize-for-model.sh" "$tmp/in" --profile sd15 --out "$tmp/out"

grep -F -- "$tmp/in/landscape.png -auto-orient -resize 768x512\\^ -gravity center -extent 768x512 $tmp/out/landscape.png" "$MAGICK_LOG" >/dev/null
grep -F -- "$tmp/in/portrait.jpg -auto-orient -resize 512x768\\^ -gravity center -extent 512x768 $tmp/out/portrait.jpg" "$MAGICK_LOG" >/dev/null
grep -F -- "$tmp/in/square.webp -auto-orient -resize 768x768\\^ -gravity center -extent 768x768 $tmp/out/square.webp" "$MAGICK_LOG" >/dev/null

: >"$MAGICK_LOG"
rm -rf "$tmp/out"
mkdir -p "$tmp/out"

"$repo_root/scripts/resize-for-model.sh" "$tmp/in" --profile sdxl --mode pad --out "$tmp/out"

grep -F -- "$tmp/in/landscape.png -auto-orient -resize 1152x896 -background black -gravity center -extent 1152x896 $tmp/out/landscape.png" "$MAGICK_LOG" >/dev/null
grep -F -- "$tmp/in/portrait.jpg -auto-orient -resize 896x1152 -background black -gravity center -extent 896x1152 $tmp/out/portrait.jpg" "$MAGICK_LOG" >/dev/null
grep -F -- "$tmp/in/square.webp -auto-orient -resize 1024x1024 -background black -gravity center -extent 1024x1024 $tmp/out/square.webp" "$MAGICK_LOG" >/dev/null

"$repo_root/scripts/resize-for-model.sh" "$tmp/in" --profile bogus --out "$tmp/out" 2>"$tmp/err" && {
  echo "expected bad profile to fail" >&2
  exit 1
}
grep -F "profile must be one of" "$tmp/err" >/dev/null
