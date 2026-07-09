#!/usr/bin/env bash
# Live dtype-regression verification for STABL-crdsypux (from_pipe fp32 upcast).
#
# Runs the four generation shapes that exercise every from_pipe call site and
# the shared-module dtype contract, in an order chosen so the poisoning bug
# (if present) is *provoked and then detected*:
#
#   1. txt2img            — baseline; also the canary re-run at the end
#   2. img2img            — plain init-image path (no from_pipe)
#   3. txt2img+controlnet — _build_controlnet_pipe from_pipe site
#   4. combined           — img2img+ControlNet from_pipe site
#   5. txt2img AGAIN      — the decisive probe: on a vandalized pipe, steps 3/4
#                           upcast shared modules in place, so this re-run fails
#                           even though step 1 passed
#
# VRAM is snapshotted before/after: an fp32-poisoned pipe roughly doubles
# allocated_gb versus the fp16 baseline.
#
# Usage: scripts/st-dtype-live-verify.sh [--server URL] [--mode NAME] [--keep]
# Exits 0 if all five probes pass, 1 otherwise.

set -uo pipefail

server="${ST_SERVER:-http://enigma.lan:4200}"
mode="lcm-general"
keep=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --server) server="$2"; shift 2 ;;
    --mode) mode="$2"; shift 2 ;;
    --keep) keep=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d /tmp/st-dtype-verify.XXXXXX)"
[ "$keep" = 1 ] || trap 'rm -rf "$work"' EXIT

st() { (cd "$repo_root/cli/go" && go run ./cmd/st --server "$server" "$@"); }

vram() {
  curl -s -m 10 "$server/api/models/status" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"{d['vram']['allocated_gb']:.2f} GB allocated, mode={d['current_mode']}\")
" 2>/dev/null || echo "(status unavailable)"
}

# Fixture images. Content is irrelevant to the dtype contract; what matters is
# that init image and control map share an aspect ratio (2% rule) and the
# request size. A gradient makes depth-map semantics vaguely plausible.
python3 - "$work" <<'EOF'
import sys
from PIL import Image
work = sys.argv[1]
w, h = 512, 768
init = Image.new("RGB", (w, h))
init.putdata([(x * 255 // w, y * 255 // h, 96) for y in range(h) for x in range(w)])
init.save(f"{work}/init.png")
depth = Image.new("L", (w, h))
depth.putdata([min(255, (x + y) * 255 // (w + h)) for y in range(h) for x in range(w)])
depth.convert("RGB").save(f"{work}/depth-map.png")
EOF

echo "== server: $server  mode: $mode  workdir: $work"
echo "== VRAM before: $(vram)"

init_ref="$(st upload "$work/init.png" | tail -1)"
echo "== init fileref: $init_ref"

declare -a names cmds
names=(txt2img img2img txt2img+controlnet combined txt2img-again)
pass=0; fail=0; results=()

run_probe() {
  local name="$1"; shift
  if st gen "$@" >"$work/$name.log" 2>&1; then
    results+=("PASS  $name"); pass=$((pass+1))
  else
    results+=("FAIL  $name — $(grep -m1 'error:' "$work/$name.log" || tail -1 "$work/$name.log")")
    fail=$((fail+1))
  fi
}

common=(--size 512x768 --steps 6 --cfg 2.4 --mode "$mode" --seed 42)

run_probe txt2img            "sleeping figure under purple blankets, dtype probe" "${common[@]}" --outfile "$work/out-1.png"
run_probe img2img            "sleeping figure under purple blankets, dtype probe" "${common[@]}" --init-image "fileref:$init_ref" --outfile "$work/out-2.png"
run_probe txt2img+controlnet "sleeping figure under purple blankets, dtype probe" "${common[@]}" --control-image "depth:$work/depth-map.png" --control-strength 1.0 --outfile "$work/out-3.png"
run_probe combined           "sleeping figure under purple blankets, dtype probe" "${common[@]}" --init-image "fileref:$init_ref" --control-image "depth:$work/depth-map.png" --control-strength 1.0 --outfile "$work/out-4.png"
run_probe txt2img-again      "sleeping figure under purple blankets, dtype probe" "${common[@]}" --outfile "$work/out-5.png"

echo "== VRAM after:  $(vram)"
echo
printf '%s\n' "${results[@]}"
echo
if [ "$fail" -eq 0 ]; then
  echo "RESULT: PASS ($pass/5) — no dtype poisoning observed"
  exit 0
else
  echo "RESULT: FAIL ($fail/5 failed) — if txt2img passed but txt2img-again failed, a from_pipe path vandalized the shared pipe (STABL-crdsypux)"
  exit 1
fi
