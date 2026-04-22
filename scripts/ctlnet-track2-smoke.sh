#!/usr/bin/env bash
set -euo pipefail

base_url="http://127.0.0.1:8000"
ws_url=""
explicit_mode=""
skip_ws=0
timeout_s=20

passes=0
failures=0
skips=0

usage() {
  cat <<'EOF'
Usage: scripts/ctlnet-track2-smoke.sh [options]

Track 2 ControlNet smoke for a live local server.

What it checks:
  1. /api/modes exposes ControlNet policy
  2. HTTP canny source -> artifact -> 501 detail.controlnet_artifacts
  3. HTTP depth source -> artifact -> 501 detail.controlnet_artifacts
  4. HTTP emitted map_asset_ref reuse path is accepted to dispatch stub
  5. HTTP disabled-policy request rejects with 400
  6. WS job:submit emits controlnet_artifacts in job:error

Options:
  --base-url <url>   HTTP base URL (default: http://127.0.0.1:8000)
  --ws-url <url>     WS URL (default: derived from --base-url + /v1/ws)
  --mode <name>      Force a single enabled mode for canny/depth checks
  --skip-ws          Skip the websocket smoke
  --timeout <secs>   Per-request timeout (default: 20)
  --help             Show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --base-url)
      base_url="$2"
      shift 2
      ;;
    --ws-url)
      ws_url="$2"
      shift 2
      ;;
    --mode)
      explicit_mode="$2"
      shift 2
      ;;
    --skip-ws)
      skip_ws=1
      shift
      ;;
    --timeout)
      timeout_s="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

pass() {
  passes=$((passes + 1))
  printf 'PASS  %s\n' "$1"
}

fail() {
  failures=$((failures + 1))
  printf 'FAIL  %s\n' "$1" >&2
}

skip() {
  skips=$((skips + 1))
  printf 'SKIP  %s\n' "$1"
}

require_cmd curl
require_cmd jq
require_cmd python

if [ -z "$ws_url" ]; then
  ws_url="$(python - "$base_url" <<'PY'
import sys
from urllib.parse import urlparse

base = sys.argv[1]
u = urlparse(base)
scheme = "wss" if u.scheme == "https" else "ws"
path = u.path.rstrip("/")
print(f"{scheme}://{u.netloc}{path}/v1/ws")
PY
)"
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

make_png() {
  python - "$1" <<'PY'
import io
import sys
from PIL import Image

out_path = sys.argv[1]
buf = io.BytesIO()
Image.new("RGB", (32, 32), color=(64, 96, 160)).save(buf, format="PNG")
with open(out_path, "wb") as f:
    f.write(buf.getvalue())
PY
}

request_json() {
  local method="$1"
  local url="$2"
  local body="${3:-}"
  local body_file="$tmpdir/body.json"
  local status

  if [ -n "$body" ]; then
    status="$(curl -sS --max-time "$timeout_s" -o "$body_file" -w '%{http_code}' \
      -H 'Content-Type: application/json' \
      -X "$method" \
      --data "$body" \
      "$url")"
  else
    status="$(curl -sS --max-time "$timeout_s" -o "$body_file" -w '%{http_code}' \
      -X "$method" \
      "$url")"
  fi

  printf '%s\n' "$status"
}

upload_file() {
  local file_path="$1"
  local body_file="$tmpdir/upload.json"
  curl -sS --max-time "$timeout_s" -o "$body_file" \
    -F "file=@${file_path};type=image/png" \
    "$base_url/v1/upload" >/dev/null
  jq -r '.fileRef // empty' "$body_file"
}

modes_file="$tmpdir/modes.json"
if curl -sS --max-time "$timeout_s" "$base_url/api/modes" >"$modes_file"; then
  pass "GET /api/modes"
else
  fail "GET /api/modes"
  exit 1
fi

if [ -n "$explicit_mode" ]; then
  canny_mode="$explicit_mode"
  depth_mode="$explicit_mode"
else
  canny_mode="$(jq -r '
    .modes
    | to_entries[]
    | select(.value.controlnet_policy.enabled == true)
    | select((.value.controlnet_policy.allowed_control_types // {}) | has("canny"))
    | .key
  ' "$modes_file" | head -n 1)"
  depth_mode="$(jq -r '
    .modes
    | to_entries[]
    | select(.value.controlnet_policy.enabled == true)
    | select((.value.controlnet_policy.allowed_control_types // {}) | has("depth"))
    | .key
  ' "$modes_file" | head -n 1)"
fi

disabled_mode="$(jq -r '
  .modes
  | to_entries[]
  | select((.value.controlnet_policy.enabled // false) == false)
  | .key
' "$modes_file" | head -n 1)"

printf 'INFO  canny_mode=%s\n' "${canny_mode:-<none>}"
printf 'INFO  depth_mode=%s\n' "${depth_mode:-<none>}"
printf 'INFO  disabled_mode=%s\n' "${disabled_mode:-<none>}"

png_path="$tmpdir/source.png"
make_png "$png_path"

source_ref="$(upload_file "$png_path")"
if [ -n "$source_ref" ]; then
  pass "POST /v1/upload"
else
  fail "POST /v1/upload"
  exit 1
fi

http_source_artifact_smoke() {
  local control_type="$1"
  local preprocess_id="$2"
  local mode_name="$3"
  local attach_id="$4"
  local out_body="$tmpdir/${control_type}.json"
  local payload status artifact_ref

  payload="$(jq -n \
    --arg prompt "controlnet smoke ${control_type}" \
    --arg mode "$mode_name" \
    --arg attachment_id "$attach_id" \
    --arg control_type "$control_type" \
    --arg source_ref "$source_ref" \
    --arg preprocess_id "$preprocess_id" \
    '{
      prompt: $prompt,
      mode: $mode,
      controlnets: [
        {
          attachment_id: $attachment_id,
          control_type: $control_type,
          source_asset_ref: $source_ref,
          preprocess: {id: $preprocess_id, options: {}}
        }
      ]
    }'
  )"

  status="$(curl -sS --max-time "$timeout_s" -o "$out_body" -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    -X POST \
    --data "$payload" \
    "$base_url/generate")"

  if [ "$status" != "501" ]; then
    fail "HTTP ${control_type} source->artifact expected 501, got $status"
    return 1
  fi

  if ! jq -e \
    --arg attachment_id "$attach_id" \
    --arg control_type "$control_type" \
    --arg preprocess_id "$preprocess_id" \
    '.detail.error | contains("ControlNet provider not yet implemented") and
     (.detail.controlnet_artifacts | length == 1) and
     (.detail.controlnet_artifacts[0].attachment_id == $attachment_id) and
     (.detail.controlnet_artifacts[0].control_type == $control_type) and
     (.detail.controlnet_artifacts[0].preprocessor_id == $preprocess_id) and
     ((.detail.controlnet_artifacts[0].asset_ref | length) > 0)' \
    "$out_body" >/dev/null; then
    fail "HTTP ${control_type} source->artifact payload shape"
    return 1
  fi

  artifact_ref="$(jq -r '.detail.controlnet_artifacts[0].asset_ref' "$out_body")"
  printf '%s\n' "$artifact_ref"
  pass "HTTP ${control_type} source->artifact"
}

http_map_reuse_smoke() {
  local mode_name="$1"
  local attach_id="$2"
  local emitted_ref="$3"
  local out_body="$tmpdir/reuse.json"
  local payload status

  payload="$(jq -n \
    --arg prompt 'controlnet smoke reuse' \
    --arg mode "$mode_name" \
    --arg attachment_id "$attach_id" \
    --arg emitted_ref "$emitted_ref" \
    '{
      prompt: $prompt,
      mode: $mode,
      controlnets: [
        {
          attachment_id: $attachment_id,
          control_type: "canny",
          map_asset_ref: $emitted_ref
        }
      ]
    }'
  )"

  status="$(curl -sS --max-time "$timeout_s" -o "$out_body" -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    -X POST \
    --data "$payload" \
    "$base_url/generate")"

  if [ "$status" != "501" ]; then
    fail "HTTP map_asset_ref reuse expected 501, got $status"
    return 1
  fi

  if ! jq -e '.detail | type == "string" and contains("ControlNet provider not yet implemented")' \
    "$out_body" >/dev/null; then
    fail "HTTP map_asset_ref reuse payload shape"
    return 1
  fi

  pass "HTTP map_asset_ref reuse"
}

http_disabled_policy_smoke() {
  local mode_name="$1"
  local out_body="$tmpdir/disabled.json"
  local payload status

  payload="$(jq -n \
    --arg prompt 'controlnet smoke disabled policy' \
    --arg mode "$mode_name" \
    --arg source_ref "$source_ref" \
    '{
      prompt: $prompt,
      mode: $mode,
      controlnets: [
        {
          attachment_id: "cn_disabled",
          control_type: "canny",
          source_asset_ref: $source_ref,
          preprocess: {id: "canny", options: {}}
        }
      ]
    }'
  )"

  status="$(curl -sS --max-time "$timeout_s" -o "$out_body" -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    -X POST \
    --data "$payload" \
    "$base_url/generate")"

  if [ "$status" != "400" ]; then
    fail "HTTP disabled policy expected 400, got $status"
    return 1
  fi

  if ! jq -e '.detail | type == "string" and contains("does not enable ControlNet")' \
    "$out_body" >/dev/null; then
    fail "HTTP disabled policy payload shape"
    return 1
  fi

  pass "HTTP disabled policy"
}

switch_mode_for_ws() {
  local mode_name="$1"
  local body status

  body="$(jq -n --arg mode "$mode_name" '{mode: $mode}')"
  status="$(request_json POST "$base_url/api/modes/switch" "$body")"
  if [ "$status" = "200" ]; then
    sleep 1
    return 0
  fi
  return 1
}

ws_artifact_smoke() {
  local source_ref_arg="$1"
  local out_file="$tmpdir/ws.json"

  if ! WS_URL="$ws_url" SOURCE_REF="$source_ref_arg" OUT_FILE="$out_file" TIMEOUT_S="$timeout_s" python - <<'PY'
import asyncio
import json
import os
import sys

import websockets

ws_url = os.environ["WS_URL"]
source_ref = os.environ["SOURCE_REF"]
out_file = os.environ["OUT_FILE"]
timeout_s = float(os.environ["TIMEOUT_S"])

payload = {
    "type": "job:submit",
    "id": "ctlnet-smoke",
    "jobType": "generate",
    "params": {
        "prompt": "controlnet smoke ws",
        "size": "512x512",
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "seed": 12345678,
        "controlnets": [
            {
                "attachment_id": "cn_ws",
                "control_type": "canny",
                "source_asset_ref": source_ref,
                "preprocess": {"id": "canny", "options": {}},
            }
        ],
    },
}


async def main() -> int:
    async with websockets.connect(ws_url, open_timeout=timeout_s, close_timeout=timeout_s) as ws:
        ack = None
        err = None
        end_time = asyncio.get_running_loop().time() + timeout_s

        while asyncio.get_running_loop().time() < end_time:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout_s))
            if msg.get("type") == "status":
                break

        await ws.send(json.dumps(payload))

        while asyncio.get_running_loop().time() < end_time:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout_s))
            if msg.get("type") == "job:ack":
                ack = msg
            elif msg.get("type") == "job:error":
                err = msg
                break

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({"ack": ack, "error": err}, f)
        return 0 if ack is not None and err is not None else 1


raise SystemExit(asyncio.run(main()))
PY
  then
    fail "WS canny source->artifact transport"
    return 1
  fi

  if ! jq -e '
    .ack.type == "job:ack" and
    .error.type == "job:error" and
    (.error.error | contains("ControlNet provider not yet implemented")) and
    (.error.controlnet_artifacts | length == 1) and
    (.error.controlnet_artifacts[0].attachment_id == "cn_ws") and
    ((.error.controlnet_artifacts[0].asset_ref | length) > 0)
  ' "$out_file" >/dev/null; then
    fail "WS canny source->artifact payload shape"
    return 1
  fi

  pass "WS canny source->artifact"
}

canny_artifact_ref=""
if [ -n "$canny_mode" ]; then
  canny_artifact_ref="$(http_source_artifact_smoke "canny" "canny" "$canny_mode" "cn_http_canny" | tail -n 1)"
else
  skip "HTTP canny source->artifact (no enabled canny mode)"
fi

if [ -n "$depth_mode" ]; then
  http_source_artifact_smoke "depth" "depth" "$depth_mode" "cn_http_depth" >/dev/null
else
  skip "HTTP depth source->artifact (no enabled depth mode)"
fi

if [ -n "$canny_artifact_ref" ]; then
  http_map_reuse_smoke "$canny_mode" "cn_http_reuse" "$canny_artifact_ref"
else
  skip "HTTP map_asset_ref reuse (no canny artifact ref)"
fi

if [ -n "$disabled_mode" ]; then
  http_disabled_policy_smoke "$disabled_mode"
else
  skip "HTTP disabled policy (no disabled mode discovered)"
fi

if [ "$skip_ws" -eq 1 ]; then
  skip "WS canny source->artifact (--skip-ws)"
elif [ -z "$canny_mode" ]; then
  skip "WS canny source->artifact (no enabled canny mode)"
else
  if switch_mode_for_ws "$canny_mode"; then
    ws_artifact_smoke "$source_ref"
  else
    skip "WS canny source->artifact (failed to switch mode first)"
  fi
fi

printf '\nSUMMARY passes=%d skips=%d failures=%d\n' "$passes" "$skips" "$failures"

if [ "$failures" -ne 0 ]; then
  exit 1
fi
