#!/usr/bin/env bash
# Start the Z-Image certification server on enigma (STABL-vnakbitb, plan Task 8).
# Run on the Mac: ssh enigma.lan 'bash -s' < scripts/zimage-cert-enigma.sh
# The script starts one container from the published image. It does not build.
# Prerequisites on enigma:
#   - Stop the dev container first. Both use port 4200 and the GPU.
#   - Detached worktree at $REPO. Config and output at $OUT (conf, images).
#   - $OUT/conf/modes.yml uses model_root /models and lora_root /models/loras.
#   - Use an st config without steps and cfg. Config defaults override mode defaults.
# Image start.sh must run 'python -m server.run'. The script checks this.
# Do not use 'uvicorn --log-config'. Only the live-test image has that file.
# USER, LOGNAME and HOME are set because UID 1000 has no passwd entry in the image.
# Status: the user waived the frozen P1-P5 runs. See the plan doc, Task 8.
set -euo pipefail

readonly CONTAINER='stability-toys-cert-STABL-vnakbitb'
readonly REPO='/home/hdd/workspace/Stability-Toys-cert-STABL-vnakbitb'
readonly OUT='/home/hdd/generations/STABL-vnakbitb'

if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "Certification container exists: $CONTAINER" >&2
  exit 1
fi

readonly IMAGE='harbor.lan/stability-toys:vnakbitb'
if ! docker run --rm --entrypoint cat "$IMAGE" /app/start.sh | grep -q 'python -m server.run'; then
  echo "Image start.sh does not run 'python -m server.run': $IMAGE" >&2
  exit 1
fi

exec docker run -d \
  --name "$CONTAINER" \
  --user 1000:1000 \
  --gpus all \
  --network observ-net \
  --publish 4200:4200 \
  --security-opt label=disable \
  --env USER=hdd \
  --env LOGNAME=hdd \
  --env HOME=/store \
  --env QUEUE_MAX=4 \
  --env YUME_ENABLED=false \
  --env NUM_WORKERS=1 \
  --env SR_NUM_WORKERS=1 \
  --env SR_ENABLED=1 \
  --env SR_QUEUE_MAX=8 \
  --env CUDA_SR_MODEL=/models/upscale_models/RealESRGAN_x2plus.pth \
  --env CUDA_SR_TILE=128 \
  --env CUDA_SR_FP16=1 \
  --env CUDA_SR_LIFECYCLE=per_request \
  --env DEFAULT_TIMEOUT=3600 \
  --env REDIS_URL=redis://redis:6379/0 \
  --env REDIS_SOCKET_TIMEOUT=2 \
  --env REDIS_CONNECT_TIMEOUT=2 \
  --env REDIS_PREFIX=lcm \
  --env MODEL_ROOT=/models \
  --env MODE_CONFIG_PATH=/conf \
  --env BACKEND=CUDA \
  --env PORT=4200 \
  --env FS_STORAGE_DIR=/store \
  --env ZIMAGE_ENCODER_STORE=/store/encoder-store \
  --env STORAGE_PROVIDER=FILESYSTEM \
  --env ASSET_STORE_PROVIDER=FILESYSTEM \
  --env FS_STORAGE_TTL_S=604800 \
  --env FS_STORAGE_CLEANUP_INTERVAL_S=3600 \
  --env WORKER_ISOLATION=subprocess \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env CUDA_ENABLE_XFORMERS=1 \
  --env CUDA_ATTENTION_SLICING=1 \
  --env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
  --env LOG_LEVEL=DEBUG \
  --env LOG_REQUESTS=1 \
  --env LOG_BODY_MAX=8192 \
  --env LOG_PATH_PREFIXES=/generate,/superres,/sdapi \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  --env TRACING_ENABLED=1 \
  --env METRICS_ENABLED=true \
  --env OTEL_PROXY_ENDPOINT=http://otel-collector:4318/v1/traces \
  --env OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318 \
  --volume "$OUT/conf:/conf:rw" \
  --volume /media/cold1/ComfyUI/models:/models:rw \
  --volume "$OUT/images:/store:rw" \
  --volume "$REPO/comfyui-workflows:/workflows:rw" \
  --volume "$REPO/backends:/app/backends:rw" \
  --volume "$REPO/invokers:/app/invokers:rw" \
  --volume "$REPO/persistence:/app/persistence:rw" \
  --volume "$REPO/server:/app/server:rw" \
  --volume "$REPO/spikes:/app/spikes:ro" \
  --volume "$REPO/utils:/app/utils:rw" \
  --volume "$REPO/lcm-sr-ui/dist:/opt/lcm-sr-server/ui-dist:ro" \
  "$IMAGE" \
  python -m server.run
