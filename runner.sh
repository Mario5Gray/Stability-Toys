# runner.sh
PLATFORM=$(uname -m)
if [ -z ${MODELS_HOST_PATH} ]; then
  echo set MODELS_HOST_PATH
  exit 1
fi

if [[ "$PLATFORM" == "x86_64" ]]; then
  export EXTRA_ARGS='--env-file env.cuda --env-file env.custom --env-file env.dev --name lcm-sd-ui --gpus=all darkbit1001/lcm-sd-ui:latest'

else 
  export EXTRA_ARGS='--env-file env.custom --env-file env.dev --env-file env.rknn --name lcm-sd-ui darkbit1001/lcm-sd-ui:latest'
fi

set -x
docker run --rm -it \
  --network dream-lab_appnet \
  -p 4200:4200 \
  --privileged \
  $@ \
  -v ./conf:/app/conf:rw,Z \
  -v ./logs:/app/logs:rw,Z \
  -v ./store:/store:rw,Z \
  -v ./workflows:/workflows:ro,Z \
  -v /media/assets/AI/SDXL/models/clip/clip-vit-base-patch32:/clip:ro,Z \
  -v "${MODELS_HOST_PATH}:/models:ro,Z" ${EXTRA_ARGS} ${RUNNER_CMD:-bash}
set +x
