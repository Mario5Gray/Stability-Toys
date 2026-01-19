# runner.sh
if [ -z ${MODEL_ROOT} ]; then 
  echo set MODEL_ROOT
  exit 1
fi
if [ -z ${CUDA_CKPT_PATH} ]; then
  echo set CUDA_CKPT_PATH
  exit 1
fi

if [[ "$(uname -m)" == "x86_64" ]]; then
  export EXTRA_ARGS='--env-file env.cuda --env-file env.lcm --name cuda-lcm-sd-ui darkbit1001/cuda-lcm-sd-ui:latest'
else 
  export EXTRA_ARGS='---env-file env.lcm --name rknn-lcm-sd-ui darkbit1001/rknn-lcm-sd-ui:latest'
fi



set -x
docker run --rm -it \
  --gpus all \
  --network host \
  --privileged \
  -e CUDA_CKPT_PATH=${CUDA_CKPT_PATH} \
  $@ \
  -v "${MODEL_ROOT}:/models:ro,Z" ${EXTRA_ARGS} bash  
set +x