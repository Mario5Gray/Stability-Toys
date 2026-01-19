#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE=docker-compose.yml
set -a            # auto-export all variables
source env.lcm
if [[ "$(uname -m)" == "x86_64" ]]; then
  source env.cuda
  COMPOSE_FILE=docker-cuda.yml
fi
set +a


if [[ "$0" == *docker* ]]; then
  docker compose up ${COMPOSE_FILE} 
else
  exec uvicorn lcm_sr_server:app \
    --host 0.0.0.0 \
    --port "${PORT:-4200}" \
    --loop uvloop \
    --http h11 \
    --timeout-keep-alive 240 \
    --log-level info
    #--no-access-log
fi
