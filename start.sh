#!/usr/bin/env bash
set -euo pipefail

if [[ "$0" == *docker* ]]; then
  docker compose up -d
else
  set -a            # auto-export all variables
  source env.lcm
  set +a

  exec uvicorn lcm_sr_server:app \
    --host 0.0.0.0 \
    --port "${PORT:-4200}" \
    --no-access-log
fi
