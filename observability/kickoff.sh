#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OBS="$ROOT/observability"

# Ensure shared network exists
docker network inspect observ-net >/dev/null 2>&1 || docker network create observ-net

# Start Tempo
docker compose -f "$OBS/docker-compose.tempo.yml" up -d

# Start OTEL Collector (no Loki here; assumes external Loki)
docker compose -f "$OBS/docker-compose.otel.yml" up -d

echo "Observability stack up."
