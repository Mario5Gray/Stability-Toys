#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OBS="$ROOT/observability"

docker compose -f "$OBS/docker-compose.otel.yml" down

docker compose -f "$OBS/docker-compose.tempo.yml" down

echo "Observability stack stopped."
