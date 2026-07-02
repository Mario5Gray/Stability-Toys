#!/usr/bin/env bash
set -euo pipefail

host="enigma"
repo_path="~/workspace/Stability-Toys"
remote_name="origin"
worktrees_dir=".worktrees"
branch=""
manual_only=0
skip_base_build=0

usage() {
  cat <<'EOF'
Usage: scripts/enigma-dev-verify.sh [options]

Options:
  --host <host>
  --repo-path <path>
  --remote <name>
  --branch <name>
  --worktrees-dir <path>
  --manual-step-only
  --skip-base-build
  --help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --host) host="$2"; shift 2 ;;
    --repo-path) repo_path="$2"; shift 2 ;;
    --remote) remote_name="$2"; shift 2 ;;
    --branch) branch="$2"; shift 2 ;;
    --worktrees-dir) worktrees_dir="$2"; shift 2 ;;
    --manual-step-only) manual_only=1; shift ;;
    --skip-base-build) skip_base_build=1; shift ;;
    --help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

helper_dir="$(cd "$(dirname "$0")" && pwd)"
remote_worktree_bin="${REMOTE_WORKTREE_BIN:-$helper_dir/remote-worktree.sh}"
sync_args=(
  --host "$host"
  --repo-path "$repo_path"
  --remote "$remote_name"
  --worktrees-dir "$worktrees_dir"
)
if [ -n "$branch" ]; then
  sync_args+=(--branch "$branch")
fi

sync_output="$("$remote_worktree_bin" "${sync_args[@]}")"
sync_host="${sync_output%%:*}"
worktree_path="${sync_output#*:}"
base_build_command=""
if [ "$skip_base_build" -eq 0 ]; then
  base_build_command="docker compose -f docker-cuda.yml build"
fi

if [ "$manual_only" -eq 0 ]; then
  ssh "$sync_host" 'bash -s' <<EOF
set -euo pipefail

cd "$worktree_path"
$base_build_command
docker compose -f docker-compose.dev.yml up -d --build

attempt=0
while [ "\$attempt" -lt 30 ]; do
  status=\$(docker inspect -f '{{.State.Health.Status}}' lcm-sd-dev 2>/dev/null || true)
  if [ "\$status" = "healthy" ]; then
    break
  fi
  attempt=\$((attempt + 1))
  sleep 2
done

status=\$(docker inspect -f '{{.State.Health.Status}}' lcm-sd-dev 2>/dev/null || true)
if [ "\$status" != "healthy" ]; then
  echo "lcm-sd-dev did not become healthy" >&2
  docker logs --tail 50 lcm-sd-dev >&2 || true
  exit 1
fi

docker logs --tail 50 lcm-sd-dev
EOF
fi

printf 'Manual step remaining:\n'
printf '1. ssh %s\n' "$sync_host"
printf '2. cd %s\n' "$worktree_path"
printf '3. edit conf/modes.yaml and save one reversible change\n'
printf '4. docker logs -f lcm-sd-dev\n'
printf '5. confirm the config watcher reloads without restarting the container\n'
