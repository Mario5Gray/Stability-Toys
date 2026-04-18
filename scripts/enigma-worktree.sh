#!/usr/bin/env bash
set -euo pipefail

host="enigma"
repo_path="~/workspace/Stability-Toys"
remote_name="origin"
branch=""
dry_run=0

usage() {
  cat <<'EOF'
Usage: scripts/enigma-worktree.sh [options]

Options:
  --branch <name>
  --host <host>
  --repo-path <path>
  --remote <name>
  --dry-run
  --help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --branch)
      branch="$2"
      shift 2
      ;;
    --host)
      host="$2"
      shift 2
      ;;
    --repo-path)
      repo_path="$2"
      shift 2
      ;;
    --remote)
      remote_name="$2"
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
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

if [ -z "$branch" ]; then
  branch="$(git branch --show-current)"
fi

if [ "$dry_run" -eq 1 ]; then
  printf 'host=%s\n' "$host"
  printf 'repo_path=%s\n' "$repo_path"
  printf 'remote=%s\n' "$remote_name"
  printf 'branch=%s\n' "$branch"
  printf 'git push %s %s\n' "$remote_name" "$branch"
  printf 'ssh %s ...\n' "$host"
  exit 0
fi

echo "implementation pending" >&2
exit 1
