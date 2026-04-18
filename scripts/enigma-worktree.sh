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

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "local preflight failed: not inside a git repository" >&2
  exit 1
fi

if [ -z "$branch" ]; then
  branch="$(git branch --show-current)"
fi

if [ -z "$branch" ]; then
  echo "local preflight failed: could not resolve branch; detached HEAD" >&2
  exit 1
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

git push "$remote_name" "$branch"

remote_path="$(ssh "$host" "branch=$(printf %q "$branch") repo_path=$(printf %q "$repo_path") remote_name=$(printf %q "$remote_name") bash -s" <<'EOF'
set -euo pipefail

repo_root="$(eval "printf '%s' $repo_path")"
worktree_path="$repo_root/.worktrees/$branch"

if ! git -C "$repo_root" rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "remote prepare failed: repo path is not a git repository: $repo_root" >&2
  exit 1
fi

git -C "$repo_root" fetch "$remote_name"
mkdir -p "$repo_root/.worktrees"

if [ -d "$worktree_path" ]; then
  if [ -n "$(git -C "$worktree_path" status --porcelain)" ]; then
    echo "remote prepare failed: worktree is dirty: $worktree_path" >&2
    exit 1
  fi
  git -C "$worktree_path" switch "$branch"
  git -C "$worktree_path" reset --hard "$remote_name/$branch"
else
  git -C "$repo_root" worktree add -B "$branch" "$worktree_path" "$remote_name/$branch"
fi

printf '%s\n' "$worktree_path"
EOF
)"

printf '%s:%s\n' "$host" "$remote_path"
