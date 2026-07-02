#!/usr/bin/env bash
set -euo pipefail

host="enigma"
repo_path="~/workspace/Stability-Toys"
remote_name="origin"
worktrees_dir=".worktrees"
branch=""
dry_run=0

usage() {
  cat <<'EOF'
Usage: scripts/remote-worktree.sh [options]

Options:
  --host <host>
  --repo-path <path>
  --remote <name>
  --branch <name>
  --worktrees-dir <path>
  --dry-run
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
    --dry-run) dry_run=1; shift ;;
    --help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

git rev-parse --show-toplevel >/dev/null

if [ -z "$branch" ]; then
  branch="$(git branch --show-current)"
fi

if [ -z "$branch" ]; then
  echo "could not resolve branch; pass --branch when running from detached HEAD" >&2
  exit 1
fi

if [ "$dry_run" -eq 1 ]; then
  printf 'host=%s\n' "$host"
  printf 'repo_path=%s\n' "$repo_path"
  printf 'remote=%s\n' "$remote_name"
  printf 'worktrees_dir=%s\n' "$worktrees_dir"
  printf 'branch=%s\n' "$branch"
  printf 'git push %s %s\n' "$remote_name" "$branch"
  printf 'ssh %s ...\n' "$host"
  exit 0
fi

git push "$remote_name" "$branch"

remote_path="$(
  ssh "$host" \
    REPO_ROOT="$repo_path" \
    REMOTE_NAME="$remote_name" \
    BRANCH="$branch" \
    WORKTREES_DIR="$worktrees_dir" \
    'bash -s' <<'EOF'
set -euo pipefail

repo_root="${REPO_ROOT/#\~/$HOME}"
remote_name="$REMOTE_NAME"
branch="$BRANCH"
worktrees_dir="$WORKTREES_DIR"

if [ ! -d "$repo_root" ]; then
  echo "remote repo path does not exist: $repo_root" >&2
  exit 1
fi

if ! git -C "$repo_root" rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "remote repo path is not a git repo: $repo_root" >&2
  exit 1
fi

git -C "$repo_root" fetch "$remote_name" >&2
mkdir -p "$repo_root/$worktrees_dir"
worktree_path="$repo_root/$worktrees_dir/$branch"

if [ ! -d "$worktree_path" ]; then
  git -C "$repo_root" worktree add -B "$branch" "$worktree_path" "$remote_name/$branch" >&2
else
  if [ -n "$(git -C "$worktree_path" status --porcelain)" ]; then
    echo "remote worktree is dirty: $worktree_path" >&2
    exit 1
  fi
  git -C "$worktree_path" switch "$branch" >&2
  git -C "$worktree_path" reset --hard "$remote_name/$branch" >&2
fi

printf '%s\n' "$worktree_path"
EOF
)"

printf '%s:%s\n' "$host" "$remote_path"
