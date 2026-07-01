# Remote Worktree Sync and Enigma Dev Verify Implementation Plan

> **For agentic workers:** REQUIRED SKILL: `superpowers:executing-plans` (inline execution only). **Repo override:** `AGENTS.md` forbids subagent-driven development; do **not** use `superpowers:subagent-driven-development` for this plan. Execute each task inline in the current session with TDD red/green/commit per task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic remote worktree sync helper plus a Stability-Toys-specific `enigma` verification wrapper that prepares a remote CUDA worktree, starts the bind-mounted dev compose flow on the GPU host, and reports the remaining manual `modes.yaml` verification step.

**Architecture:** `scripts/remote-worktree.sh` is the reusable primitive: it pushes a branch, SSHes once to a remote repo root, creates or refreshes a branch worktree, and prints `host:absolute-path`. `scripts/enigma-dev-verify.sh` wraps that primitive for this repo, runs the exact `docker-cuda.yml` and `docker-compose.dev.yml` flow on the remote host, waits for the `lcm-sd-dev` container health to turn healthy, prints recent logs, and then emits an explicit manual handoff for the final `modes.yaml` edit instead of mutating remote config automatically.

**Tech Stack:** Bash, Git CLI, SSH, Docker Compose, pytest.

---

## File Structure

- Create: `scripts/remote-worktree.sh`
  Purpose: generic local entrypoint for branch push plus remote worktree create/refresh.
- Create: `scripts/enigma-dev-verify.sh`
  Purpose: repo-specific GPU-host verifier that consumes `remote-worktree.sh`.
- Modify: `tests/test_enigma_worktree_script.py`
  Purpose: retarget the old `enigma`-specific helper contract to the new generic helper and cover the generic flags.
- Create: `tests/test_enigma_dev_verify_script.py`
  Purpose: stub-based tests for the repo-specific wrapper’s command shape and operator handoff.
- Modify: `docs/TESTING_IN_DOCKER.md`
  Purpose: operator-facing documentation for the remote GPU verification workflow and its manual final step.

### Task 1: Build the generic remote worktree helper

**Files:**
- Create: `scripts/remote-worktree.sh`
- Modify: `tests/test_enigma_worktree_script.py`

- [ ] **Step 1: Retarget the existing worktree tests to the new script and add generic flag coverage**

In `tests/test_enigma_worktree_script.py`, change the script target and add one override test.

Replace:

```python
SCRIPT = ROOT / "scripts" / "enigma-worktree.sh"
```

with:

```python
SCRIPT = ROOT / "scripts" / "remote-worktree.sh"
```

Append:

```python
def test_full_sync_uses_host_and_worktrees_dir_overrides(tmp_path):
    log_path = tmp_path / "calls.log"
    log_path.touch()
    bin_dir = _make_stub_bin(tmp_path)
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TEST_LOG": str(log_path),
        "TEST_BRANCH": "gallery-ux-polish",
    }

    result = subprocess.run(
        [
            str(SCRIPT),
            "--host", "gpu-box",
            "--repo-path", "/srv/stability",
            "--worktrees-dir", "wtrees",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "gpu-box:/srv/stability/wtrees/gallery-ux-polish"
    log = log_path.read_text()
    assert "ssh gpu-box" in log
    remote_script = (tmp_path / "remote-script.sh").read_text()
    assert 'worktree_path="$repo_root/$worktrees_dir/$branch"' in remote_script
```

Also update the existing override test’s expected path expression from `.worktrees` to keep passing when `--worktrees-dir` is not supplied:

```python
assert 'worktree_path="$repo_root/.worktrees/$branch"' in remote_script
```

- [ ] **Step 2: Run the generic-helper tests to verify they fail**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_enigma_worktree_script.py -q
```

Expected: FAIL because `scripts/remote-worktree.sh` does not exist yet and the test file now points at it.

- [ ] **Step 3: Create `scripts/remote-worktree.sh` with the generic CLI contract**

Create `scripts/remote-worktree.sh`:

```bash
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
    'bash -s' <<'"'"'EOF'"'"'
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

git -C "$repo_root" fetch "$remote_name"
mkdir -p "$repo_root/$worktrees_dir"
worktree_path="$repo_root/$worktrees_dir/$branch"

if [ ! -d "$worktree_path" ]; then
  git -C "$repo_root" worktree add -B "$branch" "$worktree_path" "$remote_name/$branch"
else
  if [ -n "$(git -C "$worktree_path" status --porcelain)" ]; then
    echo "remote worktree is dirty: $worktree_path" >&2
    exit 1
  fi
  git -C "$worktree_path" switch "$branch"
  git -C "$worktree_path" reset --hard "$remote_name/$branch"
fi

printf '%s\n' "$worktree_path"
EOF
)"

printf '%s:%s\n' "$host" "$remote_path"
```

Then mark it executable:

```bash
chmod +x scripts/remote-worktree.sh
```

- [ ] **Step 4: Run the generic-helper tests to verify they pass**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_enigma_worktree_script.py -q
```

Expected: PASS

- [ ] **Step 5: Commit the generic helper**

```bash
git add scripts/remote-worktree.sh tests/test_enigma_worktree_script.py
git commit -m "feat(remote): add generic remote worktree sync helper"
```

### Task 2: Build the repo-specific `enigma` dev verifier

**Files:**
- Create: `scripts/enigma-dev-verify.sh`
- Create: `tests/test_enigma_dev_verify_script.py`

- [ ] **Step 1: Write the failing verifier tests**

Create `tests/test_enigma_dev_verify_script.py`:

```python
from pathlib import Path
import os
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "enigma-dev-verify.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def test_verify_wraps_remote_worktree_and_runs_expected_remote_commands(tmp_path):
    log_path = tmp_path / "calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    _write_executable(
        bin_dir / "remote-worktree.sh",
        textwrap.dedent(
            """\
            #!/bin/sh
            printf 'enigma:/srv/stability/.worktrees/gallery-ux-polish\n'
            """
        ),
    )
    _write_executable(
        bin_dir / "ssh",
        textwrap.dedent(
            f"""\
            #!/bin/sh
            printf 'ssh %s\n' "$*" >> "{log_path}"
            cat > "{tmp_path}/remote-script.sh"
            """
        ),
    )

    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REMOTE_WORKTREE_BIN": str(bin_dir / "remote-worktree.sh"),
    }

    result = subprocess.run(
        [str(SCRIPT)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    remote_script = (tmp_path / "remote-script.sh").read_text()
    assert 'cd "/srv/stability/.worktrees/gallery-ux-polish"' in remote_script
    assert 'docker compose -f docker-cuda.yml build' in remote_script
    assert 'docker compose -f docker-compose.dev.yml up -d --build' in remote_script
    assert "docker inspect -f '{{.State.Health.Status}}' lcm-sd-dev" in remote_script
    assert "docker logs --tail 50 lcm-sd-dev" in remote_script


def test_verify_prints_manual_modes_yaml_handoff(tmp_path):
    log_path = tmp_path / "calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    _write_executable(
        bin_dir / "remote-worktree.sh",
        "#!/bin/sh\nprintf 'enigma:/srv/stability/.worktrees/gallery-ux-polish\\n'\n",
    )
    _write_executable(
        bin_dir / "ssh",
        textwrap.dedent(
            f"""\
            #!/bin/sh
            printf 'ssh %s\n' "$*" >> "{log_path}"
            exit 0
            """
        ),
    )

    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REMOTE_WORKTREE_BIN": str(bin_dir / "remote-worktree.sh"),
    }

    result = subprocess.run(
        [str(SCRIPT), "--dry-run-manual-step"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Manual step remaining:" in result.stdout
    assert "conf/modes.yaml" in result.stdout
    assert "docker logs -f lcm-sd-dev" in result.stdout
    assert log_path.read_text() == ""
```

- [ ] **Step 2: Run the verifier tests to verify they fail**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_enigma_dev_verify_script.py -q
```

Expected: FAIL because `scripts/enigma-dev-verify.sh` does not exist yet.

- [ ] **Step 3: Create `scripts/enigma-dev-verify.sh` with the repo-specific wrapper flow**

Create `scripts/enigma-dev-verify.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

host="enigma"
repo_path="~/workspace/Stability-Toys"
remote_name="origin"
worktrees_dir=".worktrees"
branch=""
manual_only=0

usage() {
  cat <<'EOF'
Usage: scripts/enigma-dev-verify.sh [options]

Options:
  --host <host>
  --repo-path <path>
  --remote <name>
  --branch <name>
  --worktrees-dir <path>
  --dry-run-manual-step
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
    --dry-run-manual-step) manual_only=1; shift ;;
    --help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

helper_dir="$(cd "$(dirname "$0")" && pwd)"
remote_worktree_bin="${REMOTE_WORKTREE_BIN:-$helper_dir/remote-worktree.sh}"
sync_output="$(
  "$remote_worktree_bin" \
    --host "$host" \
    --repo-path "$repo_path" \
    --remote "$remote_name" \
    --worktrees-dir "$worktrees_dir"
)"

if [ -n "$branch" ]; then
  sync_output="$(
    "$remote_worktree_bin" \
      --host "$host" \
      --repo-path "$repo_path" \
      --remote "$remote_name" \
      --branch "$branch" \
      --worktrees-dir "$worktrees_dir"
  )"
fi

sync_host="${sync_output%%:*}"
worktree_path="${sync_output#*:}"

if [ "$manual_only" -eq 0 ]; then
  ssh "$sync_host" 'bash -s' <<EOF
set -euo pipefail

cd "$worktree_path"
docker compose -f docker-cuda.yml build
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
```

Then mark it executable:

```bash
chmod +x scripts/enigma-dev-verify.sh
```

- [ ] **Step 4: Run the verifier tests to verify they pass**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_enigma_dev_verify_script.py -q
```

Expected: PASS

- [ ] **Step 5: Run both script test files to check integration-level regressions**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_enigma_worktree_script.py tests/test_enigma_dev_verify_script.py -q
```

Expected: PASS

- [ ] **Step 6: Commit the verifier wrapper**

```bash
git add scripts/enigma-dev-verify.sh tests/test_enigma_dev_verify_script.py
git commit -m "feat(enigma): add dev compose verification wrapper"
```

### Task 3: Document the remote GPU verification workflow

**Files:**
- Modify: `docs/TESTING_IN_DOCKER.md`

- [ ] **Step 1: Add an operator section for remote GPU verification**

Append this section to `docs/TESTING_IN_DOCKER.md` after the existing Docker test-path guidance:

````md
## Remote GPU Dev Verification

The bind-mounted dev workflow in [`docker-compose.dev.yml`](/Users/darkbit1001/workspace/Stability-Toys/docker-compose.dev.yml) must run from a real repo tree on the Docker host. On a laptop, use the remote helper flow instead of trying to drive the bind mounts directly through Docker context alone.

Prepare or refresh the remote worktree and run the CUDA dev verification:

```bash
scripts/enigma-dev-verify.sh --branch <branch>
```

This wrapper:

- pushes the branch to the selected Git remote
- refreshes a branch worktree on the remote host
- runs `docker compose -f docker-cuda.yml build`
- runs `docker compose -f docker-compose.dev.yml up -d --build`
- waits for `lcm-sd-dev` to report a healthy Docker health status
- prints recent container logs
- prints the remaining manual `conf/modes.yaml` watcher check

The final `modes.yaml` edit is intentionally manual in v1. It keeps the remote config mutation explicit and reversible for the operator.
````

- [ ] **Step 2: Verify the docs read cleanly**

Run:

```bash
sed -n '1,260p' docs/TESTING_IN_DOCKER.md
```

Expected: the new section is present, accurate, and does not contradict the existing local-vs-CUDA guidance.

- [ ] **Step 3: Commit the docs**

```bash
git add docs/TESTING_IN_DOCKER.md
git commit -m "docs(docker): document remote GPU dev verification flow"
```

### Task 4: Final verification

**Files:**
- No new files; verification only

- [ ] **Step 1: Run the local script-focused test suite**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_enigma_worktree_script.py tests/test_enigma_dev_verify_script.py -q
```

Expected: PASS

- [ ] **Step 2: Run the packaging contract suite to ensure the earlier dev-compose work still holds**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_cuda_packaging_contract.py -q --no-cov
```

Expected: PASS

- [ ] **Step 3: Record the live-host verification command for the operator**

Run and save for the final handoff:

```bash
printf '%s\n' 'scripts/enigma-dev-verify.sh --branch <branch>'
```

Expected: prints the exact remote verification entrypoint to use on the next real GPU-host pass.

## Self-Review

- Spec coverage:
  - generic remote worktree helper: Task 1
  - repo-specific `enigma` verifier: Task 2
  - bind-mounted dev-compose preservation: Task 2 + Task 3
  - explicit manual `modes.yaml` boundary: Task 2 + Task 3
- Placeholder scan:
  - no `TODO`, `TBD`, or “similar to above” placeholders remain
- Type and surface consistency:
  - generic helper stays `scripts/remote-worktree.sh`
  - repo-specific wrapper stays `scripts/enigma-dev-verify.sh`
  - both tasks use the same `--host`, `--repo-path`, `--remote`, `--branch`, `--worktrees-dir` vocabulary
