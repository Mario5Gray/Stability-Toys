# Enigma Worktree Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repo-local helper script that pushes the current branch, prepares the matching worktree on `enigma`, and prints the absolute remote worktree path.

**Architecture:** A single shell script, `scripts/enigma-worktree.sh`, runs locally and owns both phases: local branch push and one-session remote worktree preparation over SSH. Tests use `pytest` with stub `git` and `ssh` executables on `PATH` so behavior can be verified without touching the network or a real remote host.

**Tech Stack:** POSIX shell via `bash`, Git CLI, SSH, Python pytest.

**FP Tracking:** Parent issue `STABL-jartpofu` with child issues `STABL-rbjpaudh`, `STABL-hplmlxcj`, `STABL-bebjvjyl`, `STABL-uiwjdnns`, and `STABL-pkussfys`.

---

## File Structure

- Create: `scripts/enigma-worktree.sh`
  Laptop-side entrypoint. Parses flags, validates local Git state, pushes the branch, runs one SSH command, and prints the resolved remote worktree path.
- Create: `tests/test_enigma_worktree_script.py`
  End-to-end CLI tests using temp directories plus stub `git` and `ssh` binaries to verify command shape, dry-run behavior, and remote refresh semantics.

---

## Worktree Setup

- [ ] **Step 0: Create a feature worktree**

Run:

```bash
git worktree add .worktrees/enigma-worktree-sync -b enigma-worktree-sync
cd .worktrees/enigma-worktree-sync
```

All subsequent steps run inside `.worktrees/enigma-worktree-sync`. Paths below are repo-relative.

---

### Task 1: Add CLI skeleton and dry-run behavior (`STABL-rbjpaudh`)

**Files:**
- Create: `scripts/enigma-worktree.sh`
- Create: `tests/test_enigma_worktree_script.py`

- [ ] **Step 1: Write the failing CLI tests**

```python
# tests/test_enigma_worktree_script.py
from pathlib import Path
import os
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "enigma-worktree.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _make_stub_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "git",
        textwrap.dedent(
            """\
            #!/bin/sh
            case "$1" in
              rev-parse) printf '%s\n' "$PWD" ;;
              branch)
                if [ "$2" = "--show-current" ]; then
                  printf '%s\n' "${TEST_BRANCH:-gallery-ux-polish}"
                fi
                ;;
              *)
                printf 'git %s\n' "$*" >> "$TEST_LOG"
                ;;
            esac
            """
        ),
    )
    _write_executable(
        bin_dir / "ssh",
        textwrap.dedent(
            """\
            #!/bin/sh
            printf 'ssh %s\n' "$*" >> "$TEST_LOG"
            exit 0
            """
        ),
    )
    return bin_dir


def test_dry_run_prints_resolved_defaults_and_skips_git_push_and_ssh(tmp_path):
    log_path = tmp_path / "calls.log"
    bin_dir = _make_stub_bin(tmp_path)
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TEST_LOG": str(log_path),
        "TEST_BRANCH": "gallery-ux-polish",
    }

    result = subprocess.run(
        [str(SCRIPT), "--dry-run"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "host=enigma" in result.stdout
    assert "repo_path=~/workspace/Stability-Toys" in result.stdout
    assert "branch=gallery-ux-polish" in result.stdout
    assert "git push origin gallery-ux-polish" in result.stdout
    assert "ssh enigma" in result.stdout
    assert log_path.read_text() == ""


def test_help_exits_successfully(tmp_path):
    result = subprocess.run(
        [str(SCRIPT), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--repo-path <path>" in result.stdout
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_enigma_worktree_script.py -q
```

Expected: FAIL because `scripts/enigma-worktree.sh` does not exist.

- [ ] **Step 3: Write the minimal script skeleton**

```bash
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
```

- [ ] **Step 4: Make the script executable**

Run:

```bash
chmod +x scripts/enigma-worktree.sh
```

Expected: command succeeds with no output.

- [ ] **Step 5: Re-run the tests to verify they pass**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_enigma_worktree_script.py -q
```

Expected: both tests pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/enigma-worktree.sh tests/test_enigma_worktree_script.py
git commit -m "test: scaffold enigma worktree sync cli"
```

---

### Task 2: Add local Git preflight and push phase (`STABL-hplmlxcj`)

**Files:**
- Modify: `scripts/enigma-worktree.sh`
- Modify: `tests/test_enigma_worktree_script.py`

- [ ] **Step 1: Add failing tests for branch resolution and push**

```python
def test_uses_explicit_branch_for_push(tmp_path):
    log_path = tmp_path / "calls.log"
    bin_dir = _make_stub_bin(tmp_path)
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TEST_LOG": str(log_path),
        "TEST_BRANCH": "main",
        "TEST_REMOTE_HOME": "/home/tester",
    }

    result = subprocess.run(
        [str(SCRIPT), "--branch", "gallery-ux-polish"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    log = log_path.read_text()
    assert "git push origin gallery-ux-polish\n" in log


def test_detached_head_without_branch_flag_fails_before_push(tmp_path):
    log_path = tmp_path / "calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "git",
        textwrap.dedent(
            f"""\
            #!/bin/sh
            case "$1" in
              rev-parse) printf '%s\\n' "$PWD" ;;
              branch)
                if [ "$2" = "--show-current" ]; then
                  printf '\\n'
                fi
                ;;
              *)
                printf 'git %s\\n' "$*" >> "{log_path}"
                ;;
            esac
            """
        ),
    )
    _write_executable(
        bin_dir / "ssh",
        "#!/bin/sh\nexit 99\n",
    )
    env = os.environ | {"PATH": f"{bin_dir}:{os.environ['PATH']}"}

    result = subprocess.run(
        [str(SCRIPT)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "could not resolve branch" in result.stderr
    assert not log_path.exists()
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_enigma_worktree_script.py -q -k "explicit_branch or detached_head"
```

Expected: FAIL because the script does not yet push or reject detached HEAD.

- [ ] **Step 3: Implement the local preflight and push phase**

```bash
if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "local preflight failed: not inside a git repository" >&2
  exit 1
fi

if [ -z "$branch" ]; then
  echo "local preflight failed: could not resolve branch; detached HEAD" >&2
  exit 1
fi

git push "$remote_name" "$branch"
```

Insert this after argument parsing and before any SSH logic. Keep `--dry-run` ahead of the real push so dry-run never contacts the network.

- [ ] **Step 4: Re-run the targeted tests to verify they pass**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_enigma_worktree_script.py -q -k "explicit_branch or detached_head"
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/enigma-worktree.sh tests/test_enigma_worktree_script.py
git commit -m "feat: add local push phase for enigma sync"
```

---

### Task 3: Add one-session remote creation flow (`STABL-bebjvjyl`)

**Files:**
- Modify: `scripts/enigma-worktree.sh`
- Modify: `tests/test_enigma_worktree_script.py`

- [ ] **Step 1: Add failing tests for SSH command shape and worktree creation**

```python
def test_remote_phase_uses_single_ssh_session_and_creates_branch_worktree(tmp_path):
    log_path = tmp_path / "calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "git",
        textwrap.dedent(
            f"""\
            #!/bin/sh
            case "$1" in
              rev-parse) printf '%s\\n' "$PWD" ;;
              branch)
                if [ "$2" = "--show-current" ]; then
                  printf 'gallery-ux-polish\\n'
                fi
                ;;
              *)
                printf 'git %s\\n' "$*" >> "{log_path}"
                ;;
            esac
            """
        ),
    )
    _write_executable(
        bin_dir / "ssh",
        textwrap.dedent(
            f"""\
            #!/bin/sh
            printf 'ssh %s\\n' "$*" >> "{log_path}"
            cat > "{tmp_path}/remote-script.sh"
            printf '/home/tester/workspace/Stability-Toys/.worktrees/gallery-ux-polish\\n'
            """
        ),
    )
    env = os.environ | {"PATH": f"{bin_dir}:{os.environ['PATH']}"}

    result = subprocess.run(
        [str(SCRIPT)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "enigma:/home/tester/workspace/Stability-Toys/.worktrees/gallery-ux-polish"
    log = log_path.read_text()
    assert log.count("ssh ") == 1
    remote_script = (tmp_path / "remote-script.sh").read_text()
    assert 'git -C "$repo_root" fetch "$remote_name"' in remote_script
    assert 'git -C "$repo_root" worktree add -B "$branch" "$worktree_path" "$remote_name/$branch"' in remote_script
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_enigma_worktree_script.py -q -k "single_ssh_session"
```

Expected: FAIL because the script does not yet run SSH or create the remote worktree.

- [ ] **Step 3: Implement one-session remote creation**

```bash
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

if [ ! -d "$worktree_path/.git" ] && [ ! -f "$worktree_path/.git" ]; then
  git -C "$repo_root" worktree add -B "$branch" "$worktree_path" "$remote_name/$branch"
fi

printf '%s\n' "$worktree_path"
EOF
)"

printf '%s:%s\n' "$host" "$remote_path"
```

Keep the remote phase to one SSH invocation. Resolve `~` remotely before printing the path. Capture the remote stdout and prepend `$host:` locally so the final output is the absolute `host:path` handoff.

- [ ] **Step 4: Re-run the targeted test to verify it passes**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_enigma_worktree_script.py -q -k "single_ssh_session"
```

Expected: test passes.

- [ ] **Step 5: Commit**

```bash
git add scripts/enigma-worktree.sh tests/test_enigma_worktree_script.py
git commit -m "feat: add remote worktree creation phase"
```

---

### Task 4: Add remote refresh semantics and dirty-worktree protection (`STABL-uiwjdnns`)

**Files:**
- Modify: `scripts/enigma-worktree.sh`
- Modify: `tests/test_enigma_worktree_script.py`

- [ ] **Step 1: Add failing tests for dirty detection and branch refresh**

```python
def test_remote_refresh_aborts_when_status_porcelain_is_non_empty(tmp_path):
    log_path = tmp_path / "calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "git",
        textwrap.dedent(
            f"""\
            #!/bin/sh
            case "$1" in
              rev-parse) printf '%s\\n' "$PWD" ;;
              branch)
                if [ "$2" = "--show-current" ]; then
                  printf 'gallery-ux-polish\\n'
                fi
                ;;
              *)
                printf 'git %s\\n' "$*" >> "{log_path}"
                ;;
            esac
            """
        ),
    )
    _write_executable(
        bin_dir / "ssh",
        textwrap.dedent(
            f"""\
            #!/bin/sh
            cat > "{tmp_path}/remote-script.sh"
            exit 0
            """
        ),
    )
    env = os.environ | {"PATH": f"{bin_dir}:{os.environ['PATH']}"}

    subprocess.run(
        [str(SCRIPT)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    remote_script = (tmp_path / "remote-script.sh").read_text()
    assert 'git -C "$worktree_path" status --porcelain' in remote_script


def test_remote_refresh_switches_branch_then_resets_hard(tmp_path):
    log_path = tmp_path / "calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "git",
        textwrap.dedent(
            f"""\
            #!/bin/sh
            case "$1" in
              rev-parse) printf '%s\\n' "$PWD" ;;
              branch)
                if [ "$2" = "--show-current" ]; then
                  printf 'gallery-ux-polish\\n'
                fi
                ;;
              *)
                printf 'git %s\\n' "$*" >> "{log_path}"
                ;;
            esac
            """
        ),
    )
    _write_executable(
        bin_dir / "ssh",
        textwrap.dedent(
            f"""\
            #!/bin/sh
            cat > "{tmp_path}/remote-script.sh"
            exit 0
            """
        ),
    )
    env = os.environ | {"PATH": f"{bin_dir}:{os.environ['PATH']}"}

    subprocess.run(
        [str(SCRIPT)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    remote_script = (tmp_path / "remote-script.sh").read_text()
    assert 'git -C "$worktree_path" switch "$branch"' in remote_script
    assert 'git -C "$worktree_path" reset --hard "$remote_name/$branch"' in remote_script
```

- [ ] **Step 2: Run the targeted tests to verify they fail**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_enigma_worktree_script.py -q -k "status_porcelain or resets_hard"
```

Expected: FAIL because the remote script does not yet define dirty detection or refresh behavior.

- [ ] **Step 3: Implement remote refresh semantics**

```bash
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
```

Use `status --porcelain` for the dirty check so tracked edits and untracked non-ignored files block the refresh, while gitignored dependency directories do not.

- [ ] **Step 4: Re-run the targeted tests to verify they pass**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_enigma_worktree_script.py -q -k "status_porcelain or resets_hard"
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/enigma-worktree.sh tests/test_enigma_worktree_script.py
git commit -m "feat: add remote refresh safety for enigma sync"
```

---

### Task 5: Verify the full script contract (`STABL-pkussfys`)

**Files:**
- Modify: `tests/test_enigma_worktree_script.py`
- Modify: `scripts/enigma-worktree.sh`

- [ ] **Step 1: Add the final full-flow test**

```python
def test_full_sync_uses_repo_path_override_and_remote_override(tmp_path):
    log_path = tmp_path / "calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "git",
        textwrap.dedent(
            f"""\
            #!/bin/sh
            case "$1" in
              rev-parse) printf '%s\\n' "$PWD" ;;
              branch)
                if [ "$2" = "--show-current" ]; then
                  printf 'gallery-ux-polish\\n'
                fi
                ;;
              *)
                printf 'git %s\\n' "$*" >> "{log_path}"
                ;;
            esac
            """
        ),
    )
    _write_executable(
        bin_dir / "ssh",
        textwrap.dedent(
            f"""\
            #!/bin/sh
            printf 'ssh %s\\n' "$*" >> "{log_path}"
            cat > "{tmp_path}/remote-script.sh"
            printf '/srv/stability/.worktrees/gallery-ux-polish\\n'
            """
        ),
    )
    env = os.environ | {"PATH": f"{bin_dir}:{os.environ['PATH']}"}

    result = subprocess.run(
        [
            str(SCRIPT),
            "--remote", "upstream",
            "--repo-path", "/srv/stability",
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "git push upstream gallery-ux-polish\n" in log_path.read_text()
    assert result.stdout.strip() == "enigma:/srv/stability/.worktrees/gallery-ux-polish"
    remote_script = (tmp_path / "remote-script.sh").read_text()
    assert 'git -C "$repo_root" fetch "$remote_name"' in remote_script
    assert 'worktree_path="$repo_root/.worktrees/$branch"' in remote_script
```

- [ ] **Step 2: Run the full test to verify it fails**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_enigma_worktree_script.py -q -k "repo_path_override"
```

Expected: FAIL until the script correctly threads `--remote` and `--repo-path` through both phases.

- [ ] **Step 3: Make the minimal fixes for the full contract**

Adjust `scripts/enigma-worktree.sh` until the full test passes without weakening any earlier assertions. Do not add force-push, remote test execution, or bootstrap hooks; those remain deferred.

- [ ] **Step 4: Run the full test suite**

Run:

```bash
source /Users/darkbit1001/miniforge3/bin/activate base && python -m pytest tests/test_enigma_worktree_script.py -q
```

Expected: all tests in `tests/test_enigma_worktree_script.py` pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/enigma-worktree.sh tests/test_enigma_worktree_script.py
git commit -m "feat: finish enigma worktree sync helper"
```

---

## Self-Review

- Spec coverage:
  - Local push before remote worktree work: covered in Task 2.
  - One SSH session remote phase: covered in Task 3.
  - Absolute remote path output: covered in Task 3 and Task 5.
  - Dirty-worktree abort using `status --porcelain`: covered in Task 4.
  - Branch worktree creation via `git worktree add -B`: covered in Task 3.
  - Same remote name on both hosts and `--repo-path` override: covered in Task 5.
- Placeholder scan:
  - No `TODO`, `TBD`, or “handle appropriately” placeholders remain.
- Type consistency:
  - `host`, `repo_path`, `remote_name`, `branch`, and `worktree_path` are used consistently across all tasks.
