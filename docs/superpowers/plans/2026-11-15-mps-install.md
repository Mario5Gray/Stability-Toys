# MPS Install Target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. AGENTS.md forbids subagent-driven development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `make install-mps`, which copies the server code, conf, and env files to a prefix, names a conda env (and creates it when absent), and writes `bin/run`.

**Architecture:** Extract the `install-server` gate (refusals, blocks, prefix resolve) into one `define` block that both targets expand. `install-mps` adds a conda step, replaces `app/` from tracked files, and writes `bin/run`. `VERSION` is written last.

**Tech Stack:** GNU make 3.81 compatible recipes in POSIX sh, git, conda, pytest.

**Spec:** `docs/superpowers/specs/2026-11-15-mps-install-design.md` (builds on `docs/superpowers/specs/2026-11-15-server-install-design.md`)

**FP issue:** STABL-hagblcxg. Branch `feat/stabl-mps-install`, stacked on `feat/stabl-yztmgiec-server-install`.

## Global Constraints

- Entry: `make install-mps [PREFIX=/opt/stability-toys] [CONDA_ENV=stability-toys] [FORCE=1]`. Plain `=` for `PREFIX`, `CONDA_ENV`, and `CONDA`, so only the command line overrides them.
- `CONDA` default: the `conda` on `PATH` at make time. None found → `install-mps: conda not found`, exit 2.
- Env absent from `conda env list` → create, then pip in this order: `torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0`, `-r requirements.txt`, `-r requirements-tracing.txt`, `--no-deps -r requirements-conditioning.txt`. Env present → no change.
- `app/` is replaced every install from tracked files of `server persistence backends invokers utils scripts` and top-level `*.py`, `*.sh`, working-tree content.
- Only when absent: `conf/<file>`, empty `env.custom`, `store/`, `certs/`. Every install: `env.mps`, `CONDA_ENV`, `bin/run`. `VERSION` last.
- Gate, blocks, `FORCE`, and refusals behave exactly as `install-server`. Its 14 tests must stay green.
- No inventory, hashes, server start, or UI.

## Review Focus

1. `bin/run` started from another directory (launchd, a cron, a different shell): it must `cd` to the prefix itself. Task 1 test `test_install_mps_run_script_invocation` runs it from `/`.
2. `conda` found only as a shell function (`conda activate` style), not a binary: `command -v conda` returns `conda`, not a path. `bin/run` must hold an absolute path. Task 1 test asserts `bin/run` holds the stub's absolute path.
3. A file removed from `server/` in the checkout must disappear from `app/` on upgrade. Task 1 test `test_install_mps_upgrade_replaces_app`.
4. A failed env create must not leave a `VERSION` that blocks the retry. Task 1 test `test_install_mps_failed_env_create_keeps_no_version`.
5. Imports from the installed `app/` alone, without the checkout on `sys.path`. Task 2 step 3 imports `server.lcm_sr_server` from a scratch prefix with the real env.

---

### Task 1: Shared gate and `install-mps` target

**Files:**
- Modify: `Makefile` (extract gate from `install-server` into `define INSTALL_GATE`, add `install-mps` after `install-server`)
- Create: `tests/test_make_install_mps.py`

**Interfaces:**
- Consumes: helpers `source_repo`, `run_install`, `short_sha`, `commit_change`, `_git`, `_out` from `tests/test_make_install_server.py` (import them).
- Produces: target `install-mps`. Variables `CONDA_ENV`, `CONDA`. Output lines: the `install-server` messages with `install-mps:` as the prefix word, `install-mps: conda not found` (exit 2), `created conda env <env>` or `kept conda env <env>`, `copied`/`kept` per file, `installed <v> to <p>`, `start: "<p>/bin/run"`.
- `INSTALL_GATE` takes the target name as `$(1)` and leaves shell variables `top`, `p`, `v` set for the rest of the recipe.

Test fixture additions in the new file:

- `mps_repo(tmp_path) -> Path`: `source_repo(tmp_path)`, then add and commit `env.mps` (`BACKEND=mps\nMODE_CONFIG_PATH=${MODE_CONFIG_PATH:-conf}\n`), `requirements.txt`, `requirements-tracing.txt`, `requirements-conditioning.txt` (one line each), `server/__init__.py`, `server/app.py`, `backends/b.py`, `utils/u.py`, `persistence/p.py`, `invokers/i.py`, `scripts/check_mps_paths.py`, `top.py`, `top.sh`.
- `stub_conda(tmp_path, envs: list[str], create_rc: int = 0) -> Path`: writes an executable sh script `tmp_path/"bin"/"conda"` that appends `"$@"` as one line to `tmp_path/"conda.log"`. On `env list` it prints `# conda environments:` then one `<name> /x/<name>` line per env. On `create` it exits `create_rc`. On `run` it exits 0.
- `log_lines(tmp_path) -> list[str]`: lines of `conda.log`, or `[]`.
- Every install call passes `f"CONDA={stub}"`.

- [x] **Step 1: Write the failing tests in `tests/test_make_install_mps.py`**

```python
def test_install_mps_fresh_install_copies_app_and_files(tmp_path):
    # envs=["base", "stability-toys"]; untracked src/server/scratch.py added before install
    # returncode 0; app/server/app.py, app/backends/b.py, app/utils/u.py, app/persistence/p.py,
    # app/invokers/i.py, app/scripts/check_mps_paths.py, app/top.py, app/top.sh exist
    # app/server/scratch.py absent; conf/modes.yml == src; env.mps == src; env.custom == ""
    # store/ and certs/ are dirs; CONDA_ENV text == "stability-toys\n"; VERSION == short_sha
    # bin/run is executable and contains str(stub) and "-n stability-toys"
    # f'start: "{prefix}/bin/run"' in stdout

def test_install_mps_reuses_existing_env(tmp_path):
    # envs=["base", "stability-toys"]; no log line starts with "create" or contains "pip"
    # "kept conda env stability-toys" in stdout

def test_install_mps_creates_missing_env_in_image_order(tmp_path):
    # envs=["base"]; pip lines in order:
    #   "run -n stability-toys python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0"
    #   "run -n stability-toys python -m pip install -r <src>/requirements.txt"
    #   "run -n stability-toys python -m pip install -r <src>/requirements-tracing.txt"
    #   "run -n stability-toys python -m pip install --no-deps -r <src>/requirements-conditioning.txt"
    # "create -y -n stability-toys python=3.12" precedes them; "created conda env stability-toys" in stdout

def test_install_mps_names_custom_env(tmp_path):
    # envs=["base"]; CONDA_ENV=dreamer; CONDA_ENV text == "dreamer\n"; "create -y -n dreamer python=3.12" in log
    # bin/run contains "-n dreamer"

def test_install_mps_failed_env_create_keeps_no_version(tmp_path):
    # envs=["base"], create_rc=1; returncode != 0; VERSION absent

def test_install_mps_refuses_without_conda(tmp_path):
    # CONDA=<tmp_path>/missing/conda; returncode == 2; "install-mps: conda not found" in output

def test_install_mps_upgrade_replaces_app(tmp_path):
    # install; edit prefix/conf/modes.yml and prefix/env.custom to "edited"
    # git rm server/app.py, commit_change server/new.py "n", commit_change env.mps "v2"; install
    # app/server/app.py absent; app/server/new.py == "n"; env.mps == "v2"
    # conf/modes.yml and env.custom still "edited"

def test_install_mps_same_version_is_noop(tmp_path):
    # install; write prefix/env.mps "sentinel"; install; "already installed" in stdout; env.mps still "sentinel"

def test_install_mps_force_installs_dirty_working_tree(tmp_path):
    # install; write src/server/app.py "dirty" (uncommitted); install FORCE=1
    # returncode 0; app/server/app.py == "dirty"

def test_install_mps_run_script_invocation(tmp_path):
    # install; run [prefix/bin/run] with cwd="/" and env PORT=4299
    # last log line starts with "run --no-capture-output -n stability-toys sh -c"
    # and contains "--app-dir app" and "server.lcm_sr_server:app" and "check_mps_paths.py"
    # stub records its own cwd (pwd) in conda.log as "cwd=<path>"; that path == prefix

def test_install_mps_not_in_aggregate_install():
    # make --dry-run install in REPO_ROOT: "install-mps" not in stdout
```

The stub writes `cwd=$(pwd -P)` to the log on every `run` call, for the last test.

- [x] **Step 2: Run the tests and confirm they fail**

Run: `conda activate stability-toys && python -m tests.run tests/test_make_install_mps.py -- -q`
Expected: 10 FAIL with `No rule to make target 'install-mps'`. `test_install_mps_not_in_aggregate_install` PASSES.

- [x] **Step 3: Extract `INSTALL_GATE` and add `install-mps` to `Makefile`**

- `define INSTALL_GATE` holds the current `install-server` lines from `top=` through the same-version block, with `install-server:` messages replaced by `$(1):`. `install-server` becomes `@set -e; $(call INSTALL_GATE,install-server) <its copy lines>`.
- `CONDA_ENV = stability-toys` and `CONDA = $(shell command -v conda 2>/dev/null)`. In the recipe: `[ -x "$(CONDA)" ] || { echo "install-mps: conda not found" >&2; exit 2; }`. If `command -v` returns a bare name (shell function), resolve it with `"$(CONDA)" info --base` + `/condabin/conda`.
- Env step: `"$(CONDA)" env list | awk -v e="$(CONDA_ENV)" '$$1==e{f=1} END{exit !f}'` decides kept vs created. Create and pip lines as in Global Constraints, each through `"$(CONDA)" run -n "$(CONDA_ENV)" python -m pip install`, with requirements paths `$$top/<file>`.
- App step: `rm -rf "$$p/app"`, then `git -C "$$top" ls-files -z server persistence backends invokers utils scripts ':(glob)*.py' ':(glob)*.sh' | (cd "$$top" && xargs -0 tar -cf -) | tar -xf - -C "$$p/app"` after `mkdir -p "$$p/app"`.
- Seed conf, `env.custom`, dirs as `install-server` does. Copy `env.mps`. Write `CONDA_ENV`.
- `bin/run`: write with `printf` a `#!/bin/sh` script that does `cd "<p>"` and `exec "<conda>" run --no-capture-output -n "<env>" sh -c '<the three spec lines>'`. `chmod +x`.
- `VERSION` last, then `installed` and `start:` lines.

- [x] **Step 4: Run the tests and confirm they pass**

Run: `python -m tests.run tests/test_make_install_mps.py tests/test_make_install_server.py tests/test_makefile_entrypoints.py -- -q`
Expected: 32 passed (11 + 14 + 7).

- [x] **Step 5: Commit**

```bash
git add Makefile tests/test_make_install_mps.py
git commit -m "feat(install): make install-mps with named conda env (STABL-hagblcxg) — next: Task 2 docs and live check"
```

### Task 2: Docs and live check on this Mac

**Files:**
- Modify: `README.md` (MPS subsection under `## Server install`)
- Modify: `docs/MPS_BACKEND.md` (pointer to `make install-mps`)
- Modify: `docs/superpowers/plans/2026-11-15-mps-install.md` (tick boxes)

**Interfaces:**
- Consumes: Task 1 target and output lines.
- Produces: none.

- [x] **Step 1: Review stale drift prose in `docs/MPS_BACKEND.md`**

Run: `git show c43ccce -- Makefile` and read the `make mps` prose in `docs/MPS_BACKEND.md`.
Expected: prose matches the `0.0.0.0` host default, or update it. Then add the `make install-mps` pointer.

- [x] **Step 2: Write the README MPS subsection**

STE style: what `install-mps` copies, the conda env rule, `CONDA_ENV=`, `bin/run`, `PORT=`, and that the UI is not served.

- [x] **Step 3: Live install into a scratch prefix with the real env**

Run: `make install-mps PREFIX="<scratchpad>/mps-install"` (env `stability-toys` exists, so it is kept). Then:
`conda run -n stability-toys python -c "import sys; sys.path.insert(0, '<prefix>/app'); import server.lcm_sr_server"` with cwd `<prefix>` and `PYTHONPATH` unset.
Expected: install prints `kept conda env stability-toys` and `installed <sha>`. The import exits 0. A second install prints `already installed`.

- [x] **Step 4: Drift and suites**

Run: `drift check` (stale count not above the pre-change count, `docs/MPS_BACKEND.md` relinked only after Step 1 review). Run the CPU container suite.
Expected: no `FAILED` ID outside `docs/superpowers/plans/2026-11-12-zimage-cuda-container/baseline-failures.txt`.

- [x] **Step 5: Commit**

```bash
git add README.md docs/MPS_BACKEND.md docs/superpowers/plans/2026-11-15-mps-install.md drift.lock
git commit -m "docs(install): MPS install docs (STABL-hagblcxg) — next: finishing-branch menu"
```

A real install to `/opt/stability-toys` and a server start wait for the operator's go.
