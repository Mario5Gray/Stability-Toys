# Server Install Target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. AGENTS.md forbids subagent-driven development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `make install-server`, which puts the Docker server runtime files in `/opt/stability-toys` without writing into the checkout.

**Architecture:** One Makefile target with a single shell recipe (`set -e`). The recipe checks three refusals (not a git checkout, PREFIX inside the checkout, missing source), then two blocks (dirty tracked tree, same version), then copies by the spec's copy rules, and writes `VERSION` last.

**Tech Stack:** GNU make / BSD make compatible recipe in POSIX sh, git, pytest.

**Spec:** `docs/superpowers/specs/2026-11-15-server-install-design.md`

**FP issue:** STABL-yztmgiec

## Global Constraints

- Entry: `make install-server [PREFIX=/opt/stability-toys] [FORCE=1]`. `PREFIX` default `/opt/stability-toys`.
- Version: `git rev-parse --short HEAD`, stored in `$(PREFIX)/VERSION`, written last.
- Dirty means `git status --porcelain --untracked-files=no` is not empty.
- A block (dirty, or version equal to `$(PREFIX)/VERSION`) prints the reason, copies nothing, and exits 0.
- `FORCE=1` means "proceed to install". The install then runs as programmed. Force has no other behavior. Refusals (exit 2) are errors, not blocks, so force does not skip them.
- Copy only when absent: `compose.yaml` (from `docker-cuda.yml`), `conf/<file>` for each `git ls-files conf` entry, empty `env.custom`.
- Copy every install: `env.prod`, `env.cuda`.
- Create empty when absent: `store/`, `model/`, `workflows/`.
- Never write into the checkout. Never start the server, pull or build an image, or write `.env`.
- `install-server` is not part of the aggregate `install` target.

## Review Focus

1. A `PREFIX` path with a space: every path must be quoted. Task 1 test `test_install_server_fresh_install` uses a prefix named `pre fix`.
2. A relative or nested `PREFIX` inside the checkout (`PREFIX=./opt`): refuse with nonzero exit, write nothing. Task 1 test `test_install_server_refuses_prefix_inside_checkout`.
3. A source tree with no `.git` (tarball copy): refuse with nonzero exit, no `VERSION` with a fallback like `dev`. Task 1 test `test_install_server_refuses_non_git_source`.
4. A copy that fails midway (prefix `conf` exists as a regular file): nonzero exit, old `VERSION` kept, so the next run retries. Task 1 test `test_install_server_failed_copy_keeps_old_version`.
5. Root-owned `/opt` on macOS and Fedora: `mkdir` fails without sudo. Task 2 documents the one-time `sudo mkdir -p /opt/stability-toys && sudo chown "$USER" /opt/stability-toys` step.

---

### Task 1: `install-server` target

**Files:**
- Modify: `Makefile` (new target after `install-controlnet-scripts`, before `prod-build`)
- Create: `tests/test_make_install_server.py`

**Interfaces:**
- Consumes: none.
- Produces: make target `install-server`, variables `PREFIX` (default `/opt/stability-toys`) and `FORCE` (any non-empty value). Output lines that tests match:
  - `install-server: checkout has tracked changes; use FORCE=1` (dirty block)
  - `install-server: <version> already installed in <prefix>; use FORCE=1` (same-version block)
  - `install-server: PREFIX is inside the checkout: <prefix>` (refusal, exit 2)
  - `install-server: not a git checkout` (refusal, exit 2)
  - `copied <path>` and `kept <path>` per file, then `installed <version> to <prefix>` and `start: cd "<prefix>" && docker compose up -d`

Test fixture (in the test file, not shared):

- `source_repo(tmp_path) -> Path`: makes `tmp_path/"src"`, copies `Makefile`, `Makefile.test`, `docker-cuda.yml`, `env.prod`, `env.cuda`, and every `git ls-files conf` file from `REPO_ROOT`. Runs `git init -q`, `git add -A`, and a commit with `-c user.name=t -c user.email=t@t`. Returns the path.
- `run_install(src: Path, prefix: Path, *extra: str) -> CompletedProcess[str]`: runs `["make", "-s", "install-server", f"PREFIX={prefix}", *extra]` with `cwd=src`, `capture_output=True`, `text=True`, `check=False`, and `env={**os.environ, "GIT_CEILING_DIRECTORIES": str(src.parent)}` so git never finds a repository above `src`.
- `short_sha(src: Path) -> str`: `git rev-parse --short HEAD` in `src`.
- `commit_change(src: Path, rel: str, text: str) -> None`: writes `text` to `src/rel`, then `git add -A` and commits.

- [x] **Step 1: Write the failing tests in `tests/test_make_install_server.py`**

Each test gets `prefix = tmp_path / "pre fix"` unless stated.

```python
def test_install_server_fresh_install(tmp_path):
    src = source_repo(tmp_path); prefix = tmp_path / "pre fix"
    r = run_install(src, prefix)
    assert r.returncode == 0, r.stderr
    assert (prefix / "compose.yaml").read_text() == (src / "docker-cuda.yml").read_text()
    for name in ("controlnets.yaml", "keymap.yml", "modes-test.yml", "modes.yml"):
        assert (prefix / "conf" / name).read_text() == (src / "conf" / name).read_text()
    assert (prefix / "env.prod").read_text() == (src / "env.prod").read_text()
    assert (prefix / "env.cuda").read_text() == (src / "env.cuda").read_text()
    assert (prefix / "env.custom").read_text() == ""
    for d in ("store", "model", "workflows"):
        assert (prefix / d).is_dir() and not any((prefix / d).iterdir())
    assert (prefix / "VERSION").read_text().strip() == short_sha(src)
    assert f'start: cd "{prefix}" && docker compose up -d' in r.stdout

def test_install_server_never_writes_checkout(tmp_path):
    # after a fresh install, `git status --porcelain` (untracked included) in src is ""

def test_install_server_same_version_is_noop(tmp_path):
    # install, overwrite prefix/env.prod with "sentinel", install again
    # returncode 0, "already installed" in stdout, env.prod still "sentinel"

def test_install_server_dirty_tree_blocks(tmp_path):
    # modify tracked src/env.prod without commit, install
    # returncode 0, "checkout has tracked changes; use FORCE=1" in stdout, prefix/VERSION absent

def test_install_server_untracked_files_do_not_block(tmp_path):
    # add untracked src/conf/local.yml, install
    # returncode 0, prefix/VERSION == short_sha, prefix/conf/local.yml absent

def test_install_server_force_ignores_dirty_and_same_version(tmp_path):
    # install; write prefix/env.prod "sentinel"; modify tracked src/env.prod to "dirty"
    # install FORCE=1: returncode 0, prefix/env.prod == "dirty", VERSION == short_sha

def test_install_server_upgrade_keeps_operator_files(tmp_path):
    # install; write "edited" to prefix/compose.yaml, prefix/conf/modes.yml, prefix/env.custom
    # commit_change docker-cuda.yml, conf/modes.yml, env.prod to "v2"; install
    # compose.yaml, conf/modes.yml, env.custom still "edited"; env.prod == "v2"
    # VERSION == new short_sha

def test_install_server_upgrade_seeds_new_conf_file(tmp_path):
    # install; commit_change conf/new.yml "x"; install; prefix/conf/new.yml == "x"

def test_install_server_force_keeps_operator_files(tmp_path):
    # install; write "edited" to prefix/conf/modes.yml; install FORCE=1
    # prefix/conf/modes.yml still "edited"

def test_install_server_refuses_prefix_inside_checkout(tmp_path):
    # run_install(src, src / "opt")
    # returncode != 0, "PREFIX is inside the checkout" in stderr+stdout, (src/"opt") absent

def test_install_server_refuses_non_git_source(tmp_path):
    # source_repo, then shutil.rmtree(src/".git"); install
    # returncode != 0, "not a git checkout" in stderr+stdout, prefix/VERSION absent

def test_install_server_failed_copy_keeps_old_version(tmp_path):
    # install; old = VERSION; commit_change env.prod "v2"
    # shutil.rmtree(prefix/"conf"); write regular file prefix/"conf"; install
    # returncode != 0, VERSION still old

def test_install_server_not_in_aggregate_install():
    # make --dry-run install in REPO_ROOT: "install-server" text and "/opt/stability-toys" not in stdout
```

- [x] **Step 2: Run the tests and confirm they fail**

Run: `conda activate stability-toys && python -m tests.run tests/test_make_install_server.py -- -q`
Expected: every test except `test_install_server_not_in_aggregate_install` FAILS with `No rule to make target 'install-server'`.

- [x] **Step 3: Add the `install-server` target to `Makefile`**

- `PREFIX ?= /opt/stability-toys` and a `## Install the Docker server runtime to $(PREFIX) (FORCE=1 ignores blocks)` help comment.
- One recipe in one shell (`set -e`, lines joined with `; \`). Order:
  1. `git rev-parse --show-toplevel` fails → print `install-server: not a git checkout` to stderr, `exit 2`.
  2. Resolve the prefix to a physical absolute path without creating it: make it absolute (`case` on a leading `/`, else prefix with `$$PWD/`), then replace its nearest existing ancestor with `cd <ancestor> && pwd -P`. On macOS `/var` is a symlink to `/private/var`, and `git rev-parse --show-toplevel` returns the physical path. If the result equals or starts with the top-level plus `/` → print the refusal, `exit 2`.
  3. `v=$$(git rev-parse --short HEAD)`. Unless `FORCE` is set: dirty → print the block, `exit 0`. `[ -f "$$p/VERSION" ] && [ "$$(cat "$$p/VERSION")" = "$$v" ]` → print the block, `exit 0`.
  4. `mkdir -p "$$p/conf" "$$p/store" "$$p/model" "$$p/workflows"`.
  5. Copy-when-absent helper as a shell function `seed src dst` (prints `kept` or `copied`). Apply to `docker-cuda.yml` → `compose.yaml` and to each `git ls-files conf` path (`conf/x` → `$$p/conf/x`, `mkdir -p` its dir for nested paths).
  6. `[ -e "$$p/env.custom" ] || : > "$$p/env.custom"`.
  7. `cp env.prod env.cuda "$$p/"` and print `copied` for each.
  8. `printf '%s\n' "$$v" > "$$p/VERSION"`, then the `installed` and `start:` lines.
- Do not add `install-server` to the `install:` prerequisites.

- [x] **Step 4: Run the tests and confirm they pass**

Run: `python -m tests.run tests/test_make_install_server.py -- -q`
Expected: 13 passed.

Run: `python -m tests.run tests/test_makefile_entrypoints.py -- -q`
Expected: all passed (no regression in existing make targets).

- [x] **Step 5: Commit**

```bash
git add Makefile tests/test_make_install_server.py
git commit -m "feat(install): make install-server to /opt/stability-toys (STABL-yztmgiec) — next: Task 2 docs"
```

### Task 2: Docs and a live install on this host

**Files:**
- Modify: `README.md` (new `## Server install` section beside the `st` section near line 762)
- Modify: `docs/superpowers/plans/2026-11-15-server-install.md` (tick boxes)

**Interfaces:**
- Consumes: Task 1 target and its output lines.
- Produces: none.

- [x] **Step 1: Check drift bindings**

Run: `drift refs README.md` and `drift refs Makefile`
Expected: list of bound docs. Update any bound prose that describes the `install` targets.

- [x] **Step 2: Write the README section**

Content, in STE style, one instruction per sentence:
- What `make install-server` copies, and the copy-rule table from the spec (always, only when absent).
- The two blocks and `FORCE=1`.
- One-time prep for root-owned `/opt`: `sudo mkdir -p /opt/stability-toys && sudo chown "$USER" /opt/stability-toys`.
- Edit `compose.yaml` host paths by hand before the first `docker compose up -d`. Templating is STABL-vhyfdvmt.
- Image builds stay in the checkout (`make prod-build`).

- [x] **Step 3: Run the install on this host into a scratch prefix**

Run: `make install-server PREFIX="$TMPDIR/st-install"` twice.
Expected: first run prints `copied` lines and `installed <sha>`. Second run prints `already installed`. `git status --porcelain --untracked-files=no` in the checkout is empty.

- [x] **Step 4: Run drift check and the full CPU suite**

Run: `drift check` (stale count must not rise above the pre-change count) and `docker compose -f docker-compose.test.yml run --rm test`.
Expected: no `FAILED` ID outside `docs/superpowers/plans/2026-11-12-zimage-cuda-container/baseline-failures.txt`.

- [x] **Step 5: Commit**

```bash
git add README.md docs/superpowers/plans/2026-11-15-server-install.md
git commit -m "docs(install): server install section (STABL-yztmgiec) — next: finishing-branch menu"
```

A real install to `/opt/stability-toys` on enigma or this Mac waits for the operator's go.
