# Server install target design

Status: approved 2026-11-15 (short design, operator brainstorm). FP issue: STABL-yztmgiec.

## Intent

The operator wants a clean server directory at `/opt/stability-toys`.
The operator edits conf and env files there.
Those edits must not dirty the git checkout.

## Goals

- One Makefile target, `make install-server`, puts the Docker server runtime files in a prefix.
- The install is idempotent.
- An install is also the upgrade path.

## Non-goals

- The CLI tools. `make install` already installs them to `~/.local/bin`.
- Configuration templating. STABL-vhyfdvmt owns it.
- Removing the OpenAPI snapshot and generated client from the repository. STABL-nqgthjsb owns it.
- Moving tracked files that name host-local directories out of the repository. STABL-alczbitv owns it.
- A formal deployment routine. Until one exists, the operator edits compose parameters by hand.

## Anti-goals

- Certification, staging directories, content hashes, or any measurement of files on the install path.
- An install that overwrites operator edits to conf, compose, or `env.custom`.

## Constraints

- Makefile target only. No installer script.
- The install never writes into the checkout.

## Approaches

| Approach | Trade-off | Outcome |
|---|---|---|
| Makefile target | Small. Matches `install-st` and `install-qrng`. Version logic fits in shell recipe lines. | Picked by operator. |
| Bash script called by make | More room for logic. Adds a file this change does not need. | Dropped. |
| Python installer | Needs the conda env on the host. | Dropped. |

## Design

### Entry

```bash
make install-server [PREFIX=/opt/stability-toys] [FORCE=1]
```

`PREFIX` defaults to `/opt/stability-toys`.

### Version gate

- The version is `git rev-parse --short HEAD` of the checkout.
- The install records the version in `$(PREFIX)/VERSION`.
- Two conditions block the install:
  - The checkout is dirty (`git status --porcelain` is not empty).
  - The version equals the content of `$(PREFIX)/VERSION`.
- A blocked install prints the reason and exits 0. It copies nothing.
- `FORCE=1` ignores every block.
- The install writes `VERSION` last. A failed copy leaves the old version, so the next run retries.

### Copy rules

| Install path | Source | Rule |
|---|---|---|
| `compose.yaml` | `docker-cuda.yml` | Copy only when absent. |
| `conf/<file>` | each file in `conf/` | Copy only when absent. |
| `env.custom` | none | Create empty only when absent. Compose fails when an `env_file` is missing. |
| `env.prod`, `env.cuda` | tracked files | Copy on every install. |
| `store/`, `model/`, `workflows/` | none | Create empty only when absent. They match the compose defaults. |
| `VERSION` | `git rev-parse --short HEAD` | Write last on every install. |

`FORCE=1` does not change the copy rules. It only removes the blocks.

### Output

The target prints each path that it copied or kept.
It ends with the command to start the server from the prefix: `cd $(PREFIX) && docker compose up -d`.

### Not done

- No `.env` file, no host-path seeding.
- No server start or restart.
- No image pull or build. Image builds stay in the checkout (`make prod-build`), because the compose `build:` context needs the full source tree.
- No dev compose. It bind-mounts source, so it is a checkout workflow.
- No MPS. It runs natively, not in Docker.
- `install-server` is not part of the aggregate `make install` target.

## Left to the builder

- Shell details of the recipe, such as quoting and messages.
- Tests: drive the real target from pytest against a temporary git repository and a temporary prefix. Follow `tests/test_makefile_entrypoints.py`.

## Builder's calls (Claude, 2026-11-15)

- Compose is copied only when absent. The operator answered "correct" to that proposal.
- A blocked install exits 0, because a no-op is a success for an idempotent install.
- `FORCE=1` does not overwrite conf, compose, or `env.custom`. The operator rule "conf is only copied when absent" has no force exception.
