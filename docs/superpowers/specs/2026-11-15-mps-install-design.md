# MPS install target design

Status: approved 2026-11-15 (short design, operator brainstorm). FP issue: STABL-hagblcxg.
Builds on the server install design: `docs/superpowers/specs/2026-11-15-server-install-design.md` (STABL-yztmgiec).

## Intent

The operator wants the native MPS server installed like the Docker server: a clean prefix with conf and env files that the operator edits, and an install that is also the upgrade.
MPS has no image, so the install carries the server code.
The install names a conda env. If that env does not exist, the install creates it and uses it.

## Goals

- `make install-mps` copies the server code, conf, and env files to a prefix and writes a run script.
- The install names a conda env and creates it when absent.

## Non-goals

- Inventory, manifests, hashes, or security checks. The operator treats the app as insecure. Only checks that keep execution stable stay.
- Starting the server, or a launchd service.
- The web UI. The server serves it only from the fixed path `/opt/lcm-sr-server/ui-dist`. `make mps` does not serve it either.
- Syncing an existing env to new requirements. An existing env is reused as it is.

## Approaches

| Approach | Trade-off | Outcome |
|---|---|---|
| Install copies the code to `$PREFIX/app` | Installed `VERSION` is what runs. Checkout branch switches do not touch the running server. | Picked by operator. |
| Run script points at the checkout | Light. The server runs whatever the checkout holds, not the installed version. | Dropped. |

## Design

### Entry

```bash
make install-mps [PREFIX=/opt/stability-toys] [CONDA_ENV=stability-toys] [FORCE=1]
```

The version gate, the two blocks, `FORCE`, and the two refusals are the same as `install-server`.
Both targets share one gate in the Makefile.

### Conda env

- `CONDA_ENV` defaults to `stability-toys`.
- `CONDA` defaults to the `conda` found on `PATH`. If none is found, the install refuses with exit 2.
- The install writes the env name to `$PREFIX/CONDA_ENV`.
- If `conda env list` does not list the env, the install creates it and installs packages in the image order:
  1. `conda create -y -n <env> python=3.12`
  2. `pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0`
  3. `pip install -r requirements.txt`
  4. `pip install -r requirements-tracing.txt`
  5. `pip install --no-deps -r requirements-conditioning.txt`
- If the env exists, the install does not change it.
- A failed create or pip step stops the install before `VERSION` is written.

### Copy rules

| Install path | Source | Rule |
|---|---|---|
| `app/` | tracked files in `server/`, `persistence/`, `backends/`, `invokers/`, `utils/`, `scripts/`, and top-level `*.py` and `*.sh`, working-tree content | Replaced on every install. |
| `conf/<file>` | tracked files in `conf/` | Copied only when absent. |
| `env.mps` | tracked file | Copied on every install. |
| `env.custom` | none | Created empty only when absent. |
| `store/`, `certs/` | none | Created empty only when absent. |
| `CONDA_ENV` | `CONDA_ENV` variable | Written on every install. |
| `bin/run` | generated | Written on every install. |
| `VERSION` | `git rev-parse --short HEAD` | Written last. |

### Run script

`$PREFIX/bin/run` holds the absolute `conda` path found at install time.
It changes into `$PREFIX`, then runs through `conda run --no-capture-output -n <env>`:

```sh
set -a; . ./env.mps; . ./env.custom; set +a
python app/scripts/check_mps_paths.py
exec python -m uvicorn --app-dir app server.lcm_sr_server:app --host 0.0.0.0 --port "${PORT:-4200}"
```

The relative paths in `env.mps` (`conf`, `./store`, `$PWD/certs`) then resolve inside the prefix.

## Left to the builder

- Shell details of the recipe and the run script.
- Tests use a stub `conda` passed as `CONDA=` so they create no real env.

## Builder's calls (Claude, 2026-11-15)

- `requirements-test.txt` is not installed. The install is a server runtime, and the image does not install it either.
- `PORT` overrides the port in `bin/run`, so a smoke start can avoid a running server on 4200.
- `app/` copies working-tree content of tracked files, so `FORCE=1` on a dirty tree installs the dirty edits.
