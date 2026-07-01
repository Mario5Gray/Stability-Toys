# Remote Worktree Sync and Enigma Dev Verify Design

## Summary

This design adds two shell entrypoints:

1. `scripts/remote-worktree.sh`
2. `scripts/enigma-dev-verify.sh`

The first is a generic repo-local helper that pushes a branch, prepares a matching remote worktree over SSH, and prints the resolved remote worktree path. The second is a repo-specific verifier that uses that remote worktree to run the deferred CUDA dev-compose validation on `enigma`.

The design preserves the bind-mounted `docker-compose.dev.yml` contract. That is the main constraint. Because the dev compose file mounts repo-relative host paths such as `./conf`, `./server`, `./backends`, and `./lcm-sr-ui/dist`, the remote host must have a real checkout or worktree that Docker can mount from locally on that host. A pure image-first deploy or a laptop-only `docker context` flow does not validate the same path.

## Goals

- Add a reusable remote worktree sync helper that can target hosts other than `enigma`
- Keep the helper generic enough to reuse across remote GPU boxes and remote repo paths
- Preserve the current bind-mounted `docker-compose.dev.yml` development workflow
- Add a repo-specific verification wrapper for the deferred GPU-host validation from the fast dev Docker builds work
- Keep the generic helper free of Docker, compose, and repo-specific health-check logic
- Produce a workflow that an operator can run repeatedly without manual Git and SSH setup steps each time

## Non-Goals

- General remote command orchestration for arbitrary repos
- Replacing SSH with Docker contexts as the primary execution boundary
- Eliminating the remote repo prerequisite on the host
- Automatically provisioning Docker, NVIDIA runtime support, or the remote clone on the host
- Syncing uncommitted local changes without a Git push
- Cross-repo extraction of the generic helper in this slice
- Rewriting `docker-compose.dev.yml` to support laptop-originated remote bind mounts

## Current State

- `docker-compose.dev.yml` is intentionally bind-mount centric and assumes a local repo tree on the Docker host.
- The deferred verification step from the fast dev Docker builds plan must run on a real NVIDIA host.
- This repo previously had an `enigma`-specific worktree sync design and tests, but the script itself was removed.
- `tests/test_enigma_worktree_script.py` still captures the original worktree-sync contract and can serve as a starting point for the generic helper.
- `docker context` is still useful for inspection and follow-up Docker commands, but it is not sufficient on its own to drive the bind-mounted dev-compose path from the laptop.

## Approach Options

### Option 1: Remote worktree plus remote Docker execution

Push the branch, create or refresh a remote worktree over SSH, then run `docker compose` on the remote host from inside that worktree.

Pros:
- Preserves the exact dev-compose contract
- Exercises the real bind-mounted workflow
- Keeps repo changes small and explicit

Cons:
- Requires a remote clone and SSH access
- Uses SSH as the control plane

### Option 2: Docker context with repo-specific absolute-path overrides

Keep Docker as the primary interface, but introduce a repo-specific compose override that swaps local relative bind mounts for absolute remote host paths.

Pros:
- More Docker-native operator surface
- Could reduce direct SSH use for compose commands

Cons:
- Adds a second compose path that can drift from the local dev path
- Still depends on a remote repo tree
- More repo-specific complexity for little gain

### Option 3: Image-first remote smoke path

Build and ship an image, then run it remotely without bind mounts.

Pros:
- Good for deployment smoke tests
- Removes bind-mount assumptions

Cons:
- Does not validate the dev-compose workflow this slice exists to verify
- Loses the hot-reload and mounted-config path

### Recommendation

Use Option 1. It is the smallest change that validates the exact workflow already defined by `docker-compose.dev.yml`.

## Architecture

### Generic helper: `scripts/remote-worktree.sh`

This script is the reusable primitive.

Responsibilities:
- Resolve the target branch
- Push branch state to a selected Git remote
- Open one SSH session to the selected host
- Create or refresh a remote worktree rooted under a configurable worktree directory
- Abort safely if the existing remote worktree is dirty
- Print the resolved remote worktree path as `host:absolute-path`

Explicit non-responsibilities:
- No Docker or compose commands
- No health-check logic
- No repo-specific config mutation
- No GPU checks

### Repo-specific verifier: `scripts/enigma-dev-verify.sh`

This script is the Stability-Toys operator wrapper.

Responsibilities:
- Call `scripts/remote-worktree.sh`
- Parse the returned remote worktree path
- SSH to the remote host and run the repo’s dev-compose verification steps from inside that worktree
- Report health and log outcomes relevant to the deferred verification slice
- Keep the verification steps aligned with this repo’s documented contract

Explicit non-responsibilities:
- It is not a generic remote execution framework
- It is not intended for non-CUDA hosts in v1
- It does not become the generic interface for other repos

## Generic Helper CLI Contract

Path:
- `scripts/remote-worktree.sh`

Defaults:
- `--host enigma`
- `--repo-path ~/workspace/Stability-Toys`
- `--remote origin`
- `--worktrees-dir .worktrees`
- branch resolved from the current local branch when `--branch` is omitted

Flags:
- `--host <host>`
- `--repo-path <path>`
- `--remote <name>`
- `--branch <name>`
- `--worktrees-dir <path>`
- `--dry-run`
- `--help`

Output:
- success prints `host:absolute-path`
- dry-run prints the resolved values and the commands that would be executed

Behavior:
1. Verify the current directory is inside a Git worktree.
2. Resolve the branch from `--branch` or `git branch --show-current`.
3. Reject detached HEAD when no branch can be resolved.
4. Push the branch with `git push <remote> <branch>`.
5. SSH to the selected host in one session.
6. Verify the remote repo path exists and is a Git repository.
7. Run `git fetch <remote>` in the remote repo.
8. Ensure the remote worktrees directory exists.
9. If the target worktree does not exist, create it from `<remote>/<branch>`.
10. If the target worktree exists, abort if it is dirty; otherwise reset it to `<remote>/<branch>`.
11. Print the final resolved remote worktree path.

Refresh semantics:
- Refresh is in-place, not delete-and-recreate.
- The existing worktree must be clean before reset.
- The branch state on the remote host must match the pushed branch tip exactly after refresh.

## Repo-Specific Verification Contract

Path:
- `scripts/enigma-dev-verify.sh`

Scope:
- This script validates the deferred GPU-host check from the fast dev Docker builds work.
- It is CUDA-only in v1.

Expected remote assumptions:
- `enigma` has Docker and NVIDIA runtime support already configured.
- A base clone exists at `~/workspace/Stability-Toys` unless overridden.
- The remote operator has permission to run Docker.

Flow:
1. Prepare or refresh the remote worktree via `scripts/remote-worktree.sh`.
2. SSH to the remote host and change directory into the returned worktree.
3. Build the base image required by the dev compose flow with `docker compose -f docker-cuda.yml build`.
4. Start or rebuild the dev container with `docker compose -f docker-compose.dev.yml up -d --build`.
5. Wait for the server health endpoint at `http://127.0.0.1:4200/docs`.
6. Collect relevant diagnostics from `docker logs lcm-sd-dev`.
7. Exercise the `modes.yaml` reload check.

For step 7, v1 should keep the mutation boundary conservative:
- The script may either stop with an explicit operator handoff for the `modes.yaml` edit, or implement a reversible temporary probe that restores the file after the watcher fires.
- The final implementation must choose one path explicitly. It must not mutate config ambiguously or leave a dirty remote worktree without warning.

## Source-of-Truth Boundary

The source of truth for the remote run is the pushed Git branch plus the remote worktree created from it.

This matters because:
- bind mounts resolve on the remote host, not on the laptop
- `docker-compose.dev.yml` must run from a real repo tree on the Docker host
- local uncommitted edits are intentionally excluded from this workflow

The operator boundary is therefore:
- Git moves committed branch state to the remote host
- SSH prepares the remote worktree
- Docker runs against that remote filesystem

## Failure Handling

`scripts/remote-worktree.sh` should fail hard on:
- not being inside a Git repo
- detached HEAD with no explicit branch
- push rejection or push failure
- SSH connection failure
- missing remote repo path
- remote path not being a Git repo
- remote fetch failure
- dirty existing remote worktree
- remote worktree creation or refresh failure

`scripts/enigma-dev-verify.sh` should fail hard on:
- failure to prepare the remote worktree
- Docker build failure
- Docker compose startup failure
- health endpoint not becoming ready
- inability to inspect the expected container logs
- config reload probe failure, if that probe is automated in v1

## Testing Strategy

### Generic helper

Automated tests should remain stub-based and local:
- dry-run output
- help output
- explicit branch handling
- detached HEAD rejection
- one-session SSH invocation
- remote worktree creation path
- dirty remote worktree rejection
- refresh behavior for a clean existing worktree

The existing `tests/test_enigma_worktree_script.py` contract should be adapted to the renamed generic helper rather than discarded.

### Repo-specific verifier

Most of the verifier can be covered by command-shape tests with stubbed `ssh` and helper invocations:
- it calls `remote-worktree.sh`
- it parses `host:path`
- it runs the expected remote Docker commands in order
- it targets `docker-cuda.yml`, `docker-compose.dev.yml`, `lcm-sd-dev`, and the docs health endpoint

The GPU-backed live check remains a manual or operator-driven verification step and is intentionally separate from the local automated suite.

## Local Versus Shared Future

For this slice, both scripts live in this repo.

The intended long-term split is:
- `scripts/remote-worktree.sh`: generic enough to extract later if it proves useful across repos
- `scripts/enigma-dev-verify.sh`: repo-specific and expected to remain local to Stability-Toys

This avoids premature extraction while still keeping the reusable logic isolated from repo orchestration details.

## Files

Create:
- `scripts/remote-worktree.sh`
- `scripts/enigma-dev-verify.sh`

Modify:
- `tests/test_enigma_worktree_script.py`
- repo docs that need to point operators at the new verification workflow, if any are needed during implementation

## Acceptance Criteria

- A generic helper exists that can sync a branch-backed remote worktree to hosts other than `enigma`
- The generic helper remains free of Docker-specific logic
- A repo-specific verification wrapper exists for the CUDA dev-compose validation path
- The design preserves the bind-mounted `docker-compose.dev.yml` contract
- The implementation does not depend on local uncommitted changes
- The generic helper is structured so later extraction is possible without redesign
