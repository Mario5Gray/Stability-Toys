# Enigma Worktree Sync Design

## Summary

This design adds a repo-local helper script, `scripts/enigma-worktree.sh`, that runs from the laptop, pushes the current branch to `origin`, then SSHes to `enigma` and creates or refreshes a matching remote worktree under `~/workspace/Stability-Toys/.worktrees/<branch>`.

The first version is intentionally narrow. It targets a single remote host by default, works against a single remote repo checkout by default, and updates one branch at a time. The goal is to make remote testing on `enigma` routine without requiring manual Git and SSH sequences for every worktree.

## Goals

- Add a single repo-local helper for laptop-to-`enigma` worktree sync
- Default the remote repo root to `~/workspace/Stability-Toys`
- Push the selected branch to the configured Git remote before touching the remote worktree
- Create the remote worktree if it does not exist
- Refresh the remote worktree to exactly match the pushed branch if it already exists
- Print the final remote worktree path so the next remote command is obvious
- Provide a concrete path that works for the existing `gallery-ux-polish` worktree flow

## Non-Goals

- Running tests on `enigma` in v1
- Force-push support in v1
- Managing multiple remote repos from one invocation
- Config-driven host and repo maps in v1
- Automatic integration with the `continuous` workflow engine in v1
- Syncing uncommitted laptop changes without a Git push

## Current State

- This repo already uses project-local worktrees under `.worktrees/`.
- `.worktrees/` is already ignored by Git, so creating additional remote worktrees under the same relative directory is consistent with current practice.
- The existing gallery UX plan expects a local worktree named `.worktrees/gallery-ux-polish`.
- The remote clone on `enigma` is a prerequisite for this workflow; v1 does not provision `~/workspace/Stability-Toys` automatically.
- Remote testing on `enigma` currently requires manual `git push`, `ssh`, `git fetch`, and `git worktree add` or branch reset steps.

## Proposed Approach

Use one shell script, `scripts/enigma-worktree.sh`, as the laptop-side entrypoint.

The script owns two phases:

1. **Push phase**: determine the target branch and push it to the selected Git remote.
2. **Remote prepare phase**: SSH to `enigma`, fetch updates in the remote clone, and create or refresh `.worktrees/<branch>`.

This split keeps the behavior aligned with normal Git expectations:

- `origin` remains the transport boundary between laptop and remote host
- `enigma` only consumes pushed branch state
- the remote worktree is always derived from the pushed branch tip, not local uncommitted state
- v1 assumes the Git remote name selected by `--remote` is the same on the laptop and on `enigma`
- the remote prepare phase should run in one SSH session so remote failures report as one contiguous step

## CLI Contract

Script path:

- `scripts/enigma-worktree.sh`

Defaults:

- host: `enigma`
- repo path: `~/workspace/Stability-Toys`
- Git remote: `origin`
- branch: current local branch

Flags:

- `--branch <name>`
- `--host <host>`
- `--repo-path <path>`
- `--remote <name>`
- `--dry-run`
- `--help`

`--dry-run` should print resolved values and commands only. It should not push, SSH, fetch, or mutate either host.

The script should be invokable as:

```bash
scripts/enigma-worktree.sh
```

and for the concrete gallery example:

```bash
scripts/enigma-worktree.sh --branch gallery-ux-polish
```

## Workflow

The v1 execution flow is:

1. Verify the local working directory is inside a Git worktree.
2. Resolve the target branch from `--branch` or the current local branch.
3. Reject detached HEAD if no branch can be resolved.
4. Push `<branch>` to `<remote>` using a non-force `git push <remote> <branch>`.
5. SSH to `enigma`.
6. Verify `--repo-path` exists and is a Git repository on the remote host.
7. Run `git fetch <remote>` in the remote repo.
8. Ensure `.worktrees/` exists under the remote repo root.
9. If `.worktrees/<branch>` does not exist, create it from `<remote>/<branch>`.
10. If `.worktrees/<branch>` already exists, abort if it is dirty; otherwise hard reset `.worktrees/<branch>` to `<remote>/<branch>`.
11. Print the final remote worktree path.

## Remote Refresh Semantics

If the remote worktree already exists, v1 should update it in place instead of removing and recreating it.

Recommended remote behavior:

- ensure the branch exists locally on `enigma`
- `git fetch <remote>`
- abort if `.worktrees/<branch>` has uncommitted or staged changes
- inside `.worktrees/<branch>`, switch to the target branch if needed
- hard reset the worktree to `<remote>/<branch>`

This makes repeated invocations idempotent while ensuring testing uses the branch state that was just pushed.

## Failure Handling

The script should stop on the first failed step and print which phase failed.

Expected hard failures:

- not inside a Git repo
- no branch could be resolved
- local push failed
- local push rejected because the branch diverged and v1 does not force-push; the user must resolve divergence manually
- SSH connection failed
- remote repo path does not exist
- remote repo path is not a Git repo
- remote fetch failed
- remote worktree is dirty
- remote worktree create or refresh failed

## Example Outcome

For the existing gallery branch, the expected success output should end with a remote path like:

```text
enigma:/home/<remote-user>/workspace/Stability-Toys/.worktrees/gallery-ux-polish
```

The script should resolve `~` on the remote host before printing this path. That output is the handoff point for manual test commands on `enigma`.

## Deferred Follow-Up

The later `continuous` integration should build on this contract rather than replace it.

Deferred extensions:

- optional remote test command execution after worktree refresh
- config-driven host and repo aliases shared across repos
- per-worktree bootstrap hooks for remote dependencies and symlink setup such as `lcm-sr-ui/node_modules`
- support for pushing a worktree branch to a different named remote
- orchestration that submits or mirrors prepared worktrees into the `continuous` workflow engine
