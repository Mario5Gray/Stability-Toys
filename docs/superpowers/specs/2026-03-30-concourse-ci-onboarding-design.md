# Concourse CI Onboarding Design

**Date:** 2026-03-30
**Status:** Approved

## Summary

Install and onboard Concourse CI on mindgate (self-hosted, Docker Compose) as the primary CI/CD system for GitHub-hosted repos. Pipelines are parameterized templates shared across repos. Local git worktrees can be submitted as first-class build jobs via `fly execute`. All Docker images push to a single Harbor project: `darkbit1001/ci/`.

---

## Infrastructure

**Host:** mindgate — Intel i9-11900k, 32GB RAM, RTX 3070 8GB, running Docker.

**Deployment:** Docker Compose with three services:

| Service | Role |
|---|---|
| `concourse-db` | PostgreSQL 15 — pipeline state, credentials |
| `concourse-web` | ATC scheduler + web UI, exposed on port 8080 |
| `concourse-worker` | Local worker (privileged, for Docker-in-Docker builds) |

GitHub webhooks target `http://mindgate-ip:8080`. The `fly` CLI and web UI are accessed at the same address.

Scaling: additional workers can be added as containers on mindgate or pointed at the web node from remote machines (useful for a future cloud VM stage) without redeploying the core.

---

## Pipeline Architecture

### Shared Template

A single `pipeline-template.yml` in the `concourse-pipelines` repo is parameterized per repo. Each application repo has a corresponding vars file under `vars/` in `concourse-pipelines`:

```yaml
# vars/stability-toys.yml
repo-name: stability-toys
git-url: git@github.com:darkbit1001/Stability-Toys
branch: main
harbor-image: harbor.yourdomain.com/darkbit1001/ci/stability-toys
test-task: tasks/test-python.yml
```

Pipeline registration command (run from within `concourse-pipelines/`):
```bash
fly -t mindgate set-pipeline \
  -p stability-toys \
  -c pipeline-template.yml \
  -l vars/stability-toys.yml \
  -l ~/.concourse/credentials.yml
```

Adding a new repo takes under a minute.

### Standard Pipeline Flow

```
git resource (GitHub push on ((branch)))
  → test task       # language-specific, see below
  → build task      # docker build using repo's Dockerfile
  → push resource   # harbor.yourdomain.com/darkbit1001/ci/((repo-name)):((branch))-((git-sha))
```

### Per-Language Test Tasks

| Language | Task file | Test command |
|---|---|---|
| Python | `ci/tasks/test-python.yml` | `pytest` |
| Go | `ci/tasks/test-go.yml` | `go test ./...` |
| Rust | `ci/tasks/test-rust.yml` | `cargo test` |
| Kotlin/Java | `ci/tasks/test-java.yml` | `./gradlew test` |

Each task runs in an appropriate base image. The `((test-task))` parameter in `pipeline-template.yml` selects which task file to use, so a repo with no tests can set `test-task: ci/tasks/noop.yml`.

### Image Tagging Convention

| Trigger | Tag |
|---|---|
| Push to `main` | `main-<git-sha>` |
| Push to other branch | `<branch-name>-<git-sha>` |
| Worktree submission | user-specified (see below) |

---

## Worktree Job Submission

`fly execute` streams a local directory to Concourse and runs a task against it without requiring a git push. This is used for worktree builds.

```bash
fly -t mindgate execute \
  -c ci/tasks/build-and-push.yml \
  -i repo=. \
  --var harbor-image=harbor.yourdomain.com/darkbit1001/ci/stability-toys \
  --var tag=darkbit1001-my-feature
```

This produces:
```
harbor.yourdomain.com/darkbit1001/ci/stability-toys:darkbit1001-my-feature
```

Downstream servers that watch for the known tag will pick it up. The tag format is intentionally user-controlled so it can be anything meaningful (feature name, ticket ID, timestamp, etc.).

A thin wrapper script (`ci/submit-worktree.sh`) will handle the `fly execute` invocation with sensible defaults (auto-detect repo name, default tag to `$(whoami)-$(git branch --show-current)`).

---

## Secrets Management

`~/.concourse/credentials.yml` lives on mindgate only — never committed to any repo.

```yaml
harbor-host: harbor.yourdomain.com
harbor-user: darkbit1001
harbor-password: ...
github-private-key: |
  -----BEGIN OPENSSH PRIVATE KEY-----
  ...
github-webhook-token: ...
```

Secrets are loaded at `fly set-pipeline` time and stored encrypted in Concourse's PostgreSQL DB. They are never echoed in pipeline logs. All pipeline YAML references secrets as `((secret-name))` interpolation — no hardcoded values.

Future: integrate HashiCorp Vault as a Concourse credential manager backend for rotation and audit logging.

---

## Repository Structure

CI configuration lives in a dedicated `concourse-pipelines` repo (not inside each application repo). This keeps pipeline maintenance centralized.

```
concourse-pipelines/
  pipeline-template.yml         # shared template for all repos
  tasks/
    test-python.yml
    test-go.yml
    test-rust.yml
    test-java.yml
    noop.yml                    # for repos without tests
    build-and-push.yml          # shared by standard pipeline and worktree submission
  vars/
    stability-toys.yml
    repo-two.yml
    ...
  scripts/
    register-pipeline.sh        # wraps fly set-pipeline
    submit-worktree.sh          # wraps fly execute for worktree builds
  docker-compose.yml            # Concourse deployment on mindgate
  README.md
```

---

## Onboarding Sequence

1. Deploy Concourse on mindgate via Docker Compose
2. Install `fly` CLI locally, target mindgate (`fly -t mindgate login`)
3. Create `~/.concourse/credentials.yml` on mindgate and locally
4. Create `concourse-pipelines` repo, add shared template and task files
5. Register first repo pipeline, verify GitHub webhook triggers a build
6. Verify Docker image appears in Harbor under `darkbit1001/ci/`
7. Test `fly execute` worktree submission, verify tag in Harbor
8. Repeat step 5–6 for remaining 3 initial repos
9. Document `register-pipeline.sh` usage for future repos

---

## Out of Scope

- Downstream deployment mechanism (servers watching Harbor for tags) — this is a separate design
- Vault integration — deferred to a future iteration
- Cloud VM worker — deferred until build queue pressure warrants it
- Notification/alerting on build failure — deferred
