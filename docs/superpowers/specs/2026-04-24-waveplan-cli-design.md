# waveplan CLI — Design

**Date:** 2026-04-24
**Status:** Approved

## Purpose

A CLI tool for managing execution of tasks defined in a `*-execution-waves.json` plan file.
Agents use it to claim, complete, and inspect tasks with DAG-aware gating.

## Operations

| Command | Description |
|---------|-------------|
| `waveplan peek` | Display the next available task (JSON) without claiming it |
| `waveplan pop <agent_name>` | Claim the next available task, record `taken_by` + `started_at` |
| `waveplan fin <task_id>` | Mark task as done, record `finished_at`, unlock dependents |
| `waveplan get` | List all taken/completed tasks with timestamps in report format |

## State Storage

A separate JSON state file lives alongside the plan:

```
docs/superpowers/plans/<plan_name>.state.json
```

State file format:

```json
{
  "plan": "2026-04-22-controlnet-track-3-backend-execution-waves.json",
  "taken": {
    "T1.1": {
      "taken_by": "psi",
      "started_at": "2026-04-24 14:30"
    }
  },
  "completed": {
    "T1.1": {
      "finished_at": "2026-04-24 15:00"
    }
  }
}
```

## DAG Gate Logic

A task is **available** when:
1. It is NOT already taken or completed (not in state)
2. ALL tasks in its `depends_on` array ARE completed in the state file

`peek` / `pop` pick the available task with the **lowest wave number**, breaking ties by task ID order.

## Report Format (`get`)

```
T1.1, Write failing registry tests
started: 2026-04-24 14:30
finished: 2026-04-24 15:00
by: psi
```

No `finished_at` line if the task is still in progress.

## Error Handling

- Unknown plan file → error
- Task ID not found in plan → error
- `fin` on task with incomplete deps → error (warn user)
- `pop` when no available tasks → empty message
- `pop` with no agent name → error