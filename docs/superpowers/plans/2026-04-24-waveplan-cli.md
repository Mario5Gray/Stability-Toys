# waveplan CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A CLI tool for agents to claim, complete, and inspect tasks from a `*-execution-waves.json` plan file with DAG-aware gating.

**Architecture:** A single Python script (`scripts/waveplan`) that reads a plan JSON and a separate state JSON. It implements four subcommands (peek, pop, fin, get) with DAG gate logic that only allows tasks whose dependencies are all completed.

**Tech Stack:** Python 3, stdlib only (json, argparse, pathlib, datetime, sys)

---

## File Structure

| File | Responsibility |
|------|----------------|
| `scripts/waveplan` | Main CLI script — arg parsing, all operations |

No test file needed — the tool is a CLI with deterministic JSON I/O; testing is done by running the commands.

---

### Task 1: Scaffold the script with shebang and imports

**Files:**
- Create: `scripts/waveplan`

- [ ] **Step 1: Write the script skeleton**

```python
#!/usr/bin/env python3
"""waveplan — CLI for managing execution waves from *-execution-waves.json plans."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


PLAN_DIR = Path(__file__).resolve().parent.parent / "docs" / "superpowers" / "plans"


def load_json(path: Path) -> dict:
    """Load and return parsed JSON from path."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    """Write data as pretty-printed JSON to path."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def find_plan_file() -> Path | None:
    """Find the *-execution-waves.json file in PLAN_DIR. Returns None if none or multiple found."""
    matches = sorted(PLAN_DIR.glob("*-execution-waves.json"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        print("ERROR: no *-execution-waves.json found in docs/superpowers/plans/", file=sys.stderr)
        return None
    print(f"ERROR: multiple *-execution-waves.json files found:\n" + "\n".join(str(m) for m in matches), file=sys.stderr)
    return None


def find_state_file(plan_path: Path) -> Path:
    """Return the state file path alongside the plan."""
    return plan_path.with_suffix(".state.json")


def load_state(state_path: Path, plan_name: str) -> dict:
    """Load state file, creating it if it doesn't exist."""
    if state_path.exists():
        return load_json(state_path)
    return {"plan": plan_name, "taken": {}, "completed": {}}


def now_str() -> str:
    """Return current timestamp as YYYY-MM-DD HH:MM."""
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def is_available(task_id: str, units: dict, state: dict) -> bool:
    """Check if a task is available (not taken/completed and all deps completed)."""
    if task_id in state.get("taken", {}):
        return False
    if task_id in state.get("completed", {}):
        return False
    task = units.get(task_id)
    if task is None:
        return False
    deps = task.get("depends_on", [])
    completed = state.get("completed", {})
    return all(dep in completed for dep in deps)


def next_available_task(units: dict, state: dict) -> str | None:
    """Return the task ID with the lowest wave number that is available, or None."""
    available = [tid for tid in units if is_available(tid, units, state)]
    if not available:
        return None
    # Sort by wave (int), then by task ID string for determinism
    available.sort(key=lambda tid: (units[tid].get("wave", 9999), tid))
    return available[0]


def cmd_peek(args, units: dict, state: dict) -> None:
    """Display the next available task without claiming it."""
    tid = next_available_task(units, state)
    if tid is None:
        print("No available tasks.")
        return
    task = units[tid]
    output = {
        "task_id": tid,
        "task": task.get("task"),
        "title": task.get("title"),
        "kind": task.get("kind"),
        "wave": task.get("wave"),
        "plan_line": task.get("plan_line"),
        "depends_on": task.get("depends_on", []),
        "doc_refs": task.get("doc_refs", []),
        "fp_refs": task.get("fp_refs", []),
        "notes": task.get("notes", []),
    }
    if "command" in task:
        output["command"] = task["command"]
    print(json.dumps(output, indent=2))


def cmd_pop(args, units: dict, state: dict, state_path: Path) -> None:
    """Claim the next available task for the given agent."""
    agent = args.agent
    tid = next_available_task(units, state)
    if tid is None:
        print("No available tasks.")
        return
    task = units[tid]
    ts = now_str()
    state.setdefault("taken", {})[tid] = {
        "taken_by": agent,
        "started_at": ts,
    }
    print(f"Claimed {tid}: {task.get('title')} by {agent}")
    save_json(state_path, state)


def cmd_fin(args, units: dict, state: dict) -> None:
    """Mark a task as completed."""
    tid = args.task_id
    if tid not in units:
        print(f"ERROR: task '{tid}' not found in plan.", file=sys.stderr)
        sys.exit(1)
    if tid in state.get("completed", {}):
        print(f"Task {tid} is already completed.", file=sys.stderr)
        sys.exit(1)
    # Check deps
    deps = units[tid].get("depends_on", [])
    incomplete = [d for d in deps if d not in state.get("completed", {})]
    if incomplete:
        print(f"ERROR: task {tid} has incomplete dependencies: {', '.join(incomplete)}", file=sys.stderr)
        sys.exit(1)
    ts = now_str()
    state.setdefault("completed", {})[tid] = {"finished_at": ts}
    # Remove from taken if present
    state.get("taken", {}).pop(tid, None)
    print(f"Completed {tid}: {units[tid].get('title')}")


def cmd_get(args, units: dict, state: dict) -> None:
    """Report all taken/completed tasks."""
    taken = state.get("taken", {})
    completed = state.get("completed", {})
    all_ids = sorted(set(list(taken.keys()) + list(completed.keys())))
    if not all_ids:
        print("No tasks taken or completed.")
        return
    for tid in all_ids:
        task = units.get(tid, {})
        title = task.get("title", "<unknown>")
        print(f"{tid}, {title}")
        info = taken.get(tid, {})
        if "started_at" in info:
            print(f"started: {info['started_at']}")
        comp = completed.get(tid, {})
        if "finished_at" in comp:
            print(f"finished: {comp['finished_at']}")
        agent = info.get("taken_by", "")
        if agent:
            print(f"by: {agent}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="waveplan — manage execution waves")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("peek", help="Show next available task")
    pop_p = sub.add_parser("pop", help="Claim next available task")
    pop_p.add_argument("agent", help="Agent name claiming the task")
    fin_p = sub.add_parser("fin", help="Mark task as completed")
    fin_p.add_argument("task_id", help="Task ID to complete")
    sub.add_parser("get", help="Report all taken/completed tasks")

    args = parser.parse_args()

    plan_path = find_plan_file()
    if plan_path is None:
        sys.exit(1)

    plan = load_json(plan_path)
    units = plan.get("units", {})
    state_path = find_state_file(plan_path)
    state = load_state(state_path, plan_path.name)

    commands = {
        "peek": cmd_peek,
        "pop": cmd_pop,
        "fin": cmd_fin,
        "get": cmd_get,
    }
    commands[args.command](args, units, state)

    # Save state back
    save_json(state_path, state)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make the script executable**

```bash
chmod +x scripts/waveplan
```

- [ ] **Step 3: Verify the script loads without errors**

```bash
python scripts/waveplan --help
```

Expected output:
```
usage: waveplan [-h] {peek,pop,fin,get} ...

waveplan — manage execution waves

positional arguments:
  {peek,pop,fin,get}
```

---

### Task 2: Test state persistence and basic operations

**Files:**
- Use: `scripts/waveplan`

- [ ] **Step 1: Test state persistence**

```bash
python scripts/waveplan pop psi
python scripts/waveplan get
```

Expected `get` output:
```
T1.1, Write failing registry tests
started: 2026-04-24 14:30
by: psi
```

- [ ] **Step 2: Verify state file was created**

```bash
cat docs/superpowers/plans/2026-04-22-controlnet-track-3-backend-execution-waves.json.state.json
```

Expected: JSON with `taken.T1.1` containing `taken_by: "psi"` and `started_at`.

---

### Task 3: Test DAG gating with the actual plan file

**Files:**
- Use: `docs/superpowers/plans/2026-04-22-controlnet-track-3-backend-execution-waves.json`

- [ ] **Step 1: Reset state and test peek**

```bash
rm -f docs/superpowers/plans/2026-04-22-controlnet-track-3-backend-execution-waves.json.state.json
python scripts/waveplan peek
```

Expected: Should output T1.1 (wave 1, no deps) as JSON.

- [ ] **Step 2: Test pop and verify state file created**

```bash
python scripts/waveplan pop psi
cat docs/superpowers/plans/2026-04-22-controlnet-track-3-backend-execution-waves.json.state.json
```

Expected state file contains `taken.T1.1`.

- [ ] **Step 3: Test that T1.2 is not available (depends on T1.1)**

```bash
python scripts/waveplan peek
```

Expected: Should output T1.2's parent task T3.1 (also wave 1, no deps) — NOT T1.2.

- [ ] **Step 4: Test fin and verify dependent unlocks**

```bash
python scripts/waveplan fin T1.1
python scripts/waveplan peek
```

Expected: After completing T1.1, T1.2 (wave 2, depends on T1.1) should now be the next available task.

- [ ] **Step 5: Test error — fin on task with incomplete deps**

```bash
python scripts/waveplan fin T1.2
```

Expected: Error message about incomplete dependencies.

- [ ] **Step 6: Test error — fin on unknown task**

```bash
python scripts/waveplan fin T99.1
```

Expected: Error message "task 'T99.1' not found in plan."

- [ ] **Step 7: Test error — pop with no available tasks**

Complete all wave 1 tasks, then run:

```bash
python scripts/waveplan pop psi
```

Expected: "No available tasks."

---

### Task 4: Handle multiple plan files and edge cases

**Files:**
- Modify: `scripts/waveplan`

- [ ] **Step 1: Add `--plan` flag to specify plan file explicitly**

Update `main` to accept `--plan` and pass it to `find_plan_file`:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="waveplan — manage execution waves")
    parser.add_argument("--plan", help="Path to the *-execution-waves.json file")
    sub = parser.add_subparsers(dest="command", required=True)
    # ... rest unchanged
```

Update `find_plan_file` to accept optional path:

```python
def find_plan_file(plan_arg: str | None) -> Path | None:
    if plan_arg:
        p = Path(plan_arg)
        if not p.exists():
            print(f"ERROR: plan file '{plan_arg}' not found.", file=sys.stderr)
            return None
        return p
    return _find_auto_plan()


def _find_auto_plan() -> Path | None:
    matches = sorted(PLAN_DIR.glob("*-execution-waves.json"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        print("ERROR: no *-execution-waves.json found in docs/superpowers/plans/", file=sys.stderr)
        return None
    print(f"ERROR: multiple *-execution-waves.json files found:\n" + "\n".join(str(m) for m in matches), file=sys.stderr)
    return None
```

Update the `main()` call to `find_plan_file(plan_path=plan_arg)`:

```python
    plan_path = find_plan_file(args.plan)
```

- [ ] **Step 2: Test explicit plan path**

```bash
python scripts/waveplan --plan docs/superpowers/plans/2026-04-22-controlnet-track-3-backend-execution-waves.json peek
```

- [ ] **Step 3: Test with non-existent plan**

```bash
python scripts/waveplan --plan /tmp/nonexistent.json peek
```

Expected: "ERROR: plan file '/tmp/nonexistent.json' not found."

---

### Task 5: Final cleanup and verification

**Files:**
- Modify: `scripts/waveplan`

- [ ] **Step 1: Ensure clean state after testing**

```bash
rm -f docs/superpowers/plans/2026-04-22-controlnet-track-3-backend-execution-waves.json.state.json
```

- [ ] **Step 2: Full end-to-end test**

```bash
# Peek shows T1.1
python scripts/waveplan peek | python -c "import sys,json; d=json.load(sys.stdin); assert d['task_id']=='T1.1'"

# Pop it
python scripts/waveplan pop psi

# Peek shows next available (T3.1 or T7.4 — both wave 1, no deps)
python scripts/waveplan peek | python -c "import sys,json; d=json.load(sys.stdin); assert d['task_id'] in ('T3.1','T7.4')"

# Complete T1.1
python scripts/waveplan fin T1.1

# Now T1.2 should be available
python scripts/waveplan peek | python -c "import sys,json; d=json.load(sys.stdin); assert d['task_id']=='T1.2'"

# Get shows all activity
python scripts/waveplan get
```

- [ ] **Step 3: Commit**

```bash
git add scripts/waveplan
git commit -m "feat(superpowers): add waveplan CLI for task execution management"
```