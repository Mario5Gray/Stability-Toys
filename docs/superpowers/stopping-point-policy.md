# Stopping-Point Policy

This policy governs how agents maintain session continuity on this project. It has two layers: continuous discipline (always active) and a closing ritual (triggered on graceful session end). Together they guarantee that any new agent can reconstruct full working state in three commands.

---

## Scope

**This policy covers:** session end — both sudden (user closes without warning) and graceful (user signals intent to stop).

**This policy does not cover:** subagent dispatch, context compaction, or handoff to human reviewers. Those are handled by FP comment discipline and commit tagging respectively.

---

## Layer 1 — Continuous Discipline

These three rules are always active. No exceptions.

### Rule 1: Every commit is resumable

The commit message must include:
- The FP issue ID
- What changed
- The exact next step

**Invalid:** `feat(controlnet): progress on asset store`
**Valid:** `feat(controlnet): add AssetStore insert/resolve/cleanup (STABL-mrgpncim) — next: Task 2 eviction tests`

A cold agent reading `git log` must be able to resume without reading anything else.

### Rule 2: Every FP comment is actionable

Every FP comment must answer three questions:

1. What just landed?
2. What decisions were made, and why?
3. What is the exact next step?

**Invalid:** `Made progress on preprocessing layer.`
**Valid:**
```
Implemented preprocess_controlnet_attachments(). Chose to keep it in a
separate controlnet_preprocessing.py rather than constraints.py — different
concern (asset mutation vs. validation). Next: Task 9 step 3, wire into
lcm_sr_server.py enforce block.
```

"Made progress" is never a valid comment.

### Rule 3: Plan checkboxes are ground truth

- Check off each step immediately when complete. Do not batch.
- The active plan doc must reflect reality at all times, not just at session end.
- A cold agent `Read`-ing the plan must be able to find the exact current position without any other context.

---

## Layer 2 — Closing Ritual

When the user signals session end ("that's all", "we're done", "wrap up", "stop here", "let's stop"), run this ritual **before sending the final response**.

### Step 1 — Commit any uncommitted changes

If the working tree is dirty, commit. The commit message follows Rule 1.

### Step 2 — Write a stopping-point FP comment

Write one comment on each active in-progress issue using this exact format:

```
STOP: <what was completed this session>
NEXT: <exact next action — command or step number in the active plan>
DECIDED: <non-obvious choices made with rationale — omit line if none>
BLOCKED: <anything waiting on external input — omit line if none>
```

Example:

```
STOP: Tasks 1–3 complete (AssetStore, eviction, upload_routes migration).
NEXT: Task 4 step 1 — write failing tests in tests/test_controlnet_preprocessors.py.
DECIDED: AssetStore byte_budget default 512 MB matches spec §3; upload TTL unchanged at 300s.
```

### Step 3 — Verify plan checkboxes

Scan the active plan doc. Check off any completed steps not yet marked. No step may be left in an ambiguous state at session end.

### Step 4 — Confirm to the user

One sentence only:

> "Stopped at [task/step]. Resume: `fp context <id>`."

Nothing more.

---

## Resumption Protocol

At the start of any session, before touching code, run:

```bash
fp issue list --status in-progress     # what is claimed
fp context <active-id>                 # last STOP comment = exact position
git log --oneline -8                   # what landed
```

If the last FP comment contains `NEXT:`, that is the first action.
If no `NEXT:` is present, read the active plan doc and find the first unchecked step.

Do not begin implementation until these three commands have been run. Inferring state from the code is not acceptable. FP is the authoritative source.

---

## Summary

| Layer | When | Mechanism |
|---|---|---|
| Continuous discipline | Always | Commit messages + FP comments + plan checkboxes |
| Closing ritual | Graceful session end | 4-step checklist before final response |
| Resumption | Session start | 3-command reconstruction ritual |

The discipline layer handles sudden stops. The closing ritual handles graceful ends. The resumption protocol closes the loop.
