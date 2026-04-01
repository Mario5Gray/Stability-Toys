# Super-Resolution CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `python -m server.superres_cli` command that runs the existing shared super-resolution service directly from the shell for one input image.

**Architecture:** Keep the CLI thin. Reuse the existing service layer and factor only the SR environment loading needed so the server and CLI resolve backend/config the same way. Validate behavior with stubbed unit tests rather than hardware-dependent integration tests.

**Tech Stack:** Python, argparse, pytest, existing `server.superres_http` and `server.superres_service` modules

---

### Task 1: Add failing CLI tests

**Files:**
- Create: `tests/test_superres_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_run_once_writes_output_and_prints_summary(...):
    ...

def test_main_rejects_missing_input(...):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_superres_cli.py -q`
Expected: FAIL because `server.superres_cli` does not exist yet.

- [ ] **Step 3: Commit**

```bash
git add tests/test_superres_cli.py
git commit -m "Add failing tests for superres CLI"
```

### Task 2: Implement shared SR env loading and CLI

**Files:**
- Modify: `server/superres_http.py`
- Create: `server/superres_cli.py`
- Test: `tests/test_superres_cli.py`

- [ ] **Step 1: Add shared SR env loading helper**

Move the server SR env parsing needed by both the HTTP server and CLI into `server/superres_http.py`.

- [ ] **Step 2: Implement the minimal CLI**

Implement `python -m server.superres_cli` using `argparse`, the shared SR env loader, `initialize_superres_service(...)`, and `submit_superres(...)`.

- [ ] **Step 3: Run CLI tests to verify they pass**

Run: `python3 -m pytest tests/test_superres_cli.py -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add server/superres_http.py server/superres_cli.py tests/test_superres_cli.py
git commit -m "Add shared superres command-line interface"
```

### Task 3: Verify integration surfaces

**Files:**
- Modify: `README.md`
- Test: `tests/test_superres_http.py`
- Test: `tests/test_superres_service.py`
- Test: `tests/test_superres_cli.py`

- [ ] **Step 1: Document CLI usage**

Add one short README section showing command invocation and env expectations.

- [ ] **Step 2: Run focused verification**

Run: `python3 -m pytest tests/test_superres_service.py tests/test_superres_http.py tests/test_superres_cli.py -q`
Expected: PASS

Run: `python3 -m py_compile server/superres_service.py server/superres_http.py server/superres_cli.py`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add README.md tests/test_superres_service.py tests/test_superres_http.py tests/test_superres_cli.py server/superres_service.py server/superres_http.py server/superres_cli.py
git commit -m "Document superres CLI usage"
```
