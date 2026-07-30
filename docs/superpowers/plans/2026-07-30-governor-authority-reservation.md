# Governor Authority Reservation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **`AGENTS.md` forbids sub-agent-driven development on this project — do not use
> superpowers:subagent-driven-development here.**

**Goal:** Make a generate that targets a mode resolve, admit, and stamp against that
mode's authority — established atomically with the switch's enqueue — so the first
`st gen --mode X` succeeds instead of dying on `StaleResolutionError` or a spurious
"ControlNet provider not yet implemented".

**Architecture:** The Governor reserves a mode switch's resolution epoch *and* its
resolved model at **enqueue** time rather than at load time. Admission binds a targeted
generate to that reservation, so its stamp equals the epoch `_load_mode` will publish and
its ControlNet family / generation defaults come from the mode that will actually run it.
The barrier's epoch comparison is untouched — only what gets stamped changes.

**Tech Stack:** Python 3.11, pytest, `concurrent.futures`, `threading.RLock`; Go 1.2x
for the CLI (cobra).

**Spec:** `docs/superpowers/specs/2026-07-30-governor-authority-reservation-design.md`
**FP:** `STABL-ltefhpkk`, `STABL-iuiwzthc` (both in-progress)
**Branch:** `fix/governor-authority-reservation` (already created, spec committed at `ad70fe1`)

## Global Constraints

- **Python env:** `conda activate stability-toys` first. Expected
  `CONDA_PREFIX=/Users/darkbit1001/miniforge3/envs/stability-toys`. Use `python`, not
  `python3`.
- **Never use `grep`/`find` for symbol lookup.** Use `mcp__language-server-python`
  definition/references or `mcp__treesitter-mcp__find_usages`. `grep` is fine for log
  strings and config values.
- **TDD is mandatory.** Every task is red → green → commit. Never write implementation
  before its failing test.
- **Commit messages** carry the FP issue id, what changed, and the exact next step
  (`docs/superpowers/stopping-point-policy.md` Rule 1), and end with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- **Do not self-advance the waveplan** and do not merge. Stop at "ready for review".
- **`drift check` is NOT a gate on this branch.** It exits 1 on pre-existing stale
  anchors from the device-memory work (`7c6cf42`, `3f0c310`, `e882a75`, `bd7b322`), which
  PHI owns. `backends/governor.py` has zero drift bindings, so this work adds no debt.
- **Epochs are monotone and never reused**, including across failed loads.
- **`get_current_mode()` must keep meaning "the actually-loaded mode".** It has nine call
  sites that depend on it. New observability goes through `get_pending_mode()`.
- **Full suite baseline:** `python -m pytest tests/ -q`. One pre-existing failure in
  `test_mode_config` (hunyuandit) is expected on this baseline and is not yours.

---

## File Structure

| File | Responsibility | Change |
| --- | --- | --- |
| `backends/governor.py` | Reservation state, `_resolve_target`, `_reserve_authority`, `_reserve_and_enqueue_switch`, `_terminal_authority`, `_drop_reservation`, `admit_generation`, `get_pending_mode`, `ModeLoadFailedError`, `_load_mode(reservation=)`, dead-epoch guard, status | Modify |
| `backends/worker_pool.py` | Delegating facade | Add `admit_generation`, `get_pending_mode` passthroughs |
| `server/ws_routes.py` | WS admission | `_build_generate_request` passes `mode`; `handle_job_submit` uses `admit_generation` |
| `server/lcm_sr_server.py` | HTTP admission | Replace the blocking switch + snapshot capture with `admit_generation` |
| `cli/go/cmd/st/gen.go` | CLI generate | Drop the pre-emptive `SwitchMode` + `CurrentMode` round-trip |
| `tests/test_governor.py` | Governor unit tests | Add reservation/admission/barrier tests |
| `tests/test_ws_routes.py` | WS admission tests | Add target-mode admission tests |
| `cli/go/cmd/st/gen_test.go` | CLI test | Assert no pre-emptive switch call |

---

## Task 1: Reservation state, resolution split, and the `switch_mode` guard

Establishes the reservation primitives and stops `switch_mode` from minting a
reservation that will never be published. Also lands the committed reproduction of
`STABL-ltefhpkk` as an `xfail(strict=True)` test, so the bug is pinned in the repo from
the first commit and the suite stays green.

**Files:**
- Modify: `backends/governor.py` (add `ModeLoadFailedError`; `__init__` state;
  `_resolve_target`, `_terminal_authority`, `_reserve_authority`, `_drop_reservation`,
  `get_pending_mode`; rewrite `switch_mode` at `:782-786`)
- Test: `tests/test_governor.py`

**Interfaces:**
- Consumes: existing `ActiveModelSnapshot`, `resolve_model`, `_job_lock`,
  `_worker_available()`.
- Produces:
  - `ModeLoadFailedError(RuntimeError)`
  - `Governor._resolve_target(mode_name: str) -> tuple[ModeConfig, ResolvedModel, LocalModelBinding]` — disk I/O, **never** called under `_job_lock`
  - `Governor._terminal_authority() -> Optional[ActiveModelSnapshot]` — caller **must** hold `_job_lock`
  - `Governor._reserve_authority(mode_name: str) -> ActiveModelSnapshot` — reserve without enqueueing
  - `Governor._drop_reservation(reservation: ActiveModelSnapshot, *, dead: bool) -> None`
  - `Governor.get_pending_mode() -> Optional[str]`
  - `Governor._pending_authorities: list[ActiveModelSnapshot]`, `Governor._dead_epochs: set[int]`

- [ ] **Step 1: Add the shared test helper for multi-mode configs**

`_make_mock_mode_config()` at `tests/test_governor.py:241` returns the same mode for every
name. Reservation tests need distinct modes per name. Append this next to it:

```python
def _make_multi_mode_config(*names, default=None):
    """A mode config where get_mode(name) returns a DISTINCT mode per name.

    Reservation tests need to tell modes apart; _make_mock_mode_config returns one
    shared mode for every name, which cannot express a switch.

    Modes are SimpleNamespace, not Mock, deliberately: _resolve_target deepcopies the
    mode, and deepcopy of a Mock does not reliably preserve a post-construction
    `.name` attribute — which is exactly what these tests assert on. The existing
    _make_subprocess_governor helper uses SimpleNamespace for the same class of reason.
    """
    from backends.conditioning.contracts import ConditioningConfig

    modes = {
        name: SimpleNamespace(
            name=name,
            model_path=f"/models/{name}.safetensors",
            loras=[],
            conditioning=ConditioningConfig(),
            controlnet_policy=None,
        )
        for name in names
    }

    config = Mock()
    config.get_mode.side_effect = lambda n: modes[n]  # KeyError for unknown, as today
    config.get_default_mode.return_value = default or names[0]
    config._modes = modes
    return config


def _resolve_by_path(model_path: str, mode):
    """resolve_model stand-in whose family_id is derived from the mode, so tests can
    assert WHICH mode a reservation resolved against."""
    from backends.model_resolution import LocalModelBinding

    resolved = Mock()
    resolved.profile.family_id = f"family-of-{getattr(mode, 'name', 'unknown')}"
    return resolved, LocalModelBinding(model_path)
```

- [ ] **Step 2: Write the committed reproduction of STABL-ltefhpkk**

Append to `tests/test_governor.py`. It is `xfail(strict=True)`, so today it *fails* (as
required) while the suite stays green; Task 4 removes the marker.

```python
@pytest.mark.xfail(
    strict=True,
    reason="STABL-ltefhpkk: generate stamped against pre-switch authority. "
           "Marker removed in Task 4 when admit_generation lands.",
)
def test_generate_behind_queued_switch_is_not_stale():
    """A generate targeting mode B, admitted while a switch to B is queued ahead of
    it, must execute — not raise StaleResolutionError."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        worker = Mock()
        worker.run_job = Mock(return_value="png")
        worker.configure_conditioning = None
        from backends.worker_handle import InProcessWorkerHandle
        handle = InProcessWorkerHandle(worker_factory=Mock(return_value=worker))

        gov = Governor(
            handle=handle,
            mode_config=_make_multi_mode_config("mode-a", "mode-b", default="mode-a"),
            registry=_make_mock_registry(),
        )
        try:
            gov.switch_mode("mode-b")
            authority = gov.admit_generation("mode-b")
            job = GenerationJob(
                req=Mock(), resolution_epoch=authority.resolution_epoch
            )
            assert gov.submit_job(job).result(timeout=5.0) == "png"
        finally:
            gov.shutdown()
```

- [ ] **Step 3: Run it — confirm it fails for the right reason**

```bash
conda activate stability-toys
python -m pytest tests/test_governor.py::test_generate_behind_queued_switch_is_not_stale -v
```

Expected: **XFAIL**. Now confirm the *cause* is the bug and not a typo — run it with the
marker temporarily bypassed and read the traceback:

```bash
python -m pytest tests/test_governor.py::test_generate_behind_queued_switch_is_not_stale -v --runxfail
```

Expected: `AttributeError: 'Governor' object has no attribute 'admit_generation'`. That is
the correct pre-Task-4 failure. (After Task 4 it must pass; if it ever fails with
`StaleResolutionError` instead, the reservation is not being published — go back to Task 2.)

- [ ] **Step 4: Write the failing tests for the reservation primitives**

```python
def _reservation_governor(*names, default=None):
    """A Governor on a stub handle with distinct modes. Caller must gov.shutdown()."""
    gov = Governor(
        handle=StubHandle(),
        mode_config=_make_multi_mode_config(*names, default=default),
        registry=_make_mock_registry(),
    )
    return gov


def test_reserve_authority_bumps_epoch_and_appends():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            before = gov._resolution_epoch
            reservation = gov._reserve_authority("mode-b")
            assert reservation.resolution_epoch == before + 1
            assert reservation.mode_name == "mode-b"
            assert reservation.resolved.profile.family_id == "family-of-mode-b"
            assert gov._pending_authorities[-1] is reservation
        finally:
            gov.shutdown()


def test_terminal_authority_prefers_pending_over_active():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            with gov._job_lock:
                assert gov._terminal_authority().mode_name == "mode-a"
            reservation = gov._reserve_authority("mode-b")
            with gov._job_lock:
                assert gov._terminal_authority() is reservation
            assert gov.get_pending_mode() == "mode-b"
        finally:
            gov.shutdown()


def test_get_pending_mode_is_none_with_no_reservation():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", default="mode-a")
        try:
            assert gov.get_pending_mode() is None
        finally:
            gov.shutdown()


def test_drop_reservation_removes_by_identity_and_marks_dead():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            reservation = gov._reserve_authority("mode-b")
            gov._drop_reservation(reservation, dead=True)
            assert reservation not in gov._pending_authorities
            assert reservation.resolution_epoch in gov._dead_epochs
            # epoch is NOT rolled back — monotone, never reused
            assert gov._resolution_epoch == reservation.resolution_epoch
        finally:
            gov.shutdown()
```

- [ ] **Step 5: Write the failing tests for the `switch_mode` guard (spec §3.3)**

```python
def test_switch_mode_to_active_mode_reserves_nothing():
    """Spec §3.3: switching to the already-loaded mode must NOT reserve. The dispatch
    fast-path returns already_loaded without calling _load_mode, so a reservation made
    here would never be published — and any generate bound to it would be stamped N+1
    against active N. That is the bug, self-inflicted."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", default="mode-a")
        try:
            before = gov._resolution_epoch
            result = gov.switch_mode("mode-a").result(timeout=2.0)
            assert result == {"mode": "mode-a", "status": "already_loaded"}
            assert gov._pending_authorities == []
            assert gov._resolution_epoch == before
        finally:
            gov.shutdown()


def test_switch_mode_to_pending_mode_reports_already_queued():
    """Terminal is a pending reservation for the same mode: report already_QUEUED, not
    already_loaded — the mode is not loaded yet and status must not claim otherwise."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            gov._stop.set()  # freeze dispatch so the reservation stays pending
            gov._reserve_authority("mode-b")
            before = gov._resolution_epoch
            result = gov.switch_mode("mode-b").result(timeout=2.0)
            assert result == {"mode": "mode-b", "status": "already_queued"}
            assert gov._resolution_epoch == before
        finally:
            gov.shutdown()


def test_switch_mode_to_active_but_evicted_mode_still_reloads():
    """Regression guard: when the active mode's worker was idle-evicted, switching to
    it must still enqueue a reload. The guard mirrors the dispatch fast-path condition
    at governor.py:606, which requires _worker_available()."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", default="mode-a")
        try:
            gov._stop.set()
            gov._handle.unload()  # simulate idle eviction: state -> "dead"
            assert not gov._worker_available()
            before = gov._resolution_epoch
            gov.switch_mode("mode-a")
            assert gov._resolution_epoch == before + 1
            assert gov.get_pending_mode() == "mode-a"
        finally:
            gov.shutdown()


def test_switch_mode_force_always_reserves():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", default="mode-a")
        try:
            gov._stop.set()
            before = gov._resolution_epoch
            gov.switch_mode("mode-a", force=True)
            assert gov._resolution_epoch == before + 1
        finally:
            gov.shutdown()


def test_switch_mode_unknown_mode_still_raises_keyerror():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", default="mode-a")
        try:
            with pytest.raises(KeyError):
                gov.switch_mode("nope")
        finally:
            gov.shutdown()
```

- [ ] **Step 6: Run them — confirm they fail**

```bash
python -m pytest tests/test_governor.py -k "reserve or terminal or pending_mode or switch_mode_to or switch_mode_force or switch_mode_unknown" -v
```

Expected: FAIL — `AttributeError: 'Governor' object has no attribute '_reserve_authority'`
(and siblings). `test_switch_mode_unknown_mode_still_raises_keyerror` should already PASS;
it is a regression guard.

- [ ] **Step 7: Add the exception and the reservation state**

In `backends/governor.py`, beside `StaleResolutionError` at `:46`:

```python
class ModeLoadFailedError(RuntimeError):
    """A queued job was admitted against an authority whose load never completed."""
```

In `Governor.__init__`, immediately after `self._active_snapshot` / `self._resolution_epoch`
are set (`:283-284`) — **before** the `self._load_mode(default_mode)` call at `:314`:

```python
        # Reservations: authority that a queued mode switch WILL publish. The
        # terminal reservation is what a job admitted NOW executes against.
        self._pending_authorities: list[ActiveModelSnapshot] = []
        self._dead_epochs: set[int] = set()
```

- [ ] **Step 8: Add the resolution split and reservation helpers**

Insert above `_load_mode` (`:332`):

```python
    # --- Authority reservation (spec §3.1-§3.3) ---

    def _resolve_target(self, mode_name: str):
        """Detect + resolve a target mode. Performs disk I/O (detect_model) and
        therefore MUST NOT be called while holding _job_lock."""
        mode = deepcopy(self._mode_config.get_mode(mode_name))
        assert mode.model_path is not None
        resolved, binding = resolve_model(mode.model_path, mode)
        return mode, resolved, binding

    def _terminal_authority(self) -> Optional[ActiveModelSnapshot]:
        """The authority a job admitted NOW will execute against: the last queued
        reservation, else the published snapshot. Caller MUST hold _job_lock."""
        if self._pending_authorities:
            return self._pending_authorities[-1]
        return self._active_snapshot

    def _reserve_authority(self, mode_name: str) -> ActiveModelSnapshot:
        """Reserve an epoch + resolved model WITHOUT enqueueing a switch. Used by
        _load_mode's reservation-less callers, where there is no enqueue to pair
        with atomically."""
        mode, resolved, binding = self._resolve_target(mode_name)
        with self._job_lock:
            self._resolution_epoch += 1
            reservation = ActiveModelSnapshot(
                mode_name=mode_name,
                mode=mode,
                resolved=resolved,
                binding=binding,
                resolution_epoch=self._resolution_epoch,
            )
            self._pending_authorities.append(reservation)
            self._last_activity = time.monotonic()
            return reservation

    def _drop_reservation(self, reservation: ActiveModelSnapshot, *, dead: bool) -> None:
        """Remove a reservation by IDENTITY. The epoch is never rolled back — epochs
        are monotone and never reused, including across failures."""
        with self._job_lock:
            self._pending_authorities = [
                r for r in self._pending_authorities if r is not reservation
            ]
            if dead:
                self._dead_epochs.add(reservation.resolution_epoch)

    def get_pending_mode(self) -> Optional[str]:
        """The mode a queued switch is heading to, or None. Distinct from
        get_current_mode(), which stays 'the actually-loaded mode'."""
        with self._job_lock:
            if self._pending_authorities:
                return self._pending_authorities[-1].mode_name
            return None

    def _switch_shortcircuit(self, mode_name: str, worker_ok: bool) -> Optional[dict]:
        """The result to return INSTEAD of reserving, or None to proceed with a
        reservation (spec §3.3). Caller MUST hold _job_lock.

        `worker_ok` is passed in rather than read here: _worker_available() calls into
        the handle, and the handle must never be invoked while _job_lock is held
        (backplane Subscriber<->lock invariant). The dispatch fast-path at :606 also
        reads it unlocked, so the same slight staleness already governs this decision.
        """
        terminal = self._terminal_authority()
        if terminal is None or terminal.mode_name != mode_name:
            return None
        if self._pending_authorities:
            # A switch to this mode is already queued — bind to it, reserve nothing.
            return {"mode": mode_name, "status": "already_queued"}
        if worker_ok:
            # Already active with a live worker: the dispatch fast-path would return
            # already_loaded WITHOUT calling _load_mode, so a reservation made here
            # could never be published.
            return {"mode": mode_name, "status": "already_loaded"}
        # Active mode, but the worker was idle-evicted: fall through and reload.
        return None
```

- [ ] **Step 9: Rewrite `switch_mode` with the short-circuit guard**

Replace `switch_mode` at `backends/governor.py:782-786` entirely:

```python
    def switch_mode(self, mode_name: str, force: bool = False) -> Future:
        logger.info(f"[Governor] Queueing mode switch to: {mode_name} (force={force})")
        self._mode_config.get_mode(mode_name)  # KeyError for unknown mode (unchanged)

        if not force:
            worker_ok = self._worker_available()  # handle call — NOT under _job_lock
            with self._job_lock:
                shortcircuit = self._switch_shortcircuit(mode_name, worker_ok)
            if shortcircuit is not None:
                fut: Future = Future()
                fut.set_result(shortcircuit)
                return fut

        return self._reserve_and_enqueue_switch(mode_name, force=force)
```

`_reserve_and_enqueue_switch` becomes atomic in Task 4. For this task, add a temporary
shim directly below `switch_mode` so Task 1's tests can run in isolation. It already
returns a `Future`, so its signature does not change in Tasks 2 or 4:

```python
    def _reserve_and_enqueue_switch(self, mode_name: str, *, force: bool = False) -> Future:
        """TEMPORARY (Task 1) — Task 2 attaches the reservation to the job, Task 4
        makes reserve+enqueue one critical section."""
        reservation = self._reserve_authority(mode_name)
        job = ModeSwitchJob(target_mode=mode_name, force=force)
        try:
            return self.submit_job(job)
        except Exception:
            self._drop_reservation(reservation, dead=True)
            raise
```

- [ ] **Step 10: Run the task's tests**

```bash
python -m pytest tests/test_governor.py -k "reserve or terminal or pending_mode or switch_mode" -v
```

Expected: PASS (the `xfail` repro from Step 2 stays XFAIL).

- [ ] **Step 11: Run the full Governor + pool suites for regressions**

```bash
python -m pytest tests/test_governor.py tests/test_worker_pool.py -q
```

Expected: all pass, one XFAIL. If `test_governor_owns_epoch_and_snapshot` fails, check
that `_resolve_target` calls `get_mode` **before** bumping the epoch — that test relies on
an unknown-mode `KeyError` leaving the epoch at 0.

- [ ] **Step 12: Commit**

```bash
git add backends/governor.py tests/test_governor.py
git commit -m "feat(governor): authority reservation primitives + switch_mode guard (STABL-ltefhpkk)

Adds _resolve_target/_reserve_authority/_terminal_authority/_drop_reservation/
get_pending_mode, ModeLoadFailedError, and the spec §3.3 guard that stops
switch_mode minting a reservation the dispatch fast-path would never publish.
Commits the STABL-ltefhpkk repro as xfail(strict) so the bug is pinned in-repo.

next: Task 2 — _load_mode publishes the reservation instead of bumping the epoch.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `_load_mode` publishes the reservation

Moves the epoch bump off the load and onto the reservation, so a stamp taken at
admission equals the epoch the load publishes.

**Files:**
- Modify: `backends/governor.py` (`ModeSwitchJob` `:195-215`; `_load_mode` `:332-392`;
  dispatch mode-switch branch `:602-612`)
- Test: `tests/test_governor.py`

**Interfaces:**
- Consumes: Task 1's `_reserve_authority`, `_drop_reservation`, `_pending_authorities`.
- Produces:
  - `ModeSwitchJob.reservation: Optional[ActiveModelSnapshot]` (real dataclass field,
    replacing Task 1's `_reservation` shim)
  - `Governor._load_mode(mode_name: str, reservation: Optional[ActiveModelSnapshot] = None)`

- [ ] **Step 1: Write the failing tests**

```python
def test_load_mode_publishes_the_reserved_epoch():
    """The published snapshot carries the RESERVED epoch — not a fresh bump."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            gov._stop.set()  # freeze dispatch; drive _load_mode directly
            reservation = gov._reserve_authority("mode-b")
            gov._load_mode("mode-b", reservation=reservation)
            snapshot = gov.get_active_model_snapshot()
            assert snapshot is reservation
            assert snapshot.resolution_epoch == reservation.resolution_epoch
            assert gov._pending_authorities == []
            assert gov.get_pending_mode() is None
            assert gov.get_current_mode() == "mode-b"
        finally:
            gov.shutdown()


def test_load_mode_with_reservation_does_not_re_resolve():
    """_load_mode reuses reservation.resolved/.binding, so detect_model leaves the
    dispatch thread entirely."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path) as spy:
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            gov._stop.set()
            reservation = gov._reserve_authority("mode-b")
            calls_after_reserve = spy.call_count
            gov._load_mode("mode-b", reservation=reservation)
            assert spy.call_count == calls_after_reserve
        finally:
            gov.shutdown()


def test_load_mode_without_reservation_reserves_inline():
    """__init__ and direct callers still work: no reservation means reserve inline."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            gov._stop.set()
            before = gov._resolution_epoch
            gov._load_mode("mode-b")
            assert gov.get_active_model_snapshot().resolution_epoch == before + 1
            assert gov._pending_authorities == []
        finally:
            gov.shutdown()


def test_demand_reload_does_not_change_the_epoch():
    """Spec §3.2 / matrix case 10: _reload_from_snapshot is epoch-NEUTRAL. Queued
    generates stamped at epoch N must survive an eviction/reload cycle; a reserve here
    would bump to N+1 and reject every one of them."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", default="mode-a")
        try:
            gov._stop.set()
            epoch_before = gov.get_active_model_snapshot().resolution_epoch
            gov._unload_current_worker()          # simulate idle eviction
            gov._reload_from_snapshot()
            assert gov.get_active_model_snapshot().resolution_epoch == epoch_before
            assert gov._resolution_epoch == epoch_before
            assert gov._pending_authorities == []
        finally:
            gov.shutdown()


def test_mode_switch_job_carries_its_reservation():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            gov._stop.set()
            gov.switch_mode("mode-b")
            queued = list(gov.q.queue)
            switches = [j for j in queued if isinstance(j, ModeSwitchJob)]
            assert len(switches) == 1
            assert switches[0].reservation is gov._pending_authorities[-1]
        finally:
            gov.shutdown()
```

- [ ] **Step 2: Run them — confirm they fail**

```bash
python -m pytest tests/test_governor.py -k "load_mode or demand_reload or carries_its_reservation" -v
```

Expected: FAIL — `TypeError: _load_mode() got an unexpected keyword argument 'reservation'`
and `AttributeError: 'ModeSwitchJob' object has no attribute 'reservation'`.
`test_demand_reload_does_not_change_the_epoch` should already PASS — it is a regression
guard proving `_reload_from_snapshot` stays untouched.

- [ ] **Step 3: Add the `reservation` field to `ModeSwitchJob`**

In `backends/governor.py:195-201`, add the field (keep it last so positional construction
is unaffected):

```python
@dataclass
class ModeSwitchJob(Job):
    """Job for switching model mode."""
    target_mode: str
    on_complete: Optional[Callable] = None
    force: bool = False  # Reload even if target_mode == current_mode
    # The authority this switch WILL publish, reserved atomically with its enqueue.
    # None for switches built outside the Governor (tests, legacy callers): _load_mode
    # then reserves inline.
    reservation: Optional["ActiveModelSnapshot"] = None
```

- [ ] **Step 4: Rewrite `_load_mode` to publish the reservation**

Replace `backends/governor.py:332-392`. The changes: accept `reservation`; reserve inline
when absent; take `mode`/`resolved`/`binding` from the reservation instead of calling
`resolve_model`; publish the reservation object itself; pop it; prune dead epochs.

```python
    def _load_mode(self, mode_name: str, reservation: Optional[ActiveModelSnapshot] = None):
        """Load a mode: build the worker via handle.start(), then publish the
        reservation as the active snapshot.

        The epoch is reserved by the caller (or inline here) BEFORE the load, so a
        generate admitted against the reservation carries the epoch this publishes.
        """
        logger.info(f"[Governor] Loading mode: {mode_name}")
        if reservation is None:
            reservation = self._reserve_authority(mode_name)
        mode = reservation.mode
        resolved, binding = reservation.resolved, reservation.binding

        self._unload_current_worker()  # unregister old mode + tear down worker
        with self._job_lock:
            self._active_snapshot = None

        # Load-time measurement reads a FRESH snapshot() — the one sanctioned
        # fan-out exception (spec §4.1 / MUST-FIX-2).
        allocated_before = _worker_allocated(self._dm.snapshot())

        try:
            self._handle.start(resolved, binding, mode)
        except Exception as e:
            logger.error(f"[Governor] Failed to load mode '{mode_name}': {e}", exc_info=True)
            self._handle.unload()
            with self._job_lock:
                self._current_mode = None
                self._active_snapshot = None
            self._drop_reservation(reservation, dead=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

        vram_allocated = _worker_allocated(self._dm.snapshot())
        vram_used = max(0, vram_allocated - allocated_before)
        logger.info(f"[Governor] VRAM after load: allocated={vram_allocated/1024**3:.2f}GB")

        if mode.loras:
            logger.info(f"[Governor] Loading {len(mode.loras)} LoRAs for mode {mode_name}")

        self._registry.register_model(
            name=mode_name,
            model_path=mode.model_path or "",
            vram_bytes=vram_used,
            worker_id=0,
            loras=[lora.path for lora in mode.loras],
        )

        with self._job_lock:
            self._current_mode = mode_name
            self._active_snapshot = reservation
            self._pending_authorities = [
                r for r in self._pending_authorities if r is not reservation
            ]
            # Prune dead epochs below the published one: monotone epochs plus
            # terminal-only admission mean no NEW job can carry them (spec §3.5).
            self._dead_epochs = {
                e for e in self._dead_epochs if e >= reservation.resolution_epoch
            }

        logger.info(f"[Governor] Mode '{mode_name}' loaded (epoch={reservation.resolution_epoch})")

        self._start_dispatch_thread()
```

Note: `vram_total = self._registry.get_total_vram()` was dead (assigned, never used) in the
original; it is dropped. Keep everything else byte-equivalent.

- [ ] **Step 5: Pass the reservation through the dispatch branch**

At `backends/governor.py:610`, change the `_load_mode` call:

```python
                        result = job.execute(self._handle.worker)
                        self._load_mode(job.target_mode, reservation=job.reservation)
```

- [ ] **Step 6: Attach the reservation to the job**

Replace the temporary `_reserve_and_enqueue_switch` body so the job carries its
reservation. `switch_mode` is unchanged — the return type was already `Future`.

```python
    def _reserve_and_enqueue_switch(self, mode_name: str, *, force: bool = False) -> Future:
        """TEMPORARY (Task 2) — Task 4 makes reserve+enqueue one critical section."""
        reservation = self._reserve_authority(mode_name)
        job = ModeSwitchJob(target_mode=mode_name, force=force, reservation=reservation)
        try:
            return self.submit_job(job)
        except Exception:
            self._drop_reservation(reservation, dead=True)
            raise
```

- [ ] **Step 7: Run the task's tests**

```bash
python -m pytest tests/test_governor.py -k "load_mode or demand_reload or carries_its_reservation" -v
```

Expected: PASS.

- [ ] **Step 8: Run the full suite**

```bash
python -m pytest tests/ -q
```

Expected: only the known `test_mode_config` hunyuandit failure plus the Task 1 XFAIL.
Frozen patch targets that reference `backends.governor._load_mode` still work — the
signature gained a keyword-only-by-convention argument with a default.

- [ ] **Step 9: Commit**

```bash
git add backends/governor.py tests/test_governor.py
git commit -m "feat(governor): _load_mode publishes the reserved authority (STABL-ltefhpkk)

The epoch bump moves off the load and onto the reservation: _load_mode now
publishes the reservation object itself, reusing its resolved/binding so
detect_model leaves the dispatch thread. Demand reload stays epoch-neutral
(regression test added). Failed loads drop their reservation as dead.

next: Task 3 — dead-epoch guard + ModeLoadFailedError at the barrier.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Dead-epoch guard and `ModeLoadFailedError`

Closes the confirmed fallthrough: today the barrier is guarded on `snapshot is not None`,
so after a failed load it is skipped entirely and the job lands in the subprocess branch
with a misleading `RuntimeError("No worker available for generation")`.

**Files:**
- Modify: `backends/governor.py` (barrier `:631-640`)
- Test: `tests/test_governor.py`

**Interfaces:**
- Consumes: Task 1's `_dead_epochs`, `ModeLoadFailedError`; Task 2's failure-path drop.
- Produces: no new public API. The barrier now raises `ModeLoadFailedError` before the
  existing `StaleResolutionError` comparison.

- [ ] **Step 1: Write the failing tests**

```python
def _submit_and_capture(gov, epoch):
    """Submit a generate stamped at `epoch` and return the exception it terminates with."""
    job = GenerationJob(req=Mock(), resolution_epoch=epoch)
    fut = gov.submit_job(job)
    try:
        fut.result(timeout=5.0)
    except Exception as exc:
        return exc
    return None


def test_generate_stamped_at_dead_epoch_raises_mode_load_failed():
    """Matrix case 7: the target's load failed; the queued generate must say so, not
    fall through to the subprocess branch."""
    from backends.governor import ModeLoadFailedError

    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            gov._handle.start = Mock(side_effect=RuntimeError("checkpoint is corrupt"))
            reservation = gov._reserve_authority("mode-b")
            with pytest.raises(RuntimeError, match="checkpoint is corrupt"):
                gov._load_mode("mode-b", reservation=reservation)
            assert reservation.resolution_epoch in gov._dead_epochs

            exc = _submit_and_capture(gov, reservation.resolution_epoch)
            assert isinstance(exc, ModeLoadFailedError)
        finally:
            gov.shutdown()


def test_generate_with_no_active_snapshot_raises_mode_load_failed():
    """Matrix case 8: no authority at all means the job cannot run. Today the barrier
    is SKIPPED (guarded on `snapshot is not None`) and the job reaches the subprocess
    branch at governor.py:664, failing with a misleading 'No worker available'."""
    from backends.governor import ModeLoadFailedError

    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", default="mode-a")
        try:
            with gov._job_lock:
                gov._active_snapshot = None
            exc = _submit_and_capture(gov, 1)
            assert isinstance(exc, ModeLoadFailedError)
            assert "No worker available" not in str(exc)
        finally:
            gov.shutdown()


def test_barrier_still_rejects_a_genuinely_superseded_generate():
    """Matrix case 3: the barrier keeps its teeth. A generate admitted for one epoch,
    superseded by an unrelated switch, must still raise StaleResolutionError."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            stale_epoch = gov.get_active_model_snapshot().resolution_epoch
            gov.switch_mode("mode-b").result(timeout=5.0)
            exc = _submit_and_capture(gov, stale_epoch)
            assert isinstance(exc, StaleResolutionError)
        finally:
            gov.shutdown()


def test_dead_epochs_pruned_on_publish():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            gov._stop.set()
            gov._dead_epochs.add(1)
            reservation = gov._reserve_authority("mode-b")
            gov._load_mode("mode-b", reservation=reservation)
            assert 1 not in gov._dead_epochs
        finally:
            gov.shutdown()
```

- [ ] **Step 2: Run them — confirm they fail**

```bash
python -m pytest tests/test_governor.py -k "dead_epoch or no_active_snapshot or genuinely_superseded or pruned_on_publish" -v
```

Expected: the two `ModeLoadFailedError` tests FAIL (they get `RuntimeError`/no exception).
`test_barrier_still_rejects_a_genuinely_superseded_generate` should already PASS.

- [ ] **Step 3: Restructure the barrier**

Replace `backends/governor.py:631-640`:

```python
                    # Stale-epoch barrier. The dead-epoch / no-authority guard runs
                    # FIRST: a job whose target mode failed to load has no authority to
                    # run against, and the old `snapshot is not None` guard let it fall
                    # through to the subprocess branch with a misleading error.
                    if generation_job is not None:
                        with self._job_lock:
                            snapshot = self._active_snapshot
                            dead = generation_job.resolution_epoch in self._dead_epochs
                        if dead or snapshot is None:
                            raise ModeLoadFailedError(
                                f"job {generation_job.job_id} was admitted against epoch "
                                f"{generation_job.resolution_epoch}, whose mode load did "
                                f"not complete"
                            )
                        if snapshot.resolution_epoch != generation_job.resolution_epoch:
                            raise StaleResolutionError(
                                f"job {generation_job.job_id} stamped epoch "
                                f"{generation_job.resolution_epoch} != active epoch "
                                f"{snapshot.resolution_epoch}"
                            )
```

- [ ] **Step 4: Run the task's tests**

```bash
python -m pytest tests/test_governor.py -k "dead_epoch or no_active_snapshot or genuinely_superseded or pruned_on_publish" -v
```

Expected: PASS.

- [ ] **Step 5: Run the full suite and check for the behavior change**

```bash
python -m pytest tests/ -q
```

The new no-authority rejection is a genuine behavior change: any existing test that
submitted a `GenerationJob` with `_active_snapshot is None` and expected it to run will
now fail. If one does, read it — if it was asserting the old fallthrough, update it to
expect `ModeLoadFailedError` and note it in the commit. Do **not** weaken the guard.

- [ ] **Step 6: Commit**

```bash
git add backends/governor.py tests/test_governor.py
git commit -m "fix(governor): reject dead-epoch and no-authority jobs explicitly (STABL-ltefhpkk)

The barrier was guarded on 'snapshot is not None', so after a failed load it was
skipped and the job fell through to the subprocess branch, failing with a
misleading RuntimeError('No worker available for generation'). A dead-epoch guard
now runs first and raises ModeLoadFailedError; the epoch-equality comparison and
StaleResolutionError are unchanged. Dead epochs pruned on publish.

next: Task 4 — admit_generation + atomic reserve/enqueue.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: `admit_generation` and atomic reserve + enqueue

The Governor-side fix lands here. Removes the Task 1/2 shim and makes reserve+enqueue one
critical section, so two concurrent admitters cannot invert queue order against
`_pending_authorities`.

**Files:**
- Modify: `backends/governor.py` (`_reserve_and_enqueue_switch`, new `admit_generation`)
- Modify: `tests/test_governor.py` (remove the Task 1 xfail marker)
- Test: `tests/test_governor.py`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces:
  - `Governor.admit_generation(target_mode: Optional[str]) -> Optional[ActiveModelSnapshot]`
  - `Governor._reserve_and_enqueue_switch(mode_name: str, *, force: bool = False) -> Future`

- [ ] **Step 1: Remove the xfail marker from the Task 1 repro**

In `tests/test_governor.py`, delete the `@pytest.mark.xfail(...)` decorator above
`test_generate_behind_queued_switch_is_not_stale`. Leave the test body unchanged.

- [ ] **Step 2: Write the failing tests**

```python
def test_admit_generation_with_no_target_returns_active_snapshot():
    """Spec §3.4: a generate naming no mode means 'the current mode'. It must NOT bind
    to a pending switch — that is the wrong-model hazard."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            gov._stop.set()
            active = gov.get_active_model_snapshot()
            gov._reserve_authority("mode-b")  # a switch is pending
            assert gov.admit_generation(None) is active
        finally:
            gov.shutdown()


def test_admit_generation_binds_to_the_pending_switch():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            gov._stop.set()
            reservation = gov._reserve_authority("mode-b")
            assert gov.admit_generation("mode-b") is reservation
        finally:
            gov.shutdown()


def test_admit_generation_binds_to_the_active_mode():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", default="mode-a")
        try:
            gov._stop.set()
            active = gov.get_active_model_snapshot()
            assert gov.admit_generation("mode-a") is active
            assert gov._pending_authorities == []
        finally:
            gov.shutdown()


def test_admit_generation_creates_the_switch_for_an_untargeted_mode():
    """The Governor owns the switch: naming a mode that is neither active nor pending
    reserves it AND enqueues the ModeSwitchJob."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            gov._stop.set()
            authority = gov.admit_generation("mode-b")
            assert authority.mode_name == "mode-b"
            assert authority.resolved.profile.family_id == "family-of-mode-b"
            switches = [j for j in list(gov.q.queue) if isinstance(j, ModeSwitchJob)]
            assert len(switches) == 1
            assert switches[0].reservation is authority
        finally:
            gov.shutdown()


def test_admit_generation_is_idempotent_for_the_same_target():
    """Two generates targeting the same not-yet-active mode share ONE switch."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            gov._stop.set()
            first = gov.admit_generation("mode-b")
            second = gov.admit_generation("mode-b")
            assert first is second
            switches = [j for j in list(gov.q.queue) if isinstance(j, ModeSwitchJob)]
            assert len(switches) == 1
        finally:
            gov.shutdown()


def test_admit_generation_rolls_back_the_reservation_on_queue_full():
    """Matrix case 11: a dangling reservation would poison terminal authority for every
    later admission."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            gov._stop.set()
            while True:                      # fill the bounded queue
                try:
                    gov.q.put_nowait(CustomJob(handler=lambda: None))
                except queue.Full:
                    break
            with pytest.raises(queue.Full):
                gov.admit_generation("mode-b")
            assert gov._pending_authorities == []
            assert gov.get_pending_mode() is None
        finally:
            gov.shutdown()


def test_untargeted_generate_superseded_by_a_switch_is_still_rejected():
    """Matrix case 4: mode=None binds to the active snapshot, so a later switch still
    supersedes it. The barrier must reject."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            authority = gov.admit_generation(None)
            gov.switch_mode("mode-b").result(timeout=5.0)
            exc = _submit_and_capture(gov, authority.resolution_epoch)
            assert isinstance(exc, StaleResolutionError)
        finally:
            gov.shutdown()
```

Add `import queue` to the test module's imports if it is not already present.

- [ ] **Step 3: Run them — confirm they fail**

```bash
python -m pytest tests/test_governor.py -k "admit_generation or untargeted_generate or behind_queued_switch" -v
```

Expected: FAIL — `AttributeError: 'Governor' object has no attribute 'admit_generation'`.
`test_generate_behind_queued_switch_is_not_stale` now fails outright (marker removed) —
that is the headline RED for `STABL-ltefhpkk`.

- [ ] **Step 4: Make reserve + enqueue atomic**

Replace the Task 2 `_reserve_and_enqueue_switch` body:

```python
    def _reserve_and_enqueue_switch(self, mode_name: str, *, force: bool = False) -> Future:
        """Reserve the target's authority and enqueue its switch as ONE critical
        section (spec §3.4).

        Holding _job_lock across the bounded q.put cannot deadlock: the dispatch loop
        never holds _job_lock while blocked on q.get. Splitting the two would let a
        concurrent admitter interleave, inverting queue order against
        _pending_authorities.

        Bypasses submit_job deliberately — submit_job cannot hold the lock across its
        put, and a ModeSwitchJob needs no backplane channel (that is GenerationJob-only).
        """
        mode, resolved, binding = self._resolve_target(mode_name)  # disk I/O, no lock
        worker_ok = self._worker_available()  # handle call — NOT under _job_lock
        with self._job_lock:
            if not force:
                # Re-check under the lock: another thread may have reserved or
                # published this target while we were resolving. Without it, a
                # reservation could be minted for a mode the dispatch fast-path will
                # short-circuit — never published, and doom for anything bound to it.
                shortcircuit = self._switch_shortcircuit(mode_name, worker_ok)
                if shortcircuit is not None:
                    fut: Future = Future()
                    fut.set_result(shortcircuit)
                    return fut

            self._resolution_epoch += 1
            reservation = ActiveModelSnapshot(
                mode_name=mode_name,
                mode=mode,
                resolved=resolved,
                binding=binding,
                resolution_epoch=self._resolution_epoch,
            )
            self._pending_authorities.append(reservation)
            self._last_activity = time.monotonic()

            job = ModeSwitchJob(target_mode=mode_name, force=force, reservation=reservation)
            try:
                if self.queue_timeout_s > 0:
                    self.q.put(job, timeout=self.queue_timeout_s)
                else:
                    self.q.put_nowait(job)
            except queue.Full:
                self._pending_authorities = [
                    r for r in self._pending_authorities if r is not reservation
                ]
                self._dead_epochs.add(reservation.resolution_epoch)
                raise
            logger.debug(f"[Governor] Mode switch queued: {mode_name} (epoch={reservation.resolution_epoch})")
            return job.fut
```

- [ ] **Step 5: Add `admit_generation`**

Insert directly below `_reserve_and_enqueue_switch`:

```python
    def admit_generation(self, target_mode: Optional[str]) -> Optional[ActiveModelSnapshot]:
        """The authority a generate admitted NOW will execute against (spec §3.4).

        target_mode is None            -> the active snapshot (today's behavior). A
                                          generate naming no mode means 'the current
                                          mode'; binding it to a pending switch would
                                          silently run it on the wrong model.
        target_mode == terminal's mode -> the terminal authority (active OR a switch
                                          to it already queued).
        anything else                  -> reserve + enqueue the switch; return the
                                          reservation.

        Side-effecting by design: reserve-and-enqueue must be atomic, and a split
        accessor-plus-switch API reintroduces the interleave window this closes.
        """
        if target_mode is None:
            return self.get_active_model_snapshot()
        with self._job_lock:
            terminal = self._terminal_authority()
            if terminal is not None and terminal.mode_name == target_mode:
                return terminal
        self._reserve_and_enqueue_switch(target_mode)
        with self._job_lock:
            return self._terminal_authority()
```

- [ ] **Step 6: Run the task's tests**

```bash
python -m pytest tests/test_governor.py -k "admit_generation or untargeted_generate or behind_queued_switch" -v
```

Expected: PASS — including `test_generate_behind_queued_switch_is_not_stale`, which is the
`STABL-ltefhpkk` acceptance.

- [ ] **Step 7: Run the full suite**

```bash
python -m pytest tests/ -q
```

Expected: only the known `test_mode_config` hunyuandit failure. No XFAILs remain.

- [ ] **Step 8: Commit**

```bash
git add backends/governor.py tests/test_governor.py
git commit -m "feat(governor): admit_generation binds a generate to its target authority (STABL-ltefhpkk)

Reserve + enqueue are now one critical section, so concurrent admitters cannot
invert queue order against _pending_authorities. admit_generation binds a targeted
generate to the terminal authority, creating the switch when the target is neither
active nor pending; mode=None still binds to the active snapshot so the barrier
keeps rejecting genuinely superseded jobs.

STABL-ltefhpkk acceptance test is green (xfail marker removed).

next: Task 5 — wire ws_routes + lcm_sr_server to admit_generation (STABL-iuiwzthc).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Transport wiring — WS and HTTP admission

Closes `STABL-iuiwzthc`. The target mode already arrives in the WS `job:submit` params and
is discarded at `_build_generate_request`; this task reads it and routes both transports
through one admission path.

**Files:**
- Modify: `server/ws_routes.py` (`_build_generate_request` `:366-387`; `handle_job_submit`
  `:184-247`)
- Modify: `server/lcm_sr_server.py` (`generate` `:547-578`)
- Modify: `backends/worker_pool.py` (facade passthroughs)
- Test: `tests/test_ws_routes.py`

**Interfaces:**
- Consumes: `Governor.admit_generation`, `Governor.get_pending_mode`.
- Produces: `WorkerPool.admit_generation`, `WorkerPool.get_pending_mode` (delegating).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ws_routes.py`, following the existing fake-pool pattern in that file:

```python
def test_build_generate_request_passes_the_target_mode():
    """The mode is already on the wire (CLI params) and was silently dropped."""
    from server.ws_routes import _build_generate_request

    req = _build_generate_request({"prompt": "hi", "mode": "mode-b"})
    assert req.mode == "mode-b"


def test_build_generate_request_mode_defaults_to_none():
    from server.ws_routes import _build_generate_request

    assert _build_generate_request({"prompt": "hi"}).mode is None


def test_ws_admission_uses_the_target_authority_for_controlnet():
    """STABL-iuiwzthc / matrix case 2: a ControlNet generate admitted while the target
    is still loading must resolve bindings against the TARGET family, not fall into the
    'no active snapshot' branch that reports the provider as unimplemented."""
    from unittest.mock import MagicMock
    from tests.snapshot_test_helpers import make_active_snapshot, make_family_provider

    target_mode = SimpleNamespace(name="mode-b", controlnet_policy=None)
    target_authority = make_active_snapshot(target_mode, family_id="sdxl", epoch=6)

    pool = MagicMock()
    pool.get_active_model_snapshot.return_value = None   # mid-load: authority is None
    pool.admit_generation.return_value = target_authority

    authority = pool.admit_generation("mode-b")
    assert authority is not None
    assert authority.resolved.profile.family_id == "sdxl"
    assert authority.resolution_epoch == 6
    # The provider admits ControlNet for the TARGET family.
    provider = make_family_provider(family_id="sdxl", supports_controlnet=True)
    assert provider.family_binding("sdxl").execution_capabilities.supports_controlnet


def test_worker_pool_facade_delegates_admission():
    from backends.worker_pool import WorkerPool

    assert hasattr(WorkerPool, "admit_generation")
    assert hasattr(WorkerPool, "get_pending_mode")


def test_generation_defaults_come_from_the_target_mode():
    """Matrix case 6 / spec §1.2: admission binding to the LIVE mode meant
    finalize_mode_generate_request took size/steps/guidance from the OUTGOING mode.
    Masked today only because the barrier rejects the job — so it must be tested
    directly, not through a passing generate."""
    from server.generation_constraints import finalize_mode_generate_request
    from server.ws_routes import _build_generate_request
    from tests.snapshot_test_helpers import make_active_snapshot

    # finalize_mode_generate_request (server/generation_constraints.py:4-24) reads
    # mode.default_size / default_steps / default_guidance, and validates the result
    # against mode.resolution_options — omit that and it raises ValueError on the size.
    outgoing = SimpleNamespace(
        name="mode-a", default_size="512x512", default_steps=4,
        default_guidance=1.0, controlnet_policy=None,
        resolution_options=[{"size": "512x512"}],
    )
    target = SimpleNamespace(
        name="mode-b", default_size="1024x1024", default_steps=25,
        default_guidance=7.5, controlnet_policy=None,
        resolution_options=[{"size": "1024x1024"}],
    )
    target_authority = make_active_snapshot(target, family_id="hunyuandit", epoch=6)

    pool = MagicMock()
    pool.get_active_model_snapshot.return_value = make_active_snapshot(outgoing, epoch=5)
    pool.admit_generation.return_value = target_authority

    req = _build_generate_request({"prompt": "hi", "mode": "mode-b"})
    authority = pool.admit_generation(req.mode)
    finalize_mode_generate_request(
        req, authority.mode,
        env_default_size="512x512", env_default_steps=4, env_default_guidance=1.0,
    )
    assert req.size == "1024x1024"
    assert req.num_inference_steps == 25
    assert req.guidance_scale == 7.5
```

The module path and all four attribute names above were verified against
`server/generation_constraints.py:4-24` while writing this plan — they are facts, not
assumptions. The substitutions only fire when the incoming value still equals the env
default, which is why the test passes the same `env_default_*` values that
`_build_generate_request` uses.

- [ ] **Step 2: Run them — confirm they fail**

```bash
python -m pytest tests/test_ws_routes.py -k "target_mode or mode_defaults_to_none or target_authority or facade_delegates" -v
```

Expected: FAIL — `AssertionError` on `req.mode` (it is `None`), and
`AttributeError` on the facade methods.

- [ ] **Step 3: Pass the mode through `_build_generate_request`**

In `server/ws_routes.py:369-387`, add the field to the `GenerateRequest(...)` call:

```python
    return GenerateRequest(
        prompt=params.get("prompt", ""),
        negative_prompt=params.get("negative_prompt"),
        mode=params.get("mode"),
        scheduler_id=params.get("scheduler_id"),
```

- [ ] **Step 4: Route WS admission through `admit_generation`**

In `server/ws_routes.py`, replace line `:186`:

```python
            snapshot = state.worker_pool.get_active_model_snapshot()
```

with (note `req` must be built first so its `mode` is available — move the
`_build_generate_request` call above the admission read):

```python
            req = _build_generate_request(params)
            snapshot = state.worker_pool.admit_generation(getattr(req, "mode", None))
```

and delete the now-duplicated `req = _build_generate_request(params)` on the following
line. Then replace the epoch stamp at `:242-246`:

```python
                resolution_epoch=snapshot.resolution_epoch,
```

The `current_resolution_epoch()` fallback is removed: with `admit_generation`, `snapshot`
is `None` only when there is genuinely no model, and that path already sets
`pre_submit_job_error` via `ensure_controlnet_dispatch_supported`. Guard the construction
so a `None` authority cannot reach the stamp — wrap the `GenerationJob(...)` block:

```python
        if pre_submit_job_error is None and snapshot is None:
            pre_submit_job_error = "No model loaded"
        if pre_submit_job_error is None:
```

- [ ] **Step 5: Route HTTP admission through the same call**

In `server/lcm_sr_server.py`, replace the blocking switch block at `:552-578`:

```python
    if supports_modes and hasattr(runtime, "admit_generation"):
        # One admission path with the WS route: bind to the mode this request TARGETS,
        # established atomically with the switch. Replaces the former blocking
        # switch_mode(...).result(30s) — the switch is now queued ahead of the job and
        # the job is stamped against the authority that switch will publish.
        try:
            snapshot = runtime.admit_generation(req.mode)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"Mode '{req.mode}' not found in modes.yaml",
            )
        except Exception as e:
            logger.error(f"[/generate] Admission failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Admission failed: {e}")
    else:
        snapshot = (
            runtime.get_active_model_snapshot()
            if supports_modes and hasattr(runtime, "get_active_model_snapshot")
            else None
        )
```

- [ ] **Step 6: Add the facade passthroughs**

In `backends/worker_pool.py`, beside `get_active_model_snapshot` (`:108-112`):

```python
    def admit_generation(self, target_mode):
        return self._governor.admit_generation(target_mode)

    def get_pending_mode(self):
        return self._governor.get_pending_mode()
```

- [ ] **Step 7: Run the task's tests**

```bash
python -m pytest tests/test_ws_routes.py -k "target_mode or mode_defaults_to_none or target_authority or facade_delegates" -v
```

Expected: PASS.

- [ ] **Step 8: Run the WS, HTTP, and Governor suites**

```bash
python -m pytest tests/test_ws_routes.py tests/test_governor.py tests/test_worker_pool.py -q
```

Existing WS tests use `MagicMock` pools; `admit_generation` will auto-mock and return a
`MagicMock`, which is truthy and has a `.resolution_epoch`. If a test asserts on the
stamped epoch, wire it through `install_mode_backed` in `tests/snapshot_test_helpers.py`
by adding:

```python
    pool.admit_generation.return_value = snapshot
```

next to the existing `pool.get_active_model_snapshot.return_value = snapshot` at
`tests/snapshot_test_helpers.py:80`. Prefer fixing the helper once over patching tests
individually.

- [ ] **Step 9: Run the full suite**

```bash
python -m pytest tests/ -q
```

Expected: only the known `test_mode_config` hunyuandit failure.

- [ ] **Step 10: Commit**

```bash
git add server/ws_routes.py server/lcm_sr_server.py backends/worker_pool.py tests/
git commit -m "fix(ws,http): admit generates against the mode they target (STABL-iuiwzthc)

The target mode already arrived in the WS job:submit params and was dropped at
_build_generate_request; it is now read and both transports admit through one
Governor call. A ControlNet generate submitted during a mode switch resolves its
bindings against the TARGET family instead of hitting the 'no active snapshot'
branch that reported the provider as unimplemented. HTTP /generate drops its
blocking switch_mode(...).result(30s) for the same path.

next: Task 6 — pending-mode observability + drop the CLI's pre-emptive switch.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Pending-mode observability and the CLI

Closes the silent "unloaded" window and removes the CLI's pre-emptive switch, which is
the call that creates the racing `ModeSwitchJob`.

**Files:**
- Modify: `backends/governor.py` (`_build_runtime_status` `:523-542`)
- Modify: `server/model_routes.py` (`get_models_status` `:130-154`)
- Modify: `cli/go/cmd/st/gen.go:670-677`
- Test: `tests/test_governor.py`, `cli/go/cmd/st/gen_test.go`

**Interfaces:**
- Consumes: `Governor.get_pending_mode()` (Task 1).
- Produces: `pending_mode` key in the runtime-status payload and in
  `GET /api/models/status`.

- [ ] **Step 1: Write the failing Python tests**

```python
def test_runtime_status_reports_the_pending_mode_during_a_switch():
    """Matrix case 9: while a switch is queued, status must say WHICH mode is coming
    instead of reporting nothing loaded. _load_mode unregisters the outgoing mode
    ~30s before registering the new one, and that window was silent."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            gov._stop.set()
            gov.admit_generation("mode-b")
            status = gov._build_runtime_status()
            assert status["pending_mode"] == "mode-b"
            assert status["current_mode"] == "mode-a"
        finally:
            gov.shutdown()


def test_runtime_status_pending_mode_is_none_when_settled():
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", default="mode-a")
        try:
            assert gov._build_runtime_status()["pending_mode"] is None
        finally:
            gov.shutdown()


def test_reserving_refreshes_last_activity():
    """The idle watchdog must not evict a mode that was just requested."""
    with patch("backends.governor.resolve_model", side_effect=_resolve_by_path):
        gov = _reservation_governor("mode-a", "mode-b", default="mode-a")
        try:
            gov._stop.set()
            gov._last_activity = time.monotonic() - 10_000
            gov.admit_generation("mode-b")
            assert time.monotonic() - gov._last_activity < 5.0
        finally:
            gov.shutdown()
```

- [ ] **Step 2: Run them — confirm they fail**

```bash
python -m pytest tests/test_governor.py -k "pending_mode or last_activity" -v
```

Expected: FAIL — `KeyError: 'pending_mode'`. `test_reserving_refreshes_last_activity`
should already PASS (Task 1 Step 8 set `_last_activity` in `_reserve_authority`, and Task 4
set it in `_reserve_and_enqueue_switch`); it is a regression guard.

- [ ] **Step 3: Add `pending_mode` to the runtime status**

In `backends/governor.py:527-539`, add the key beside `current_mode`:

```python
        payload = {
            "status": status,
            "is_loaded": self.is_model_loaded(),
            "current_mode": self._current_mode,
            "pending_mode": self.get_pending_mode(),
            "queue_size": self.get_queue_size(),
```

- [ ] **Step 4: Surface it on `GET /api/models/status`**

In `server/model_routes.py:130-135`, add the field. Read it defensively so non-mode-system
runtimes are unaffected:

```python
    return {
        "backend": provider.backend_id,
        "backend_version": backend_version,
        "current_mode": current_mode,
        "pending_mode": (
            runtime.get_pending_mode() if hasattr(runtime, "get_pending_mode") else None
        ),
        "is_loaded": runtime.is_model_loaded(),
```

- [ ] **Step 5: Run the Python tests**

```bash
python -m pytest tests/test_governor.py -k "pending_mode or last_activity" -v
python -m pytest tests/ -q
```

Expected: PASS; full suite shows only the known `test_mode_config` failure.

- [ ] **Step 6: Write the failing Go test**

Append to `cli/go/cmd/st/gen_test.go`:

```go
// The server now owns switch+admit atomically (STABL-ltefhpkk): the CLI must NOT
// pre-emptively switch, because that pre-switch is exactly what created the racing
// ModeSwitchJob. params["mode"] already ships in the WS submit frame.
func TestGenDoesNotPreSwitchMode(t *testing.T) {
	var hitSwitch, hitStatus bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/modes/switch":
			hitSwitch = true
		case "/api/models/status":
			hitStatus = true
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"current_mode":"mode-a","is_loaded":true}`))
	}))
	defer srv.Close()

	params := stclient.GenParams{"prompt": "hi", "mode": "mode-b"}
	client := stclient.New(srv.URL)

	// The pre-submit phase must make no switch and no current-mode round-trip.
	if err := preSubmitModeSideEffects(context.Background(), client, params); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if hitSwitch {
		t.Error("CLI called POST /api/modes/switch; the server owns the switch now")
	}
	if hitStatus {
		t.Error("CLI called GET /api/models/status for CurrentMode; round-trip should be gone")
	}
	if params["mode"] != "mode-b" {
		t.Errorf("params[mode] = %v, want mode-b (must still ship in the submit frame)", params["mode"])
	}
}
```

This references a helper that does not exist yet — that is deliberate. Step 8 introduces
`preSubmitModeSideEffects` as the seam, replacing the inline block.

- [ ] **Step 7: Run it — confirm it fails**

```bash
cd cli/go && go test ./cmd/st/... -run TestGenDoesNotPreSwitchMode -v
```

Expected: FAIL to compile — `undefined: preSubmitModeSideEffects`.

- [ ] **Step 8: Replace the pre-emptive switch with a no-op seam**

In `cli/go/cmd/st/gen.go`, replace lines 670-677 entirely:

```go
	// Mode switching is server-side (STABL-ltefhpkk). params["mode"] ships in the
	// submit frame and the Governor admits the generate against the mode it targets,
	// establishing the switch atomically. The CLI must not pre-switch: that call is
	// what created the racing ModeSwitchJob the server then had to reject.
	if err := preSubmitModeSideEffects(ctx, client, params); err != nil {
		return err
	}
```

and add the seam near the other helpers in the same file:

```go
// preSubmitModeSideEffects is the (now empty) pre-submit mode hook. It exists so the
// "CLI does not pre-switch" invariant is testable rather than proven by absence.
func preSubmitModeSideEffects(ctx context.Context, client *stclient.Client, params stclient.GenParams) error {
	return nil
}
```

- [ ] **Step 9: Run the Go tests**

```bash
cd cli/go && go build ./... && go test ./... 
```

Expected: PASS. If `ctx`, `client`, or `stclient` become unused elsewhere in `gen.go`,
the compiler will say so — remove the dead references it names, nothing more.

- [ ] **Step 10: Commit**

```bash
git add backends/governor.py server/model_routes.py cli/go/cmd/st/gen.go tests/test_governor.py cli/go/cmd/st/gen_test.go
git commit -m "feat(governor,cli): expose pending_mode; drop the CLI pre-emptive switch (STABL-ltefhpkk, STABL-iuiwzthc)

_load_mode unregisters the outgoing mode ~30s before registering the new one, so
status reported nothing loaded for the whole window with no log line. get_pending_mode
now surfaces the target through _build_runtime_status and GET /api/models/status, and
reserving refreshes _last_activity so the idle watchdog cannot evict a mode that was
just requested.

The CLI stops calling POST /api/modes/switch before generating: that call created the
racing ModeSwitchJob. params[mode] already ships in the submit frame.

next: live acceptance on enigma, then report ready for review.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Live acceptance and FP closeout

The unit suite cannot prove the enigma repro is dead. This task is verification only — no
new code unless it finds a defect.

**Files:** none (FP + plan checkboxes only)

- [ ] **Step 1: Run the full suite one final time**

```bash
conda activate stability-toys
python -m pytest tests/ -q
```

Record the exact pass/fail counts for the FP evidence. Only the known `test_mode_config`
hunyuandit failure is acceptable.

- [ ] **Step 2: Run the live repro on enigma**

Deploy the branch, then run the exact sequence from the `STABL-ltefhpkk` 2026-07-30
comment:

```bash
st switch lcm-general
st gen 'a forest scene'
st generate --mode HunyuanDiT 'a forest scene'      # must produce an image FIRST try
```

Capture the server log. Required: no `StaleResolutionError`, no
`[Governor] Job failed`, an image returned on the first request.

- [ ] **Step 3: Run the ControlNet repro (STABL-iuiwzthc)**

```bash
st gen --mode SDXL --control-ref canny:<ref> 'a forest scene'   # first time, triggers the switch
```

Required: no "ControlNet provider not yet implemented". Also confirm the WS-disconnect
symptom from the 2026-07-27 comment is gone; if it survives, file it as a new FP issue
with the log rather than widening this branch.

- [ ] **Step 4: Check the unload-after-gen behavior**

After the bundled generate succeeds, poll status:

```bash
curl -s localhost:8000/api/models/status | jq '{current_mode, pending_mode, is_loaded}'
```

Required: `is_loaded: true` and `current_mode` is the target. If the model still unloads
silently, the §1.3 hypothesis was wrong — capture `pending_mode` plus
`MODEL_IDLE_TIMEOUT_SECS` from the deployment and file a new FP issue with that evidence.
Do not guess a second mechanism into this branch.

- [ ] **Step 5: Comment on both FP issues**

Fill every angle-bracket slot with the real value recorded in Steps 1-4 — a comment that
still contains a placeholder is not a valid closeout.

```bash
fp issue assign STABL-ltefhpkk --rev <head-sha>
fp comment STABL-ltefhpkk "STOP: Authority reservation landed (Tasks 1-6). The Governor
reserves a switch's epoch + resolved model at ENQUEUE and admission binds a targeted
generate to that reservation, so the stamp equals the epoch _load_mode publishes. Barrier
epoch comparison unchanged.
EVIDENCE: <N> passed / <M> failed (only the known test_mode_config hunyuandit failure);
live on enigma — 'st switch lcm-general; st gen ...; st generate --mode HunyuanDiT ...'
produced an image on the FIRST request, no StaleResolutionError in the server log.
DECIDED: mode=None still binds the ACTIVE snapshot, so a bare gen racing someone else's
switch is still rejected — binding it to terminal authority would run it on the wrong
model. switch_mode short-circuits before reserving (a reservation the dispatch fast-path
never publishes is the same bug, self-inflicted). Demand reload left epoch-neutral.
NEXT: human review of branch fix/governor-authority-reservation; do NOT merge or mark
done until review completes."

fp issue assign STABL-iuiwzthc --rev <head-sha>
fp comment STABL-iuiwzthc "STOP: Cured by the same reservation fix as STABL-ltefhpkk. The
reservation covers the whole load window, so a targeted generate never reads the transient
None snapshot, and ControlNet bindings resolve against the TARGET family. The
ws_routes.py:220-225 'no active snapshot' branch now fires only for its intended case —
genuinely no model, or a non-CUDA family.
EVIDENCE: <N> passed; live on enigma — 'st gen --mode SDXL --control-ref canny:<ref> ...'
produced an image on the FIRST request, no 'ControlNet provider not yet implemented'.
WS-disconnect symptom from the 2026-07-27 comment: <gone | still present, filed as
STABL-xxxx>.
DECIDED: fix direction 1 from the filing (target-mode-aware admission); direction 3 falls
out as a consequence rather than a separate mechanism.
NEXT: human review of branch fix/governor-authority-reservation; do NOT merge or mark done
until review completes."
```

Each comment states what landed, the live evidence (commands + observed output), and the
exact next step, per `docs/superpowers/stopping-point-policy.md` Rule 2.

- [ ] **Step 6: Report ready for review**

Do **not** mark the FP issues done, do not merge, do not self-advance the waveplan. Report
to the human that the branch is ready for review, with the suite counts and the live
acceptance output.

---

## Verification Summary

| Spec § | Requirement | Task |
| --- | --- | --- |
| §3.1 | Reservation state, terminal authority | 1 |
| §3.2 | Resolution split; `_load_mode` publishes the reservation; demand reload epoch-neutral | 1, 2 |
| §3.3 | `switch_mode` short-circuits before reserving | 1 |
| §3.4 | `admit_generation`; `mode=None` binds active; `queue.Full` unwind | 4 |
| §3.5 | Dead-epoch guard, `ModeLoadFailedError`, pruning | 3 |
| §3.6 | `get_pending_mode`, `pending_mode` in status, `_last_activity` refresh | 1, 6 |
| §5 | WS + HTTP wiring, facade passthroughs, CLI pre-switch removal | 5, 6 |
| §7 case 1 | Generate behind a queued switch | 1 (xfail) → 4 (green) |
| §7 case 2 | ControlNet admitted mid-load | 5 |
| §7 cases 3, 4 | Barrier still rejects superseded jobs | 3, 4 |
| §7 case 5 | Same-mode reservation guard | 1 |
| §7 case 6 | Target-mode generation defaults | 5 |
| §7 cases 7, 8 | Failed load / no authority | 3 |
| §7 case 9 | Mid-switch observability | 6 |
| §7 case 10 | Demand reload epoch-neutral | 2 |
| §7 case 11 | `queue.Full` rollback | 4 |
| §1.3 | Silent unload window | 6, 7 |
