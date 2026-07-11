# Karras Scheduler Variants Design

**Date:** 2026-07-10
**Status:** Approved
**FP:** STABL-jltuulda

## Problem

Operators can request only the scheduler IDs currently hardcoded in
`backends/scheduler_registry.py`. Each registry entry is just a Diffusers class
path string, and `build_scheduler()` always does:

```python
scheduler_cls.from_config(deepcopy(config))
```

That means there is no registry-local way to apply constructor overrides such as
`use_karras_sigmas=True`. Today the only way to get Karras sigmas is to bake the
setting into a checkpoint's `scheduler_config.json`, which then affects every
future scheduler reconstructed from that baseline config.

The immediate user need is narrower: expose Karras-sigma variants for the
existing DPM++ schedulers, especially `dpmpp_sde` on SDXL, without changing the
behavior of current scheduler IDs.

## Goal

1. Add a registry-owned per-ID override surface for scheduler construction.
2. Introduce canonical scheduler IDs `dpmpp_2m_karras` and
   `dpmpp_sde_karras`.
3. Make the new IDs selectable anywhere scheduler IDs already flow today:
   request parsing, mode allowlists, worker selection, and PNG metadata.
4. Preserve byte-compatible behavior for all existing scheduler IDs and default
   mode choices.

## Non-goals

- Arbitrary per-request scheduler kwargs.
- Editing checkpoint `scheduler_config.json` files.
- Adding new scheduler families beyond the two Karras variants.
- Changing HTTP, WS, OpenAPI, CLI flag, or PNG metadata schemas.
- Changing the default scheduler for any mode in this slice.

## Existing Context

Current relevant surfaces:

- `backends/scheduler_registry.py` owns canonical scheduler IDs, normalizes the
  requested ID, imports the Diffusers class, and builds a scheduler from a deep
  copy of the baseline config.
- `backends/cuda_worker.py` resolves the requested/default scheduler ID,
  enforces `allowed_scheduler_ids`, rebuilds from the worker's captured baseline
  scheduler config, and records the selected `scheduler_id` into the emitted
  PNG `lcm` metadata for both SD1.5 and SDXL render paths.
- `conf/modes.yml` currently exposes `dpmpp_2m` and `dpmpp_sde` on the `SDXL`
  mode but not Karras variants.
- `cli/go/USAGE.md`, the request models, and the Go client already treat
  `scheduler_id` as an opaque string. No transport or client code assumes a
  fixed scheduler enum.

This is therefore a registry-and-policy slice, not a transport or API slice.

## Design

### Registry entries become structured specs

Replace the string-only registry map with a structured entry type:

```python
@dataclass(frozen=True)
class SchedulerSpec:
    class_path: str
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)
```

Use this for every entry in a renamed `SCHEDULER_SPECS` mapping.

Why a dataclass instead of a `(class_path, extra_kwargs)` tuple:

- field names keep the registry readable once some entries need overrides and
  others do not
- the surface is easier to extend later without tuple-index churn
- it keeps the "no behavior change for existing IDs" requirement explicit by
  allowing empty `extra_kwargs`

`list_scheduler_ids()` remains a sorted list of mapping keys. Unknown-ID error
behavior stays unchanged apart from listing the expanded key set.

`get_scheduler_class()` remains the class-only resolver: it looks up the
`SchedulerSpec`, imports `spec.class_path`, and returns the scheduler class.
`build_scheduler()` remains responsible for applying `spec.extra_kwargs` during
construction.

### Scheduler construction uses spec-local kwargs

`build_scheduler(scheduler_id, config)` keeps the current deep-copy behavior for
`config` isolation, but it now resolves the full `SchedulerSpec` and calls:

```python
scheduler_cls.from_config(deepcopy(config), **spec.extra_kwargs)
```

This keeps the override authority local to the canonical ID. The baseline
config captured from the loaded pipeline remains untouched, so selecting a
Karras variant for one job does not mutate the worker's stored baseline for the
next job.

### New canonical IDs

Add two new registry entries:

- `dpmpp_2m_karras`
  - class:
    `diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler`
  - kwargs: `{"use_karras_sigmas": True}`
- `dpmpp_sde_karras`
  - class:
    `diffusers.schedulers.scheduling_dpmsolver_singlestep.DPMSolverSinglestepScheduler`
  - kwargs: `{"use_karras_sigmas": True}`

The existing `dpmpp_2m` and `dpmpp_sde` entries keep empty kwargs and therefore
preserve current behavior.

The `dpmpp_sde_karras` class mapping deliberately inherits the existing
`dpmpp_sde` choice of `DPMSolverSinglestepScheduler`. This slice does not
re-evaluate that mapping against Diffusers' separate `DPMSolverSDEScheduler`.

No aliasing layer is needed. These are first-class canonical IDs and should
appear anywhere the system lists or validates scheduler IDs.

### Mode policy changes

Add the two Karras IDs to the `allowed_scheduler_ids` list for the `SDXL` mode
in `conf/modes.yml`.

Keep `default_scheduler_id: dpmpp_2m` unchanged. The point of this slice is to
make the variants selectable, not to change the repo default or silently shift
existing output characteristics.

No SD1.5 mode changes are required in shared repo config because the current
shared mode file does not expose DPM++ selection there. Future operators may
add the new IDs to their own mode allowlists without any further code change.

### Worker and metadata behavior stay unchanged

No `cuda_worker.py` behavior change is required beyond consuming the expanded
registry key set:

- requested IDs still normalize through `normalize_scheduler_id()`
- allowlist enforcement still happens before scheduler construction
- the selected canonical ID is still written into PNG `lcm.scheduler_id`

That means a request using `dpmpp_sde_karras` should be observable in emitted
PNG metadata without adding any new metadata fields.

## Testing

### Registry tests

Extend `tests/test_scheduler_registry.py` to cover:

- migrate the existing fixture that clears and restores `SCHEDULER_IMPORTS` so
  it instead installs structured entries in `SCHEDULER_SPECS`
- structured registry entries resolve correctly
- `build_scheduler()` forwards `extra_kwargs` into `from_config()`
- the config argument is still deep-copied before construction
- existing IDs with empty kwargs remain behavior-compatible
- `list_scheduler_ids()` includes the new Karras IDs

The most important regression guard is the kwargs passthrough test, because
that is the new construction seam this issue introduces.

### Worker selection tests

Extend `tests/test_cuda_worker_capabilities.py` so one focused test proves a
Karras ID survives the full worker selection path:

- request `scheduler_id="dpmpp_sde_karras"`
- allowlist includes only `dpmpp_sde_karras`
- `build_scheduler()` is called with the normalized Karras ID and the baseline
  config copy
- the rebuilt scheduler is installed on the pipeline and the selected ID is
  returned

The existing rejection test already covers the negative allowlist case; it does
not need Karras-specific duplication.

### Shared mode policy coverage

The repository currently has no test that snapshots or loads the shared
`conf/modes.yml` scheduler allowlists. Adding that broader config-coverage seam
is out of scope for this slice. The mode-policy change is verified directly by
reviewing the `SDXL` entry in `conf/modes.yml`; synthetic allowlist fixtures in
other tests do not need updates.

### Metadata proof

Add or extend one worker render-path test that inspects the emitted PNG `lcm`
chunk and confirms `scheduler_id` records the canonical Karras ID exactly as
selected. This proves the new IDs flow through like existing scheduler IDs
without extra metadata plumbing.

## Documentation

No transport or CLI contract docs require structural change because
`scheduler_id` is already an opaque string surface.

Repo-local documentation changes are optional and should stay narrow:

- if a user-facing doc enumerates the shared `SDXL` mode's allowed schedulers,
  update that list
- do not add a new top-level scheduler guide for this slice

`cli/go/USAGE.md` does not need a contract update for correctness; it already
documents `--scheduler` generically.

## Risks and Constraints

- `use_karras_sigmas=True` must be forwarded only for the new canonical IDs.
  Applying it by mutating the baseline config would create sticky behavior
  across later scheduler selections, which is explicitly out of bounds.
- The registry representation change is internal but broad enough that tests
  should protect listing, unknown-ID errors, and class resolution together.
- Shared repo config should expose the new IDs only where operators have
  already opted into DPM++ selection. This avoids widening scheduler policy in
  unrelated modes by accident.

## Acceptance

This design is complete when:

- `build_scheduler("dpmpp_sde_karras", config)` constructs the existing
  singlestep Diffusers scheduler with `use_karras_sigmas=True`
- existing scheduler IDs construct exactly as before
- the shared `SDXL` mode allowlist includes the two new Karras IDs while the
  default remains `dpmpp_2m`
- a generation using a Karras ID records that exact canonical ID in PNG
  metadata
