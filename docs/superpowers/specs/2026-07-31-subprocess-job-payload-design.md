# Subprocess job envelope carries the whole job — design

**Issue:** STABL-spxwqlan (found during STABL-qfjfflrx)
**Date:** 2026-07-31
**Status:** approved (2026-07-31)

## Problem

`encode_job` carries `(req, job_id, resolution_epoch)`. The child rebuilds from exactly
those, so `init_image` and `controlnet_bindings` take their dataclass defaults — `None`
and `[]`. The worker reads them as `getattr(job, 'init_image', None)` and
`getattr(job, "controlnet_bindings", []) or []`, and `None`/`[]` is the legitimate txt2img
shape, so nothing raises.

Under `WORKER_ISOLATION=subprocess`, an img2img request silently becomes txt2img and a
ControlNet request silently generates uncontrolled. The user gets a plausible image that
ignored what they asked for.

Reachable because the facet-3 plan deferred these fields (correct while the handle was
test-only) and `STABL-ptoicrho` then made the handle a production opt-in without closing
the gap. Task 8's acceptance was txt2img, so it could not have caught it.

## The deferral was priced wrong

`STABL-qfjfflrx` records that `ControlNetBinding` "may hold PIL/bytes, need a wire form".
It does not. It is eight plain fields (`str`/`bytes`/`float`) and round-trips pickle
cleanly — verified before designing:

```
pickle.loads(pickle.dumps(binding)) == binding   -> True
```

So the blocker that justified deferring does not exist for the current type.

## Approach

**`JOB_SCHEMA_VERSION = 2`** — the body becomes
`(req, job_id, resolution_epoch, init_image, controlnet_bindings)`.

Version 1 is rejected rather than accepted-with-defaults. Accepting it would reintroduce
exactly this bug for a mixed-version pair, and the existing decoder already fails loudly
on an unknown version, which is the behaviour to keep.

`init_image` rides the pipe with the rest of the body rather than shared memory. That is a
deliberate simplicity choice at this size: an init image is request-scale (hundreds of KB),
the pipe already carries the pickled `GenerateRequest`, and `SharedMemBlob` exists for the
*result* path where the producer/consumer handoff needed it. If init images grow to
multi-MB this is the first thing to revisit.

## The real guard: a completeness test

A runtime rejection of unsupported fields would be dead code the moment the envelope can
carry them. The durable protection is structural — enumerate `GenerationJob`'s dataclass
fields and assert each is either carried by the envelope or named in an explicit
`_NOT_CARRIED` allowlist with a reason.

That test fails when someone adds a field to `GenerationJob` and forgets the envelope,
which is precisely how this defect was born. `fut` is the one legitimate exclusion — a
`Future` is parent-side machinery and cannot cross a process boundary.

## Tests

1. **Real spawn boundary.** A job carrying `init_image` and two `controlnet_bindings` is
   submitted through `SubprocessWorkerHandle`; the child reports back what it received.
   Mocked transport is what missed this originally, so the test uses an actual child.
2. **Completeness.** Every `GenerationJob` field is carried or explicitly excused.
3. **Version rejection.** A v1 envelope is refused by the v2 decoder — no silent
   default-filling.
4. **Round-trip.** `ControlNetBinding` survives encode/decode with `control_image_bytes`
   intact, since a dropped control image is the failure mode with no visible symptom.

## Non-goals

- Moving `init_image` to shared memory (see above).
- The rest of the `STABL-qfjfflrx` seam audit, which this interrupted.
