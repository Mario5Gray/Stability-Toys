from __future__ import annotations

import pickle
from dataclasses import dataclass

# Versioned job wire-form (spec §4.2). A leading schema_version byte gates the
# pickle body.
#
# v2 (STABL-spxwqlan) added init_image + controlnet_bindings. v1 carried only
# (req, job_id, resolution_epoch), so those two took their dataclass DEFAULTS in
# the child — None and [] — which is the legitimate txt2img shape. An img2img
# request therefore silently became txt2img, and a ControlNet request silently
# generated uncontrolled, with no error raised anywhere.
JOB_SCHEMA_VERSION = 2

# The completeness contract, asserted by tests/test_subprocess_worker_handle.py.
# Every GenerationJob field must appear in exactly one of these, so ADDING a field
# without wire support fails a test instead of silently degrading a job. That
# omission is how STABL-spxwqlan was born.
CARRIED_JOB_FIELDS = (
    "req",
    "job_id",
    "resolution_epoch",
    "init_image",
    "controlnet_bindings",
)

NOT_CARRIED_JOB_FIELDS = {
    # Parent-side machinery: a Future cannot cross a process boundary. The child
    # reports its terminal through the backplane sink and the parent's bridge
    # fulfils this future on the near side.
    "fut": "parent-side Future; the backplane sink carries the terminal instead",
    # Derived from the class in Job.__post_init__, identical on both sides.
    "job_type": "derived in __post_init__, not transported",
}


@dataclass
class DecodedJob:
    req: object
    job_id: str
    resolution_epoch: int
    init_image: bytes | None = None
    controlnet_bindings: list | None = None


def encode_job(job) -> bytes:
    """Encode a GenerationJob into the versioned wire-form.

    T0 (a7814c6) resolved the req wire-form: GenerateRequest pickles cleanly across
    the spawn boundary, so the envelope carries the INSTANCE directly.

    ControlNetBinding likewise pickles cleanly — it is eight plain str/bytes/float
    fields. The facet-3 plan deferred it assuming it "may hold PIL/bytes" and would
    need a bespoke wire form; that is not true of the current type, and the
    deferral is what produced STABL-spxwqlan.

    init_image rides the pipe with the rest of the body rather than shared memory:
    it is request-scale, and the pipe already carries the pickled GenerateRequest.
    SharedMemBlob exists for the RESULT path, where the producer/consumer handoff
    actually needed it. Revisit if init images become multi-MB.
    """
    body = pickle.dumps((
        job.req,
        job.job_id,
        job.resolution_epoch,
        getattr(job, "init_image", None),
        list(getattr(job, "controlnet_bindings", []) or []),
    ))
    return bytes([JOB_SCHEMA_VERSION]) + body


def decode_job(raw: bytes) -> DecodedJob:
    """Decode the versioned wire-form.

    Rejects unknown schema versions so a version mismatch across a spawn boundary
    fails loudly. v1 is REFUSED rather than accepted-and-default-filled: filling the
    missing fields with None/[] is precisely the silent degradation of
    STABL-spxwqlan, which a mixed-version pair would otherwise reproduce exactly.
    """
    if not raw:
        raise ValueError("empty job envelope")
    version = raw[0]
    if version != JOB_SCHEMA_VERSION:
        raise ValueError(
            f"unknown job schema_version {version} (this build speaks "
            f"{JOB_SCHEMA_VERSION}); refusing to decode rather than default-fill "
            f"fields the sender may have meant to send"
        )
    req, job_id, resolution_epoch, init_image, controlnet_bindings = pickle.loads(raw[1:])
    return DecodedJob(
        req=req,
        job_id=job_id,
        resolution_epoch=resolution_epoch,
        init_image=init_image,
        controlnet_bindings=controlnet_bindings,
    )
