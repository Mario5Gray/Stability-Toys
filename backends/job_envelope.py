from __future__ import annotations

import pickle
from dataclasses import dataclass

# Versioned job wire-form (spec §4.2). A leading schema_version byte gates the
# pickle body; post-M1 additions are additive (init_image / controlnet_bindings
# ride a later version, not this one — see plan "Deferred").
JOB_SCHEMA_VERSION = 1


@dataclass
class DecodedJob:
    req: object
    job_id: str
    resolution_epoch: int


def encode_job(job) -> bytes:
    """Encode a GenerationJob into the versioned wire-form.

    T0 (a7814c6) resolved the req wire-form: GenerateRequest pickles cleanly
    across the spawn boundary, so the envelope carries the INSTANCE directly
    (no model_dump/model_validate fallback). The body is `(req, job_id,
    resolution_epoch)`; the big payload (init_image / result bytes) rides shared
    memory, not this pipe.
    """
    body = pickle.dumps((job.req, job.job_id, job.resolution_epoch))
    return bytes([JOB_SCHEMA_VERSION]) + body


def decode_job(raw: bytes) -> DecodedJob:
    """Decode the versioned wire-form. Rejects unknown schema versions so a
    version mismatch across a spawn boundary fails loudly, not silently."""
    if not raw:
        raise ValueError("empty job envelope")
    version = raw[0]
    if version != JOB_SCHEMA_VERSION:
        raise ValueError(f"unknown job schema_version {version}")
    req, job_id, resolution_epoch = pickle.loads(raw[1:])
    return DecodedJob(req=req, job_id=job_id, resolution_epoch=resolution_epoch)
