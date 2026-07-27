"""Task 3 (facet-3, STABL-rgvxuedo): versioned job wire-form — encode_job/decode_job.

Carries {req, job_id, resolution_epoch} behind a leading schema_version byte
(spec §4.2). T0 (a7814c6) resolved the wire-form: GenerateRequest pickles
cleanly across the spawn boundary, so the envelope carries the INSTANCE
directly (no model_dump/model_validate fallback).

NOTE: the plan's draft test used GenerateRequest(prompt=..., steps=4,
width=512, height=512); the real model (server/lcm_sr_server.py:136) fields are
num_inference_steps + size ("512x512") — same correction T0 applied. The plan's
unused `from unittest.mock import Mock` import is dropped (dead import).
"""
from backends.job_envelope import encode_job, decode_job, JOB_SCHEMA_VERSION
from backends.governor import GenerationJob


def test_job_envelope_round_trips_minimal_fields():
    from server.lcm_sr_server import GenerateRequest
    req = GenerateRequest(prompt="x", num_inference_steps=4, size="512x512")
    job = GenerationJob(req=req, resolution_epoch=7)
    raw = encode_job(job)
    assert raw[0] == JOB_SCHEMA_VERSION           # leading version byte
    d = decode_job(raw)
    assert d.job_id == job.job_id
    assert d.resolution_epoch == 7
    assert d.req.prompt == "x"


def test_job_envelope_rejects_unknown_version():
    import pytest
    with pytest.raises(ValueError):
        decode_job(bytes([99]) + b"garbage")
