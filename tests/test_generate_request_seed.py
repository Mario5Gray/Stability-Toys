"""Seed range validation for GenerateRequest.

SD seeds span the full uint32 range (0 .. 2**32-1). numpy's RandomState (rknn
worker) requires seed <= 2**32-1, and torch's manual_seed (cuda worker) accepts
more, so 2**32-1 is the correct, downstream-safe ceiling.
"""
import pytest
from pydantic import ValidationError

from server.lcm_sr_server import GenerateRequest


def test_accepts_uint32_seed_above_int32_max():
    # Regression: 3097772406 is a valid uint32 seed that exceeded the old
    # int32 (2**31-1) cap.
    req = GenerateRequest(prompt="x", seed=3097772406)
    assert req.seed == 3097772406


def test_accepts_uint32_max_seed():
    req = GenerateRequest(prompt="x", seed=2**32 - 1)
    assert req.seed == 2**32 - 1


def test_rejects_seed_above_uint32_max():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="x", seed=2**32)


def test_rejects_negative_seed():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="x", seed=-1)
