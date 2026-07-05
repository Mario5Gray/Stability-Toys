# tests/test_controlnet_acceptance.py
"""
Track 2 acceptance: reusable artifact flow and eviction coverage.

These tests verify the core Track 2 contract:
  - canny/depth source-ref requests produce stored control_map assets
  - emitted artifact refs are stable and resolvable after preprocessing
  - eviction under byte-budget pressure respects LRU and pinning rules

Generation is intentionally NOT tested here (Track 3 stub returns 501).
The goal is to prove Track 2's asset/preprocessing layer is solid before
Track 3 lands execution.
"""

import io
import pytest
from PIL import Image
from unittest.mock import Mock

from server.asset_store import BucketPolicy, InMemoryAssetStore
from server.controlnet_models import ControlNetAttachment, ControlNetPreprocessRequest
from server.controlnet_preprocessors import ControlMapResult, PreprocessorRegistry
from server.controlnet_preprocessing import preprocess_controlnet_attachments


def _solid_png(w: int = 32, h: int = 32) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=(50, 100, 150)).save(buf, format="PNG")
    return buf.getvalue()


def _fake_reg(pid: str, output_bytes: bytes = b"fake-cmap") -> PreprocessorRegistry:
    result = ControlMapResult(
        preprocessor_id=pid, control_type=pid,
        image_bytes=output_bytes, width=32, height=32,
    )
    p = Mock(preprocessor_id=pid, control_type=pid)
    p.run = Mock(return_value=result)
    reg = PreprocessorRegistry()
    reg.register(p)
    return reg


def _req(controlnets):
    class R:
        pass
    r = R()
    r.controlnets = controlnets
    return r


# ------------------------------------------------------------------ #
# Canny: source → artifact
# ------------------------------------------------------------------ #

def test_canny_source_ref_produces_reusable_artifact():
    store = InMemoryAssetStore()
    source_ref = store.write("upload", _solid_png())

    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        source_asset_ref=source_ref,
        preprocess=ControlNetPreprocessRequest(id="canny"),
    )
    artifacts = preprocess_controlnet_attachments(_req([att]), store, registry=_fake_reg("canny"))

    assert len(artifacts) == 1
    emitted_ref = artifacts[0].asset_ref

    entry = store.resolve(emitted_ref)
    assert entry.bucket == "control_map"
    assert entry.data == b"fake-cmap"
    assert entry.metadata["preprocessor_id"] == "canny"
    assert entry.metadata["source_asset_ref"] == source_ref


# ------------------------------------------------------------------ #
# Depth: source → artifact
# ------------------------------------------------------------------ #

def test_depth_source_ref_produces_reusable_artifact():
    store = InMemoryAssetStore()
    source_ref = store.write("upload", _solid_png())

    att = ControlNetAttachment(
        attachment_id="cn_2",
        control_type="depth",
        source_asset_ref=source_ref,
        preprocess=ControlNetPreprocessRequest(id="depth"),
    )
    artifacts = preprocess_controlnet_attachments(_req([att]), store, registry=_fake_reg("depth", b"depth-map"))

    entry = store.resolve(artifacts[0].asset_ref)
    assert entry.bucket == "control_map"
    assert entry.data == b"depth-map"
    assert entry.metadata["preprocessor_id"] == "depth"


# ------------------------------------------------------------------ #
# Reuse: emitted ref survives and can be supplied as map_asset_ref
# ------------------------------------------------------------------ #

def test_emitted_artifact_ref_reusable_as_map_asset_ref():
    """A ref emitted in one request can be supplied as map_asset_ref in the next."""
    store = InMemoryAssetStore()
    source_ref = store.write("upload", _solid_png())
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        source_asset_ref=source_ref,
        preprocess=ControlNetPreprocessRequest(id="canny"),
    )
    artifacts = preprocess_controlnet_attachments(_req([att]), store, registry=_fake_reg("canny"))
    emitted_ref = artifacts[0].asset_ref

    att2 = ControlNetAttachment(
        attachment_id="cn_2",
        control_type="canny",
        map_asset_ref=emitted_ref,
    )
    artifacts2 = preprocess_controlnet_attachments(_req([att2]), store, registry=_fake_reg("canny"))
    assert artifacts2 == []
    assert store.resolve(emitted_ref).data == b"fake-cmap"


# ------------------------------------------------------------------ #
# Eviction under byte-budget pressure
# ------------------------------------------------------------------ #

def test_lru_eviction_removes_least_recently_used_control_map_when_budget_exceeded():
    # Per-bucket budgets: the control_map bucket is sized to 20 so the three
    # emitted maps (7 + 7 + 10 = 24) exceed it and the LRU map (ref2) is evicted,
    # leaving ref1 (7) + ref3 (10) = 17. The pinned upload source lives in its own
    # bucket and no longer shares the control-map budget.
    store = InMemoryAssetStore(buckets={
        "upload": BucketPolicy("upload", byte_budget=25, ttl_s=300),
        "control_map": BucketPolicy("control_map", byte_budget=20, ttl_s=None),
    })
    source_ref = store.write("upload", b"src")  # 3 bytes
    store.pin(source_ref)

    att1 = ControlNetAttachment(
        attachment_id="cn_1", control_type="canny",
        source_asset_ref=source_ref,
        preprocess=ControlNetPreprocessRequest(id="canny"),
    )
    artifacts1 = preprocess_controlnet_attachments(_req([att1]), store, registry=_fake_reg("canny", b"x" * 7))
    ref1 = artifacts1[0].asset_ref

    att2 = ControlNetAttachment(
        attachment_id="cn_2", control_type="canny",
        source_asset_ref=source_ref,
        preprocess=ControlNetPreprocessRequest(id="canny"),
    )
    artifacts2 = preprocess_controlnet_attachments(_req([att2]), store, registry=_fake_reg("canny", b"y" * 7))
    ref2 = artifacts2[0].asset_ref

    store.resolve(ref1)  # make ref1 the most recently used control map

    att3 = ControlNetAttachment(
        attachment_id="cn_3", control_type="canny",
        source_asset_ref=source_ref,
        preprocess=ControlNetPreprocessRequest(id="canny"),
    )
    artifacts3 = preprocess_controlnet_attachments(_req([att3]), store, registry=_fake_reg("canny", b"q" * 10))
    ref3 = artifacts3[0].asset_ref

    assert store.total_bytes() <= 25
    assert store.resolve(ref1).data == b"x" * 7
    assert store.resolve(ref3).data == b"q" * 10
    with pytest.raises(KeyError, match="not found or evicted"):
        store.resolve(ref2)
    store.unpin(source_ref)


# Note: the former test_pinned_ref_survives_eviction was removed in the move to
# per-bucket budgets. It asserted cross-bucket global-budget eviction (a pinned
# upload source and an emitted control_map sharing one budget) — a scenario that
# no longer exists now that each bucket has an independent budget. Pinned survival
# and fail-closed-under-pin-pressure are covered by unit tests in
# tests/test_asset_store.py (test_admission_fails_closed_when_pins_exceed_budget,
# test_unpin_allows_later_eviction).


# ------------------------------------------------------------------ #
# Two attachments in one request — both emit
# ------------------------------------------------------------------ #

def test_two_attachments_emit_two_artifacts():
    store = InMemoryAssetStore()
    src1 = store.write("upload", _solid_png())
    src2 = store.write("upload", _solid_png())

    reg = PreprocessorRegistry()
    for pid in ("canny", "depth"):
        r = ControlMapResult(preprocessor_id=pid, control_type=pid,
                             image_bytes=f"{pid}-out".encode(), width=32, height=32)
        p = Mock(preprocessor_id=pid, control_type=pid)
        p.run = Mock(return_value=r)
        reg.register(p)

    att1 = ControlNetAttachment(
        attachment_id="cn_1", control_type="canny",
        source_asset_ref=src1,
        preprocess=ControlNetPreprocessRequest(id="canny"),
    )
    att2 = ControlNetAttachment(
        attachment_id="cn_2", control_type="depth",
        source_asset_ref=src2,
        preprocess=ControlNetPreprocessRequest(id="depth"),
    )
    artifacts = preprocess_controlnet_attachments(_req([att1, att2]), store, registry=reg)
    assert len(artifacts) == 2
    assert artifacts[0].attachment_id == "cn_1"
    assert artifacts[1].attachment_id == "cn_2"
