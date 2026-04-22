import io
from unittest.mock import Mock

import pytest
from PIL import Image
from pydantic import ConfigDict

from server.asset_store import AssetStore
from server.controlnet_models import ControlNetAttachment, ControlNetPreprocessRequest
from server.controlnet_preprocessors import ControlMapResult, PreprocessorRegistry


def _solid_png(w: int = 8, h: int = 8) -> bytes:
    img = Image.new("RGB", (w, h), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fake_registry(preprocessor_id: str = "canny") -> PreprocessorRegistry:
    result = ControlMapResult(
        preprocessor_id=preprocessor_id,
        control_type=preprocessor_id,
        image_bytes=b"cmap-output",
        width=8,
        height=8,
    )
    fake = Mock()
    fake.preprocessor_id = preprocessor_id
    fake.control_type = preprocessor_id
    fake.run = Mock(return_value=result)
    reg = PreprocessorRegistry()
    reg.register(fake)
    return reg


def _req(controlnets):
    class R:
        pass

    r = R()
    r.controlnets = controlnets
    return r


def test_preprocess_source_ref_emits_artifact():
    from server.controlnet_preprocessing import preprocess_controlnet_attachments

    store = AssetStore()
    source_ref = store.insert("upload", _solid_png())

    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        source_asset_ref=source_ref,
        preprocess=ControlNetPreprocessRequest(id="canny", options={}),
    )
    req = _req([att])
    artifacts = preprocess_controlnet_attachments(req, store, registry=_fake_registry("canny"))

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.attachment_id == "cn_1"
    assert artifact.control_type == "canny"
    assert artifact.preprocessor_id == "canny"
    assert artifact.source_asset_ref == source_ref
    assert len(artifact.asset_ref) == 32


def test_preprocess_result_stored_as_control_map():
    from server.controlnet_preprocessing import preprocess_controlnet_attachments

    store = AssetStore()
    source_ref = store.insert("upload", _solid_png())
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        source_asset_ref=source_ref,
        preprocess=ControlNetPreprocessRequest(id="canny"),
    )
    artifacts = preprocess_controlnet_attachments(_req([att]), store, registry=_fake_registry("canny"))
    emitted_ref = artifacts[0].asset_ref
    entry = store.resolve(emitted_ref)
    assert entry.kind == "control_map"
    assert entry.data == b"cmap-output"
    assert entry.metadata["control_type"] == "canny"
    assert entry.metadata["source_asset_ref"] == source_ref


def test_preprocess_updates_attachment_map_asset_ref():
    from server.controlnet_preprocessing import preprocess_controlnet_attachments

    store = AssetStore()
    source_ref = store.insert("upload", _solid_png())
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        source_asset_ref=source_ref,
        preprocess=ControlNetPreprocessRequest(id="canny"),
    )
    req = _req([att])
    artifacts = preprocess_controlnet_attachments(req, store, registry=_fake_registry("canny"))
    assert req.controlnets[0].map_asset_ref == artifacts[0].asset_ref


def test_preprocess_normalizes_attachment_onto_direct_map_path():
    from server.controlnet_preprocessing import preprocess_controlnet_attachments

    store = AssetStore()
    source_ref = store.insert("upload", _solid_png())
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        source_asset_ref=source_ref,
        preprocess=ControlNetPreprocessRequest(id="canny"),
    )
    req = _req([att])

    preprocess_controlnet_attachments(req, store, registry=_fake_registry("canny"))

    assert req.controlnets[0].map_asset_ref is not None
    assert req.controlnets[0].source_asset_ref is None
    assert req.controlnets[0].preprocess is None


def test_preprocess_normalizes_attachment_with_validate_assignment_enabled():
    from server.controlnet_preprocessing import preprocess_controlnet_attachments

    class _StrictAttachment(ControlNetAttachment):
        model_config = ConfigDict(validate_assignment=True)

    store = AssetStore()
    source_ref = store.insert("upload", _solid_png())
    att = _StrictAttachment(
        attachment_id="cn_1",
        control_type="canny",
        source_asset_ref=source_ref,
        preprocess=ControlNetPreprocessRequest(id="canny"),
    )
    req = _req([att])

    preprocess_controlnet_attachments(req, store, registry=_fake_registry("canny"))

    assert req.controlnets[0].map_asset_ref is not None
    assert req.controlnets[0].source_asset_ref is None
    assert req.controlnets[0].preprocess is None


def test_map_asset_ref_attachment_is_skipped():
    from server.controlnet_preprocessing import preprocess_controlnet_attachments

    store = AssetStore()
    map_ref = store.insert("control_map", b"existing-map")
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        map_asset_ref=map_ref,
    )
    artifacts = preprocess_controlnet_attachments(_req([att]), store, registry=_fake_registry())
    assert artifacts == []


def test_missing_source_ref_raises_value_error():
    from server.controlnet_preprocessing import preprocess_controlnet_attachments

    store = AssetStore()
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        source_asset_ref="no-such-ref",
        preprocess=ControlNetPreprocessRequest(id="canny"),
    )
    with pytest.raises(ValueError, match="not found or evicted"):
        preprocess_controlnet_attachments(_req([att]), store, registry=_fake_registry())


def test_unknown_preprocessor_id_raises():
    from server.controlnet_preprocessing import preprocess_controlnet_attachments

    store = AssetStore()
    source_ref = store.insert("upload", b"img")
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        source_asset_ref=source_ref,
        preprocess=ControlNetPreprocessRequest(id="no-such-preprocessor"),
    )
    with pytest.raises(ValueError, match="unknown preprocessor"):
        preprocess_controlnet_attachments(_req([att]), store, registry=_fake_registry())
