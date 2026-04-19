import pytest
from pydantic import ValidationError

from server.controlnet_models import ControlNetAttachment, ControlNetPreprocessRequest


def test_attachment_accepts_map_ref_path():
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="canny",
        map_asset_ref="asset_abc",
        strength=0.8,
        start_percent=0.0,
        end_percent=1.0,
    )
    assert att.map_asset_ref == "asset_abc"
    assert att.source_asset_ref is None
    assert att.preprocess is None


def test_attachment_accepts_source_plus_preprocess_path():
    att = ControlNetAttachment(
        attachment_id="cn_1",
        control_type="depth",
        source_asset_ref="asset_src",
        preprocess=ControlNetPreprocessRequest(id="depth", options={}),
    )
    assert att.source_asset_ref == "asset_src"
    assert att.preprocess.id == "depth"


def test_attachment_rejects_neither_source_nor_map():
    with pytest.raises(ValidationError, match="map_asset_ref or source_asset_ref"):
        ControlNetAttachment(attachment_id="cn_1", control_type="canny")


def test_attachment_rejects_both_source_and_map():
    with pytest.raises(ValidationError, match="exactly one of"):
        ControlNetAttachment(
            attachment_id="cn_1",
            control_type="canny",
            map_asset_ref="asset_a",
            source_asset_ref="asset_b",
            preprocess=ControlNetPreprocessRequest(id="canny"),
        )


def test_attachment_rejects_source_without_preprocess():
    with pytest.raises(ValidationError, match="preprocess"):
        ControlNetAttachment(
            attachment_id="cn_1",
            control_type="canny",
            source_asset_ref="asset_a",
        )


def test_attachment_rejects_strength_out_of_range():
    with pytest.raises(ValidationError):
        ControlNetAttachment(
            attachment_id="cn_1",
            control_type="canny",
            map_asset_ref="asset_a",
            strength=-0.1,
        )


def test_attachment_rejects_inverted_percent_range():
    with pytest.raises(ValidationError, match="start_percent"):
        ControlNetAttachment(
            attachment_id="cn_1",
            control_type="canny",
            map_asset_ref="asset_a",
            start_percent=0.8,
            end_percent=0.2,
        )


def test_attachment_rejects_blank_attachment_id():
    with pytest.raises(ValidationError):
        ControlNetAttachment(
            attachment_id="",
            control_type="canny",
            map_asset_ref="asset_a",
        )
