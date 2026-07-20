"""Unit tests for the Hunyuan acceptance warning policy (spec section 8).

Runs anywhere — no CUDA, no model, no diffusers import. The live acceptance is
GPU-gated, so the policy deciding whether a run passes must be testable here.
"""

from hunyuan_warning_policy import blocking_ignored_warnings

CONFIG_ATTRS_KNOWN = (
    "The config attributes {'learn_sigma': True, 'norm_type': 'layer_norm'} were "
    "passed to HunyuanDiT2DControlNetModel, but are not expected and will be "
    "ignored. Please verify your config.json configuration file."
)

CONFIG_ATTRS_UNKNOWN = (
    "The config attributes {'learn_sigma': True, 'mystery_key': 7} were "
    "passed to HunyuanDiT2DControlNetModel, but are not expected and will be "
    "ignored. Please verify your config.json configuration file."
)

ROTARY_DROPPED = (
    "cross_attention_kwargs ['image_rotary_emb'] are not expected by "
    "XFormersAttnProcessor and will be ignored."
)

UNKNOWN_IGNORED = "Some other thing will be ignored for reasons unknown."


def test_known_config_attrs_are_allowed():
    # Spec section 8 sanctions exactly these two for the validated Canny artifact.
    assert blocking_ignored_warnings([CONFIG_ATTRS_KNOWN]) == []


def test_quoted_config_values_are_not_mistaken_for_keys():
    # 'layer_norm' is a VALUE in the message; only dict keys count.
    assert blocking_ignored_warnings([CONFIG_ATTRS_KNOWN]) == []


def test_unexpected_config_attr_blocks():
    blocking = blocking_ignored_warnings([CONFIG_ATTRS_UNKNOWN])
    assert blocking == [CONFIG_ATTRS_UNKNOWN]


def test_dropped_cross_attention_kwarg_always_blocks():
    blocking = blocking_ignored_warnings([ROTARY_DROPPED])
    assert blocking == [ROTARY_DROPPED]


def test_unrecognized_ignored_warning_fails_closed():
    assert blocking_ignored_warnings([UNKNOWN_IGNORED]) == [UNKNOWN_IGNORED]


def test_unrelated_warnings_are_untouched():
    assert blocking_ignored_warnings(["You have disabled the safety checker"]) == []


def test_mixed_stream_reports_only_blocking_entries():
    blocking = blocking_ignored_warnings(
        [CONFIG_ATTRS_KNOWN, ROTARY_DROPPED, "unrelated chatter"]
    )
    assert blocking == [ROTARY_DROPPED]
