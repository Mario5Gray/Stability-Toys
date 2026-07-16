import sys
from types import ModuleType

import pytest

from spikes import hunyuandit_controlnet_spike as spike


def _fake_module(name: str, **attrs) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_import_gate_rejects_t5_tokenizer_placeholder(monkeypatch, capsys):
    class Loadable:
        @classmethod
        def from_pretrained(cls):
            return cls()

    class Placeholder:
        pass

    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        _fake_module(
            "diffusers",
            __version__="test",
            HunyuanDiT2DControlNetModel=Loadable,
            HunyuanDiTControlNetPipeline=Loadable,
            HunyuanDiTPipeline=Loadable,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _fake_module(
            "transformers",
            __version__="test",
            BertModel=Loadable,
            T5EncoderModel=Loadable,
            T5Tokenizer=Placeholder,
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        spike.import_gate()

    assert exc_info.value.code == 2
    assert "T5Tokenizer is not loadable" in capsys.readouterr().out
