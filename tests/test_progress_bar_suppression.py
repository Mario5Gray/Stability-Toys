"""The diffusers progress bar is the last non-JSON writer on the log stream.

Found by the STABL-xqqqqvse live container proof: with LOG_FORMAT=json, three of
39 lines were not JSON. Two were prints (fixed by STABL-gjuxibsb). The third was
tqdm:

    Loading pipeline components...:   0%|          | 0/6 [00:00<?, ?it/s]...

It writes to the stream directly rather than through `logging`, so no logging
configuration can capture it — but it CAN be turned off, which is better than
documenting a wart in a stream we call structured.

Gated on LOG_FORMAT, because the bar is genuinely useful to a human watching a
dev container in text mode. It is only corruption when something is parsing.
"""
from unittest.mock import MagicMock

import pytest

from backends.cuda_worker import CudaWorkerBase


class _Pipe:
    """Records set_progress_bar_config calls."""

    def __init__(self):
        self.calls = []

    def set_progress_bar_config(self, **kwargs):
        self.calls.append(kwargs)


def _suppress(pipe):
    """Call the seam the way a worker does, without building a worker."""
    return CudaWorkerBase._maybe_disable_progress_bar(pipe)


def test_json_mode_disables_the_progress_bar(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "json")
    pipe = _Pipe()
    _suppress(pipe)
    assert pipe.calls == [{"disable": True}]


def test_text_mode_LEAVES_THE_BAR_ALONE(monkeypatch):
    """The bar is useful to a human watching a dev container. It is only
    corruption when something is parsing the stream."""
    monkeypatch.delenv("LOG_FORMAT", raising=False)
    pipe = _Pipe()
    _suppress(pipe)
    assert pipe.calls == []


def test_an_UNRECOGNISED_log_format_leaves_the_bar_alone(monkeypatch):
    """resolve_log_format degrades unknown values to text, and this must agree
    with it rather than guess separately."""
    monkeypatch.setenv("LOG_FORMAT", "logfmt")
    pipe = _Pipe()
    _suppress(pipe)
    assert pipe.calls == []


def test_it_uses_the_SAME_resolver_the_formatter_uses(monkeypatch):
    """The gate and the actual output format must not be able to disagree.
    Patching the resolver — not the env — proves this reads THAT function rather
    than re-deriving LOG_FORMAT on its own."""
    import backends.cuda_worker as cw

    monkeypatch.delenv("LOG_FORMAT", raising=False)   # env says text...
    monkeypatch.setattr(cw, "resolve_log_format", lambda: "json")  # ...resolver says json
    pipe = _Pipe()
    _suppress(pipe)
    assert pipe.calls == [{"disable": True}]


def test_a_pipe_without_the_method_is_tolerated(monkeypatch):
    """Not every pipeline object exposes set_progress_bar_config, and a missing
    cosmetic optimisation must never fail a model load."""
    monkeypatch.setenv("LOG_FORMAT", "json")
    _suppress(object())          # must not raise


def test_a_raising_pipe_is_tolerated(monkeypatch):
    """Same reason. This runs inside the load path; suppressing a progress bar is
    not worth an exception that costs a model load."""
    monkeypatch.setenv("LOG_FORMAT", "json")
    pipe = MagicMock()
    pipe.set_progress_bar_config.side_effect = RuntimeError("boom")
    _suppress(pipe)              # must not raise


def test_the_shared_seam_is_wired_into_every_family():
    """ast check: _setup_pipe_memory_opts is the one post-load path all four
    load sites route through, so the suppression belongs there. If a family ever
    stops calling it, this fails rather than silently leaking a progress bar."""
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parent.parent / "backends" / "cuda_worker.py"
    tree = ast.parse(src.read_text(), filename=str(src))

    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_setup_pipe_memory_opts"
    )
    called = {
        n.func.attr for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "_maybe_disable_progress_bar" in called, (
        "_setup_pipe_memory_opts no longer suppresses the progress bar; every "
        "family's load path goes through it, so the suppression belongs there"
    )
