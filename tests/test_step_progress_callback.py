"""STABL-zueslhah Task 2 — map a diffusers step callback to a Progress emitter.

Family-agnostic by signature probing (decision B): prefer the modern
callback_on_step_end, fall back to the legacy callback/callback_steps, and skip
cleanly when the pipeline supports neither. Tested at the helper level so no full
worker/pipeline is needed; the call-site injection is mechanical.
"""
from backends.step_progress import inject_step_progress


def test_modern_callback_on_step_end():
    calls = []

    class _Pipe:
        def __call__(self, callback_on_step_end=None, **kw):
            ...

    kwargs = {}
    inject_step_progress(_Pipe(), kwargs, lambda s, t, st: calls.append((s, t, st)), total=30)
    assert "callback_on_step_end" in kwargs

    cb = kwargs["callback_on_step_end"]
    # diffusers calls cb(pipe, step_index, timestep, callback_kwargs) -> dict
    assert cb("PIPE", 0, 999, {"latents": 1}) == {"latents": 1}
    cb("PIPE", 29, 0, {})
    assert calls == [(1, 30, "denoise"), (30, 30, "denoise")]  # 1-based step


def test_legacy_callback_callback_steps():
    calls = []

    class _Pipe:
        def __call__(self, callback=None, callback_steps=1, **kw):
            ...

    kwargs = {}
    inject_step_progress(_Pipe(), kwargs, lambda s, t, st: calls.append((s, t, st)), total=10)
    assert kwargs["callback_steps"] == 1
    # legacy signature: cb(step, timestep, latents)
    kwargs["callback"](2, 500, "latents")
    assert calls == [(3, 10, "denoise")]


def test_none_progress_is_noop():
    kwargs = {}
    inject_step_progress(object(), kwargs, None, 10)
    assert kwargs == {}


def test_unsupported_pipe_skips_without_crashing():
    class _Pipe:
        def __call__(self, **kw):  # no named callback param
            ...

    kwargs = {}
    inject_step_progress(_Pipe(), kwargs, lambda *a: None, 10)
    assert kwargs == {}


def test_progress_error_never_propagates():
    """A misbehaving consumer must never break generation."""
    class _Pipe:
        def __call__(self, callback_on_step_end=None, **kw):
            ...

    def _boom(*a):
        raise RuntimeError("consumer blew up")

    kwargs = {}
    inject_step_progress(_Pipe(), kwargs, _boom, total=5)
    # invoking the callback must swallow the error and still return callback_kwargs
    assert kwargs["callback_on_step_end"]("P", 0, 0, {"k": 1}) == {"k": 1}
