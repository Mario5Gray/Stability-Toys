"""Map a diffusers step callback to a backplane Progress emitter (STABL-zueslhah).

Stdlib only (inspect) — kept out of cuda_worker so it imports without torch.

`inject_step_progress` probes the pipeline's __call__ signature and adds the
right callback to pipe_kwargs, family-agnostic (decision B): the modern
`callback_on_step_end` when supported, else the legacy `callback`/`callback_steps`,
else nothing. The emitted step is 1-based (step 0 of N -> progress(1, N)). A
misbehaving consumer never breaks generation — the callback swallows exceptions.
"""
import inspect
from typing import Any, Callable, Optional

ProgressEmitter = Callable[[int, int, str], None]


def inject_step_progress(
    pipe: Any,
    pipe_kwargs: dict,
    progress: Optional[ProgressEmitter],
    total: int,
) -> None:
    if progress is None:
        return
    try:
        params = inspect.signature(pipe.__call__).parameters
    except (ValueError, TypeError):
        return

    total_i = int(total)

    def _emit(step_index: int) -> None:
        try:
            progress(step_index + 1, total_i, "denoise")  # 1-based
        except Exception:
            pass  # a bad consumer must never break generation

    if "callback_on_step_end" in params:
        def _modern(_pipe, step, _timestep, callback_kwargs):
            _emit(step)
            return callback_kwargs
        pipe_kwargs["callback_on_step_end"] = _modern
    elif "callback" in params and "callback_steps" in params:
        def _legacy(step, _timestep, _latents):
            _emit(step)
        pipe_kwargs["callback"] = _legacy
        pipe_kwargs["callback_steps"] = 1
