"""Map a diffusers step callback to a backplane Progress emitter (STABL-zueslhah)
and to a cancellation predicate (STABL-jredufxb).

Stdlib only (inspect) — kept out of cuda_worker so it imports without torch.

`inject_step_progress` probes the pipeline's __call__ signature and adds the
right callback to pipe_kwargs, family-agnostic (decision B): the modern
`callback_on_step_end` when supported, else the legacy `callback`/`callback_steps`,
else nothing. The emitted step is 1-based (step 0 of N -> progress(1, N)). A
misbehaving consumer never breaks generation — the callback swallows exceptions.

The same callback is the only re-entry point into a running generation, so it is
also where a timed-out job is reaped: `should_cancel()` is consulted at each step
boundary and raises. That is the whole of the reap's granularity — a single long
step, VAE decode, or a wedged CUDA call is not interruptible here.
"""
import inspect
from concurrent.futures import CancelledError
from typing import Any, Callable, Optional

ProgressEmitter = Callable[[int, int, str], None]


def inject_step_progress(
    pipe: Any,
    pipe_kwargs: dict,
    progress: Optional[ProgressEmitter],
    total: int,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    if progress is None and should_cancel is None:
        return
    try:
        params = inspect.signature(pipe.__call__).parameters
    except (ValueError, TypeError):
        return

    total_i = int(total)

    def _emit(step_index: int) -> None:
        if progress is None:
            return
        try:
            progress(step_index + 1, total_i, "denoise")  # 1-based
        except Exception:
            pass  # a bad consumer must never break generation

    def _check_cancel() -> None:
        # STABL-jredufxb. Deliberately NOT inside _emit: that swallows every
        # exception so a misbehaving progress consumer cannot break generation,
        # and a cancel raised there would be eaten with it.
        #
        # concurrent.futures.CancelledError, not a bespoke type: classify_exception()
        # maps only that (or a class named CancelledError) to CANCELLED, and the
        # subprocess parent path does no cancel_requested remap. The message must
        # never contain "out of memory" — the dispatch loop's _oom substring test
        # runs before its cancel branch.
        if should_cancel is not None and should_cancel():
            raise CancelledError("job cancelled at step boundary")

    if "callback_on_step_end" in params:
        def _modern(_pipe, step, _timestep, callback_kwargs):
            _check_cancel()
            _emit(step)
            return callback_kwargs
        pipe_kwargs["callback_on_step_end"] = _modern
    elif "callback" in params and "callback_steps" in params:
        def _legacy(step, _timestep, _latents):
            _check_cancel()
            _emit(step)
        pipe_kwargs["callback"] = _legacy
        pipe_kwargs["callback_steps"] = 1
