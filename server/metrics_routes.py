"""The /metrics scrape endpoint (STABL-asawxgvp).

Renders in-memory gauges written by MetricsSampler. It performs NO device or
consumer round-trip — see server/metrics_sampler.py for why that separation is
load-bearing.

ROUTE ORDERING: this router must be included BEFORE lcm_sr_server's
`app.mount("/", StaticFiles(...))`. That mount matches every path and Starlette
matches routes in registration order, so anything registered after it is
unreachable whenever the UI dist is present — true in the deployed image, false
on a dev box where the mount is skipped.
"""
import logging
from typing import Any, Callable, Optional

from fastapi import APIRouter, HTTPException, Response

from server.metrics import get_metrics

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/metrics")
def metrics_endpoint() -> Response:
    met = get_metrics()
    if not met.enabled:
        raise HTTPException(404, detail="metrics disabled")
    body, content_type = met.render()
    # no-store: a cached scrape flat-lines every gauge for the cache lifetime.
    return Response(content=body, media_type=content_type,
                    headers={"Cache-Control": "no-store"})


def build_runtime_stats_fn(
    pool_getter: Callable[[], object],
) -> Callable[[], Optional[dict]]:
    """Adapt the worker pool / Governor into the sampler's stats callable.

    Kept here rather than in the sampler so server/metrics_sampler.py stays free
    of any backends/ knowledge.

    `pool_getter` must READ an already-built pool — pass
    `lambda: getattr(app.state, "worker_pool", None)`, never
    `backends.worker_pool.get_worker_pool`. That accessor CONSTRUCTS a pool (and
    loads a model) when none exists, so using it here would build a whole worker
    just to read a queue depth, on backends that have none.

    Returns None rather than raising on every failure: the runtime may not be up
    when the sampler's first pass runs, and a wedged Governor must degrade to
    "no queue gauges" rather than killing the sampling pass that also carries the
    device metrics.
    """
    def _stats() -> Optional[dict]:
        try:
            pool = pool_getter()
        except Exception:
            return None
        if pool is None:
            return None
        try:
            # WorkerPool holds its Governor on _governor (backends/worker_pool.py:71);
            # the fallback also lets a bare Governor be passed straight through.
            gov: Any = getattr(pool, "_governor", pool)
            with gov._job_lock:
                in_flight = sum(
                    1 for r in gov._job_records.values()
                    if r.executing_since is not None
                )
            return {"queue_depth": gov.get_queue_size(), "jobs_in_flight": in_flight}
        except Exception:
            logger.debug("[Metrics] runtime stats unavailable", exc_info=True)
            return None

    return _stats
