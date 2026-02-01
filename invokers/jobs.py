import copy
import time
import threading
from typing import Any, Callable, Dict, Optional, List, Tuple

STALE_S = 60          # no heartbeat for 60s => stale
HARD_S  = 15 * 60     # 15 min hard cap

JOBS_LOCK = threading.RLock()
JOBS: Dict[str, Dict[str, Any]] = {}

# Optional callback for WS push notifications on job updates
_on_update: Optional[Callable[[str, dict], None]] = None


def set_on_update(cb: Optional[Callable[[str, dict], None]]) -> None:
    """Register a callback invoked after every job mutation. Called with (job_id, snapshot)."""
    global _on_update
    _on_update = cb


def _fire_update(job_id: str) -> None:
    """Fire the update callback with a snapshot of the job, if registered."""
    cb = _on_update
    if cb is None:
        return
    j = JOBS.get(job_id)
    if j is None:
        return
    try:
        cb(job_id, copy.deepcopy(j))
    except Exception:
        pass  # never let callback errors break job logic

def jobs_get(job_id: str) -> Optional[Dict[str, Any]]:
    with JOBS_LOCK:
        j = JOBS.get(job_id)
        # Return a snapshot so caller/serializer doesn't race with writers
        return copy.deepcopy(j) if j is not None else None

def jobs_put(job_id: str, job: Dict[str, Any]) -> None:
    with JOBS_LOCK:
        JOBS[job_id] = job

def jobs_update(job_id: str, patch: Dict[str, Any]) -> None:
    """Shallow merge patch into the top-level dict."""
    with JOBS_LOCK:
        j = JOBS.get(job_id)
        if j is None:
            return
        j.update(patch)
        _fire_update(job_id)

def jobs_items_snapshot() -> List[Tuple[str, Dict[str, Any]]]:
    """
    Safe iteration helper.
    Returns a list of (job_id, job_dict_copy) so callers can iterate
    without holding the lock.
    """
    with JOBS_LOCK:
        return [(jid, copy.deepcopy(j)) for jid, j in JOBS.items()]


def jobs_mark_error_if_running(job_id: str, message: str) -> None:
    with JOBS_LOCK:
        j = JOBS.get(job_id)
        if not j:
            return
        if j.get("status") in ("done", "error", "canceled"):
            return
        now = time.time()
        j["status"] = "error"
        j["error"] = message
        j["finished_at"] = now
        j["updated_at"] = now

def jobs_update_path(job_id: str, path: str, value: Any) -> None:
    """
    Update nested fields safely: jobs_update_path(id, "progress.fraction", 0.5)
    Creates intermediate dicts if missing.
    """
    keys = path.split(".")
    with JOBS_LOCK:
        j = JOBS.get(job_id)
        if j is None:
            return
        cur = j
        for k in keys[:-1]:
            nxt = cur.get(k)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[k] = nxt
            cur = nxt
        cur[keys[-1]] = value
        _fire_update(job_id)

def jobs_append_unique(job_id: str, path: str, item: Any) -> None:
    """
    Append to list at path if last isn't the same (good for node_progression).
    """
    keys = path.split(".")
    with JOBS_LOCK:
        j = JOBS.get(job_id)
        if j is None:
            return
        cur = j
        for k in keys[:-1]:
            nxt = cur.get(k)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[k] = nxt
            cur = nxt
        lst = cur.get(keys[-1])
        if not isinstance(lst, list):
            lst = []
            cur[keys[-1]] = lst
        if not lst or lst[-1] != item:
            lst.append(item)