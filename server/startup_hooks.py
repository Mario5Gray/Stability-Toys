import threading
import time
from invokers.jobs import jobs_items_snapshot, jobs_mark_error_if_running

STALE_S = 60
HARD_S = 15 * 60


def _jobs_reaper_loop():
    while True:
        now = time.time()
        for jid, j in jobs_items_snapshot():
            status = j.get("status")
            if status not in ("queued", "running"):
                continue

            created = float(j.get("created_at") or now)
            hb = float(j.get("heartbeat_at") or j.get("updated_at") or j.get("started_at") or created)

            if now - created > HARD_S:
                jobs_mark_error_if_running(jid, "Job timed out (hard timeout).")
            elif status == "running" and now - hb > STALE_S:
                jobs_mark_error_if_running(jid, "Job stalled (no heartbeat).")

        time.sleep(5)

def start_jobs_reaper():
    t = threading.Thread(target=_jobs_reaper_loop, daemon=True)
    t.start()
