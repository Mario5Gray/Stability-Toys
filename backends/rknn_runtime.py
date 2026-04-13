from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import Future
from typing import Any, List, Optional

from transformers import CLIPTokenizer

from backends.base import Job, ModelPaths, PipelineWorker


def build_rknn_context_cfgs_for_rk3588(num_workers: int) -> List[dict]:
    core_masks = ["NPU_CORE_0", "NPU_CORE_1", "NPU_CORE_2"]
    cfgs = []
    for i in range(num_workers):
        cfgs.append(
            {
                "multi_context": True,
                "core_mask": core_masks[i % len(core_masks)],
                "context_name": f"w{i}",
                "worker_id": i,
            }
        )
    return cfgs


class PipelineService:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        paths: ModelPaths,
        num_workers: int,
        queue_max: int,
        rknn_context_cfgs: Optional[List[dict]] = None,
        use_rknn_context_cfgs: bool = True,
    ):
        self.workers: List[PipelineWorker] = []
        self.threads: List[threading.Thread] = []
        self._stop = threading.Event()

        self.paths = paths
        self.num_workers = max(1, int(num_workers))
        self.q: "queue.Queue[Job]" = queue.Queue(maxsize=int(queue_max))

        with open(self.paths.scheduler_config, "r") as f:
            self.scheduler_config = json.load(f)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

        if rknn_context_cfgs is None:
            rknn_context_cfgs = build_rknn_context_cfgs_for_rk3588(self.num_workers)
        if len(rknn_context_cfgs) != self.num_workers:
            raise ValueError("rknn_context_cfgs must match num_workers length")

        for i in range(self.num_workers):
            from backends.rknn_worker import RKNNPipelineWorker

            worker = RKNNPipelineWorker(
                worker_id=i,
                paths=self.paths,
                scheduler_config=self.scheduler_config,  # type: ignore[arg-type]
                tokenizer=self.tokenizer,  # type: ignore[arg-type]
                rknn_context_cfg=rknn_context_cfgs[i],  # type: ignore[index]
                use_rknn_context_cfgs=use_rknn_context_cfgs,
            )
            self.workers.append(worker)

        for i in range(self.num_workers):
            thread = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            thread.start()
            self.threads.append(thread)

    @classmethod
    def get_instance(
        cls,
        paths: ModelPaths,
        num_workers: int,
        queue_max: int,
        rknn_context_cfgs: Optional[List[dict]] = None,
        use_rknn_context_cfgs: bool = True,
    ) -> "PipelineService":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(
                    paths=paths,
                    num_workers=num_workers,
                    queue_max=queue_max,
                    rknn_context_cfgs=rknn_context_cfgs,
                    use_rknn_context_cfgs=use_rknn_context_cfgs,
                )
            return cls._instance

    def shutdown(self):
        self._stop.set()
        while True:
            try:
                job = self.q.get_nowait()
            except queue.Empty:
                break
            if not job.fut.done():
                job.fut.set_exception(RuntimeError("Service shutting down"))
            self.q.task_done()

    def submit(self, req: Any, timeout_s: float = 0.25) -> Future:
        fut: Future = Future()
        job = Job(req=req, fut=fut, submitted_at=time.time())
        try:
            self.q.put(job, timeout=timeout_s)
        except queue.Full:
            fut.set_exception(RuntimeError("Queue full"))
        return fut

    def _worker_loop(self, worker_idx: int):
        worker = self.workers[worker_idx]
        while not self._stop.is_set():
            try:
                job = self.q.get(timeout=0.1)
            except queue.Empty:
                continue

            if job.fut.cancelled():
                self.q.task_done()
                continue

            try:
                png, seed = worker.run_job(job)  # type: ignore[arg-type]
                if not job.fut.done():
                    job.fut.set_result((png, seed))
            except Exception as exc:
                if not job.fut.done():
                    job.fut.set_exception(exc)
            finally:
                self.q.task_done()
