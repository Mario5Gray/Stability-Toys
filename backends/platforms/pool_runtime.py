"""Queue-backed generation runtime shared by platform providers."""

from __future__ import annotations

from typing import Any


class PoolGenerationRuntime:
    def __init__(
        self, *, pool: Any | None = None, parent_execution_policy: Any = None, **_: Any
    ) -> None:
        if pool is None:
            from backends.worker_pool import get_worker_pool

            if parent_execution_policy is None:
                pool = get_worker_pool()
            else:
                pool = get_worker_pool(parent_execution_policy=parent_execution_policy)
        self._pool = pool

    def get_active_model_snapshot(self) -> Any:
        return self._pool.get_active_model_snapshot()

    def wait_for_result(self, fut: Any) -> Any:
        """Wait without charging queue time to the generation budget."""
        return self._pool.wait_for_result(fut)

    def submit_generate(
        self,
        req: Any,
        *,
        snapshot: Any = None,
        controlnet_bindings: Any = None,
        timeout_s: float | None = None,
    ) -> Any:
        """Submit one job against a captured active-model snapshot."""
        from backends.worker_pool import GenerationJob

        if snapshot is None:
            snapshot = self._pool.get_active_model_snapshot()
        if snapshot is None:
            runtime_name = type(self).__name__
            raise RuntimeError(
                f"{runtime_name}.submit_generate requires an active model "
                "snapshot; no model is loaded"
            )

        if controlnet_bindings is None:
            controlnet_bindings = self._resolve_bindings(req, snapshot)

        job = GenerationJob(
            req=req,
            controlnet_bindings=list(controlnet_bindings),
            resolution_epoch=snapshot.resolution_epoch,
        )
        return self._pool.submit_job(job, timeout_s=timeout_s)

    @staticmethod
    def _resolve_bindings(req: Any, snapshot: Any) -> list:
        if not getattr(req, "controlnets", None):
            return []
        from server.asset_store import get_store
        from server.controlnet_execution import resolve_controlnet_bindings

        return resolve_controlnet_bindings(
            req,
            mode=snapshot.mode,
            store=get_store(),
            active_family=snapshot.resolved.profile.family_id,
        )

    def switch_mode(self, mode_name: str, force: bool = False) -> Any:
        return self._pool.switch_mode(mode_name, force=force)

    def get_current_mode(self) -> str | None:
        return self._pool.get_current_mode()

    def is_model_loaded(self) -> bool:
        return self._pool.is_model_loaded()

    def get_queue_size(self) -> int:
        return self._pool.get_queue_size()

    def shutdown(self) -> None:
        self._pool.shutdown()
