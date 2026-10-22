from __future__ import annotations

from functools import partial
from typing import Any

from backends.model_registry import PlaceholderModelRegistry
from backends.platforms.base import BackendCapabilities, FamilyPlatformBinding
from backends.platforms.cpu import PlaceholderSuperResRuntime
from backends.platforms.mps_policy import MPS_TORCH_POLICY
from backends.platforms.pool_runtime import PoolGenerationRuntime


class MPSProvider:
    backend_id: str = "mps"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(False, True, False, False)

    def family_binding(self, family_id: str) -> FamilyPlatformBinding | None:
        from backends.platforms.mps_bindings import MPS_FAMILY_BINDINGS

        return MPS_FAMILY_BINDINGS.get(family_id)

    def create_worker_factory(self, *args: Any, **kwargs: Any) -> Any:
        from backends.platforms.mps_bindings import MPS_FAMILY_BINDINGS
        from backends.worker_factory import create_bound_worker

        return partial(
            create_bound_worker,
            backend_id=self.backend_id,
            family_bindings=MPS_FAMILY_BINDINGS,
        )

    def create_model_registry(self) -> PlaceholderModelRegistry:
        return PlaceholderModelRegistry(self.backend_id)

    def create_generation_runtime(self, **kwargs: Any) -> PoolGenerationRuntime:
        return PoolGenerationRuntime(parent_execution_policy=MPS_TORCH_POLICY, **kwargs)

    def create_superres_runtime(self, *args: Any, **kwargs: Any) -> PlaceholderSuperResRuntime:
        return PlaceholderSuperResRuntime(self.backend_id)
