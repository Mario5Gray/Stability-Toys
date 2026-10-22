"""Worker factory: dispatch by neutral family through backend bindings.

The factory consumes an already-resolved model and a backend binding table. It
builds from ``binding.model_path`` and thawed ``resolved.info``. It resolves the
dotted ``worker_ref`` only during worker creation. Thus server boot, status
reads, and rejected requests never import worker code. It never re-detects the
model family.
"""

import importlib
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from backends.model_resolution import LocalModelBinding, ResolvedModel, thaw_model_info
from backends.platforms.base import FamilyPlatformBinding, UnsupportedFamilyError

if TYPE_CHECKING:
    from backends.base import PipelineWorker

logger = logging.getLogger(__name__)


def _resolve_worker_class(worker_ref: str):
    module_path, class_name = worker_ref.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_bound_worker(
    worker_id: int,
    resolved: ResolvedModel,
    binding: LocalModelBinding,
    *,
    backend_id: str,
    family_bindings: Mapping[str, FamilyPlatformBinding],
) -> "PipelineWorker":
    """Create one worker from the supplied backend binding table.

    Raise ``UnsupportedFamilyError`` when the table has no family binding.
    """

    family_id = resolved.profile.family_id
    cell = family_bindings.get(family_id)
    if cell is None:
        raise UnsupportedFamilyError(
            f"family {family_id!r} has no {backend_id.upper()} platform binding"
        )

    worker_class = _resolve_worker_class(cell.worker_ref)
    model_info = thaw_model_info(resolved.info, binding)
    # The resolved model alone selects the family.
    # Pass its profile instead of using a worker subclass default.
    worker = worker_class(
        worker_id=worker_id,
        model_path=binding.model_path,
        model_info=model_info,
        family_profile=resolved.profile,
    )
    logger.info(
        f"[WorkerFactory] Created {cell.worker_ref} for family {family_id!r} "
        f"(worker {worker_id})"
    )
    return worker


def create_cuda_worker(
    worker_id: int,
    resolved: ResolvedModel,
    binding: LocalModelBinding,
) -> "PipelineWorker":
    """Create a CUDA worker through the compatibility entry point."""
    from backends.platforms.cuda_bindings import CUDA_FAMILY_BINDINGS

    return create_bound_worker(
        worker_id,
        resolved,
        binding,
        backend_id="cuda",
        family_bindings=CUDA_FAMILY_BINDINGS,
    )
