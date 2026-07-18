"""Worker factory: dispatch by neutral family via the CUDA binding table.

The factory consumes an already-resolved model. It looks up the canonical CUDA
cell from ``resolved.profile.family_id``, resolves the dotted ``worker_ref`` with
importlib *only here* (so server boot, status reads, and rejected requests never
import Torch/Diffusers worker code), and builds the worker from the node-local
``binding.model_path`` plus the thawed ``resolved.info``. It never re-detects.
"""

import importlib
import logging
from typing import TYPE_CHECKING

from backends.model_resolution import LocalModelBinding, ResolvedModel, thaw_model_info
from backends.platforms.base import UnsupportedFamilyError
from backends.platforms.cuda_bindings import CUDA_FAMILY_BINDINGS

if TYPE_CHECKING:
    from backends.base import PipelineWorker

logger = logging.getLogger(__name__)


def _resolve_worker_class(worker_ref: str):
    module_path, class_name = worker_ref.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_cuda_worker(
    worker_id: int,
    resolved: ResolvedModel,
    binding: LocalModelBinding,
) -> "PipelineWorker":
    """Create the CUDA worker bound to ``resolved.profile.family_id``.

    Raises ``UnsupportedFamilyError`` for a known family that has no CUDA cell.
    """

    family_id = resolved.profile.family_id
    cell = CUDA_FAMILY_BINDINGS.get(family_id)
    if cell is None:
        raise UnsupportedFamilyError(
            f"family {family_id!r} has no CUDA platform binding"
        )

    worker_class = _resolve_worker_class(cell.worker_ref)
    model_info = thaw_model_info(resolved.info, binding)
    worker = worker_class(
        worker_id=worker_id,
        model_path=binding.model_path,
        model_info=model_info,
    )
    logger.info(
        f"[WorkerFactory] Created {cell.worker_ref} for family {family_id!r} "
        f"(worker {worker_id})"
    )
    return worker
