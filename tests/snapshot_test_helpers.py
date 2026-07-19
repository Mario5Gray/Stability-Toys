"""Shared scaffolding for mode-backed admission tests.

Manufactures the active-model snapshot + a family-binding provider so tests
exercise the snapshot-era admission seam (get_active_model_snapshot +
provider.family_binding) instead of the retired get_current_mode/get_mode_config
path.
"""

from __future__ import annotations

from types import SimpleNamespace

from backends.platforms.base import (
    BackendCapabilities,
    ExecutionCapabilities,
    FamilyPlatformBinding,
)


def make_active_snapshot(mode, *, family_id: str = "sdxl", epoch: int = 1):
    """Build a minimal ActiveModelSnapshot-shaped object around a mode.

    Only the fields admission reads are populated: mode, mode_name, the resolved
    profile's family_id, and the resolution epoch.
    """
    mode_name = getattr(mode, "name", None) or "test-mode"
    return SimpleNamespace(
        mode_name=mode_name,
        mode=mode,
        resolved=SimpleNamespace(profile=SimpleNamespace(family_id=family_id)),
        binding=SimpleNamespace(model_path=f"/models/{mode_name}.safetensors"),
        resolution_epoch=epoch,
    )


def make_family_provider(
    *,
    family_id: str = "sdxl",
    supports_img2img: bool = True,
    supports_controlnet: bool = True,
    supports_combined: bool = True,
    backend_id: str = "cuda",
):
    """A provider whose family_binding(family_id) returns a cell with the given
    execution capabilities; other families return None (unsupported)."""
    cell = FamilyPlatformBinding(
        worker_ref="backends.cuda_worker.DiffusersSDXLCudaWorker",
        execution_capabilities=ExecutionCapabilities(
            supports_img2img, supports_controlnet, supports_combined
        ),
    )

    def family_binding(fid: str):
        return cell if fid == family_id else None

    return SimpleNamespace(
        backend_id=backend_id,
        capabilities=lambda: BackendCapabilities(True, True, True, True),
        family_binding=family_binding,
    )


def make_mode_backed_runtime(mode, *, family_id: str = "sdxl", epoch: int = 1):
    """A mock generation runtime exposing the snapshot seam for HTTP generate()."""
    from unittest.mock import MagicMock

    rt = MagicMock(spec=[
        "switch_mode", "get_current_mode", "get_active_model_snapshot", "submit_generate",
    ])
    rt.get_current_mode.return_value = getattr(mode, "name", None) or "test-mode"
    rt.get_active_model_snapshot.return_value = make_active_snapshot(
        mode, family_id=family_id, epoch=epoch
    )
    return rt


def install_mode_backed(state, pool, mode, *, family_id="sdxl", epoch=1, **caps):
    """Wire a snapshot + provider onto app.state/pool for a mode-backed test."""
    snapshot = make_active_snapshot(mode, family_id=family_id, epoch=epoch)
    pool.get_active_model_snapshot.return_value = snapshot
    pool.current_resolution_epoch.return_value = epoch
    # Display-only: the connect-time system:status frame reads get_current_mode();
    # keep it JSON-serializable so the status frame doesn't break the socket.
    pool.get_current_mode.return_value = snapshot.mode_name
    state.backend_provider = make_family_provider(family_id=family_id, **caps)
    return snapshot
