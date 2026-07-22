"""Detects test stubs escaping into code that needs the real diffusers.

Unit modules install a MagicMock at `sys.modules["diffusers"]` so they can import
backends.cuda_worker without the library. Two ways that escapes:

  1. `sys.modules.setdefault("diffusers", MagicMock())` checks whether diffusers
     is already *imported*, not whether it is *installed*, so it stubs a fully
     installed library whenever a stubbing module imports first.
  2. `sys.modules["diffusers"].SomePipeline = _Fake` mutates whatever object is
     there, permanently installing fakes into the real library when present.

Either one can fail the live CUDA acceptance with an error that looks like a
worker defect. See STABL-tmrnepae.
"""

import importlib.metadata
import sys

import pytest


def _real_diffusers_installed() -> bool:
    """Ask the package metadata, not the import system.

    `importlib.util.find_spec` raises ValueError when a stub without __spec__
    already occupies sys.modules -- i.e. it breaks precisely when the leak this
    module detects is present. Distribution metadata is unaffected by stubs.
    """
    try:
        importlib.metadata.version("diffusers")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


REAL_DIFFUSERS_INSTALLED = _real_diffusers_installed()


def _is_stub(module) -> bool:
    return type(module).__module__.startswith("unittest.mock")


@pytest.mark.skipif(
    not REAL_DIFFUSERS_INSTALLED, reason="no real diffusers to protect"
)
def test_setdefault_does_not_protect_an_installed_library():
    """Pins the mechanism, so the false assumption cannot be reintroduced.

    `setdefault` is a no-op only when the module is already in sys.modules.
    Installation is irrelevant to it, which is exactly the misunderstanding
    behind the leak.
    """
    from unittest.mock import MagicMock

    saved = sys.modules.pop("diffusers", None)
    try:
        sys.modules.setdefault("diffusers", MagicMock())
        assert _is_stub(sys.modules["diffusers"]), (
            "setdefault stubbed an installed library -- this is the leak's "
            "root cause, not a safe no-op"
        )
    finally:
        sys.modules.pop("diffusers", None)
        if saved is not None:
            sys.modules["diffusers"] = saved


@pytest.mark.skipif(
    not REAL_DIFFUSERS_INSTALLED, reason="no real diffusers to protect"
)
@pytest.mark.xfail(
    reason=(
        "STABL-tmrnepae / STABL-sgdavnvz: stubs leak across the session. "
        "Not fixable by per-module teardown: pytest imports every test file at "
        "collection, so a stubbing module can bind a MagicMock torch into a real "
        "shared library (e.g. safetensors) before any fixture runs, and "
        "restoring sys.modules afterward cannot unbind it. Autouse containment "
        "was tried and failed 32 tests for exactly this reason. The real fix is "
        "collection-time process isolation (pytest-forked / xdist loadfile). "
        "Meanwhile the live acceptance calls conftest.restore_pristine_modules() "
        "so the GPU path is protected. Flips to XPASS once isolation lands."
    ),
    strict=False,
)
def test_real_diffusers_is_not_a_stub_at_session_scope():
    """Fails when an earlier module left a stub in place.

    Ordering-sensitive by nature: it can only observe leaks from modules pytest
    imported before this one. That is enough to catch the full-suite case, which
    is the one that reaches the live CUDA acceptance.
    """
    module = sys.modules.get("diffusers")
    if module is None:
        pytest.skip("diffusers not imported yet in this session")
    assert not _is_stub(module), (
        "sys.modules['diffusers'] is a MagicMock while the real library is "
        "installed; a unit-test stub has leaked and any code needing real "
        "diffusers in this session will fail in a confusing place"
    )
