import os
from unittest.mock import patch

import pytest


def test_get_backend_provider_requires_explicit_backend():
    from backends.platform_registry import get_backend_provider, reset_backend_provider

    reset_backend_provider()
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError, match="BACKEND must be set explicitly"):
            get_backend_provider()


def test_get_backend_provider_rejects_unsupported_backend():
    from backends.platform_registry import get_backend_provider, reset_backend_provider

    reset_backend_provider()
    with patch.dict(os.environ, {"BACKEND": "auto"}, clear=True):
        with pytest.raises(RuntimeError, match="Unsupported BACKEND='auto'"):
            get_backend_provider()


def test_get_backend_provider_resolves_known_backend():
    from backends.platform_registry import get_backend_provider, reset_backend_provider

    reset_backend_provider()
    with patch.dict(os.environ, {"BACKEND": "cuda"}, clear=True):
        provider = get_backend_provider()

    assert provider.backend_id == "cuda"
    assert provider.capabilities().supports_generation is True


def test_cpu_provider_resolves_with_placeholder_capabilities():
    from backends.platform_registry import get_backend_provider, reset_backend_provider

    reset_backend_provider()
    with patch.dict(os.environ, {"BACKEND": "cpu"}, clear=True):
        provider = get_backend_provider()

    caps = provider.capabilities()
    assert provider.__class__.__name__ == "CPUProvider"
    assert provider.backend_id == "cpu"
    assert caps.supports_generation is False
    assert caps.supports_superres is False
