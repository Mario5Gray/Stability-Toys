"""DeviceMemory contract tests. Torch-free: this file must never import torch."""
import subprocess
import sys

from backends.device_memory import (
    ConsumerMemory,
    DeviceMemorySnapshot,
    MemoryTopology,
)


def _snap(consumers=(), total=24 * 1024**3, free=10 * 1024**3):
    return DeviceMemorySnapshot(
        device_uuid="GPU-test",
        topology=MemoryTopology.DISCRETE,
        total_bytes=total,
        free_bytes=free,
        consumers=consumers,
    )


def test_used_bytes_derived():
    assert _snap().used_bytes == 14 * 1024**3


def test_unattributed_subtracts_reserved_not_allocated():
    # reserved=5GB, allocated=3GB: the cached-but-free 2GB belongs to the
    # consumer; unattributed must be computed against reserved.
    c = ConsumerMemory(label="worker", pid=123, allocated_bytes=3 * 1024**3,
                       reserved_bytes=5 * 1024**3)
    assert _snap(consumers=(c,)).unattributed_bytes == 9 * 1024**3


def test_unattributed_clamps_at_zero():
    c = ConsumerMemory(label="worker", pid=1, allocated_bytes=20 * 1024**3,
                       reserved_bytes=20 * 1024**3)  # over-report vs used=14GB
    assert _snap(consumers=(c,)).unattributed_bytes == 0


def test_stale_defaults_false():
    c = ConsumerMemory(label="worker", pid=1, allocated_bytes=0, reserved_bytes=0)
    assert c.stale is False


def test_module_is_torch_free():
    """Importing backends.device_memory must not bind torch — checked in a
    clean subprocess so other tests' torch imports can't pollute the result."""
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys, backends.device_memory; "
         "sys.exit(0 if 'torch' not in sys.modules else 1)"],
        capture_output=True,
    )
    assert result.returncode == 0, (
        f"device_memory pulled torch into sys.modules: {result.stderr.decode()}"
    )
