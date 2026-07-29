"""DeviceMemory contract tests. Torch-free: this file must never import torch."""
import subprocess
import sys

from backends.device_memory import (
    ConsumerMemory,
    CudaDeviceMemory,
    DeviceMemorySnapshot,
    MemoryTopology,
    NullDeviceMemory,
    UnifiedDeviceMemory,
    get_device_memory,
    reset_device_memory,
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


class _FakePsutil:
    class _VM:
        available = 8 * 1024**3
        total = 32 * 1024**3
    @staticmethod
    def virtual_memory():
        return _FakePsutil._VM()


def test_null_degrades_to_zero_unknown():
    dm = NullDeviceMemory()
    snap = dm.snapshot()
    assert snap.total_bytes == 0 and snap.free_bytes == 0
    assert snap.topology == MemoryTopology.UNKNOWN
    assert dm.available_for_load() == 0
    assert dm.device_name == "Unknown"
    reg = dm.register(object())  # inert, returns a Registration
    reg.close(); reg.close()     # idempotent


def test_unified_reads_psutil_host_ram():
    dm = UnifiedDeviceMemory(_psutil=_FakePsutil())
    assert dm.available_for_load() == 8 * 1024**3
    snap = dm.cached_snapshot()  # pre-seeded at construction
    assert snap.total_bytes == 32 * 1024**3
    assert snap.topology == MemoryTopology.UNIFIED


def test_singleton_selects_unified_when_no_nvml(monkeypatch):
    reset_device_memory()
    import builtins
    real_import = builtins.__import__
    def fake_import(name, *a, **k):
        if name == "pynvml":
            raise ImportError("no nvml on this host")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    dm = get_device_memory()
    assert isinstance(dm, (UnifiedDeviceMemory, NullDeviceMemory))  # psutil present → Unified
    reset_device_memory()


class _FakeNvml:
    """Fake NVML boundary. Provider unit tests stub THIS (the injected module),
    which is the one allowed pynvml seam; consumers never see pynvml."""
    class _Info:
        free = 20 * 1024**3
        total = 24 * 1024**3
    def __init__(self):
        self.inited = False
    def nvmlInit(self):
        self.inited = True
    def nvmlDeviceGetHandleByUUID(self, uuid):
        assert uuid == "GPU-fake-uuid"
        return "HANDLE"
    def nvmlDeviceGetMemoryInfo(self, handle):
        assert handle == "HANDLE"
        return self._Info()
    def nvmlDeviceGetName(self, handle):
        return b"NVIDIA GeForce RTX 3090"


def test_cuda_provider_nvml_by_uuid():
    nvml = _FakeNvml()
    dm = CudaDeviceMemory("GPU-fake-uuid", _pynvml=nvml)
    assert nvml.inited
    assert dm.device_uuid == "GPU-fake-uuid"
    assert dm.topology == MemoryTopology.DISCRETE
    assert dm.device_name == "NVIDIA GeForce RTX 3090"
    assert dm.available_for_load() == 20 * 1024**3
    snap = dm.cached_snapshot()  # pre-seeded at construction
    assert snap.total_bytes == 24 * 1024**3
    assert snap.free_bytes == 20 * 1024**3
    assert snap.consumers == ()


def test_available_for_load_does_not_fan_out():
    nvml = _FakeNvml()
    dm = CudaDeviceMemory("GPU-fake-uuid", _pynvml=nvml)

    class _ExplodingConsumer:
        def pool_stats(self):
            raise AssertionError("available_for_load() MUST NOT fan out")
        def reclaim(self):
            pass

    dm.register(_ExplodingConsumer())
    assert dm.available_for_load() == 20 * 1024**3  # no consumer round-trip
