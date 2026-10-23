"""Deterministic MPS host probe (STABL-npwtkfpf).

Prints one JSON report about the live MPS framework on this host: host and
Torch versions, MPS build and availability, the settings MPS_TORCH_POLICY
resolves, a fixed-seed float16 tensor from an MPS generator, memory stats
around a known allocation, BACKEND=mps provider selection, and the exact
unsupported-family error.

Run from the repo root in the stability-toys conda env, outside any
restricted sandbox (sandboxes hide Metal):

    conda activate stability-toys
    python scripts/mps_host_probe.py > /tmp/mps_probe_1.json

Check cross-run determinism: a second run must reproduce the tensor values.

    python scripts/mps_host_probe.py --expect /tmp/mps_probe_1.json

Exit codes keep three outcomes apart. An unavailable runtime is never a pass.

    0  pass         every check passed on a live MPS runtime
    1  failed       at least one check failed
    3  unavailable  MPS is not built or not available (sandbox, non-Mac host)

Code 2 is left to argparse usage errors. A failed check outranks unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA = "mps_host_probe/v1"
EXIT_PASS = 0
EXIT_FAILED = 1
EXIT_UNAVAILABLE = 3

SEED = 7
TENSOR_SHAPE = (4,)
ALLOC_ELEMENTS = 1024 * 1024  # float32 -> 4 MiB
ALLOC_BYTES = ALLOC_ELEMENTS * 4
UNSUPPORTED_FAMILY = "sdxl"  # sd15 is bound; sdxl carries the negative check
UNSUPPORTED_MESSAGE = f"family {UNSUPPORTED_FAMILY!r} has no MPS platform binding"

PASS, FAIL, SKIPPED = "pass", "fail", "skipped"


class CheckFailed(Exception):
    pass


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _git_state() -> tuple[str | None, bool | None]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()

    try:
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        return None, None


def _host() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "mac_ver": platform.mac_ver()[0],
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def _stats(policy: Any) -> dict[str, int]:
    current, driver = policy.memory_stats()
    return {"current_allocated": int(current), "driver_allocated": int(driver)}


def _check_policy(torch: Any, policy: Any, out: dict[str, Any]) -> None:
    settings = policy.resolve_settings(None)
    out.update(
        device=settings.device,
        dtype=str(settings.dtype),
        attention_slicing=settings.attention_slicing,
    )
    if settings.device != "mps" or settings.dtype is not torch.float16:
        raise CheckFailed("policy must resolve device mps and torch.float16")


def _seeded_values(torch: Any) -> Any:
    generator = torch.Generator(device="mps").manual_seed(SEED)
    return torch.rand(
        *TENSOR_SHAPE, generator=generator, device="mps", dtype=torch.float16
    )


def _check_seeded_tensor(
    torch: Any, expected: list[float] | None, out: dict[str, Any]
) -> None:
    tensor = _seeded_values(torch)
    values = tensor.cpu().tolist()
    repeat = _seeded_values(torch).cpu().tolist()
    out.update(
        dtype=str(tensor.dtype),
        device=tensor.device.type,
        values=values,
        repeatable_in_process=values == repeat,
    )
    if expected is not None:
        out["matches_expected"] = values == expected
    if out["device"] != "mps" or tensor.dtype is not torch.float16:
        raise CheckFailed("seeded tensor must be float16 on mps")
    if not out["repeatable_in_process"]:
        raise CheckFailed("same seed gave different values in one process")
    if out["matches_expected"] is False:
        raise CheckFailed(f"values differ from expected {expected}")


def _check_memory(torch: Any, policy: Any, out: dict[str, Any]) -> None:
    out["before"] = _stats(policy)
    block = torch.zeros(ALLOC_ELEMENTS, dtype=torch.float32, device="mps")
    out["after_alloc"] = _stats(policy)
    del block
    policy.release_cache()
    out["after_release"] = _stats(policy)
    grew = out["after_alloc"]["current_allocated"] - out["before"]["current_allocated"]
    if grew < ALLOC_BYTES:
        raise CheckFailed(f"allocation of {ALLOC_BYTES} bytes accounted only {grew}")
    if (
        out["after_release"]["current_allocated"]
        >= out["after_alloc"]["current_allocated"]
    ):
        raise CheckFailed("release did not lower current allocated memory")


def _check_provider(out: dict[str, Any]) -> None:
    from backends.platform_registry import get_backend_provider, reset_backend_provider
    from backends.platforms.mps import MPSProvider

    previous = os.environ.get("BACKEND")
    os.environ["BACKEND"] = "mps"
    reset_backend_provider()
    try:
        provider = get_backend_provider()
        out.update(backend=provider.backend_id, provider=type(provider).__name__)
    finally:
        if previous is None:
            os.environ.pop("BACKEND", None)
        else:
            os.environ["BACKEND"] = previous
        reset_backend_provider()
    if not isinstance(provider, MPSProvider) or provider.backend_id != "mps":
        raise CheckFailed("BACKEND=mps must select MPSProvider")


def _check_unsupported_family(out: dict[str, Any]) -> None:
    from backends.platforms.base import UnsupportedFamilyError
    from backends.worker_factory import create_mps_worker

    resolved = SimpleNamespace(profile=SimpleNamespace(family_id=UNSUPPORTED_FAMILY))
    try:
        create_mps_worker(0, resolved, None)  # type: ignore[arg-type]
    except UnsupportedFamilyError as exc:
        out["message"] = str(exc)
    else:
        raise CheckFailed(
            f"family {UNSUPPORTED_FAMILY!r} did not raise UnsupportedFamilyError"
        )
    if out["message"] != UNSUPPORTED_MESSAGE:
        raise CheckFailed(f"expected message {UNSUPPORTED_MESSAGE!r}")


def _run(check: dict[str, Any], fn: Any, *args: Any) -> None:
    try:
        fn(*args, check)
        check["status"] = PASS
    except Exception as exc:  # noqa: BLE001 - every failure belongs in the report
        check.update(status=FAIL, error=_error(exc))


def load_expected_values(path: str | Path) -> list[float]:
    report = json.loads(Path(path).read_text())
    return list(report["checks"]["seeded_tensor"]["values"])


def run_probe(expected_values: list[float] | None = None) -> tuple[int, dict[str, Any]]:
    commit, dirty = _git_state()
    checks: dict[str, dict[str, Any]] = {
        "policy": {
            "status": SKIPPED,
            "error": None,
            "device": None,
            "dtype": None,
            "attention_slicing": None,
        },
        "seeded_tensor": {
            "status": SKIPPED,
            "error": None,
            "seed": SEED,
            "shape": list(TENSOR_SHAPE),
            "dtype": None,
            "device": None,
            "values": None,
            "repeatable_in_process": None,
            "matches_expected": None,
        },
        "memory": {
            "status": SKIPPED,
            "error": None,
            "alloc_bytes": ALLOC_BYTES,
            "before": None,
            "after_alloc": None,
            "after_release": None,
        },
        "provider": {
            "status": SKIPPED,
            "error": None,
            "backend": None,
            "provider": None,
        },
        "unsupported_family": {
            "status": SKIPPED,
            "error": None,
            "family": UNSUPPORTED_FAMILY,
            "message": None,
        },
    }
    runtime: dict[str, Any] = {
        "version": None,
        "mps_built": None,
        "mps_available": None,
    }
    mps_checks = ("policy", "seeded_tensor", "memory")

    try:
        import torch

        runtime.update(
            version=torch.__version__,
            mps_built=bool(torch.backends.mps.is_built()),
            mps_available=bool(torch.backends.mps.is_available()),
        )
    except Exception as exc:  # noqa: BLE001
        for name in mps_checks:
            checks[name].update(
                status=FAIL, error=f"torch import failed: {_error(exc)}"
            )
        torch = None

    available = bool(runtime["mps_built"] and runtime["mps_available"])
    if torch is not None and available:
        from backends.platforms.mps_policy import MPS_TORCH_POLICY

        _run(checks["policy"], _check_policy, torch, MPS_TORCH_POLICY)
        _run(checks["seeded_tensor"], _check_seeded_tensor, torch, expected_values)
        _run(checks["memory"], _check_memory, torch, MPS_TORCH_POLICY)
    _run(checks["provider"], _check_provider)
    _run(checks["unsupported_family"], _check_unsupported_family)

    if any(check["status"] == FAIL for check in checks.values()):
        status, code = "failed", EXIT_FAILED
    elif not available:
        status, code = "unavailable", EXIT_UNAVAILABLE
    else:
        status, code = "pass", EXIT_PASS
    report = {
        "schema": SCHEMA,
        "status": status,
        "exit_code": code,
        "commit": commit,
        "dirty": dirty,
        "host": _host(),
        "torch": runtime,
        "checks": checks,
    }
    return code, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic MPS host probe")
    parser.add_argument(
        "--expect",
        metavar="REPORT",
        help="earlier probe JSON; fail unless the seeded tensor values match it",
    )
    args = parser.parse_args(argv)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    expected = load_expected_values(args.expect) if args.expect else None
    code, report = run_probe(expected_values=expected)
    print(json.dumps(report, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    sys.exit(main())
