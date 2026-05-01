import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_dry_run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "--dry-run", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_make_test_passes_selected_pytest_target_to_local_test_service():
    result = _make_dry_run(
        "test",
        "TEST=tests/test_cuda_worker_controlnet.py",
        "PYTEST_ARGS=-q",
    )

    assert result.returncode == 0, result.stderr
    assert (
        "docker compose -f docker-compose.test.yml run --rm test "
        "python -m pytest tests/test_cuda_worker_controlnet.py -q"
    ) in result.stdout


def test_make_test_cuda_passes_selected_pytest_target_to_cuda_test_service():
    result = _make_dry_run(
        "test-cuda",
        "TEST=tests/test_cuda_worker_controlnet.py",
        "PYTEST_ARGS=-q",
    )

    assert result.returncode == 0, result.stderr
    assert (
        "docker compose -f docker-compose.test.yml run --rm test-cuda "
        "python -m pytest tests/test_cuda_worker_controlnet.py -q"
    ) in result.stdout


def test_dev_test_file_defaults_to_verbose_without_coverage_flags():
    result = _make_dry_run(
        "-f",
        "Makefile.test",
        "dev-test-file",
        "FILE=test_cuda_worker_controlnet.py",
    )

    assert result.returncode == 0, result.stderr
    assert (
        "docker compose -f docker-compose.test.yml run --rm test "
        "python -m pytest tests/test_cuda_worker_controlnet.py -v"
    ) in result.stdout
    assert "--cov=backends" not in result.stdout
