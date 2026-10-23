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
        "python -m tests.run tests/test_cuda_worker_controlnet.py -- -q"
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
        "python -m tests.run tests/test_cuda_worker_controlnet.py -- -q"
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
        "python -m tests.run tests/test_cuda_worker_controlnet.py -- -v"
    ) in result.stdout
    assert "--cov=backends" not in result.stdout


# --- Server script entrypoint (STABL-acvmutwa) ------------------------------
# README.md told a contributor to run `python server/lcm_sr_server.py`. That
# command puts `server/` on sys.path instead of the repo root, so `import
# server` raises at the first absolute import. No edit inside the file fixes
# it. The working form is `python -m server.lcm_sr_server`, which puts the repo
# root on sys.path and still runs the `__main__` block.


def test_the_module_form_imports_under_package_semantics():
    """`runpy.run_module` resolves the name exactly as `python -m` does. The
    run_name is set away from "__main__" so uvicorn never starts."""
    result = subprocess.run(
        [
            "python",
            "-c",
            "import runpy; "
            "runpy.run_module('server.lcm_sr_server', run_name='__not_main__')",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_the_readme_advertises_the_module_form():
    """The README is the first page a contributor reads. It must not print a
    command that cannot run."""
    readme = (REPO_ROOT / "README.md").read_text()

    assert "python -m server.lcm_sr_server" in readme
    assert "python server/lcm_sr_server.py" not in readme


def test_the_main_block_imports_the_logging_config_by_its_package_path():
    """`logging_config` lives in `server/`. The bare name resolves only when the
    working directory is `server/`, which is not where `python -m` stands. This
    import sits inside the `__main__` block, so no import test reaches it."""
    import ast

    source = (REPO_ROOT / "server" / "lcm_sr_server.py").read_text()
    tree = ast.parse(source)

    names = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and "logging_config" in node.module
    }
    assert names == {"server.logging_config"}, names


def test_the_server_file_uses_no_relative_imports():
    """The file mixed one relative import into otherwise absolute `server.*`
    imports. One form only, so the next reader cannot guess wrong."""
    import ast

    source = (REPO_ROOT / "server" / "lcm_sr_server.py").read_text()
    relative = [
        node.module or "."
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.level > 0
    ]
    assert relative == [], relative
