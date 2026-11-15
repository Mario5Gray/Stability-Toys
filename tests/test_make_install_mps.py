"""Tests for `make install-mps` (STABL-hagblcxg).

Each test drives the real target in a temporary git repository. A stub
`conda` script stands in for conda, so no test builds a real env. Spec:
docs/superpowers/specs/2026-11-15-mps-install-design.md
"""

import os
import subprocess
from pathlib import Path

from tests.test_make_install_server import (
    REPO_ROOT,
    _git,
    _out,
    commit_change,
    short_sha,
    source_repo,
)

MPS_FILES = {
    "env.mps": "BACKEND=mps\nMODE_CONFIG_PATH=${MODE_CONFIG_PATH:-conf}\n",
    "requirements.txt": "r1\n",
    "requirements-tracing.txt": "r2\n",
    "requirements-conditioning.txt": "r3\n",
    "server/__init__.py": "",
    "server/app.py": "app\n",
    "backends/b.py": "b\n",
    "utils/u.py": "u\n",
    "persistence/p.py": "p\n",
    "invokers/i.py": "i\n",
    "scripts/check_mps_paths.py": "print('ok')\n",
    "top.py": "t\n",
    "top.sh": "t\n",
}


def mps_repo(tmp_path: Path) -> Path:
    src = source_repo(tmp_path)
    for rel, text in MPS_FILES.items():
        path = src / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "mps fixture")
    return src


def stub_conda(tmp_path: Path, envs: list[str], create_rc: int = 0) -> Path:
    stub = tmp_path / "bin" / "conda"
    stub.parent.mkdir(parents=True, exist_ok=True)
    listing = "\n".join(f"{name} /x/{name}" for name in envs)
    stub.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> "{tmp_path}/conda.log"\n'
        'case "$1" in\n'
        "  env)\n"
        '    echo "# conda environments:"\n'
        f'    cat <<EOF\n{listing}\nEOF\n'
        "    ;;\n"
        f"  create) exit {create_rc} ;;\n"
        f'  run) echo "cwd=$(pwd -P)" >> "{tmp_path}/conda.log"; exit 0 ;;\n'
        "esac\n"
        "exit 0\n"
    )
    stub.chmod(0o755)
    return stub


def log_lines(tmp_path: Path) -> list[str]:
    log = tmp_path / "conda.log"
    return log.read_text().splitlines() if log.exists() else []


def run_mps(src: Path, prefix: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "-s", "install-mps", f"PREFIX={prefix}", *extra],
        cwd=src,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GIT_CEILING_DIRECTORIES": str(src.parent)},
    )


def _setup(tmp_path, envs, create_rc=0):
    src = mps_repo(tmp_path)
    stub = stub_conda(tmp_path, envs, create_rc)
    return src, stub, tmp_path / "pre fix"


def test_install_mps_fresh_install_copies_app_and_files(tmp_path):
    src, stub, prefix = _setup(tmp_path, ["base", "stability-toys"])
    (src / "server" / "scratch.py").write_text("untracked\n")
    r = run_mps(src, prefix, f"CONDA={stub}")
    assert r.returncode == 0, _out(r)
    for rel in (
        "server/app.py",
        "backends/b.py",
        "utils/u.py",
        "persistence/p.py",
        "invokers/i.py",
        "scripts/check_mps_paths.py",
        "top.py",
        "top.sh",
    ):
        assert (prefix / "app" / rel).is_file(), rel
    assert not (prefix / "app" / "server" / "scratch.py").exists()
    assert (prefix / "conf" / "modes.yml").read_text() == (src / "conf" / "modes.yml").read_text()
    assert (prefix / "env.mps").read_text() == (src / "env.mps").read_text()
    assert (prefix / "env.custom").read_text() == ""
    assert (prefix / "store").is_dir()
    assert (prefix / "certs").is_dir()
    assert (prefix / "CONDA_ENV").read_text() == "stability-toys\n"
    assert (prefix / "VERSION").read_text().strip() == short_sha(src)
    run = prefix / "bin" / "run"
    assert os.access(run, os.X_OK)
    text = run.read_text()
    assert str(stub) in text
    assert "-n stability-toys" in text
    assert f'start: "{prefix}/bin/run"' in r.stdout


def test_install_mps_reuses_existing_env(tmp_path):
    src, stub, prefix = _setup(tmp_path, ["base", "stability-toys"])
    r = run_mps(src, prefix, f"CONDA={stub}")
    assert r.returncode == 0, _out(r)
    for line in log_lines(tmp_path):
        assert not line.startswith("create"), line
        assert "pip" not in line, line
    assert "kept conda env stability-toys" in r.stdout


def test_install_mps_creates_missing_env_in_image_order(tmp_path):
    src, stub, prefix = _setup(tmp_path, ["base"])
    r = run_mps(src, prefix, f"CONDA={stub}")
    assert r.returncode == 0, _out(r)
    pip = "run -n stability-toys python -m pip install"
    want = [
        "create -y -n stability-toys python=3.12",
        f"{pip} torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0",
        f"{pip} -r {src}/requirements.txt",
        f"{pip} -r {src}/requirements-tracing.txt",
        f"{pip} --no-deps -r {src}/requirements-conditioning.txt",
    ]
    got = [x for x in log_lines(tmp_path) if x.startswith(("create", "run"))]
    assert got == want
    assert "created conda env stability-toys" in r.stdout


def test_install_mps_names_custom_env(tmp_path):
    src, stub, prefix = _setup(tmp_path, ["base"])
    r = run_mps(src, prefix, f"CONDA={stub}", "CONDA_ENV=dreamer")
    assert r.returncode == 0, _out(r)
    assert (prefix / "CONDA_ENV").read_text() == "dreamer\n"
    assert "create -y -n dreamer python=3.12" in log_lines(tmp_path)
    assert "-n dreamer" in (prefix / "bin" / "run").read_text()


def test_install_mps_failed_env_create_keeps_no_version(tmp_path):
    src, stub, prefix = _setup(tmp_path, ["base"], create_rc=1)
    r = run_mps(src, prefix, f"CONDA={stub}")
    assert r.returncode != 0, _out(r)
    assert not (prefix / "VERSION").exists()


def test_install_mps_refuses_without_conda(tmp_path):
    src = mps_repo(tmp_path)
    r = run_mps(src, tmp_path / "pre fix", f"CONDA={tmp_path}/missing/conda")
    assert r.returncode == 2, _out(r)
    assert "install-mps: conda not found" in _out(r)


def test_install_mps_upgrade_replaces_app(tmp_path):
    src, stub, prefix = _setup(tmp_path, ["base", "stability-toys"])
    assert run_mps(src, prefix, f"CONDA={stub}").returncode == 0
    (prefix / "conf" / "modes.yml").write_text("edited")
    (prefix / "env.custom").write_text("edited")
    _git(src, "rm", "-q", "server/app.py")
    _git(src, "commit", "-q", "-m", "rm app")
    commit_change(src, "server/new.py", "n")
    commit_change(src, "env.mps", "v2")
    r = run_mps(src, prefix, f"CONDA={stub}")
    assert r.returncode == 0, _out(r)
    assert not (prefix / "app" / "server" / "app.py").exists()
    assert (prefix / "app" / "server" / "new.py").read_text() == "n"
    assert (prefix / "env.mps").read_text() == "v2"
    assert (prefix / "conf" / "modes.yml").read_text() == "edited"
    assert (prefix / "env.custom").read_text() == "edited"


def test_install_mps_same_version_is_noop(tmp_path):
    src, stub, prefix = _setup(tmp_path, ["base", "stability-toys"])
    assert run_mps(src, prefix, f"CONDA={stub}").returncode == 0
    (prefix / "env.mps").write_text("sentinel")
    r = run_mps(src, prefix, f"CONDA={stub}")
    assert r.returncode == 0, _out(r)
    assert "already installed" in r.stdout
    assert (prefix / "env.mps").read_text() == "sentinel"


def test_install_mps_force_installs_dirty_working_tree(tmp_path):
    src, stub, prefix = _setup(tmp_path, ["base", "stability-toys"])
    assert run_mps(src, prefix, f"CONDA={stub}").returncode == 0
    (src / "server" / "app.py").write_text("dirty")
    r = run_mps(src, prefix, f"CONDA={stub}", "FORCE=1")
    assert r.returncode == 0, _out(r)
    assert (prefix / "app" / "server" / "app.py").read_text() == "dirty"


def test_install_mps_run_script_invocation(tmp_path):
    src, stub, prefix = _setup(tmp_path, ["base", "stability-toys"])
    assert run_mps(src, prefix, f"CONDA={stub}").returncode == 0
    (tmp_path / "conda.log").unlink()
    r = subprocess.run(
        [str(prefix / "bin" / "run")],
        cwd="/",
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PORT": "4299"},
    )
    assert r.returncode == 0, _out(r)
    lines = log_lines(tmp_path)
    cmd = "\n".join(lines)  # the sh -c argument spans several log lines
    assert cmd.startswith("run --no-capture-output -n stability-toys sh -c")
    assert "--app-dir app" in cmd
    assert "server.lcm_sr_server:app" in cmd
    assert "check_mps_paths.py" in cmd
    cwd = next(x for x in lines if x.startswith("cwd="))
    assert cwd == f"cwd={os.path.realpath(prefix)}"


def test_install_mps_not_in_aggregate_install():
    r = subprocess.run(
        ["make", "--dry-run", "install"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "install-mps" not in r.stdout
