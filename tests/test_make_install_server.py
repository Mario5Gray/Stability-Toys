"""Tests for `make install-server` (STABL-yztmgiec).

Each test drives the real target in a temporary git repository that holds
copies of the files the target reads. Spec:
docs/superpowers/specs/2026-11-15-server-install-design.md
"""

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ("Makefile", "Makefile.test", "docker-cuda.yml", "env.prod", "env.cuda")
CONF_FILES = ("controlnets.yaml", "keymap.yml", "modes-test.yml", "modes.yml")


def _git(src: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=src,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "GIT_CEILING_DIRECTORIES": str(src.parent)},
    ).stdout


def source_repo(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "conf").mkdir(parents=True)
    for name in SOURCE_FILES:
        shutil.copy2(REPO_ROOT / name, src / name)
    for name in CONF_FILES:
        shutil.copy2(REPO_ROOT / "conf" / name, src / "conf" / name)
    _git(src, "init", "-q")
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "fixture")
    return src


def run_install(src: Path, prefix: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "-s", "install-server", f"PREFIX={prefix}", *extra],
        cwd=src,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "GIT_CEILING_DIRECTORIES": str(src.parent)},
    )


def short_sha(src: Path) -> str:
    return _git(src, "rev-parse", "--short", "HEAD").strip()


def commit_change(src: Path, rel: str, text: str) -> None:
    path = src / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", f"change {rel}")


def _out(r: subprocess.CompletedProcess[str]) -> str:
    return r.stdout + r.stderr


def test_install_server_fresh_install(tmp_path):
    src = source_repo(tmp_path)
    prefix = tmp_path / "pre fix"
    r = run_install(src, prefix)
    assert r.returncode == 0, _out(r)
    assert (prefix / "compose.yaml").read_text() == (src / "docker-cuda.yml").read_text()
    for name in CONF_FILES:
        assert (prefix / "conf" / name).read_text() == (src / "conf" / name).read_text()
    assert (prefix / "env.prod").read_text() == (src / "env.prod").read_text()
    assert (prefix / "env.cuda").read_text() == (src / "env.cuda").read_text()
    assert (prefix / "env.custom").read_text() == ""
    for d in ("store", "model", "workflows"):
        assert (prefix / d).is_dir()
        assert not any((prefix / d).iterdir())
    assert (prefix / "VERSION").read_text().strip() == short_sha(src)
    assert f'start: cd "{prefix}" && docker compose up -d' in r.stdout


def test_install_server_never_writes_checkout(tmp_path):
    src = source_repo(tmp_path)
    r = run_install(src, tmp_path / "pre fix")
    assert r.returncode == 0, _out(r)
    assert _git(src, "status", "--porcelain") == ""


def test_install_server_same_version_is_noop(tmp_path):
    src = source_repo(tmp_path)
    prefix = tmp_path / "pre fix"
    assert run_install(src, prefix).returncode == 0
    (prefix / "env.prod").write_text("sentinel")
    r = run_install(src, prefix)
    assert r.returncode == 0, _out(r)
    assert "already installed" in r.stdout
    assert (prefix / "env.prod").read_text() == "sentinel"


def test_install_server_dirty_tree_blocks(tmp_path):
    src = source_repo(tmp_path)
    prefix = tmp_path / "pre fix"
    (src / "env.prod").write_text("dirty")
    r = run_install(src, prefix)
    assert r.returncode == 0, _out(r)
    assert "checkout has tracked changes; use FORCE=1" in r.stdout
    assert not (prefix / "VERSION").exists()


def test_install_server_untracked_files_do_not_block(tmp_path):
    src = source_repo(tmp_path)
    prefix = tmp_path / "pre fix"
    (src / "conf" / "local.yml").write_text("local")
    r = run_install(src, prefix)
    assert r.returncode == 0, _out(r)
    assert (prefix / "VERSION").read_text().strip() == short_sha(src)
    assert not (prefix / "conf" / "local.yml").exists()


def test_install_server_force_ignores_dirty_and_same_version(tmp_path):
    src = source_repo(tmp_path)
    prefix = tmp_path / "pre fix"
    assert run_install(src, prefix).returncode == 0
    (prefix / "env.prod").write_text("sentinel")
    (src / "env.prod").write_text("dirty")
    r = run_install(src, prefix, "FORCE=1")
    assert r.returncode == 0, _out(r)
    assert (prefix / "env.prod").read_text() == "dirty"
    assert (prefix / "VERSION").read_text().strip() == short_sha(src)


def test_install_server_upgrade_keeps_operator_files(tmp_path):
    src = source_repo(tmp_path)
    prefix = tmp_path / "pre fix"
    assert run_install(src, prefix).returncode == 0
    for rel in ("compose.yaml", "conf/modes.yml", "env.custom"):
        (prefix / rel).write_text("edited")
    for rel in ("docker-cuda.yml", "conf/modes.yml", "env.prod"):
        commit_change(src, rel, "v2")
    r = run_install(src, prefix)
    assert r.returncode == 0, _out(r)
    for rel in ("compose.yaml", "conf/modes.yml", "env.custom"):
        assert (prefix / rel).read_text() == "edited"
    assert (prefix / "env.prod").read_text() == "v2"
    assert (prefix / "VERSION").read_text().strip() == short_sha(src)


def test_install_server_upgrade_seeds_new_conf_file(tmp_path):
    src = source_repo(tmp_path)
    prefix = tmp_path / "pre fix"
    assert run_install(src, prefix).returncode == 0
    commit_change(src, "conf/new.yml", "x")
    r = run_install(src, prefix)
    assert r.returncode == 0, _out(r)
    assert (prefix / "conf" / "new.yml").read_text() == "x"


def test_install_server_force_keeps_operator_files(tmp_path):
    src = source_repo(tmp_path)
    prefix = tmp_path / "pre fix"
    assert run_install(src, prefix).returncode == 0
    (prefix / "conf" / "modes.yml").write_text("edited")
    r = run_install(src, prefix, "FORCE=1")
    assert r.returncode == 0, _out(r)
    assert (prefix / "conf" / "modes.yml").read_text() == "edited"


def test_install_server_refuses_prefix_inside_checkout(tmp_path):
    src = source_repo(tmp_path)
    r = run_install(src, src / "opt")
    assert r.returncode != 0
    assert "PREFIX is inside the checkout" in _out(r)
    assert not (src / "opt").exists()


def test_install_server_refuses_non_git_source(tmp_path):
    src = source_repo(tmp_path)
    prefix = tmp_path / "pre fix"
    shutil.rmtree(src / ".git")
    r = run_install(src, prefix)
    assert r.returncode != 0
    assert "not a git checkout" in _out(r)
    assert not (prefix / "VERSION").exists()


def test_install_server_failed_copy_keeps_old_version(tmp_path):
    src = source_repo(tmp_path)
    prefix = tmp_path / "pre fix"
    assert run_install(src, prefix).returncode == 0
    old = (prefix / "VERSION").read_text()
    commit_change(src, "env.prod", "v2")
    shutil.rmtree(prefix / "conf")
    (prefix / "conf").write_text("not a dir")
    r = run_install(src, prefix)
    assert r.returncode != 0
    assert (prefix / "VERSION").read_text() == old


def test_install_server_not_in_aggregate_install():
    r = subprocess.run(
        ["make", "--dry-run", "install"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert "install-server" not in r.stdout
    assert "/opt/stability-toys" not in r.stdout
