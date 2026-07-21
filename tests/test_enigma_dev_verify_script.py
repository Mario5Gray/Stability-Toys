from pathlib import Path
import os
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "enigma-dev-verify.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _write_remote_worktree_stub(path: Path, helper_log: Path) -> None:
    _write_executable(
        path,
        textwrap.dedent(
            f"""\
            #!/bin/sh
            printf 'helper %s\\n' "$*" >> "{helper_log}"
            printf 'enigma:/srv/stability/.worktrees/gallery-ux-polish\\n'
            """
        ),
    )


def test_verify_wraps_remote_worktree_and_runs_expected_remote_commands(tmp_path):
    log_path = tmp_path / "calls.log"
    helper_log = tmp_path / "helper.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    _write_remote_worktree_stub(bin_dir / "remote-worktree.sh", helper_log)
    _write_executable(
        bin_dir / "ssh",
        textwrap.dedent(
            f"""\
            #!/bin/sh
            printf 'ssh %s\n' "$*" >> "{log_path}"
            cat > "{tmp_path}/remote-script.sh"
            """
        ),
    )

    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REMOTE_WORKTREE_BIN": str(bin_dir / "remote-worktree.sh"),
    }

    result = subprocess.run(
        [str(SCRIPT)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    helper_lines = helper_log.read_text().splitlines()
    assert len(helper_lines) == 1
    assert "--host enigma.lan" in helper_lines[0]
    assert "--repo-path /home/hdd/workspace/Stability-Toys" in helper_lines[0]
    remote_script = (tmp_path / "remote-script.sh").read_text()
    assert 'cd "/srv/stability/.worktrees/gallery-ux-polish"' in remote_script
    assert "docker compose -f docker-cuda.yml build" in remote_script
    assert "docker compose -f docker-compose.dev.yml up -d --build" in remote_script
    assert "docker inspect -f '{{.State.Health.Status}}' stability-toys-dev" in remote_script
    assert "docker logs --tail 50 stability-toys-dev" in remote_script


def test_verify_unhealthy_container_dumps_local_diagnostics(tmp_path):
    log_path = tmp_path / "calls.log"
    helper_log = tmp_path / "helper.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    _write_remote_worktree_stub(bin_dir / "remote-worktree.sh", helper_log)
    _write_executable(
        bin_dir / "ssh",
        textwrap.dedent(
            f"""\
            #!/bin/sh
            printf 'ssh %s\n' "$*" >> "{log_path}"
            cat > "{tmp_path}/remote-script.sh"
            """
        ),
    )

    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REMOTE_WORKTREE_BIN": str(bin_dir / "remote-worktree.sh"),
    }

    result = subprocess.run(
        [str(SCRIPT)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    remote_script = (tmp_path / "remote-script.sh").read_text()
    assert "dump_dev_container_diagnostics()" in remote_script
    assert "[enigma-dev-verify] stability-toys-dev did not become healthy; dumping diagnostics" in remote_script
    assert "[enigma-dev-verify] container state:" in remote_script
    assert "docker inspect -f 'status={{.State.Status}}" in remote_script
    assert "[enigma-dev-verify] recent container logs (tail 250):" in remote_script
    assert "docker logs --tail 250 stability-toys-dev" in remote_script
    diagnostic_body = remote_script.split("dump_dev_container_diagnostics() {", 1)[1].split("\n}", 1)[0]
    assert ">&2" not in diagnostic_body
    assert "dump_dev_container_diagnostics" in remote_script.split('if [ "$status" != "healthy" ]; then', 1)[1]


def test_verify_manual_step_only_skips_remote_docker_phase(tmp_path):
    log_path = tmp_path / "calls.log"
    helper_log = tmp_path / "helper.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path.touch()

    _write_remote_worktree_stub(bin_dir / "remote-worktree.sh", helper_log)
    _write_executable(
        bin_dir / "ssh",
        textwrap.dedent(
            f"""\
            #!/bin/sh
            printf 'ssh %s\n' "$*" >> "{log_path}"
            exit 0
            """
        ),
    )

    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REMOTE_WORKTREE_BIN": str(bin_dir / "remote-worktree.sh"),
    }

    result = subprocess.run(
        [str(SCRIPT), "--manual-step-only"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Manual step remaining:" in result.stdout
    assert "Terminal A: docker logs -f stability-toys-dev" in result.stdout
    assert "leave the log stream running" in result.stdout
    assert "Terminal B: edit conf/modes.yaml and save one reversible change" in result.stdout
    assert "Terminal A: confirm the config watcher reloads without restarting the container" in result.stdout
    assert log_path.read_text() == ""
    helper_lines = helper_log.read_text().splitlines()
    assert len(helper_lines) == 1


def test_verify_skip_base_build_omits_cuda_base_build(tmp_path):
    log_path = tmp_path / "calls.log"
    helper_log = tmp_path / "helper.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    _write_remote_worktree_stub(bin_dir / "remote-worktree.sh", helper_log)
    _write_executable(
        bin_dir / "ssh",
        textwrap.dedent(
            f"""\
            #!/bin/sh
            printf 'ssh %s\n' "$*" >> "{log_path}"
            cat > "{tmp_path}/remote-script.sh"
            """
        ),
    )

    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REMOTE_WORKTREE_BIN": str(bin_dir / "remote-worktree.sh"),
    }

    result = subprocess.run(
        [str(SCRIPT), "--skip-base-build"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    remote_script = (tmp_path / "remote-script.sh").read_text()
    assert "docker compose -f docker-cuda.yml build" not in remote_script
    assert "docker compose -f docker-compose.dev.yml up -d --build" in remote_script


def test_verify_does_not_source_remote_envrc_before_compose_commands(tmp_path):
    log_path = tmp_path / "calls.log"
    helper_log = tmp_path / "helper.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    _write_remote_worktree_stub(bin_dir / "remote-worktree.sh", helper_log)
    _write_executable(
        bin_dir / "ssh",
        textwrap.dedent(
            f"""\
            #!/bin/sh
            printf 'ssh %s\n' "$*" >> "{log_path}"
            cat > "{tmp_path}/remote-script.sh"
            """
        ),
    )

    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REMOTE_WORKTREE_BIN": str(bin_dir / "remote-worktree.sh"),
    }

    result = subprocess.run(
        [str(SCRIPT)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    remote_script = (tmp_path / "remote-script.sh").read_text()
    compose_up = "docker compose -f docker-compose.dev.yml up -d --build"
    assert "if [ -f .envrc ]; then" not in remote_script
    assert "set -a" not in remote_script
    assert "$remote_env_block" not in remote_script
    assert compose_up in remote_script


def test_verify_passes_local_compose_env_to_remote_shell(tmp_path):
    log_path = tmp_path / "calls.log"
    helper_log = tmp_path / "helper.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    _write_remote_worktree_stub(bin_dir / "remote-worktree.sh", helper_log)
    _write_executable(
        bin_dir / "ssh",
        textwrap.dedent(
            f"""\
            #!/bin/sh
            printf 'ssh %s\n' "$*" >> "{log_path}"
            cat > "{tmp_path}/remote-script.sh"
            """
        ),
    )

    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REMOTE_WORKTREE_BIN": str(bin_dir / "remote-worktree.sh"),
        "MODELS_HOST_PATH": "/media/models",
        "FS_HOST_PATH": "/media/creative/dreamers",
        "WORKFLOW_HOST_PATH": "/media/workflows",
    }

    result = subprocess.run(
        [str(SCRIPT)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    remote_script = (tmp_path / "remote-script.sh").read_text()
    compose_up = "docker compose -f docker-compose.dev.yml up -d --build"
    for export_line in [
        "export MODELS_HOST_PATH=/media/models",
        "export FS_HOST_PATH=/media/creative/dreamers",
        "export WORKFLOW_HOST_PATH=/media/workflows",
    ]:
        assert export_line in remote_script
        assert remote_script.index(export_line) < remote_script.index(compose_up)


def test_verify_prints_filesystem_anchor_observations_before_compose(tmp_path):
    log_path = tmp_path / "calls.log"
    helper_log = tmp_path / "helper.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    _write_remote_worktree_stub(bin_dir / "remote-worktree.sh", helper_log)
    _write_executable(
        bin_dir / "ssh",
        textwrap.dedent(
            f"""\
            #!/bin/sh
            printf 'ssh %s\n' "$*" >> "{log_path}"
            cat > "{tmp_path}/remote-script.sh"
            """
        ),
    )

    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REMOTE_WORKTREE_BIN": str(bin_dir / "remote-worktree.sh"),
    }

    result = subprocess.run(
        [str(SCRIPT)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    remote_script = (tmp_path / "remote-script.sh").read_text()
    compose_up = "docker compose -f docker-compose.dev.yml up -d --build"
    for anchor in [
        'printf \'[enigma-dev-verify] worktree=%s\\n\' "$PWD"',
        'observe_anchor "MODELS_HOST_PATH" "${MODELS_HOST_PATH:-./model}"',
        'observe_anchor "FS_HOST_PATH" "${FS_HOST_PATH:-./store}"',
        'observe_anchor "WORKFLOW_HOST_PATH" "${WORKFLOW_HOST_PATH:-./workflows}"',
    ]:
        assert anchor in remote_script
        assert remote_script.index(anchor) < remote_script.index(compose_up)
