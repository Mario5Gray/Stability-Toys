from pathlib import Path
import os
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "enigma-worktree.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _make_stub_bin(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "git",
        textwrap.dedent(
            """\
            #!/bin/sh
            case "$1" in
              rev-parse) printf '%s\n' "$PWD" ;;
              branch)
                if [ "$2" = "--show-current" ]; then
                  printf '%s\n' "${TEST_BRANCH:-gallery-ux-polish}"
                fi
                ;;
              *)
                printf 'git %s\n' "$*" >> "$TEST_LOG"
                ;;
            esac
            """
        ),
    )
    _write_executable(
        bin_dir / "ssh",
        textwrap.dedent(
            """\
            #!/bin/sh
            printf 'ssh %s\n' "$*" >> "$TEST_LOG"
            exit 0
            """
        ),
    )
    return bin_dir


def test_dry_run_prints_resolved_defaults_and_skips_git_push_and_ssh(tmp_path):
    log_path = tmp_path / "calls.log"
    log_path.touch()
    bin_dir = _make_stub_bin(tmp_path)
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TEST_LOG": str(log_path),
        "TEST_BRANCH": "gallery-ux-polish",
    }

    result = subprocess.run(
        [str(SCRIPT), "--dry-run"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "host=enigma" in result.stdout
    assert "repo_path=~/workspace/Stability-Toys" in result.stdout
    assert "branch=gallery-ux-polish" in result.stdout
    assert "git push origin gallery-ux-polish" in result.stdout
    assert "ssh enigma" in result.stdout
    assert log_path.read_text() == ""


def test_help_exits_successfully(tmp_path):
    result = subprocess.run(
        [str(SCRIPT), "--help"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--repo-path <path>" in result.stdout


def test_uses_explicit_branch_for_push(tmp_path):
    log_path = tmp_path / "calls.log"
    log_path.touch()
    bin_dir = _make_stub_bin(tmp_path)
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TEST_LOG": str(log_path),
        "TEST_BRANCH": "main",
        "TEST_REMOTE_HOME": "/home/tester",
    }

    result = subprocess.run(
        [str(SCRIPT), "--branch", "gallery-ux-polish"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    log = log_path.read_text()
    assert "git push origin gallery-ux-polish\n" in log


def test_detached_head_without_branch_flag_fails_before_push(tmp_path):
    log_path = tmp_path / "calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "git",
        textwrap.dedent(
            f"""\
            #!/bin/sh
            case "$1" in
              rev-parse) printf '%s\\n' "$PWD" ;;
              branch)
                if [ "$2" = "--show-current" ]; then
                  printf '\\n'
                fi
                ;;
              *)
                printf 'git %s\\n' "$*" >> "{log_path}"
                ;;
            esac
            """
        ),
    )
    _write_executable(
        bin_dir / "ssh",
        "#!/bin/sh\nexit 99\n",
    )
    env = os.environ | {"PATH": f"{bin_dir}:{os.environ['PATH']}"}

    result = subprocess.run(
        [str(SCRIPT)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "could not resolve branch" in result.stderr
    assert not log_path.exists()


def test_remote_phase_uses_single_ssh_session_and_creates_branch_worktree(tmp_path):
    log_path = tmp_path / "calls.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "git",
        textwrap.dedent(
            f"""\
            #!/bin/sh
            case "$1" in
              rev-parse) printf '%s\\n' "$PWD" ;;
              branch)
                if [ "$2" = "--show-current" ]; then
                  printf 'gallery-ux-polish\\n'
                fi
                ;;
              *)
                printf 'git %s\\n' "$*" >> "{log_path}"
                ;;
            esac
            """
        ),
    )
    _write_executable(
        bin_dir / "ssh",
        textwrap.dedent(
            f"""\
            #!/bin/sh
            printf 'ssh %s\\n' "$*" >> "{log_path}"
            cat > "{tmp_path}/remote-script.sh"
            printf '/home/tester/workspace/Stability-Toys/.worktrees/gallery-ux-polish\\n'
            """
        ),
    )
    env = os.environ | {"PATH": f"{bin_dir}:{os.environ['PATH']}"}

    result = subprocess.run(
        [str(SCRIPT)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "enigma:/home/tester/workspace/Stability-Toys/.worktrees/gallery-ux-polish"
    log = log_path.read_text()
    assert log.count("ssh ") == 1
    remote_script = (tmp_path / "remote-script.sh").read_text()
    assert 'git -C "$repo_root" fetch "$remote_name"' in remote_script
    assert 'git -C "$repo_root" worktree add -B "$branch" "$worktree_path" "$remote_name/$branch"' in remote_script
