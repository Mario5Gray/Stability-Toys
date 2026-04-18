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
