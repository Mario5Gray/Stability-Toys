"""Contract tests for scripts/local-host.sh.

The script provisions a dedicated host-machine Python env matching the project
pins (the shared Miniforge base drifts; see STABL-zisphapv). It must detect
architecture and CUDA, ask the operator interactively rather than take CLI args,
and mirror the container's torch/requirements install order so local pytest
agrees with `make test`.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "local-host.sh"


def _text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_script_exists_and_is_executable():
    assert SCRIPT.exists(), "scripts/local-host.sh missing"
    assert SCRIPT.stat().st_mode & 0o111, "local-host.sh must be executable"


def test_detects_os_and_arch():
    t = _text()
    assert "uname -s" in t, "must detect OS"
    assert "uname -m" in t, "must detect architecture"


def test_detects_cuda():
    assert "nvidia-smi" in _text(), "must probe for CUDA via nvidia-smi"


def test_prompts_operator_and_takes_no_config_cli_args():
    t = _text()
    assert "read " in t, "must prompt the operator interactively"
    # No positional-arg config: the operator answers prompts, not flags.
    assert "getopts" not in t, "config must come from prompts, not getopts"
    assert '"$1"' not in t and "${1" not in t, "must not read positional CLI args"


def test_matches_container_torch_and_index_urls():
    t = _text()
    assert "2.10.0" in t, "torch version must match the container (2.10.0)"
    assert "whl/cpu" in t, "CPU wheel index required"
    assert "whl/cu128" in t, "CUDA wheel index required"


def test_installs_the_three_requirements_files_in_order():
    t = _text()
    for req in ("requirements.txt", "requirements-test.txt", "requirements-conditioning.txt"):
        assert req in t, f"must install {req}"
    # Conditioning is installed --no-deps, matching the image.
    assert "--no-deps" in t, "conditioning requirements must install with --no-deps"


def test_creates_a_dedicated_env_not_the_shared_base():
    t = _text()
    assert "base" not in t.replace("database", ""), (
        "must not install into the shared conda base; create a dedicated env"
    )
    assert "stability-toys" in t, "default dedicated env name should be stability-toys"


def test_verifies_the_install_before_finishing():
    t = _text()
    assert "import torch" in t, "must verify torch imports after install"
