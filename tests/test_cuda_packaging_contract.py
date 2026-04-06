from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _requirements_lines():
    content = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    return [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_requirements_do_not_own_cuda_torch_or_xformers_packages():
    lines = _requirements_lines()

    assert "torch" not in lines
    assert "xformers" not in lines


def test_dockerfile_verifies_torch_and_xformers_after_cuda_install():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "pip install --no-cache-dir torch==" in dockerfile
    assert "xformers==" in dockerfile
    assert "import torch" in dockerfile
    assert "import xformers" in dockerfile


def test_dockerfile_fails_fast_when_cuda_build_arch_is_not_amd64():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert 'dpkg --print-architecture' in dockerfile
    assert 'CUDA backend requires linux/amd64 build platform' in dockerfile
