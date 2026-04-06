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


def test_dockerfile_redeclares_shared_git_sha_for_ui_and_server_stages():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    ui_stage = "FROM node:20-trixie-slim AS ui-build"
    server_stage = "FROM python:3.12-slim AS server"

    assert dockerfile.startswith(
        "ARG TARGETPLATFORM\nARG BACKEND\nARG CERTFILE\nARG GIT_SHA=dev\n"
    )

    ui_start = dockerfile.index(ui_stage)
    server_start = dockerfile.index(server_stage)

    assert "ARG GIT_SHA=dev" in dockerfile[:ui_start]
    assert "ARG GIT_SHA=dev" in dockerfile[ui_start:server_start]
    assert "ENV VITE_APP_VERSION=${GIT_SHA}" in dockerfile[ui_start:server_start]
    assert "ARG GIT_SHA=dev" in dockerfile[server_start:]
    assert "ENV BACKEND_VERSION=${GIT_SHA}" in dockerfile[server_start:]


def test_checked_in_compose_build_entrypoints_pass_git_sha_from_env_with_dev_fallback():
    cuda_compose = (REPO_ROOT / "docker-cuda.yml").read_text(encoding="utf-8")
    rknn_compose = (REPO_ROOT / "docker-rknn.yml").read_text(encoding="utf-8")

    expected_arg = "GIT_SHA: ${GIT_SHA:-dev}"

    assert "build:" in cuda_compose
    assert "build:" in rknn_compose
    assert expected_arg in cuda_compose
    assert expected_arg in rknn_compose
