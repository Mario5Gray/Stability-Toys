from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_dry_run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["make", "--dry-run", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


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


def _requirements_text():
    return (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")


def test_transformers_pinned_below_5_to_preserve_diffusers_single_file_clip_loading():
    """Transformers 5.x flattened CLIPTextModel (removed inner .text_model),
    breaking diffusers from_single_file CLIP loading. Pin below 5.0."""
    text = _requirements_text()

    # Find the transformers line (not commented out)
    transformers_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
        and not line.lstrip().startswith("#")
        and "transformers" in line.lower()
        and "diffusers" not in line.lower()
    ]

    assert transformers_lines, "expected a 'transformers' requirement in requirements.txt"

    for line in transformers_lines:
        # Must contain an upper bound that excludes 5.x
        assert "<5" in line or "<5.0" in line, (
            f"transformers requirement '{line}' must be pinned <5.0 to avoid "
            "the CLIPTextModel.text_model AttributeError in diffusers from_single_file"
        )


def test_dockerfile_verifies_torch_and_xformers_after_cuda_install():
    dockerfile = (REPO_ROOT / "docker/platform/python-cuda.Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "pip install --no-cache-dir torch==" in dockerfile
    assert "xformers==" in dockerfile
    assert "import torch" in dockerfile
    assert "import xformers" in dockerfile


def test_dockerfile_fails_fast_when_cuda_build_arch_is_not_amd64():
    dockerfile = (REPO_ROOT / "docker/platform/python-cuda.Dockerfile").read_text(
        encoding="utf-8"
    )

    assert 'dpkg --print-architecture' in dockerfile
    assert 'CUDA backend requires linux/amd64 build platform' in dockerfile


def test_cuda_platform_dockerfile_installs_gpgv_and_avoids_invalid_apt_gpg_override():
    dockerfile = (REPO_ROOT / "docker/platform/python-cuda.Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "gpgv" in dockerfile
    assert "APT::Key::GPGVCommand=1" not in dockerfile


def test_cuda_platform_dockerfile_is_pinned_to_bookworm_for_debian12_cuda_repo():
    dockerfile = (REPO_ROOT / "docker/platform/python-cuda.Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "FROM python:3.12-slim-bookworm" in dockerfile


def test_dockerfile_redeclares_shared_git_sha_for_ui_and_server_stages():
    dockerfile = (REPO_ROOT / "docker/runtime/app.Dockerfile").read_text(
        encoding="utf-8"
    )

    ui_stage = "FROM node:20-trixie-slim AS ui-build"
    server_stage = "FROM ${BASE_IMAGE}"

    ui_start = dockerfile.index(ui_stage)
    server_start = dockerfile.index(server_stage)

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
    assert "runtime: nvidia" in cuda_compose
    assert "reservations:" in cuda_compose
    assert "devices:" in cuda_compose
    assert "driver: nvidia" in cuda_compose
    assert "count: all" in cuda_compose
    assert "capabilities: [gpu]" in cuda_compose


def test_live_test_entrypoint_threads_shared_git_sha_through_backend_and_ui_dev():
    live_test_compose = (REPO_ROOT / "docker-compose.live-test.yml").read_text(
        encoding="utf-8"
    )
    live_test_dockerfile = (
        REPO_ROOT / "docker/runtime/live-test.Dockerfile"
    ).read_text(encoding="utf-8")

    expected_arg = "GIT_SHA: ${GIT_SHA:-dev}"

    assert "build:" in live_test_compose
    assert expected_arg in live_test_compose
    assert "- VITE_APP_VERSION=${GIT_SHA:-dev}" in live_test_compose
    assert "ARG GIT_SHA=dev" in live_test_dockerfile
    assert "BACKEND_VERSION=${GIT_SHA}" in live_test_dockerfile
    assert "BACKEND_VERSION=dev" not in live_test_dockerfile


def test_runtime_dockerfile_uses_base_image_arg():
    text = (REPO_ROOT / "docker/runtime/app.Dockerfile").read_text(encoding="utf-8")

    assert "ARG BASE_IMAGE" in text
    assert "FROM ${BASE_IMAGE}" in text
    assert 'LABEL io.platform.base-ref="${PLATFORM_BASE_REF}"' in text


def test_live_test_runtime_dockerfile_uses_base_image_arg():
    text = (REPO_ROOT / "docker/runtime/live-test.Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "ARG BASE_IMAGE" in text
    assert "FROM ${BASE_IMAGE}" in text
    assert 'LABEL io.platform.base-ref="${PLATFORM_BASE_REF}"' in text


def test_root_dockerfiles_are_marked_as_compatibility_entrypoints():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    live_test = (REPO_ROOT / "Dockerfile.live-test").read_text(encoding="utf-8")

    assert "# Compatibility entrypoint." in dockerfile
    assert "docker/runtime/app.Dockerfile" in dockerfile
    assert "# Compatibility entrypoint." in live_test
    assert "docker/runtime/live-test.Dockerfile" in live_test


def test_quick_dockerfile_copies_only_python_source_without_reinstalling_deps():
    quick = (REPO_ROOT / "Dockerfile.quick").read_text(encoding="utf-8")

    # Must FROM an existing base image (no full rebuild)
    assert "ARG BASE_IMAGE" in quick
    assert "FROM ${BASE_IMAGE}" in quick

    # Must copy the same Python source dirs as the full Dockerfile
    for src in ["conf/", "server/", "persistence/", "backends/", "invokers/", "utils/"]:
        assert f"COPY {src} /app/{src}" in quick

    # Must NOT reinstall deps or rebuild UI
    assert "pip install" not in quick
    assert "yarn" not in quick
    assert "node" not in quick.lower()
    assert "cuda-keyring" not in quick


def test_makefile_quick_build_target_uses_dockerfile_quick():
    result = _make_dry_run("quick-build")

    assert result.returncode == 0, result.stderr
    assert "Dockerfile.quick" in result.stdout
    assert "--build-arg BASE_IMAGE" in result.stdout


def test_makefile_quick_build_target_accepts_custom_image_override():
    result = _make_dry_run("quick-build", "IMAGE=custom:tag")

    assert result.returncode == 0, result.stderr
    assert "custom:tag" in result.stdout
