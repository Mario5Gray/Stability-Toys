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


def test_runtime_requirements_include_sentencepiece_for_t5_tokenizers():
    lines = _requirements_lines()

    assert any(line.startswith("sentencepiece") for line in lines)


def test_diffusers_floor_supports_hunyuandit_pipelines():
    """HunyuanDiTPipeline / HunyuanDiTControlNetPipeline / HunyuanDiT2DControlNetModel
    require Diffusers >= 0.39.0 (the spike-proven floor). Guard the floor here so a
    lower pin cannot silently ship a Diffusers without the Hunyuan family classes."""
    import re

    lines = _requirements_lines()
    diffusers_lines = [line for line in lines if line.startswith("diffusers")]

    assert diffusers_lines, "expected a 'diffusers' requirement in requirements.txt"

    line = diffusers_lines[0]
    match = re.search(r">=\s*(\d+)\.(\d+)", line)
    assert match, f"diffusers requirement '{line}' must carry a >= floor"
    major, minor = int(match.group(1)), int(match.group(2))
    assert (major, minor) >= (0, 39), (
        f"diffusers floor '{line}' is below 0.39.0, which lacks the HunyuanDiT "
        "family pipeline classes"
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


def test_controlnet_tools_stage_installs_all_script_extras():
    """The controlnet-tools Dockerfile stage must install deps for every
    script extra (depth, pose, canny) so all three scripts work in the image."""
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    # Isolate the controlnet-tools stage
    marker = "FROM server AS controlnet-tools"
    idx = dockerfile.find(marker)
    assert idx != -1, "expected controlnet-tools stage in root Dockerfile"
    stage = dockerfile[idx:]

    # depth extra deps
    assert "transformers" in stage
    assert "controlnet-aux" in stage
    assert "matplotlib" in stage

    # pose extra deps
    assert "mediapipe" in stage

    # canny extra deps
    assert "opencv-python-headless" in stage


def test_docker_cuda_yml_passes_backend_cuda_build_arg():
    import yaml

    compose = yaml.safe_load(
        (REPO_ROOT / "docker-cuda.yml").read_text(encoding="utf-8")
    )
    svc = compose["services"]["lcm-sd"]
    args = svc["build"]["args"]

    assert args.get("BACKEND") == "cuda"


def test_test_cuda_compose_requests_all_nvidia_gpus():
    import yaml

    compose = yaml.safe_load(
        (REPO_ROOT / "docker-compose.test.yml").read_text(encoding="utf-8")
    )
    service = compose["services"]["test-cuda"]
    devices = service["deploy"]["resources"]["reservations"]["devices"]

    assert service["runtime"] == "nvidia"
    assert devices == [
        {
            "driver": "nvidia",
            "count": "all",
            "capabilities": ["gpu"],
        }
    ]


def test_docker_rknn_yml_passes_backend_rknn_build_arg():
    import yaml

    compose = yaml.safe_load(
        (REPO_ROOT / "docker-rknn.yml").read_text(encoding="utf-8")
    )
    svc = compose["services"]["lcm-sd"]
    args = svc["build"]["args"]

    assert args.get("BACKEND") == "rknn"


def test_live_test_dockerfile_uses_qualified_module_path():
    text = (REPO_ROOT / "docker/runtime/live-test.Dockerfile").read_text(
        encoding="utf-8"
    )

    assert "server.lcm_sr_server:app" in text
    # The bare module name must not appear — it never worked
    bare_cmd = '"uvicorn", "lcm_sr_server:app"'
    assert bare_cmd not in text


def test_root_live_test_dockerfile_uses_qualified_module_path():
    text = (REPO_ROOT / "Dockerfile.live-test").read_text(encoding="utf-8")

    assert "server.lcm_sr_server:app" in text
    bare_cmd = '"uvicorn", "lcm_sr_server:app"'
    assert bare_cmd not in text


def test_dev_compose_uses_live_test_dockerfile():
    text = (REPO_ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8")

    assert "docker/runtime/live-test.Dockerfile" in text
    # Must NOT use the root Dockerfile (which always runs the UI build stage)
    assert "dockerfile: ./Dockerfile" not in text


def test_dev_compose_mounts_config_at_conf_not_app_conf():
    text = (REPO_ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8")

    assert "./conf:/conf" in text
    assert "./conf:/app/conf" not in text


def test_dev_compose_mounts_python_source_volumes():
    text = (REPO_ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8")

    for src in ["./server:/app/server", "./backends:/app/backends",
                "./utils:/app/utils", "./persistence:/app/persistence",
                "./invokers:/app/invokers"]:
        assert src in text, f"dev compose must mount {src}"


def test_dev_compose_mounts_prebuilt_ui_dist():
    text = (REPO_ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8")

    assert "lcm-sr-ui/dist:/opt/lcm-sr-server/ui-dist" in text
    assert ":ro" in text


def test_dev_compose_uses_dev_env_files():
    import yaml

    compose = yaml.safe_load(
        (REPO_ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8")
    )
    svc = compose["services"]["lcm-sd"]
    env_files = svc.get("env_file", [])

    assert "env.dev" in env_files
    assert "env.cuda" in env_files


def test_dev_compose_uses_dev_image_tag():
    text = (REPO_ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8")

    assert "lcm-sd-ui:dev" in text
    # Must NOT use the production tag
    assert "lcm-sd-ui:latest" not in text.split("image:")[1].split("\n")[0] if "image:" in text else True


def test_dev_compose_takes_base_image_build_arg():
    text = (REPO_ROOT / "docker-compose.dev.yml").read_text(encoding="utf-8")

    assert "BASE_IMAGE" in text
    assert "harbor.lan/lcm-sd-ui:latest" in text


def test_makefile_dev_target_uses_dev_compose():
    result = _make_dry_run("dev")

    assert result.returncode == 0, result.stderr
    assert "docker compose -f docker-compose.dev.yml up" in result.stdout


def test_makefile_dev_build_target_uses_dev_compose():
    result = _make_dry_run("dev-build")

    assert result.returncode == 0, result.stderr
    assert "docker compose -f docker-compose.dev.yml build" in result.stdout


def test_makefile_dev_down_target_uses_dev_compose():
    result = _make_dry_run("dev-down")

    assert result.returncode == 0, result.stderr
    assert "docker compose -f docker-compose.dev.yml down" in result.stdout
