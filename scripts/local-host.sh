#!/usr/bin/env bash
#
# local-host.sh — provision a dedicated host Python environment that matches the
# project pins, so local `pytest` agrees with the container `make test`.
#
# Why a dedicated env: the shared Miniforge root environment drifts as other
# projects install into it (STABL-zisphapv: it carried transformers 5.x and an
# older diffusers than the pins allow, which aborts pytest collection). This
# script never touches that shared environment; it creates an isolated one.
#
# Interactive by design: it detects the machine and then asks the operator for
# anything it cannot infer. It takes no command-line arguments.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_VERSION="2.10.0"
TORCHVISION_VERSION="0.25.0"
TORCHAUDIO_VERSION="2.10.0"
XFORMERS_VERSION="0.0.34"
CPU_INDEX="https://download.pytorch.org/whl/cpu"
CUDA_INDEX="https://download.pytorch.org/whl/cu128"

say()  { printf '\033[36m[local-host]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[local-host]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[local-host]\033[0m %s\n' "$*" >&2; exit 1; }

ask() {
    # ask <prompt> <default> -> echoes the answer (default if empty)
    local reply prompt=$1 default=$2
    read -r -p "$(printf '%s [%s]: ' "$prompt" "$default")" reply || true
    printf '%s' "${reply:-$default}"
}

ask_yn() {
    # ask_yn <prompt> <default y|n> -> returns 0 for yes
    local reply prompt=$1 default=$2
    read -r -p "$(printf '%s (%s): ' "$prompt" "$([ "$default" = y ] && echo Y/n || echo y/N)")" reply || true
    reply="${reply:-$default}"
    case "$reply" in [Yy]*) return 0 ;; *) return 1 ;; esac
}

# --- 1. detect the machine --------------------------------------------------

OS="$(uname -s)"
ARCH="$(uname -m)"
say "detected OS=${OS} arch=${ARCH}"

CUDA_PRESENT=no
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    CUDA_PRESENT=yes
    say "detected an NVIDIA GPU (nvidia-smi)"
fi

# Default compute backend from what the host can actually run. macOS never has
# CUDA; Linux offers it only when a GPU is present.
if [ "$OS" = "Darwin" ]; then
    DEFAULT_BACKEND=cpu
    [ "$ARCH" = "arm64" ] && say "Apple Silicon: torch will use the CPU/MPS wheels from PyPI"
elif [ "$CUDA_PRESENT" = yes ]; then
    DEFAULT_BACKEND=cuda
else
    DEFAULT_BACKEND=cpu
fi

# --- 2. ask the operator ----------------------------------------------------

ENV_NAME="$(ask 'Environment name' 'stability-toys')"
PY_VERSION="$(ask 'Python version' '3.12')"
BACKEND="$(ask 'Compute backend (cpu|cuda)' "$DEFAULT_BACKEND")"
if [ "$BACKEND" = cuda ] && [ "$OS" = "Darwin" ]; then
    die "CUDA is not available on macOS; re-run and choose cpu"
fi

WANT_CONDITIONING=no
if ask_yn 'Install Compel conditioning extras?' y; then
    WANT_CONDITIONING=yes
fi

# --- 3. pick an environment manager -----------------------------------------

MANAGER=""
for c in mamba micromamba conda; do
    if command -v "$c" >/dev/null 2>&1; then MANAGER="$c"; break; fi
done
# PY is the python interpreter invocation for the chosen env. Everything else
# (pip, the verify script) runs through it, so the conda and venv paths share
# one code path below.
if [ -n "$MANAGER" ]; then
    say "using ${MANAGER} for the environment"
    if ask_yn "Create/replace conda env '${ENV_NAME}'?" y; then
        "$MANAGER" create -y -n "$ENV_NAME" "python=${PY_VERSION}"
    fi
    PY=("$MANAGER" run -n "$ENV_NAME" python)
    ACTIVATE="${MANAGER} activate ${ENV_NAME}"
else
    warn "no conda/mamba found; falling back to a venv at ${REPO_ROOT}/.venv-${ENV_NAME}"
    VENV_DIR="${REPO_ROOT}/.venv-${ENV_NAME}"
    python3 -m venv "$VENV_DIR"
    PY=("${VENV_DIR}/bin/python")
    ACTIVATE="source ${VENV_DIR}/bin/activate"
fi

pip_install() {
    "${PY[@]}" -m pip install "$@"
}

# Newest pip inside the env, so wheel resolution matches CI.
pip_install --upgrade pip

# --- 4. install torch to match the container --------------------------------

say "installing torch ${TORCH_VERSION} (${BACKEND})"
if [ "$OS" = "Darwin" ]; then
    # PyPI ships the macOS (arm64/x86_64) wheels; the Linux CPU index has none.
    pip_install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}"
elif [ "$BACKEND" = cuda ]; then
    pip_install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
        "xformers==${XFORMERS_VERSION}" --index-url "$CUDA_INDEX"
else
    pip_install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url "$CPU_INDEX"
fi

# --- 5. install the project requirements, in the container's order ----------

say "installing project requirements"
pip_install -r "${REPO_ROOT}/requirements.txt"
pip_install -r "${REPO_ROOT}/requirements-test.txt"
if [ "$WANT_CONDITIONING" = yes ]; then
    # --no-deps mirrors the image: Compel's notebook dependency stays out.
    pip_install --no-deps -r "${REPO_ROOT}/requirements-conditioning.txt"
fi

# --- 6. verify --------------------------------------------------------------

say "verifying the install"
"${PY[@]}" - <<'PY'
import importlib.metadata as m
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
for pkg in ("transformers", "diffusers"):
    print(pkg, m.version(pkg))
PY

say "done. environment '${ENV_NAME}' is ready."
say "activate with:  ${ACTIVATE}"
say "point local pytest at this environment instead of the shared root env."
