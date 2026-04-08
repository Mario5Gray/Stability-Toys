FROM harbor.lan/certificate-base:latest AS certs

FROM python:3.12-slim

WORKDIR /opt/platform

COPY --from=certs /usr/local/share/ca-certificates/ /usr/local/share/ca-certificates/

RUN set -eu; \
    arch="$(dpkg --print-architecture)"; \
    if [ "$arch" != "amd64" ]; then \
        echo "CUDA backend requires linux/amd64 build platform, got ${arch}. Re-run docker build with --platform=linux/amd64 or use docker compose with platform: linux/amd64." >&2; \
        exit 1; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    build-essential \
    gpgv \
    libxext6 \
    libxrender1 \
    libsm6 \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      cuda-cudart-12-8 \
      libcublas-12-8 \
      libcufft-12-8 \
      libcurand-12-8 \
      libcusolver-12-8 \
      libcusparse-12-8 && \
    rm -rf /var/lib/apt/lists/*

RUN set -eu; \
    update-ca-certificates; \
    crt_list="$(mktemp)"; \
    ca_bundle="$(mktemp)"; \
    trap 'rm -f "$crt_list" "$ca_bundle"' EXIT; \
    find /usr/local/share/ca-certificates -type f -name '*.crt' | sort > "$crt_list"; \
    if [ -s "$crt_list" ]; then \
        while IFS= read -r crt; do \
            cat "$crt"; \
        done < "$crt_list" > "$ca_bundle"; \
        while IFS= read -r crt; do \
            openssl verify -CAfile "$ca_bundle" "$crt"; \
        done < "$crt_list"; \
    else \
        echo "No custom CA certificates found in /usr/local/share/ca-certificates; skipping verification."; \
    fi

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir nvidia-ml-py && \
    pip install --no-cache-dir --no-deps realesrgan==0.3.0

RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 xformers==0.0.34 --index-url https://download.pytorch.org/whl/cu128 && \
    python - <<'PY'
import torch
import xformers
from xformers import _cpp_lib
from xformers.ops import memory_efficient_attention

assert torch.__version__.startswith("2.10.0"), torch.__version__
assert (torch.version.cuda or "").startswith("12.8"), torch.version.cuda
assert memory_efficient_attention is not None

load_error = getattr(_cpp_lib, "_cpp_library_load_exception", None)
if load_error is not None:
    raise SystemExit(f"xformers extension failed to load: {load_error}")

print(f"Verified torch={torch.__version__} cuda={torch.version.cuda} xformers={xformers.__version__}")
PY
