FROM harbor.lan/certificate-base:latest AS certs

FROM python:3.12-slim

WORKDIR /opt/platform

COPY --from=certs /usr/local/share/ca-certificates/ /usr/local/share/ca-certificates/
COPY librknnrt.so /tmp/librknnrt.so

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    build-essential \
    libxext6 \
    libxrender1 \
    libsm6 \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    wget \
    gnupg \
    && cp /tmp/librknnrt.so /usr/lib/librknnrt.so \
    && rm -rf /var/lib/apt/lists/*

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
    pip install --no-cache-dir torch
