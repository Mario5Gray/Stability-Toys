ARG TARGETPLATFORM
ARG BACKEND
ARG CERTFILE

# CERTS
FROM harbor.lan/certificate-base:latest AS certs

# ---------- UI build stage ----------
FROM node:20-trixie-slim AS ui-build

WORKDIR /ui

# If you use Yarn classic:
RUN corepack enable && corepack prepare yarn@1.22.22 --activate

# Copy UI project (adjust paths to your repo layout)
ARG UI_DIR=lcm-sr-ui

COPY ${UI_DIR}/package.json lcm-sr-ui/yarn.lock ./
COPY ${UI_DIR}/postcss.config.cjs ./
COPY ${UI_DIR}/tailwind.config.cjs ./
COPY ${UI_DIR}/index.html ./

RUN yarn install --frozen-lockfile

COPY ${UI_DIR}/ ./

RUN yarn build

# ---------- Python server stage ----------
FROM python:3.12-slim AS server
ARG BACKEND
WORKDIR /app

COPY librknnrt.so /tmp/librknnrt.so
RUN <<SOFA
if [ "$BACKEND" = "rknn" ]; then
   apt-get update
   apt-get install -y --no-install-recommends ca-certificates curl build-essential libxext6 libxrender1 libsm6 git ffmpeg libgl1 libglib2.0-0 wget gnupg
fi 

cp /tmp/librknnrt.so /usr/lib/librknnrt.so
SOFA

RUN <<EOFA
if [ "$BACKEND" = "cuda" ]; then
    apt-get update
    apt-get install -y ca-certificates curl build-essential libxext6 libxrender1 libsm6 git ffmpeg libgl1 libglib2.0-0 wget gnupg
    wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    apt update -o APT::Key::GPGVCommand=1        
    apt-get install -y \
     cuda-cudart-12-8 \
     libcublas-12-8 \
     libcufft-12-8 \
     libcurand-12-8 \
     libcusolver-12-8 \
     libcusparse-12-8
fi
EOFA

# Copy any custom CA certs from the certs stage.
COPY --from=certs /usr/local/share/ca-certificates/ /usr/local/share/ca-certificates/

# update-ca-certificates & verify discovered custom certs
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


# Install python deps
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

RUN if [ "$BACKEND" = "cuda" ]; then \
      pip install --no-cache-dir nvidia-ml-py; \
    fi

RUN if [ "$BACKEND" = "cuda" ]; then \
      pip install --no-cache-dir --no-deps realesrgan==0.3.0; \
    fi
# Install cuda12.8 because we have to for xformers.
RUN <<EOI
if [ "$BACKEND" = "cuda" ]; then \
      pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 xformers==0.0.34 --index-url https://download.pytorch.org/whl/cu128; 
fi
EOI


# Copy server code 
COPY server/ /app/server/
COPY persistence/ /app/persistence/
COPY backends/ /app/backends/
COPY invokers/ /app/invokers/
COPY utils/ /app/utils/

COPY *.py /app/
COPY *.sh /app/
RUN chmod +x /app/start.sh

# Copy built UI into where FastAPI will serve it
RUN mkdir -p /app/logs
RUN mkdir -p /opt/lcm-sr-server/ui-dist
COPY --from=ui-build /ui/dist/ /opt/lcm-sr-server/ui-dist/

EXPOSE 4200

CMD ["/bin/bash", "-c", "/app/start.sh"]
