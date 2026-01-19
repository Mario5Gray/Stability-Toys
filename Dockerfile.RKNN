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

COPY lcm-sr-ui/ ./

RUN yarn build


# ---------- Python server stage ----------
FROM python:3.12-slim AS server
WORKDIR /app

# System deps (add what you need)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
  && rm -rf /var/lib/apt/lists/*
RUN  pip install --no-cache-dir --upgrade pip

# Install python deps
# (Put your real requirements in requirements.txt)
COPY requirements.txt /app/requirements.txt
COPY librknnrt.so /usr/lib/librknnrt.so

RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy server code
COPY *.py /app

# Copy built UI into where FastAPI will serve it
RUN mkdir -p /opt/lcm-sr-server/ui-dist
COPY --from=ui-build /ui/dist/ /opt/lcm-sr-server/ui-dist/

EXPOSE 4200

CMD ["uvicorn", "lcm_sr_server:app", "--host", "0.0.0.0", "--port", "4200", "--no-access-log"]
