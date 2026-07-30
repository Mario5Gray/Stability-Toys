ARG BASE_IMAGE

FROM ${BASE_IMAGE}
ARG GIT_SHA=dev
ARG PLATFORM_BASE_REF=unknown

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BACKEND_VERSION=${GIT_SHA}

LABEL io.platform.role="runtime"
LABEL io.platform.base-ref="${PLATFORM_BASE_REF}"

WORKDIR /app

COPY server/ /app/server/
COPY persistence/ /app/persistence/
COPY backends/ /app/backends/
COPY invokers/ /app/invokers/
COPY utils/ /app/utils/
COPY *.py /app/
COPY *.sh /app/

RUN mkdir -p /opt/lcm-sr-server/ui-dist

# Materialize the logging dict as a uvicorn-consumable file. uvicorn's
# --log-config takes a FILE, not a Python dict, and the CMD imports the app
# (so lcm_sr_server.py's `if __name__ == "__main__"` log_config never runs and
# the root logger is left unconfigured -> app INFO logs, e.g. [ModelRegistry],
# are dropped to WARNING via logging.lastResort). Generated from
# server/logging_config.py, which stays the single source of truth. Written to
# /app root (NOT a bind-mounted subdir) so the dev volume mounts don't shadow it.
RUN PYTHONPATH=/app python -c "import json, server.logging_config as m; json.dump(m.LOGGING_CONFIG, open('/app/logging_config.json', 'w'))"

EXPOSE 4200

CMD ["uvicorn", "server.lcm_sr_server:app", "--host", "0.0.0.0", "--port", "4200", "--reload", "--log-config", "/app/logging_config.json"]
