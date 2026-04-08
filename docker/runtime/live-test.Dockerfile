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

EXPOSE 4200

CMD ["uvicorn", "lcm_sr_server:app", "--host", "0.0.0.0", "--port", "4200", "--reload"]
