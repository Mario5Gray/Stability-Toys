ARG BASE_IMAGE

FROM node:20-trixie-slim AS ui-build
ARG GIT_SHA=dev
ARG UI_DIR=lcm-sr-ui

WORKDIR /ui

RUN corepack enable && corepack prepare yarn@1.22.22 --activate

ENV VITE_APP_VERSION=${GIT_SHA}

COPY ${UI_DIR}/package.json lcm-sr-ui/yarn.lock ./
COPY ${UI_DIR}/postcss.config.cjs ./
COPY ${UI_DIR}/tailwind.config.cjs ./
COPY ${UI_DIR}/index.html ./

RUN yarn install --frozen-lockfile

COPY ${UI_DIR}/ ./

RUN yarn build

FROM ${BASE_IMAGE}
ARG GIT_SHA=dev
ARG PLATFORM_BASE_REF=unknown

ENV BACKEND_VERSION=${GIT_SHA}
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

RUN chmod +x /app/start.sh && \
    mkdir -p /app/logs && \
    mkdir -p /opt/lcm-sr-server/ui-dist

COPY --from=ui-build /ui/dist/ /opt/lcm-sr-server/ui-dist/

EXPOSE 4200

CMD ["/bin/bash", "-c", "/app/start.sh"]
