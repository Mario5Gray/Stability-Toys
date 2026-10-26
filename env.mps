# Native MPS path (STABL-gpcsnipw).
#
# Docker Desktop on macOS runs a Linux VM and does not expose Metal, so no
# container on a Mac can run MPS. This file is sourced by `make mps`; it is
# never passed to `docker run --env-file` or to a compose env_file.
BACKEND=mps
MODE_CONFIG_PATH=conf
WORKER_ISOLATION=inproc
