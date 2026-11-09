# Native MPS path (STABL-gpcsnipw).
#
# Docker Desktop on macOS runs a Linux VM and does not expose Metal, so no
# container on a Mac can run MPS. This file is sourced by `make mps`; it is
# never passed to `docker run --env-file` or to a compose env_file.
BACKEND=mps
MODE_CONFIG_PATH=${MODE_CONFIG_PATH:-conf}
WORKER_ISOLATION=inproc
FS_STORAGE_DIR=${FS_STORAGE_DIR:-./store}
STORAGE_PROVIDER=${STORAGE_PROVIDER:-FILESYSTEM}

# --- Private CA trust (node2.lan / agentgateway) ---------------------------
# certs/ is machine-local and gitignored. conda's Python ships its own CA
# bundle and does not read the macOS Keychain, so combine certifi's public
# roots with the private CAs from certs/*.crt and point Python's TLS stack at
# the result. Without this, HTTPS calls to the gateway fail with
# "unable to get local issuer certificate".
if [ -d "$PWD/certs" ] && command -v python >/dev/null 2>&1; then
  ST_CA_BUNDLE="$PWD/certs/ca-bundle.pem"
  python - "$ST_CA_BUNDLE" <<'PY'
import glob, os, sys

import certifi

out_path = sys.argv[1]
with open(out_path, "w") as out:
    out.write(open(certifi.where()).read())
    for path in sorted(glob.glob(os.path.join(os.path.dirname(out_path), "*.crt"))):
        out.write("\n" + open(path).read())
PY
  SSL_CERT_FILE="$ST_CA_BUNDLE"
  REQUESTS_CA_BUNDLE="$ST_CA_BUNDLE"
fi
