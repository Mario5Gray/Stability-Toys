#!/usr/bin/env bash
# check-types.sh — run pyright over the production Python codebase.
# Passes any extra arguments through to pyright (e.g. a specific file path).
# Usage:
#   ./check-types.sh                      # check everything in pyrightconfig.json
#   ./check-types.sh backends/base.py     # check one file
#   ./check-types.sh --outputjson . | jq  # JSON output for tooling
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

[ -f "$SCRIPT_DIR/env.custom" ] && source "$SCRIPT_DIR/env.custom"
[ -f "$SCRIPT_DIR/env.lsp" ] && source "$SCRIPT_DIR/env.lsp"

PYTHON="${LSP_PYTHON:-$(which python)}"

exec pyright --pythonpath "$PYTHON" "$@"
