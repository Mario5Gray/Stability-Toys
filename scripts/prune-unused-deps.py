#!/usr/bin/env python3
"""
prune-unused-deps.py — remove known-unused packages from requirements.txt.

Dry-run by default; pass --apply to write changes.

Packages marked for removal were identified by auditing every import statement
across backends/, server/, invokers/, utils/, and persistence/ against the
declared requirements. See notes/rknn_ff.md audit section for context.

Usage:
    python scripts/prune-unused-deps.py           # dry-run, shows diff
    python scripts/prune-unused-deps.py --apply   # writes requirements.txt
"""

import argparse
import re
import sys
from pathlib import Path

REQUIREMENTS = Path(__file__).parent.parent / "requirements.txt"

# Packages confirmed unused: no import anywhere in the production codebase.
# Each entry is the package name as it appears in requirements.txt (case-insensitive prefix match).
PRUNE = {
    "nvidia-ml-py":           "GPU monitoring — not imported anywhere",
    "peft":                   "LoRA support — not wired in yet",
    "accelerate":             "Distributed training — not activated",
    "psutil":                 "System monitoring — not imported anywhere",
    "scipy":                  "Scientific computing — not imported anywhere",
    "opencv-python-headless": "Image processing — PIL handles all image work",
    "fast-histogram":         "Histogram util — not imported anywhere",
}

# Packages to keep even though they have no direct import:
#   protobuf      — explicit floor pin for transitive deps (transformers, grpc)
#   xformers      — activated via pipe.enable_xformers_memory_efficient_attention()
#   uvloop        — picked up by uvicorn[standard] at runtime
#   python-multipart — required by FastAPI form/file parsing
#   watchdog      — imported in server file-watch code


def _pkg_name(line: str) -> str:
    """Extract bare package name from a requirement line (strips version specifiers)."""
    line = line.strip()
    if not line or line.startswith("#"):
        return ""
    # strip extras like package[extra]
    m = re.match(r"^([A-Za-z0-9_.\-]+)", line)
    return m.group(1).lower() if m else ""


def prune(lines: list[str]) -> tuple[list[str], list[str]]:
    """Return (kept_lines, removed_lines)."""
    kept, removed = [], []
    prune_keys = {k.lower() for k in PRUNE}
    for line in lines:
        name = _pkg_name(line)
        if name and name in prune_keys:
            removed.append(line)
        else:
            kept.append(line)
    return kept, removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Write changes to requirements.txt")
    parser.add_argument("--file", default=str(REQUIREMENTS), help="Path to requirements file")
    args = parser.parse_args()

    req_path = Path(args.file)
    if not req_path.exists():
        print(f"ERROR: {req_path} not found", file=sys.stderr)
        sys.exit(1)

    lines = req_path.read_text().splitlines(keepends=True)
    kept, removed = prune(lines)

    if not removed:
        print("Nothing to prune — requirements.txt is already clean.")
        return

    print("Packages to remove:")
    for line in removed:
        name = _pkg_name(line)
        reason = PRUNE.get(name, "")
        print(f"  - {line.rstrip():<40}  # {reason}")

    if args.apply:
        req_path.write_text("".join(kept))
        print(f"\nWritten: {req_path}")
    else:
        print("\nDry-run — pass --apply to write changes.")


if __name__ == "__main__":
    main()
