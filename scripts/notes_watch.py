#!/usr/bin/env python3
"""Track changes in notes/ and record updates in a cookie file."""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTES_DIR = ROOT / "notes"
COOKIE_PATH = ROOT / "notes" / ".notes_tracking.json"


def _hash_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_cookie() -> dict:
    if not COOKIE_PATH.exists():
        return {"files": {}, "last_scan": None}
    try:
        with COOKIE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"files": {}, "last_scan": None}


def _save_cookie(data: dict) -> None:
    COOKIE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = COOKIE_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp_path.replace(COOKIE_PATH)


def main() -> int:
    if not NOTES_DIR.exists():
        print(f"notes directory not found at {NOTES_DIR}")
        return 1

    cookie = _load_cookie()
    prev = cookie.get("files", {})

    updates = []
    for path in NOTES_DIR.rglob("*.md"):
        if path.name.startswith("."):
            continue
        try:
            digest = _hash_file(path)
        except Exception:
            continue
        rel = str(path.relative_to(ROOT))
        if prev.get(rel) != digest:
            updates.append(rel)
        prev[rel] = digest

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    cookie["files"] = prev
    cookie["last_scan"] = now
    cookie.setdefault("updates", []).append({"ts": now, "files": updates})

    _save_cookie(cookie)

    if updates:
        print("Updated notes:")
        for rel in updates:
            print(f"- {rel}")
    else:
        print("No note changes detected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
