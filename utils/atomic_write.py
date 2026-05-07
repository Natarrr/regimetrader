"""utils/atomic_write.py — write JSON atomically via temp-file + replace.

Knuth (Turing 1974) frame: durable updates require write-then-rename so no
reader ever observes a partial file. POSIX `os.replace` and Windows
`MoveFileExW(MOVEFILE_REPLACE_EXISTING)` are both atomic on the same
filesystem, so this idiom is safe across platforms.

Usage:
    from utils.atomic_write import atomic_write_json
    atomic_write_json(Path("logs/metrics.json"), {"ok": True})
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, obj: Any, mode: int = 0o644) -> None:
    """Knuth: write JSON to `path` atomically (UTF-8, indent=2).

    Steps:
      1. Serialise to bytes (raises before any I/O if obj is not JSON-able).
      2. Create a temp file in the same directory (so os.replace is atomic).
      3. fsync the temp before rename so the bytes are on disk.
      4. os.replace(temp, path) — atomic at the POSIX/NTFS layer.

    Parent directories are created if missing. On Windows, chmod is best-effort.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, ensure_ascii=False).encode("utf-8")

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        try:
            os.chmod(tmp_name, mode)
        except (OSError, NotImplementedError):
            pass
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
