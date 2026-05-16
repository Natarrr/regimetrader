"""regime_trader/utils/io.py
Atomic JSON I/O helpers.

Write-temp-then-rename guarantees readers never see a half-written file,
even if the process is killed mid-write.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)


def save_json_atomic(path: Path | str, payload: Any, *, indent: int = 2) -> None:
    """Write *payload* to *path* atomically (tmp file + os.replace).

    Sobiech (2001) — the rename syscall is atomic on POSIX and near-atomic on
    Windows (NTFS), so readers always see a complete file.

    Args:
        path:    Destination file path.
        payload: JSON-serialisable object.
        indent:  Indentation for the output JSON (default 2).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=indent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path | str, obj: Any, mode: int = 0o644) -> None:
    """Knuth (Turing 1974) — write JSON to *path* atomically (UTF-8, indent=2).

    Alias for save_json_atomic kept for back-compat with monitoring modules.
    The *mode* argument is accepted but is a best-effort chmod on Windows.
    """
    save_json_atomic(path, obj)
    try:
        os.chmod(str(path), mode)
    except (OSError, NotImplementedError):
        pass


def load_json_safe(path: Path | str, default: Any = None) -> Any:
    """Read and parse a JSON file; return *default* on any error.

    Args:
        path:    File to read.
        default: Value to return when the file is absent or unreadable.
    """
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except Exception as exc:
        log.warning("load_json_safe(%s): %s", path, exc)
        return default
