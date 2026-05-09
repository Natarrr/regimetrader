"""regime_trader/utils/logging_cfg.py
Centralised logging configuration with secret masking.

Call configure_logging() once at application startup (streamlit_app.py or CLI
entry points). All other modules just do ``logging.getLogger(__name__)``.

Stiglitz (2001 Nobel) — information asymmetry: secrets in logs are a one-way
leak from the application to its operators / potential attackers. SecretMaskFilter
redacts live env-var values before any handler emits a record.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

# Keys whose runtime values must never appear in log output.
_SECRET_ENV_KEYS: tuple[str, ...] = (
    "FMP_API_KEY",
    "POLYGON_API_KEY",
    "ALPACA_API_KEY",
    "ALPACA_KEY_ID",
    "ALPACA_SECRET_KEY",
    "ALPACA_SECRET",
    "ANTHROPIC_API_KEY",
)


def mask_secret(value: str) -> str:
    """Replace any live secret env-var value found in *value* with '<REDACTED>'.

    Reads environment at call time so rotating secrets are handled correctly.

    Args:
        value: String potentially containing a secret.

    Returns:
        *value* with all detected secret substrings replaced.
    """
    for key in _SECRET_ENV_KEYS:
        secret = os.environ.get(key, "")
        if secret and secret in value:
            value = value.replace(secret, "<REDACTED>")
    return value


class SecretMaskFilter(logging.Filter):
    """logging.Filter that redacts live secret env-var values from log records.

    Attach to any handler to prevent API keys from leaking into log streams,
    files, or external observability platforms.

    Masks both the format string (record.msg) and positional/keyword args so
    that lazy % interpolation never materialises a secret in the final output.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = mask_secret(str(record.msg))
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(
                    mask_secret(str(a)) if isinstance(a, str) else a
                    for a in record.args
                )
            elif isinstance(record.args, dict):
                record.args = {
                    k: mask_secret(str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
        return True


def configure_logging(
    level: int | str = logging.INFO,
    fmt: Optional[str] = None,
    stream=None,
    mask_env: bool = True,
) -> None:
    """Configure the root logger for the regime_trader application.

    Installs a SecretMaskFilter by default so API keys never reach log sinks.
    Safe to call multiple times — replaces the first handler rather than
    stacking new ones (important for test isolation).

    Args:
        level:    Logging level (e.g. logging.DEBUG or "DEBUG").
        fmt:      Log format string; defaults to a concise timestamped format.
        stream:   Output stream (default: sys.stderr).
        mask_env: When True (default), install SecretMaskFilter on the handler.
    """
    if fmt is None:
        fmt = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(fmt))
    if mask_env:
        handler.addFilter(SecretMaskFilter())

    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        root.addHandler(handler)
    else:
        # Replace first handler so repeated calls in tests don't stack handlers.
        root.handlers[0] = handler


def get_logger(name: str) -> logging.Logger:
    """Return a named logger; configure root logger if not yet set up.

    Args:
        name: Logger name (typically __name__ of the calling module).

    Returns:
        Named logging.Logger instance.
    """
    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)


# Module-level logger alias used across the codebase.
_dm_logger: logging.Logger = logging.getLogger("decision_matrix")
