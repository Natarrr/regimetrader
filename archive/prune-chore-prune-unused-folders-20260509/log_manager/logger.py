"""
logger.py
─────────
Structured logging for regime_trader using loguru as the backend
and rich for coloured console output.

SETUP
─────
Call setup_logger() once at startup (in main.py) before importing any
other module.  All subsequent `from log_manager.logger import get_logger`
calls return a pre-configured loguru sink bound to the caller's name.

OUTPUT TARGETS
──────────────
  stderr (rich)     Coloured human-readable output during development.
                    Suppressed when LOG_LEVEL=WARNING or above in production.
  logs/{date}.log   Rotating daily JSON-structured log file for audit/replay.
                    Retained for LOG_RETENTION days (default 30).

STRUCTURED FIELDS  (always present in JSON log)
────────────────────────────────────────────────
  time, level, name, message
  + any kwargs passed to logger.bind(**kwargs).info(...)

Usage
─────
    from log_manager.logger import get_logger
    log = get_logger(__name__)

    log.info("Regime confirmed", regime="Bull", confidence=0.87)
    log.warning("Circuit breaker", level="REDUCE_DAY", daily_dd=-0.023)
    log.bind(symbol="SPY", order_id="abc123").info("Order submitted")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger as _loguru_logger

# Re-export the configured logger so callers can do:
#   from log_manager.logger import logger
# or:
#   from log_manager.logger import get_logger
logger = _loguru_logger


def setup_logger(
    level:     str           = "INFO",
    log_dir:   str           = "logs",
    rotation:  str           = "1 day",
    retention: str           = "30 days",
    json_logs: bool          = False,
    sink_stderr: bool        = True,
) -> None:
    """
    Configure loguru sinks.  Call once at process startup.

    Parameters
    ----------
    level      : Minimum log level ("DEBUG", "INFO", "WARNING", "ERROR").
    log_dir    : Directory for rotating log files.
    rotation   : loguru rotation spec (e.g. "1 day", "100 MB").
    retention  : loguru retention spec (e.g. "30 days").
    json_logs  : Write JSON-structured records to the log file.
    sink_stderr: Emit human-readable logs to stderr (disable in production).
    """
    # Remove the default handler
    _loguru_logger.remove()

    # ── stderr sink (rich-formatted, coloured) ────────────────────────────────
    if sink_stderr:
        _loguru_logger.add(
            sys.stderr,
            level  = level,
            format = (
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
                "<level>{message}</level>"
            ),
            colorize = True,
            backtrace = True,
            diagnose  = True,
        )

    # ── Rotating file sink (JSON or plain text) ───────────────────────────────
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    if json_logs:
        _loguru_logger.add(
            log_path / "{time:YYYY-MM-DD}.json",
            level      = level,
            rotation   = rotation,
            retention  = retention,
            serialize  = True,       # JSON records
            enqueue    = True,       # thread-safe async write
            backtrace  = True,
            diagnose   = False,      # don't leak secrets into JSON traces
        )
    else:
        _loguru_logger.add(
            log_path / "{time:YYYY-MM-DD}.log",
            level      = level,
            rotation   = rotation,
            retention  = retention,
            format     = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                "{name}:{line} — {message}"
            ),
            enqueue    = True,
            backtrace  = True,
            diagnose   = False,
        )

    _loguru_logger.info(
        "Logger initialised | level={} | log_dir={} | json={}",
        level, log_dir, json_logs,
    )


def get_logger(name: Optional[str] = None):
    """
    Return a loguru logger optionally bound to a module name.

    Parameters
    ----------
    name : Module name string (typically __name__).
           If provided the logger is bound with name=name so it appears
           in the log record's 'name' field for easy filtering.

    Returns
    -------
    A loguru Logger (or BoundLogger) instance.
    """
    if name:
        return _loguru_logger.bind(name=name)
    return _loguru_logger
