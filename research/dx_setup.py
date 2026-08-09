"""Developer experience setup: rich tracebacks + loguru configuration.

Import and call setup() at the top of any entry point to get:
- Beautiful syntax-highlighted tracebacks (replaces default Python tracebacks)
- Structured loguru logging with file rotation
- Consistent log format across all scripts

Usage:
    from research.dx_setup import setup
    setup()  # call once at startup, before any other code runs
"""
from __future__ import annotations

import sys
from pathlib import Path

_INITIALIZED = False


def setup(
    log_file: str | Path | None = None,
    log_level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "3 days",
) -> None:
    """Install rich tracebacks and configure loguru logging.

    Args:
        log_file: path to log file (None = stderr only)
        log_level: minimum log level (DEBUG, INFO, WARNING, ERROR)
        rotation: log file rotation size
        retention: how long to keep old log files
    """
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    # 0. Fix Windows console encoding (cp1252 can't handle some Unicode)
    import os as _os
    _os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    # 1. Rich tracebacks — replaces default Python exception formatting
    try:
        from rich.traceback import install as install_traceback
        install_traceback(show_locals=False, max_frames=20)
    except ImportError:
        pass

    # 2. Loguru — configure if not already configured
    try:
        from loguru import logger
        logger.remove()  # remove default handler

        # Console: colored, level-filtered
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
        )

        # File: structured JSON, rotated
        if log_file is not None:
            logger.add(
                str(log_file),
                level="DEBUG",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                       "{name}:{function}:{line} | {message}",
                rotation=rotation,
                retention=retention,
                encoding="utf-8",
            )
    except ImportError:
        pass


def get_logger():
    """Return the loguru logger (call after setup())."""
    try:
        from loguru import logger
        return logger
    except ImportError:
        import logging
        return logging.getLogger("forgeai")
