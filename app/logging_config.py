"""Structured logging with correlation IDs for request tracing."""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> str | None:
    """Get the current correlation ID."""
    return _correlation_id.get()


def set_correlation_id(cid: str | None = None) -> str:
    """Set or generate a correlation ID."""
    if cid is None:
        cid = uuid.uuid4().hex[:12]
    _correlation_id.set(cid)
    return cid


def clear_correlation_id() -> None:
    """Clear the current correlation ID."""
    _correlation_id.set(None)


class CorrelationFilter(logging.Filter):
    """Logging filter that adds correlation ID to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id() or "-"
        return True


class StructuredFormatter(logging.Formatter):
    """Structured log formatter with correlation IDs."""

    def __init__(self, fmt: str | None = None, datefmt: str | None = None) -> None:
        if fmt is None:
            fmt = (
                "%(asctime)s [%(levelname)s] %(name)s [%(correlation_id)s]: %(message)s"
            )
        super().__init__(fmt=fmt, datefmt=datefmt)


def setup_structured_logging(
    level: int = logging.INFO,
    fmt: str | None = None,
) -> None:
    """Configure structured logging with correlation IDs."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # If no handlers are configured yet (e.g. no basicConfig was called), the
    # loop below would leave the root logger empty and Python's lastResort
    # handler would silently DROP all INFO records. Ensure a stderr console
    # handler always exists so INFO logs (startup, "LLM configured", metrics)
    # actually reach the terminal.
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        root_logger.addHandler(logging.StreamHandler(sys.stderr))

    for handler in root_logger.handlers:
        handler.addFilter(CorrelationFilter())
        if isinstance(handler, logging.StreamHandler) and handler.formatter is None:
            handler.setFormatter(StructuredFormatter(fmt=fmt))

    for name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        uv_logger = logging.getLogger(name)
        uv_logger.addFilter(CorrelationFilter())


class ConnectionLogger:
    """Logger with automatic correlation ID for WebSocket connections."""

    def __init__(self, connection_id: str | None = None) -> None:
        self._connection_id = connection_id or set_correlation_id()
        self._logger = logging.getLogger("voice_assistant")

    def _log(self, level: int, msg: str, *args, **kwargs) -> None:
        token = _correlation_id.set(self._connection_id)
        try:
            self._logger.log(level, msg, *args, **kwargs)
        finally:
            _correlation_id.reset(token)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.ERROR, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.DEBUG, msg, *args, **kwargs)
