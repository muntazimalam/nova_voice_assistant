"""Structured logging with correlation IDs for request tracing."""
from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from typing import Optional

_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return _correlation_id.get()


def set_correlation_id(cid: Optional[str] = None) -> str:
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
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None) -> None:
        if fmt is None:
            fmt = "%(asctime)s [%(levelname)s] %(name)s [%(correlation_id)s]: %(message)s"
        super().__init__(fmt=fmt, datefmt=datefmt)


def setup_structured_logging(
    level: int = logging.INFO,
    fmt: Optional[str] = None,
) -> None:
    """Configure structured logging with correlation IDs."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    for handler in root_logger.handlers:
        handler.addFilter(CorrelationFilter())
        if isinstance(handler, logging.StreamHandler) and handler.formatter is None:
            handler.setFormatter(StructuredFormatter(fmt=fmt))
    
    for name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        uv_logger = logging.getLogger(name)
        uv_logger.addFilter(CorrelationFilter())


class ConnectionLogger:
    """Logger with automatic correlation ID for WebSocket connections."""
    
    def __init__(self, connection_id: Optional[str] = None) -> None:
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
