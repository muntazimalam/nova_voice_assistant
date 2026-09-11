"""Rate limiting middleware for WebSocket connections."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("voice_assistant")


@dataclass
class RateLimiter:
    """Token bucket rate limiter per connection."""
    
    max_requests: int = 60
    window_seconds: float = 60.0
    burst_size: int = 10
    
    _tokens: dict[str, float] = field(default_factory=dict)
    _last_refill: dict[str, float] = field(default_factory=dict)
    _window_start: dict[str, float] = field(default_factory=dict)
    _request_counts: dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        self._tokens = defaultdict(lambda: float(self.burst_size))
        self._last_refill = defaultdict(float)
        self._window_start = defaultdict(float)
        self._request_counts = defaultdict(int)

    def allow(self, client_id: str) -> bool:
        """Check if a request is allowed for the given client.
        
        Args:
            client_id: Unique client identifier
            
        Returns:
            True if request is allowed
        """
        now = time.time()
        
        if now - self._window_start[client_id] > self.window_seconds:
            self._window_start[client_id] = now
            self._request_counts[client_id] = 0
        
        self._request_counts[client_id] += 1
        
        if self._request_counts[client_id] > self.max_requests:
            logger.warning(
                "Rate limit exceeded for client %s: %d requests in %.0fs",
                client_id[:8], self._request_counts[client_id], self.window_seconds,
            )
            return False
        
        elapsed = now - self._last_refill[client_id]
        self._last_refill[client_id] = now
        
        self._tokens[client_id] = min(
            self.burst_size,
            self._tokens[client_id] + elapsed * (self.max_requests / self.window_seconds),
        )
        
        if self._tokens[client_id] < 1.0:
            logger.warning("Rate limit burst exhausted for client %s", client_id[:8])
            return False
        
        self._tokens[client_id] -= 1.0
        return True

    def get_usage(self, client_id: str) -> dict:
        """Get current usage stats for a client."""
        now = time.time()
        window_remaining = max(0, self.window_seconds - (now - self._window_start[client_id]))
        return {
            "requests_in_window": self._request_counts.get(client_id, 0),
            "max_requests": self.max_requests,
            "window_remaining_seconds": round(window_remaining, 1),
            "tokens_remaining": round(self._tokens.get(client_id, 0), 1),
        }

    def reset(self, client_id: str) -> None:
        """Reset rate limiter state for a client."""
        self._tokens[client_id] = float(self.burst_size)
        self._last_refill[client_id] = time.time()
        self._window_start[client_id] = time.time()
        self._request_counts[client_id] = 0


_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
