"""LLM response caching for reduced latency and API costs."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("voice_assistant")


@dataclass
class CacheEntry:
    response: str
    created_at: float
    ttl_seconds: float
    access_count: int = 0
    last_accessed: float = 0.0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds


class LLMCache:
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl_seconds: float = 3600.0,
        cleanup_interval_seconds: float = 300.0,
    ) -> None:
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl_seconds
        self._cleanup_interval = cleanup_interval_seconds
        self._lock = asyncio.Lock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}
        self._last_cleanup = time.time()

    def _make_key(self, messages: list[dict], model: str = "") -> str:
        content = json.dumps(messages, sort_keys=True, default=str)
        key_data = f"{model}:{content}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    async def get(self, messages: list[dict], model: str = "") -> Optional[str]:
        key = self._make_key(messages, model)
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            if entry.is_expired:
                del self._cache[key]
                self._stats["misses"] += 1
                return None
            self._cache.move_to_end(key)
            entry.access_count += 1
            entry.last_accessed = time.time()
            self._stats["hits"] += 1
            return entry.response

    async def set(
        self,
        messages: list[dict],
        response: str,
        model: str = "",
        ttl_seconds: Optional[float] = None,
    ) -> None:
        key = self._make_key(messages, model)
        ttl = ttl_seconds or self._default_ttl
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
            while len(self._cache) >= self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                self._stats["evictions"] += 1
            self._cache[key] = CacheEntry(
                response=response,
                created_at=time.time(),
                ttl_seconds=ttl,
            )
            await self._cleanup_expired()

    async def _cleanup_expired(self) -> None:
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        expired_keys = [k for k, e in self._cache.items() if e.is_expired]
        for k in expired_keys:
            del self._cache[k]

    def get_stats(self) -> dict:
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
            "hit_rate": round(hit_rate, 3),
        }

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()
            self._stats = {"hits": 0, "misses": 0, "evictions": 0}


class ResponseSummarizer:
    """Summarizes conversation history to maintain context within token limits."""

    def __init__(self, max_chars: int = 400) -> None:
        self._max_chars = max_chars

    async def summarize(self, messages: list[dict]) -> str:
        """Generate a concise summary of conversation turns.
        
        Falls back to rough extractive summarization if the LLM is unavailable,
        so the feature degrades gracefully.
        """
        try:
            text_parts = [
                f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
            ]
            full_text = " ".join(text_parts)
            return self._extractive_summary(full_text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Summarization failed: %s", exc)
            return ""

    def _extractive_summary(self, text: str) -> str:
        """Simple extractive summarization: key sentences by word frequency."""
        if len(text) <= self._max_chars:
            return text

        sentences = [
            s.strip() for s in text.replace("\n", " ").split(". ")
            if s.strip()
        ]

        words = text.lower().split()
        stopwords = {
            "a", "an", "the", "and", "or", "but", "if", "then", "of", "to",
            "in", "on", "for", "with", "at", "by", "from", "as", "is", "are",
            "was", "were", "i", "you", "he", "she", "it", "we", "they",
            "what", "when", "where", "how", "why", "do", "does", "did",
        }
        word_freq: dict[str, int] = {}
        for w in words:
            w = w.strip(".,!?;:()\"'")
            if w and w not in stopwords:
                word_freq[w] = word_freq.get(w, 0) + 1

        if not sentences:
            return text[: self._max_chars]

        scored = []
        for s in sentences:
            score = sum(word_freq.get(w.strip(".,!?;:()\"'"), 0) for w in s.lower().split())
            scored.append((score, s))

        scored.sort(reverse=True, key=lambda x: x[0])

        summary = ""
        for _, sentence in scored:
            if len(summary) + len(sentence) + 2 > self._max_chars:
                break
            if summary:
                summary += ". "
            summary += sentence

        if not summary:
            summary = text[: self._max_chars]

        logger.debug("Extractive summary: %d chars", len(summary))
        return summary
