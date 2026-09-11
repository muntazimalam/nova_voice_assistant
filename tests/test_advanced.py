"""Tests for the advanced pipeline features (codec, cache, VAD, echo cancel)."""
import asyncio
import time

import numpy as np
import pytest

from app.cache import LLMCache
from app.codec import AudioFrameBuffer, OpusCodec
from app.echo_cancel import EchoCanceler
from app.rate_limit import RateLimiter


# ── Opus Codec ───────────────────────────────────────────────────────────────

class TestOpusCodec:
    def test_roundtrip_silence(self):
        codec = OpusCodec()
        frame_size = codec.get_frame_size()
        silence = b"\x00" * frame_size
        opus = codec.encode_pcm_to_opus(silence)
        assert isinstance(opus, bytes)
        decoded = codec.decode_opus_to_pcm(opus)
        assert len(decoded) == frame_size

    def test_roundtrip_tone(self):
        codec = OpusCodec()
        frame_size = codec.get_frame_size()
        # 440 Hz tone at 16 kHz, one frame
        samples = np.sin(2 * np.pi * 440 * np.arange(frame_size // 2) / 16000).astype(np.int16)
        pcm = samples.tobytes()
        opus = codec.encode_pcm_to_opus(pcm)
        decoded = codec.decode_opus_to_pcm(opus)
        assert len(decoded) == frame_size


class TestAudioFrameBuffer:
    def test_feed_returns_complete_frames(self):
        buf = AudioFrameBuffer(frame_size=4)
        frames = buf.feed(b"\x01\x02\x03\x04\x05\x06")
        assert frames == [b"\x01\x02\x03\x04"]
        assert bytes(buf.flush()) == b"\x05\x06"

    def test_clear(self):
        buf = AudioFrameBuffer(frame_size=4)
        buf.feed(b"\x01\x02\x03\x04\x05\x06")
        buf.clear()
        assert buf.flush() is None


# ── LLM Cache ────────────────────────────────────────────────────────────────

class TestLLMCache:
    @pytest.mark.asyncio
    async def test_cache_hit_and_miss(self):
        cache = LLMCache(max_size=10)
        messages = [{"role": "user", "content": "What time is it?"}]

        assert await cache.get(messages) is None  # miss
        await cache.set(messages, "It is noon.")
        assert await cache.get(messages) == "It is noon."  # hit

    @pytest.mark.asyncio
    async def test_cache_expiry(self):
        cache = LLMCache(max_size=10, default_ttl_seconds=0.1)
        messages = [{"role": "user", "content": "Hi"}]
        await cache.set(messages, "Hello!")
        assert await cache.get(messages) == "Hello!"
        await asyncio.sleep(0.2)
        assert await cache.get(messages) is None  # expired

    @pytest.mark.asyncio
    async def test_cache_eviction(self):
        cache = LLMCache(max_size=2)
        for i in range(3):
            await cache.set([{"role": "user", "content": f"msg {i}"}], f"reply {i}")
        stats = cache.get_stats()
        assert stats["size"] == 2
        assert stats["evictions"] == 1


# ── Echo Canceler ────────────────────────────────────────────────────────────

class TestEchoCanceler:
    def test_no_output_no_echo(self):
        ec = EchoCanceler()
        assert not ec.is_echo(b"\x00" * 640)

    def test_register_output_marks_echo_window(self):
        ec = EchoCanceler()
        samples = (np.sin(2 * np.pi * 440 * np.arange(320) / 16000) * 8000).astype(np.int16)
        ec.set_outputting(True)
        ec.register_output(samples.tobytes())
        assert ec._is_outputting


# ── Rate Limiter ─────────────────────────────────────────────────────────────

class TestRateLimiter:
    def test_allows_within_budget(self):
        rl = RateLimiter(max_requests=5, window_seconds=60, burst_size=3)
        assert all(rl.allow("client-1") for _ in range(3))

    def test_rejects_over_budget(self):
        rl = RateLimiter(max_requests=5, window_seconds=60, burst_size=1)
        assert rl.allow("client-1")
        assert not rl.allow("client-1")  # burst exhausted immediately

    def test_reset(self):
        rl = RateLimiter(max_requests=10, window_seconds=60, burst_size=1)
        assert rl.allow("c")
        assert not rl.allow("c")  # burst exhausted
        rl.reset("c")
        assert rl.allow("c")  # token replenished after reset