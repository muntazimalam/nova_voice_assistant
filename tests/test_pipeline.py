"""End-to-end pipeline tests (STT skipped; LLM and TTS are stubbed, offline)."""
import asyncio
import json

import pytest

from app.main import VoicePipeline
from app.state import ConnectionState


class FakeWebSocket:
    """Minimal stand-in for a Starlette WebSocket that records frames."""

    def __init__(self):
        self.frames: list[dict] = []  # decoded JSON frames, in order
        self.audio_bytes: list[bytes] = []

    async def send_text(self, data: str) -> None:
        self.frames.append(json.loads(data))

    async def send_bytes(self, data: bytes) -> None:
        self.audio_bytes.append(data)

    def frames_of_type(self, ftype: str) -> list[dict]:
        return [f for f in self.frames if f.get("type") == ftype]


@pytest.fixture
def fake_ws():
    return FakeWebSocket()


def _stub_services(monkeypatch, sentences):
    """Replace the global LLM/TTS services with deterministic fake streams.

    ``sentences`` are yielded token-by-token like a real Gemini stream, and the
    TTS fake returns a unique byte-marker per sentence.
    """
    import app.main as main

    synced = []

    async def fake_llm(history):
        for sentence in sentences:
            for token in list(sentence):
                yield token

    async def fake_tts(text):
        synced.append(text)
        marker = f"<audio:{text[:12]}>".encode()
        for _ in range(3):
            yield marker

    monkeypatch.setattr(main.llm_service, "stream_reply", fake_llm)
    monkeypatch.setattr(main.tts_service, "stream_audio", fake_tts)
    return synced


async def _await_pipeline(state, timeout: float = 3.0):
    task = state.pipeline_task
    assert task is not None, "run_command should have created a pipeline task"
    await asyncio.wait_for(asyncio.shield(task), timeout=timeout)


class TestPipelineEndToEnd:
    async def test_streams_speech_with_frames_and_commits_history(self, fake_ws, monkeypatch):
        state = ConnectionState()
        pipeline = VoicePipeline(state)
        synced = _stub_services(
            monkeypatch,
            ["Hello there!", " It is sunny today.", " Let's build."],
        )

        await pipeline.run_command(fake_ws, "what time is it")
        await _await_pipeline(state)

        assert fake_ws.frames_of_type("audio_start")[0]["encoding"] in ("mp3", "wav")
        segments_started = fake_ws.frames_of_type("audio_segment_start")
        segments_ended = fake_ws.frames_of_type("audio_segment_end")
        assert len(segments_started) == len(synced) == 3
        assert len(segments_ended) == 3
        assert synced[0] == "Hello there!"
        assert fake_ws.audio_bytes  # audio was actually streamed

        reply = fake_ws.frames_of_type("chat_reply")[0]
        assert reply["content"] == "Hello there! It is sunny today. Let's build."

        # metrics surfaced everywhere the WS protocol promises them
        assert reply["metrics"]["total_roundtrip_ms"] >= 0

        assert state.stage == "IDLE"
        assert state.history == [
            {"role": "user", "content": "what time is it"},
            {"role": "assistant", "content": "Hello there! It is sunny today. Let's build."},
        ]

    async def test_interrupt_stops_speech_and_returns_to_idle(self, fake_ws, monkeypatch):
        import app.main as main

        state = ConnectionState()
        pipeline = VoicePipeline(state)

        async def slow_llm(history):
            # Emit tokens slowly so we can interrupt mid-stream.
            for token in "This is a long answer to interrupt. It keeps going. And going!":
                yield token
                await asyncio.sleep(0.001)

        async def slow_tts(text):
            while True:
                yield b"\xff" * 100
                await asyncio.sleep(0.01)

        monkeypatch.setattr(main.llm_service, "stream_reply", slow_llm)
        monkeypatch.setattr(main.tts_service, "stream_audio", slow_tts)

        await pipeline.run_command(fake_ws, "interrupt me")
        task = state.pipeline_task

        # Fire barge-in shortly after the audio stream begins.
        await asyncio.sleep(0.05)
        state.interrupt_event.set()
        await asyncio.wait_for(asyncio.shield(task), timeout=3.0)

        assert state.stage == "IDLE"
        assert any(
            "interrupted" in f.get("message", "").lower()
            for f in fake_ws.frames_of_type("state_change")
        )
        assert state.busy is False

    async def test_barge_in_grace_period_ignores_reply_echo(self, fake_ws, monkeypatch):
        import app.main as main
        import numpy as np
        from app.main import handle_audio_frame

        state = ConnectionState()
        pipeline = VoicePipeline(state)
        state.stage = "SPEAKING"
        state.wake_buffer = main.wake_service.new_buffer()
        state.speaking_started_at = state.now_ms()

        pcm = (np.full(320, 30000, dtype="<i2").tobytes())

        await handle_audio_frame(fake_ws, pipeline, state, pcm)
        assert not state.interrupt_event.is_set(), "barge-in must be ignored inside grace window"

        state.speaking_started_at = state.now_ms() - state.barge_in_grace_ms - 100
        await handle_audio_frame(fake_ws, pipeline, state, pcm)
        assert state.interrupt_event.is_set(), "barge-in should fire after the grace window"

    async def test_listening_debounce_requires_sustained_voice(self, fake_ws, monkeypatch):
        import app.main as main
        import numpy as np
        from app.main import handle_audio_frame

        state = ConnectionState()
        pipeline = VoicePipeline(state)
        state.stage = "LISTENING"
        state.wake_buffer = main.wake_service.new_buffer()

        async def fake_feed(pcm, buffer):
            buffer.feed(pcm)
            return ""

        monkeypatch.setattr(main.wake_service, "feed", fake_feed)

        pcm = np.full(320, 30000, dtype="<i2").tobytes()
        debounce = max(1, main.settings.voice_debounce_frames)

        # Isolated voiced chunks below the debounce count must NOT start capture.
        for _ in range(debounce - 1):
            await handle_audio_frame(fake_ws, pipeline, state, pcm)
        assert state.stage == "LISTENING", "debounce must swallow isolated noise bursts"
        assert state.voice_streak == debounce - 1

        # A silence chunk resets the streak.
        silence = (np.zeros(320, dtype="<i2")).tobytes()
        await handle_audio_frame(fake_ws, pipeline, state, silence)
        assert state.voice_streak == 0

        # Sustained voice across the threshold starts capture with pre-roll.
        for _ in range(debounce):
            await handle_audio_frame(fake_ws, pipeline, state, pcm)
        assert state.stage == "CAPTURING"
        assert len(state.command_buffer) > 0

    async def test_empty_text_never_generates_audio(self, fake_ws, monkeypatch):
        state = ConnectionState()
        pipeline = VoicePipeline(state)
        _stub_services(monkeypatch, ["ShouldNotBeCalled"])

        await pipeline.run_command(fake_ws, "   ")
        await _await_pipeline(state)

        assert fake_ws.frames_of_type("audio_start") == []
        assert fake_ws.frames_of_type("audio_segment_start") == []
        assert fake_ws.audio_bytes == []
        assert state.stage == "IDLE"

    async def test_busy_guard_rejects_overlap(self, fake_ws, monkeypatch):
        import app.main as main

        state = ConnectionState()
        pipeline = VoicePipeline(state)

        async def slow_llm(history):
            await asyncio.sleep(0.5)  # long enough to force a busy collision
            yield "done"

        async def never_ending_tts(text):
            yield b"x"

        monkeypatch.setattr(main.llm_service, "stream_reply", slow_llm)
        monkeypatch.setattr(main.tts_service, "stream_audio", never_ending_tts)

        await pipeline.run_command(fake_ws, "first command")
        task = state.pipeline_task
        assert task is not None

        # Second command while busy -> warning, no new task.
        await pipeline.run_command(fake_ws, "second command")
        warnings = fake_ws.frames_of_type("system_event")
        assert any("Busy" in w.get("message", "") for w in warnings)

        # Cleanup: interrupt and let the first task finish.
        state.interrupt_event.set()
        await asyncio.wait_for(asyncio.shield(task), timeout=3.0)
        assert state.stage == "IDLE"