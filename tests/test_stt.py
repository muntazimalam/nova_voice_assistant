"""Unit tests for the STT and audio-buffer helpers (no model, no network)."""

import numpy as np
import pytest

from app.config import get_settings
from app.stt import RollingAudioBuffer, SpeechToText


def _pcm(*samples: int) -> bytes:
    return np.array(samples, dtype="<i2").tobytes()


def _speech_pcm(rms: int = 1500, n: int = 1600) -> bytes:
    """Synthesize 100 ms of 16 kHz 'speech-like' noise at a target RMS."""
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, n).astype(np.float32)
    x = x / (np.linalg.norm(x) / np.sqrt(n)) * rms
    return (x.astype(np.int16)).tobytes()


class TestCalculateRms:
    def test_silence_is_zero(self):
        assert SpeechToText.calculate_rms(b"") == 0.0
        assert SpeechToText.calculate_rms(b"\x00\x00" * 100) == 0.0

    def test_tiny_chunk_is_zero(self):
        assert SpeechToText.calculate_rms(b"\x00") == 0.0

    def test_active_speech_is_loud(self):
        assert SpeechToText.calculate_rms(_speech_pcm(rms=1800)) > 1000.0

    def test_white_noise_matches_reference(self):
        got = SpeechToText.calculate_rms(_speech_pcm(rms=1200))
        assert got == pytest.approx(1200.0, rel=0.15)


class TestPcmToFloat32:
    def test_peak_amplitude_normalized_to_one(self):
        arr = SpeechToText._pcm_to_float32(_pcm(32767, -32768, 0, 0))
        assert arr.dtype == np.float32
        assert float(arr[0]) == pytest.approx(32767 / 32768)
        assert float(arr[1]) == pytest.approx(-32768 / 32768)
        assert float(arr[2]) == 0.0

    def test_length_is_half_byte_length(self):
        assert SpeechToText._pcm_to_float32(b"\x01\x02\x03\x04").shape == (2,)


class TestRollingAudioBuffer:
    def test_feed_keeps_only_window(self):
        buf = RollingAudioBuffer(window_ms=1000)  # cap = 16k * 1s * 2 B = 32 kB
        cap = buf.window_bytes
        assert cap == 32000
        buf.feed(bytes(10000))
        buf.feed(bytes(15000))  # 25 kB total < cap -> no trim
        assert len(buf) == 25000
        buf.feed(bytes(10000))  # 35 kB total > cap -> trim to last 32 kB
        assert len(buf) == cap

    def test_tail_returns_most_recent_millis(self):
        buf = RollingAudioBuffer(window_ms=1000)
        buf.feed(b"\x01\x02" * 100)  # 100 bytes = 50 samples = 3.125 ms
        assert buf.tail(10) == b"\x01\x02" * 100  # short buffer -> everything

    def test_bytes_protocol(self):
        buf = RollingAudioBuffer(window_ms=100)
        buf.feed(b"\xaa\xbb" * 10)
        assert bytes(buf) == b"\xaa\xbb" * 10


class TestTranscribe:
    def test_empty_audio_returns_empty_string(self):
        stt = SpeechToText(get_settings())
        assert stt.transcribe(b"") == ""
        assert stt.transcribe(b"\x00") == ""

    def test_transcribes_with_fake_model(self, monkeypatch):
        stt = SpeechToText(get_settings())

        class FakeSegment:
            def __init__(self, text, no_speech_prob=0.0):
                self.text = text
                self.no_speech_prob = no_speech_prob

        class FakeModel:
            def transcribe(self, audio, **kwargs):
                captured = {"pcm_is_float32": audio.dtype == np.float32}
                self.captured = captured
                return (
                    iter([FakeSegment("hello"), FakeSegment(" world", 0.4)]),
                    None,
                )

        fake = FakeModel()
        monkeypatch.setattr(stt, "_model", fake)
        monkeypatch.setattr(stt, "_load", lambda: fake)

        result = stt.transcribe(_speech_pcm())
        assert result == "hello world"
        assert fake.captured["pcm_is_float32"]

    def test_filters_low_confidence_segments(self, monkeypatch):
        stt = SpeechToText(get_settings())

        class FakeSegment:
            def __init__(self, text, no_speech_prob=0.0):
                self.text = text
                self.no_speech_prob = no_speech_prob

        class FakeModel:
            def transcribe(self, audio, **kwargs):
                return (
                    iter(
                        [
                            FakeSegment("keep me", 0.1),
                            FakeSegment("drop me", 0.95),
                        ]
                    ),
                    None,
                )

        fake = FakeModel()
        monkeypatch.setattr(stt, "_model", fake)
        monkeypatch.setattr(stt, "_load", lambda: fake)

        assert stt.transcribe(_speech_pcm()) == "keep me"
