"""Server-side acoustic echo cancellation for barge-in detection."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("voice_assistant")

_SAMPLE_RATE = 16000


@dataclass
class EchoCanceler:
    """Adaptive echo canceller for voice assistant barge-in detection.

    Tracks assistant audio output timing and applies spectral subtraction
    to suppress echo from the assistant's own TTS audio in the microphone.
    """

    sample_rate: int = _SAMPLE_RATE
    filter_length_ms: float = 200.0
    convergence_factor: float = 0.01
    echo_threshold: float = 0.3
    output_window_ms: float = 500.0

    _output_buffer: deque = field(default_factory=lambda: deque(maxlen=100))
    _output_timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    _filter_weights: np.ndarray | None = field(default=None, repr=False)
    _last_output_time: float = field(default=0.0)
    _is_outputting: bool = field(default=False)
    _has_pcm_reference: bool = field(default=False)
    _pending_wav: bytearray = field(default_factory=bytearray, repr=False)
    _wav_payload_start: int | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        filter_len = int(self.sample_rate * self.filter_length_ms / 1000)
        self._filter_weights = np.zeros(filter_len, dtype=np.float32)

    def register_output(self, pcm_int16: bytes) -> None:
        """Register a raw-PCM chunk of assistant output for echo tracking."""
        if not pcm_int16:
            return
        samples = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return
        self._output_buffer.append(samples)
        self._output_timestamps.append(time.monotonic())
        self._last_output_time = time.monotonic()
        self._is_outputting = True
        self._has_pcm_reference = True

    def register_container_output(self, chunk: bytes, encoding: str) -> None:
        """Register a TTS output chunk.

        ``encoding == "wav"`` chunks are RIFF containers from local engines
        (XTTS/Piper) — the header is stripped here so only real PCM reaches the
        references. Any other encoding (e.g. edge-tts MP3) cannot be decoded
        server-side, so only the timing is recorded and ``_has_pcm_reference``
        stays ``False`` (similarity-based echo checks are then disabled rather
        than comparing against compressed bytes interpreted as PCM).
        """
        if encoding == "wav":
            self._pending_wav.extend(chunk)
            pcm = self._consume_wav()
            self.register_output(pcm)
        else:
            # Encoded output (MP3): record timing so `is_echo` timeouts behave,
            # but do NOT feed the compressed bytes into the PCM reference.
            self._last_output_time = time.monotonic()
            self._is_outputting = True

    def _consume_wav(self) -> bytes:
        """Extract the PCM payload from a possibly-straddled RIFF stream."""
        buf = self._pending_wav
        if self._wav_payload_start is None:
            if len(buf) < 12:
                return b""
            start = self._find_wav_data_offset(bytes(buf))
            self._wav_payload_start = start if start is not None else 0
        start = self._wav_payload_start
        if start >= len(buf):
            return b""
        data = bytes(buf[start:])
        buf.clear()
        self._wav_payload_start = None
        return data

    @staticmethod
    def _find_wav_data_offset(wav: bytes) -> int | None:
        """Return byte offset of the ``data`` chunk payload in a RIFF header."""
        if len(wav) < 12 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
            return None
        offset = 12
        while offset + 8 <= len(wav):
            chunk_id = wav[offset : offset + 4]
            size = int.from_bytes(wav[offset + 4 : offset + 8], "little")
            if chunk_id == b"data":
                return offset + 8
            offset += 8 + size
            if size % 2:
                offset += 1
        return None

    def set_outputting(self, state: bool) -> None:
        """Set whether assistant is currently outputting audio."""
        self._is_outputting = state
        if not state:
            self._has_pcm_reference = False
            self._pending_wav.clear()
            self._wav_payload_start = None
            self._filter_weights = np.zeros_like(self._filter_weights)

    def is_echo(self, pcm_int16: bytes, tolerance_ms: float = 500.0) -> bool:
        """Check if mic input is likely echo from assistant output.

        Args:
            pcm_int16: Raw mic audio
            tolerance_ms: Time window to consider echo likely

        Returns:
            True if likely echo
        """
        if not self._is_outputting:
            return False

        now = time.monotonic()
        time_since_output = (now - self._last_output_time) * 1000

        if time_since_output > tolerance_ms:
            self._is_outputting = False
            return False

        # Without a genuine PCM reference there is nothing trustworthy to
        # compare against — encoding like MP3 must not be interpreted as PCM.
        if not self._has_pcm_reference or not self._output_buffer:
            return False

        mic_samples = (
            np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
        )

        if len(mic_samples) == 0:
            return False

        mic_rms = float(np.sqrt(np.mean(np.square(mic_samples))))

        output_samples = np.concatenate(list(self._output_buffer))
        output_rms = (
            float(np.sqrt(np.mean(np.square(output_samples))))
            if len(output_samples) > 0
            else 0.0
        )

        if output_rms < 0.01:
            return False

        similarity = self._compute_similarity(mic_samples, output_samples)

        is_echo = (
            similarity > self.echo_threshold
            and mic_rms < output_rms * 1.5
            and time_since_output < tolerance_ms
        )

        if is_echo:
            logger.debug(
                "Echo detected: similarity=%.2f, mic_rms=%.4f, output_rms=%.4f, delay=%.0fms",
                similarity,
                mic_rms,
                output_rms,
                time_since_output,
            )

        return is_echo

    def _compute_similarity(self, mic: np.ndarray, output: np.ndarray) -> float:
        """Compute normalized cross-correlation between mic and output."""
        weights = self._filter_weights
        if weights is None:
            return 0.0
        min_len = min(len(mic), len(output), weights.shape[0])
        if min_len == 0:
            return 0.0

        mic_seg = mic[:min_len]
        output_seg = output[:min_len]

        mic_norm = np.linalg.norm(mic_seg)
        output_norm = np.linalg.norm(output_seg)

        if mic_norm < 1e-8 or output_norm < 1e-8:
            return 0.0

        correlation = np.abs(np.dot(mic_seg, output_seg) / (mic_norm * output_norm))
        return float(correlation)

    def cancel_echo(self, pcm_int16: bytes) -> bytes:
        """Apply echo cancellation to mic audio.

        Uses spectral subtraction to suppress echo components.
        """
        if (
            not self._is_outputting
            or not self._has_pcm_reference
            or not self._output_buffer
        ):
            return pcm_int16

        mic = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0

        if len(mic) == 0:
            return pcm_int16

        output_concat = np.concatenate(list(self._output_buffer))

        min_len = min(len(mic), len(output_concat))
        if min_len < 64:
            return pcm_int16

        mic_seg = mic[:min_len]
        out_seg = output_concat[:min_len]

        mic_fft = np.fft.rfft(mic_seg)
        out_fft = np.fft.rfft(out_seg)

        mic_power = np.abs(mic_fft) ** 2
        out_power = np.abs(out_fft) ** 2

        total_out_power = np.sum(out_power)
        if total_out_power < 1e-10:
            return pcm_int16

        echo_ratio = np.minimum(
            out_power / (total_out_power / len(out_power) + 1e-10), 1.0
        )

        suppressed_power = mic_power * (1.0 - echo_ratio * 0.8)
        suppressed_power = np.maximum(suppressed_power, mic_power * 0.1)

        suppressed_fft = np.sqrt(suppressed_power) * np.exp(1j * np.angle(mic_fft))
        suppressed = np.fft.irfft(suppressed_fft, n=min_len)

        result = np.zeros_like(mic)
        result[:min_len] = suppressed
        result = np.clip(result * 32768.0, -32768, 32767).astype(np.int16)

        return result.tobytes()

    def reset(self) -> None:
        """Reset echo canceller state."""
        self._output_buffer.clear()
        self._output_timestamps.clear()
        self._filter_weights = np.zeros_like(self._filter_weights)
        self._last_output_time = 0.0
        self._is_outputting = False
        self._has_pcm_reference = False
        self._pending_wav.clear()
        self._wav_payload_start = None


_echo_canceler: EchoCanceler | None = None


def get_echo_canceler() -> EchoCanceler:
    """Get or create the global echo canceller."""
    global _echo_canceler
    if _echo_canceler is None:
        _echo_canceler = EchoCanceler()
    return _echo_canceler
