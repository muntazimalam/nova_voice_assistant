"""Speech-to-Text using faster-whisper (fully local, free).

Transcribes 16kHz mono Int16 PCM into text. The model is downloaded
automatically on first use via the Hugging Face hub.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from .config import Settings

logger = logging.getLogger("voice_assistant")

_SAMPLE_RATE = 16000


class SpeechToText:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None  # lazy load (takes a few seconds)
        self._model_name = settings.whisper_model
        self._device = settings.whisper_device
        self._compute_type = settings.whisper_compute_type
        # Serializes transcribe() across coroutines/connections (the
        # underlying ctranslate2 model is not safe for concurrent use).
        self.lock = asyncio.Lock()

    def _load(self):
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel

        logger.info(
            "Loading faster-whisper model '%s' (device=%s, compute=%s)...",
            self._model_name,
            self._device,
            self._compute_type,
        )
        self._model = WhisperModel(
            self._model_name,
            device=self._device,
            compute_type=self._compute_type,
        )
        logger.info("faster-whisper model loaded.")
        return self._model

    def load_now(self) -> None:
        """Pre-load the model so the first transcription has no cold-start stall."""
        self._load()

    @staticmethod
    def _pcm_to_float32(pcm: bytes) -> np.ndarray:
        """16-bit Int16 little-endian PCM -> float32 in [-1, 1]."""
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        return samples / 32768.0

    @staticmethod
    def _normalize(audio: np.ndarray) -> np.ndarray:
        """Peak-normalize quiet audio so real laptop mics transcribe reliably.

        Whisper is trained on near-full-scale audio; typical built-in mic
        levels sit far below that and get rejected by the VAD / produce ""
        transcripts. Boost quiet-but-non-silent clips up so speech is
        intelligible. No-op when the clip is already loud or pure zero."
        """
        if audio.size == 0:
            return audio
        peak = float(np.max(np.abs(audio)))
        if 0.03 < peak < 0.85:
            gain = 0.9 / peak
            return np.clip(audio * gain, -1.0, 1.0)
        return audio

    @staticmethod
    def calculate_rms(pcm: bytes) -> float:
        """Fast RMS calculation for 16-bit Int16 PCM."""
        if not pcm or len(pcm) < 2:
            return 0.0
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(samples))))

    def transcribe(
        self,
        pcm: bytes,
        initial_prompt: str | None = None,
        vad_filter: bool = True,
    ) -> str:
        """Transcribe raw 16kHz Int16 PCM, returning trimmed text with minimal latency.

        ``vad_filter`` enables faster-whisper's internal VAD, which trims silence
        from the audio before transcription. That is desirable for long command
        recordings but NEVER for wake-word confirmation: a short phrase buried in
        a mostly-silent rolling buffer gets deleted as "non-speech" and the wake
        never fires.
        """
        if not pcm or len(pcm) < 2:
            return ""
        audio = self._pcm_to_float32(pcm)
        if self._settings.whisper_normalize:
            audio = self._normalize(audio)
        model = self._load()
        segments, _info = model.transcribe(
            audio,
            language=self._settings.whisper_language or "en",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=vad_filter,
            vad_parameters={"min_silence_duration_ms": 250},
            without_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
        )
        parts: list[str] = []
        for segment in segments:
            if segment.no_speech_prob and segment.no_speech_prob > 0.9:
                continue
            parts.append(segment.text.strip())
        return " ".join(parts).strip()


class RollingAudioBuffer:
    """Keeps a rolling window of 16kHz Int16 PCM (seconds -> bytes)."""

    def __init__(self, window_ms: float) -> None:
        self.window_bytes = int(_SAMPLE_RATE * window_ms / 1000.0) * 2
        self._buf = bytearray()

    def __bytes__(self) -> bytes:
        return bytes(self._buf)

    def feed(self, pcm: bytes) -> None:
        self._buf.extend(pcm)
        if len(self._buf) > self.window_bytes:
            del self._buf[: len(self._buf) - self.window_bytes]

    def tail(self, millis: float) -> bytes:
        """Return the most recent `millis` milliseconds as bytes."""
        nbytes = int(_SAMPLE_RATE * millis / 1000.0) * 2
        if nbytes >= len(self._buf):
            return bytes(self._buf)
        return bytes(self._buf[-nbytes:])

    def __len__(self) -> int:
        return len(self._buf)
