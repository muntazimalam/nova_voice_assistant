"""Wake-word detection: hybrid openWakeWord + Whisper confirmation.

Strategies
----------
* ``whisper``  - continuously sniff a rolling window of audio with
  faster-whisper and fire when the transcript matches the wake phrase.
  Guaranteed to work with the existing stack; slightly heavier CPU.
* ``openwakeword`` - use an openWakeWord model (pre-trained or custom) as a
  fast first gate, then confirm with Whisper before firing.
* ``auto`` (default) - use openwakeword if a custom model exists in
  ``models/wake/``, otherwise fall back to whisper sniffing (works for any
  ``wake_phrases`` out of the box).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import Settings
from .stt import RollingAudioBuffer, SpeechToText

logger = logging.getLogger("voice_assistant")

_SAMPLE_RATE = 16000


class WakeDetector:
    def __init__(self, settings: Settings, stt: SpeechToText) -> None:
        self._settings = settings
        self._stt = stt
        self._lock = asyncio.Lock()
        self._last_sniff_ms: float = 0.0
        self._ow: Any = None
        self._ow_names: list[str] = []
        self._using_builtin_gate = False
        self._strategy = self._resolve_strategy()

        # Precompile wake-phrase matchers, e.g. r"\bhey nova\b".
        self._patterns = [
            re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
            for phrase in settings.wake_phrases
        ]

    # ------------------------------------------------------------------ setup
    def _resolve_strategy(self) -> str:
        requested = self._settings.wake_strategy.lower()
        if requested == "whisper":
            return "whisper"
        if requested == "openwakeword":
            if not self._load_openwakeword():
                logger.warning("openwakeword unavailable; falling back to whisper.")
                return "whisper"
            return "openwakeword"
        # auto: prefer a custom wake model (it is trained for the actual phrase);
        # with only the built-in gate we would never catch "hello <name>", so use
        # throttled whisper sniffing instead, which matches the configured phrases.
        if self._has_custom_model() and self._load_openwakeword():
            return "openwakeword"
        logger.warning(
            "No custom wake model in %s — using whisper sniffing for the "
            "'%s' wake phrase (this works out of the box but uses more CPU).",
            self._settings.ow_model_dir,
            "', '".join(self._settings.wake_phrases),
        )
        return "whisper"

    def _has_custom_model(self) -> bool:
        return bool(self._custom_model_files())

    def _custom_model_files(self):
        model_dir = Path(self._settings.ow_model_dir)
        if not model_dir.exists():
            return []
        return [
            p for p in model_dir.iterdir() if p.suffix.lower() in (".tflite", ".onnx")
        ]

    def _load_openwakeword(self) -> bool:
        try:
            # pyrefly: ignore [missing-import]
            from openwakeword.model import Model as OWModel
        except ImportError:
            logger.info(
                "openwakeword package not installed; using whisper wake detection."
            )
            return False

        try:
            custom = [str(p) for p in self._custom_model_files()]
            if custom:
                self._ow = OWModel(wakeword_models=custom, inference_framework="onnx")
                self._ow_names = [Path(p).stem for p in custom]
                logger.info("openwakeword loaded custom models: %s", self._ow_names)
                return True
            # No custom model: use a built-in keyword as a cheap first filter.
            # Whisper still gates the real wake phrase, so this only pre-filters.
            self._ow = OWModel(
                wakeword_models=["hey_mycroft"], inference_framework="onnx"
            )
            self._ow_names = ["hey_mycroft"]
            self._using_builtin_gate = True
            logger.warning(
                "No custom wake model found in %s. Using built-in '%s' as a "
                "filter (Whisper still gates the real wake phrase).",
                self._settings.ow_model_dir,
                self._ow_names,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("openwakeword init failed (%s); using whisper.", exc)
            return False

    # ------------------------------------------------------------------- feed
    async def feed(
        self, pcm: bytes, buffer: RollingAudioBuffer, sniff_wake: bool = True
    ) -> str:
        """Feed a streamed audio chunk into the given per-connection buffer.

        Parameters
        ----------
        pcm
            16 kHz Int16 little-endian PCM chunk.
        buffer
            The connection's own rolling wake buffer (never shared across
            clients) — this keeps multi-connection audio from mixing together.
        sniff_wake
            When ``False`` the audio is only appended to the rolling buffer for
            later pre-roll — no Whisper/openwakeword confirmation runs. Use this
            while the user is already explicitly LISTENING so STT stays free for
            the actual command transcription instead of re-sniffing for a wake
            phrase we no longer need.

        Returns
        -------
        ``"wake"``
            Wake phrase was confirmed — caller should start command capture.
        ``"speech"``
            RMS is above threshold but wake phrase not confirmed yet.
        ``""``
            Silence / below RMS threshold, or openwakeword gate not triggered.
        """
        buffer.feed(pcm)
        if not sniff_wake:
            return ""

        if self._rms(pcm) < self._settings.wake_rms_threshold:
            return ""

        if self._strategy == "openwakeword":
            if await self._ow_fired(pcm):
                # OWW gate triggered — run Whisper confirmation.
                if await self._confirm(buffer):
                    return "wake"
                # OWW fired but Whisper disagreed: still report speech.
                return "speech"
            # A built-in gate (e.g. "hey mycroft") can't match the configured
            # phrase — fall back to throttled whisper sniffing so wake still
            # works even in explicit openwakeword mode without a custom model.
            if self._using_builtin_gate:
                now = time.monotonic() * 1000.0
                if now - self._last_sniff_ms >= self._settings.wake_sniff_interval_ms:
                    self._last_sniff_ms = now
                    if await self._confirm(buffer):
                        return "wake"
            return ""

        # whisper sniffing path (throttled)
        now = time.monotonic() * 1000.0
        if now - self._last_sniff_ms < self._settings.wake_sniff_interval_ms:
            return "speech"
        self._last_sniff_ms = now
        if await self._confirm(buffer):
            return "wake"
        return "speech"

    async def _ow_fired(self, pcm: bytes) -> bool:
        try:
            samples = np.frombuffer(pcm, dtype=np.int16)
            if len(samples) < 1280:
                return False
            ow = self._ow
            if ow is None:
                return False
            predictions = ow.predict(samples)
            for name in self._ow_names:
                scores = np.asarray(predictions.get(name, [])).ravel()
                if scores.size == 0:
                    continue
                # Pre-trained models emit a single sigmoid; custom binary
                # models emit [other, wake], where the last class is "wake".
                score = float(scores[-1] if scores.size >= 2 else scores[0])
                if score >= 0.5:
                    return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("openWakeWord feed error: %s", exc)
        return False

    async def _confirm(self, buffer: RollingAudioBuffer) -> bool:
        """Whisper-confirm the wake phrase against the given rolling window."""
        async with self._lock:
            # Transcribe only the most recent slice of the window. The phrase
            # "hey nova / nova" is what just happened; transcribing all 3 s is
            # slower without helping accuracy. 2 s comfortably catches the
            # phrase plus a beat of preceding audio.
            audio = buffer.tail(2000.0) if buffer else b""
            if not audio:
                return False
            loop = asyncio.get_running_loop()
            try:
                # Acquire STT lock with a timeout to prevent waiting forever if
                # a concurrent transcription holds it.
                async with asyncio.timeout(5.0):
                    async with self._stt.lock:
                        # NOT vad_filter: faster-whisper's internal VAD trims
                        # "non-speech" from mostly-silent buffers, deleting the
                        # short wake phrase and breaking confirmation.
                        text = await loop.run_in_executor(
                            None, self._stt.transcribe, audio, None, False
                        )
            except TimeoutError:
                logger.warning("Wake confirm timed out waiting for STT lock.")
                return False
            except Exception as exc:  # noqa: BLE001
                logger.warning("Wake confirm transcription failed: %s", exc)
                return False
            return any(p.search(text) for p in self._patterns)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _rms(pcm: bytes) -> float:
        if not pcm:
            return 0.0
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(samples))))

    def new_buffer(self) -> RollingAudioBuffer:
        """Create a fresh per-connection rolling wake buffer."""
        return RollingAudioBuffer(self._settings.wake_window_millis)

    def is_speech(self, pcm: bytes) -> bool:
        """True if the chunk's loudness suggests active speech (used for barge-in)."""
        return self._rms(pcm) >= self._settings.wake_rms_threshold

    @property
    def strategy(self) -> str:
        return self._strategy
