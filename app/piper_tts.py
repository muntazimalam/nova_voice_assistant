"""Optional local neural TTS engine based on Piper (onnxruntime).

Unlike edge-tts (a cloud round-trip per clause), Piper synthesizes fully on
this machine, which removes the network hop for near-instant first-voice
latency. It is optional: if ``piper-tts`` is not installed or no model is
present, :class:`TextToSpeech` falls back to edge-tts automatically.

Setup
-----
    pip install piper-tts
    # Download a voice (plus its .onnx.json config), e.g. from
    # https://huggingface.co/rhasspy/piper-voices
    # into models/piper/ and set TTS_ENGINE=piper in .env.

The emitted audio is a complete WAV per spoken segment (decodeAudioData /
HTMLAudioElement auto-detect the container, so the existing browser playback
code needs no changes).
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave
from pathlib import Path

from .config import Settings

logger = logging.getLogger("voice_assistant")


class LocalPiper:
    """Wraps a single Piper voice for streaming, thread-safe synthesis."""

    def __init__(self, settings: Settings) -> None:
        self._model_path = Path(settings.piper_model_path)
        self._config_path = Path(settings.piper_config_path)
        if not self._config_path.exists():
            # Fall back to the standard sibling config name (<model>.onnx.json).
            self._config_path = self._model_path.with_suffix(
                self._model_path.suffix + ".json"
            )
        self._voice = None
        self._sample_rate = 22050

    # ------------------------------------------------------------------ load
    def load(self):
        """Import piper + load the ONNX voice (idempotent)."""
        if self._voice is not None:
            return self._voice
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"Piper model not found: {self._model_path} — download a voice "
                f"into models/piper/ or set TTS_ENGINE=edge."
            )
        try:
            from piper import PiperVoice
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "piper-tts is not installed. Run: pip install piper-tts "
                "(or keep TTS_ENGINE=edge for cloud TTS)."
            ) from exc

        logger.info("Loading Piper voice: %s", self._model_path)
        self._voice = PiperVoice.load(
            str(self._model_path), config_path=str(self._config_path)
        )
        self._sample_rate = getattr(
            self._voice.config, "sample_rate", self._sample_rate
        )
        logger.info("Piper voice ready (sample_rate=%s).", self._sample_rate)
        return self._voice

    def _synth_wav(self, text: str) -> bytes:
        """Synthesize ``text`` into mono int16 WAV bytes (CPU-bound)."""
        voice = self.load()
        logger.debug("Piper synthesizing %d chars...", len(text))
        with io.BytesIO() as buf:
            with wave.open(buf, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self._sample_rate)
                wav_file.writeframes(self._synthesize_stream_raw(voice, text))
            return buf.getvalue()

    @staticmethod
    def _synthesize_stream_raw(voice, text: str) -> bytes:
        """Run the (possibly several) raw-PCM generators from piper."""
        try:
            data = voice.synthesize_stream_raw(text)
        except TypeError:
            # Older piper < 1.0 also returned (audio, sample_rate) directly.
            audio, _sr = voice.synthesize(text)
            return audio.tobytes() if hasattr(audio, "tobytes") else audio
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        if hasattr(data, "__iter__"):
            chunks: list[bytes] = []
            for chunk in data:
                if isinstance(chunk, (bytes, bytearray)):
                    chunks.append(bytes(chunk))
            return b"".join(chunks)
        raise RuntimeError("Unexpected piper synthesize_stream_raw output")

    async def stream_audio(self, text: str):
        """Yield the synthesized WAV in chunks (run on the executor)."""
        if not text.strip():
            return
        loop = asyncio.get_running_loop()
        wav = await loop.run_in_executor(None, self._synth_wav, text)
        # Deliberately chunked rather than a single frame so the client's
        # streaming path reacts while the rest of the pipeline continues.
        step = 16 * 1024
        for i in range(0, len(wav), step):
            yield wav[i : i + step]
