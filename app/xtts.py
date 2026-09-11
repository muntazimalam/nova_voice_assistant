"""Local neural TTS engine based on Coqui XTTS-v2 — the most human-sounding
option that runs fully on this machine (no API key, no cloud).

XTTS-v2 is a multi-speaker model: we pick a bundled speaker reference voice
(US female "Daisy Studious" by default) to sound like a warm human assistant.
Because it runs on CPU it is slow per clause (~3-9s); this is intentional —
set ``TTS_ENGINE=edge`` (cloud, ~300ms) if you need faster replies.

Setup
-----
    pip install "transformers>=4.57,<4.60" torch torchaudio torchcodec coqui-tts
    # First run downloads the ~1.6 GB model into ~/.cache/coqui|huggingface.
    # The CPML non-commercial license applies: set COQUI_TOS_AGREED=1 after
    # reading https://coqui.ai/cpml.

The emitted audio is a full WAV per spoken segment at 24 kHz mono int16;
the browser's ``decodeAudioData`` auto-detects the WAV container, so no
client changes are required.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave

from .config import Settings

logger = logging.getLogger("voice_assistant")


class LocalXTTS:
    """Wraps Coqui XTTS-v2 for streaming, synthesized-gen WAV output.

    Thread-safety: model instantiation and ``tts()`` are CPU-bound and
    serialized through the event loop's default executor.
    """

    SAMPLE_RATE = 24000  # XTTS-v2 output rate
    MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._speaker = settings.tts_xtts_speaker
        self._language = settings.tts_xtts_language
        self._tts = None  # lazy
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ load
    def load(self):
        """Import TTS and instantiate the model (idempotent)."""
        if self._tts is not None:
            return self._tts

        # The CPML terms prompt blocks first import in a server (non-TTY).
        # Honour an explicit environment flag rather than blocking on input().
        if "COQUI_TOS_AGREED" not in __import__("os").environ:
            logger.info("XTTS not enabled without COQUI_TOS_AGREED=1 (see README).")
            raise RuntimeError(
                "XTTS requires the Coqui CPML terms. Set COQUI_TOS_AGREED=1 "
                "in your environment after reading https://coqui.ai/cpml."
            )

        from TTS.api import TTS

        logger.info("Loading XTTS-v2 (model=%s)...", self.MODEL_NAME)
        self._tts = TTS(self.MODEL_NAME)
        logger.info("XTTS-v2 loaded.")
        return self._tts

    def _synth_wav(self, text: str) -> bytes:
        """Synthesize ``text`` into 24 kHz mono int16 WAV bytes (CPU-bound)."""
        tts = self.load()
        logger.debug("XTTS synthesizing %d chars...", len(text))
        samples = tts.tts(
            text=text,
            speaker=self._speaker,
            language=self._language,
        )
        with io.BytesIO() as buf:
            with wave.open(buf, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.SAMPLE_RATE)
                wav_file.writeframes(self._float32_to_int16(samples))
            return buf.getvalue()

    @staticmethod
    def _float32_to_int16(samples) -> bytes:
        import numpy as np

        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim > 1:  # (n,)  or  (n, 1)
            arr = arr[:, 0] if arr.shape[-1] == 1 else arr.mean(axis=1)
        arr = np.clip(arr, -1.0, 1.0)
        return (arr * 32767.0).astype("<i2").tobytes()

    async def stream_audio(self, text: str):
        """Yield the synthesized WAV in chunks (run on the executor)."""
        if not text.strip():
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            wav = await loop.run_in_executor(None, self._synth_wav, text)
        step = 16 * 1024
        for i in range(0, len(wav), step):
            yield wav[i : i + step]

    @property
    def speaker(self) -> str:
        return self._speaker
