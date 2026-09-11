"""Text-to-Speech with engine selection.

* ``edge`` (default)  - edge-tts (Microsoft neural voices, free, cloud, MP3).
                        Fast (~300 ms). Default for snappy replies.
* ``xtts``            - Coqui XTTS-v2, fully local and the most human voice,
                        but CPU synthesis takes ~3-9 s per clause. First run
                        downloads ~1.6 GB of model weights. Falls back to
                        edge-tts per sentence if the model is unavailable.
* ``piper``           - optional local Piper (onnxruntime) voices, WAV.
                        Falls back to edge-tts per sentence if unavailable.

The browser plays whatever container arrives (MP3 or WAV) via the WebAudio
``decodeAudioData`` path, so no client-side changes are needed to switch.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

from .config import Settings

logger = logging.getLogger("voice_assistant")


class TextToSpeech:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.engine = settings.tts_engine.lower().strip() or "edge"
        self.encoding = "mp3" if self.engine == "edge" else "wav"
        self._piper = None  # lazy, optional
        self._xtts = None  # lazy, optional

    @property
    def piper_available(self) -> bool:
        return self._piper is not None

    def _get_piper(self):
        if self._piper is None:
            from .piper_tts import LocalPiper

            self._piper = LocalPiper(self._settings)
        return self._piper

    def _get_xtts(self):
        if self._xtts is None:
            from .xtts import LocalXTTS

            self._xtts = LocalXTTS(self._settings)
        return self._xtts

    async def stream_audio(self, text: str) -> AsyncIterator[bytes]:
        """Yield encoded audio chunks (MP3 or WAV) for ``text``.

        If the configured engine (piper or xtts) is unavailable — missing
        model / package / license flag — fall back to edge-tts so a reply
        (and first voice latency) is never lost.
        """
        if not text.strip():
            return

        if self.engine in ("piper", "xtts"):
            try:
                engine = self._get_piper() if self.engine == "piper" else self._get_xtts()
                engine.load()  # raise fast here if unconfigured
                async for chunk in engine.stream_audio(text):
                    yield chunk
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s TTS unavailable (%s); using edge-tts.", self.engine, exc)
                self.engine = "edge"
                self.encoding = "mp3"

        async for chunk in self._stream_edge_mp3(text):
            yield chunk

    async def _stream_edge_mp3(self, text: str) -> AsyncIterator[bytes]:
        import edge_tts

        communicate = edge_tts.Communicate(
            text=text,
            voice=self._settings.tts_voice,
            rate=self._settings.tts_rate,
            pitch=getattr(self._settings, "tts_pitch", "+0Hz"),
            volume=getattr(self._settings, "tts_volume", "+0%"),
        )
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]