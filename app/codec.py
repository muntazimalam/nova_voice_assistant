"""Opus audio codec for efficient WebSocket streaming.

Provides encoding/decoding of PCM audio to/from Opus frames,
reducing bandwidth by ~50% while improving noise resilience.
"""
from __future__ import annotations

import asyncio
import io
import logging
import struct
from typing import Optional

import numpy as np

logger = logging.getLogger("voice_assistant")

_SAMPLE_RATE = 16000
_CHANNELS = 1
_FRAME_DURATION_MS = 20  # 20ms frames (320 samples at 16kHz)
_FRAME_SIZE = int(_SAMPLE_RATE * _FRAME_DURATION_MS / 1000)  # 320 samples


class OpusCodec:
    """Opus encoder/decoder for real-time voice streaming.
    
    Falls back to raw PCM if opuslib is not available.
    """

    def __init__(self) -> None:
        self._encoder = None
        self._decoder = None
        self._available = False
        self._load()

    def _load(self) -> None:
        try:
            import opuslib
            self._encoder = opuslib.Encoder(_SAMPLE_RATE, _CHANNELS, opuslib.APPLICATION_VOIP)
            self._decoder = opuslib.Decoder(_SAMPLE_RATE, _CHANNELS)
            self._available = True
            logger.info("Opus codec loaded successfully.")
        except ImportError:
            logger.warning(
                "opuslib not installed; using raw PCM. "
                "Install with: pip install opuslib"
            )
        except Exception as exc:
            logger.warning("Opus codec initialization failed: %s", exc)

    @property
    def available(self) -> bool:
        return self._available

    def encode_pcm_to_opus(self, pcm_int16: bytes) -> bytes:
        """Encode 16-bit PCM to Opus frame.
        
        Args:
            pcm_int16: Raw 16-bit PCM bytes (little-endian)
            
        Returns:
            Opus-encoded frame bytes, or original PCM if codec unavailable
        """
        if not self._available or self._encoder is None:
            return pcm_int16

        try:
            samples = np.frombuffer(pcm_int16, dtype=np.int16)
            # Ensure frame is correct size
            if len(samples) != _FRAME_SIZE:
                # Pad or truncate to frame size
                if len(samples) < _FRAME_SIZE:
                    samples = np.pad(samples, (0, _FRAME_SIZE - len(samples)))
                else:
                    samples = samples[:_FRAME_SIZE]
            
            opus_data = self._encoder.encode(samples.tobytes(), _FRAME_SIZE)
            return opus_data
        except Exception as exc:
            logger.debug("Opus encode failed, falling back to PCM: %s", exc)
            return pcm_int16

    def decode_opus_to_pcm(self, opus_data: bytes) -> bytes:
        """Decode Opus frame to 16-bit PCM.
        
        Args:
            opus_data: Opus-encoded frame bytes
            
        Returns:
            Decoded 16-bit PCM bytes (little-endian)
        """
        if not self._available or self._decoder is None:
            return opus_data

        try:
            pcm_bytes = self._decoder.decode(opus_data, _FRAME_SIZE)
            return pcm_bytes
        except Exception as exc:
            logger.debug("Opus decode failed: %s", exc)
            return opus_data

    def get_frame_size(self) -> int:
        """Return the expected Opus frame size in bytes."""
        return _FRAME_SIZE * 2  # 16-bit samples = 2 bytes each


class AudioFrameBuffer:
    """Buffer for assembling Opus frames from streaming audio."""
    
    def __init__(self, frame_size: int = _FRAME_SIZE * 2) -> None:
        self._frame_size = frame_size
        self._buffer = bytearray()
        self._frames: list[bytes] = []

    def feed(self, data: bytes) -> list[bytes]:
        """Feed raw bytes and return complete frames.
        
        Args:
            data: Raw audio bytes
            
        Returns:
            List of complete frames ready for encoding
        """
        self._buffer.extend(data)
        frames = []
        
        while len(self._buffer) >= self._frame_size:
            frame = bytes(self._buffer[:self._frame_size])
            del self._buffer[:self._frame_size]
            frames.append(frame)
        
        return frames

    def flush(self) -> Optional[bytes]:
        """Return any remaining partial frame."""
        if self._buffer:
            remaining = bytes(self._buffer)
            self._buffer.clear()
            return remaining
        return None

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._frames.clear()


# Global codec instance
_codec: Optional[OpusCodec] = None


def get_codec() -> OpusCodec:
    """Get or create the global Opus codec instance."""
    global _codec
    if _codec is None:
        _codec = OpusCodec()
    return _codec
