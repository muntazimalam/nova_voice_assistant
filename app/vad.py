"""Voice Activity Detection using Silero VAD for precise endpointing.

Provides more accurate silence detection than simple RMS thresholding,
reducing false triggers and improving user experience.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("voice_assistant")

_SAMPLE_RATE = 16000
_MODEL_SAMPLE_RATE = 16000


class SileroVAD:
    """Silero Voice Activity Detection for accurate endpointing.
    
    Falls back to RMS-based detection if Silero model is unavailable.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = _SAMPLE_RATE,
        min_silence_ms: float = 450.0,
        speech_pad_ms: float = 50.0,
    ) -> None:
        self._threshold = threshold
        self._sample_rate = sample_rate
        self._min_silence_ms = min_silence_ms
        self._speech_pad_ms = speech_pad_ms
        self._model = None
        self._available = False
        self._load()

    def _load(self) -> None:
        """Load the Silero VAD model."""
        try:
            import torch
            self._torch = torch
            
            # Try to load from torch hub
            self._model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            self._available = True
            logger.info("Silero VAD model loaded successfully.")
        except ImportError:
            logger.warning(
                "torch not installed; using RMS-based VAD. "
                "Install with: pip install torch"
            )
        except Exception as exc:
            logger.warning("Silero VAD initialization failed: %s; using RMS-based VAD.", exc)

    @property
    def available(self) -> bool:
        return self._available

    def is_speech(self, pcm_int16: bytes, sample_rate: int = _SAMPLE_RATE) -> bool:
        """Detect if audio chunk contains speech.
        
        Args:
            pcm_int16: Raw 16-bit PCM bytes
            sample_rate: Audio sample rate
            
        Returns:
            True if speech detected
        """
        if not self._available or self._model is None:
            return self._rms_fallback(pcm_int16)

        try:
            samples = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32)
            samples = samples / 32768.0  # Normalize to [-1, 1]
            
            # Resample if needed
            if sample_rate != _MODEL_SAMPLE_RATE:
                samples = self._resample(samples, sample_rate, _MODEL_SAMPLE_RATE)
            
            # Convert to tensor
            tensor = self._torch.from_numpy(samples).unsqueeze(0)
            
            # Get VAD probability
            with self._torch.no_grad():
                prob = self._model(tensor, _MODEL_SAMPLE_RATE).item()
            
            return prob >= self._threshold
        except Exception as exc:
            logger.debug("Silero VAD failed, falling back to RMS: %s", exc)
            return self._rms_fallback(pcm_int16)

    def _rms_fallback(self, pcm_int16: bytes) -> bool:
        """Simple RMS-based speech detection as fallback."""
        if not pcm_int16 or len(pcm_int16) < 2:
            return False
        samples = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(samples))))
        return rms >= 250.0  # Default threshold

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple linear interpolation resampling."""
        if orig_sr == target_sr:
            return audio
        duration = len(audio) / orig_sr
        target_len = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_len)
        return np.interp(indices, np.arange(len(audio)), audio)

    def reset(self) -> None:
        """Reset VAD state for new audio stream."""
        if self._available and self._model is not None:
            try:
                self._model.reset_states()
            except Exception:
                pass


class EndpointDetector:
    """Combines Silero VAD with traditional silence detection for robust endpointing."""
    
    def __init__(
        self,
        vad: Optional[SileroVAD] = None,
        silence_timeout_ms: float = 450.0,
        min_command_ms: float = 250.0,
    ) -> None:
        self._vad = vad or SileroVAD()
        self._silence_timeout_ms = silence_timeout_ms
        self._min_command_ms = min_command_ms
        self._last_speech_ms: float = 0.0
        self._command_start_ms: float = 0.0
        self._has_speech = False

    def update(self, pcm_int16: bytes, now_ms: float) -> Optional[str]:
        """Update endpoint detector with new audio chunk.
        
        Args:
            pcm_int16: Raw 16-bit PCM bytes
            now_ms: Current time in milliseconds
            
        Returns:
            "speech" if speech detected
            "endpoint" if endpoint detected
            "" otherwise
        """
        is_speech = self._vad.is_speech(pcm_int16)
        
        if is_speech:
            self._last_speech_ms = now_ms
            self._has_speech = True
            if self._command_start_ms == 0.0:
                self._command_start_ms = now_ms
            return "speech"
        
        if self._has_speech and self._last_speech_ms > 0:
            silence_duration = now_ms - self._last_speech_ms
            command_duration = now_ms - self._command_start_ms
            
            if (silence_duration >= self._silence_timeout_ms and 
                command_duration >= self._min_command_ms):
                self.reset()
                return "endpoint"
        
        return ""

    def reset(self) -> None:
        """Reset detector state."""
        self._last_speech_ms = 0.0
        self._command_start_ms = 0.0
        self._has_speech = False
        self._vad.reset()


# Global instances
_vad: Optional[SileroVAD] = None
_endpoint_detector: Optional[EndpointDetector] = None


def get_vad() -> SileroVAD:
    """Get or create the global VAD instance."""
    global _vad
    if _vad is None:
        _vad = SileroVAD()
    return _vad


def get_endpoint_detector() -> EndpointDetector:
    """Get or create the global endpoint detector."""
    global _endpoint_detector
    if _endpoint_detector is None:
        _endpoint_detector = EndpointDetector(vad=get_vad())
    return _endpoint_detector
