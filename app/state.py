"""Per-connection runtime state for the always-on voice assistant."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from .echo_cancel import EchoCanceler
from .stt import RollingAudioBuffer


@dataclass
class ConnectionState:
    """Tracks everything for a single WebSocket client."""

    # Current pipeline stage.
    stage: str = "STANDBY"  # STANDBY, LISTENING, CAPTURING, PROCESSING, SPEAKING

    # Rolling conversation history for the LLM.
    history: list = field(default_factory=list)

    # Command audio buffer (16kHz Int16 PCM) while CAPTURING.
    command_buffer: bytearray = field(default_factory=bytearray)

    # Per-connection rolling wake buffer (never shared across clients).
    wake_buffer: Optional[RollingAudioBuffer] = None

    # Timestamps for silence detection (monotonic milliseconds).
    last_voice_at: Optional[float] = None
    capture_started_at: Optional[float] = None
    has_speech: bool = False

    # Consecutive voiced audio chunks observed while LISTENING. Capture only
    # starts once `voice_debounce_frames` are reached, so a single noise burst
    # can't start a capture that transcribes to "".
    voice_streak: int = 0

    # Consecutive voiced chunks while SPEAKING. Barge-in only fires after this
    # reaches `barge_in_required_frames`, so an echo blip of the assistant's own
    # TTS audio can no longer interrupt a reply mid-sentence.
    speaking_voice_streak: int = 0

    # Guards against concurrent processing of overlapping utterances.
    pipeline_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    busy: bool = False

    # Background task running the current utterance pipeline, so the receive
    # loop stays responsive to pings / stop while a reply is streaming.
    pipeline_task: Optional[asyncio.Task] = None

    # Set while audio is being finalized to keep concurrent finalizers
    # (watchdog + audio_end) from double-transcribing.
    finalizing: bool = False

    # Raised to request the current reply stop speaking (barge-in / interrupt).
    interrupt_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Per-connection TTS output tracking for echo cancellation. This MUST be
    # per-connection: a module-global canceller would treat every client's mic
    # as echo while any single client's reply is playing.
    echo_canceler: Optional[EchoCanceler] = None

    # Monotonic ms when the current reply started speaking. Barge-in is ignored
    # for the first `barge_in_grace_ms` so Nova's own TTS echo leaking into the
    # mic cannot interrupt the reply within the first fraction of a second.
    speaking_started_at: Optional[float] = None
    barge_in_grace_ms: float = 600.0

    # Monotonic ms when the last TTS audio chunk was sent to the client. Barge-in
    # is also ignored while audio is actively flowing (see `barge_in_gap_ms`),
    # so a continuous echo of Nova's own speech can never count as user input.
    last_audio_sent_at: Optional[float] = None
    barge_in_gap_ms: float = 250.0

    def reset_capture(self) -> None:
        self.command_buffer.clear()
        self.last_voice_at = None
        self.capture_started_at = None
        self.has_speech = False
        self.voice_streak = 0
        self.speaking_voice_streak = 0
        self.last_audio_sent_at = None

    def now_ms(self) -> float:
        return time.monotonic() * 1000.0
