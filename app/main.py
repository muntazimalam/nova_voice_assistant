import asyncio
import contextvars
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .cache import LLMCache, ResponseSummarizer
from .codec import OpusCodec, get_codec
from .config import get_settings
from .echo_cancel import EchoCanceler, get_echo_canceler
from .llm import LLMService
from .logging_config import ConnectionLogger, clear_correlation_id, set_correlation_id, setup_structured_logging
from .rate_limit import RateLimiter, get_rate_limiter
from .state import ConnectionState
from .stt import SpeechToText
from .tts import TextToSpeech
from .vad import EndpointDetector, get_endpoint_detector
from .wake import WakeDetector
from . import __version__

# Configure structured logging
setup_structured_logging(level=logging.INFO)
logger = logging.getLogger("voice_assistant")

# File paths
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Ensure directories exist
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Load settings & instantiate services (lazy models inside)
settings = get_settings()
llm_service = LLMService(settings)
stt_service = SpeechToText(settings)
tts_service = TextToSpeech(settings)
wake_service = WakeDetector(settings, stt_service)

# Advanced pipeline components
codec: Optional[OpusCodec] = None
vad: Optional[EndpointDetector] = None
echo_canceler: Optional[EchoCanceler] = None
llm_cache: Optional[LLMCache] = None
summarizer: Optional[ResponseSummarizer] = None
rate_limiter: Optional[RateLimiter] = None

# Session resumption: map session_id -> ConversationState for reconnect recovery
session_store: Dict[str, Dict[str, Any]] = {}


def _init_advanced_features() -> None:
    """Initialize optional advanced audio/cache features."""
    global codec, vad, echo_canceler, llm_cache, summarizer, rate_limiter

    logger.info(
        "LLM configured: model=%s fallbacks=%s",
        settings.gemini_model,
        settings.gemini_fallback_models,
    )

    # Opus codec
    if settings.use_opus_codec:
        codec = get_codec()
        if codec.available:
            logger.info("Opus codec enabled for audio streaming.")
        else:
            logger.warning("Opus codec unavailable; using raw PCM.")
    else:
        logger.info("Opus codec disabled by config.")

    # Silero VAD
    if settings.use_silero_vad:
        vad = get_endpoint_detector()
        logger.info("Silero VAD initialized: %s", "available" if vad._vad.available else "RMS fallback")
    else:
        logger.info("Silero VAD disabled by config.")

    # Echo cancellation
    if settings.use_echo_cancellation:
        echo_canceler = get_echo_canceler()
        logger.info("Echo cancellation initialized.")
    else:
        logger.info("Echo cancellation disabled by config.")

    # LLM cache
    if settings.llm_cache_enabled:
        llm_cache = LLMCache(
            max_size=settings.llm_cache_max_size,
            default_ttl_seconds=settings.llm_cache_ttl_seconds,
        )
        logger.info("LLM response cache initialized (max=%d, ttl=%.0fs).",
                     settings.llm_cache_max_size, settings.llm_cache_ttl_seconds)
    else:
        logger.info("LLM response cache disabled by config.")

    # Summarizer
    if settings.conversation_summary_enabled:
        summarizer = ResponseSummarizer()
        logger.info("Conversation summarizer initialized.")
    else:
        logger.info("Conversation summarizer disabled.")

    # Rate limiter
    if settings.rate_limit_enabled:
        rate_limiter = RateLimiter(
            max_requests=settings.rate_limit_max_requests,
            window_seconds=settings.rate_limit_window_seconds,
            burst_size=settings.rate_limit_burst_size,
        )
        logger.info("Rate limiter initialized (max=%d/%.0fs).",
                     settings.rate_limit_max_requests, settings.rate_limit_window_seconds)
    else:
        logger.info("Rate limiter disabled by config.")


def _warmup() -> None:
    """Pre-load heavy models on a worker thread so the first turn is instant."""
    logger.info("Warming up STT model...")
    stt_service.load_now()
    logger.info("STT model ready.")
    try:
        llm_service.load()
        logger.info("Gemini client ready.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Gemini client warmup failed (will retry on first use): %s", exc)
    # Pre-warm TTS engines if local
    if settings.tts_engine in ("xtts", "piper"):
        logger.info("Warming up %s TTS engine...", settings.tts_engine)
        try:
            if settings.tts_engine == "xtts":
                tts_service._get_xtts().load()
            else:
                tts_service._get_piper().load()
            logger.info("%s TTS engine ready.", settings.tts_engine)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s TTS warmup failed (will fall back to edge): %s", settings.tts_engine, exc)
    logger.info("Warmup complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    _init_advanced_features()
    await loop.run_in_executor(None, _warmup)

    async def _session_cleaner() -> None:
        """Periodically drop abandoned session entries so memory stays bounded."""
        while True:
            await asyncio.sleep(300.0)
            _prune_sessions(settings.session_ttl_seconds)

    cleaner_task = asyncio.create_task(_session_cleaner())
    yield
    # Cleanup on shutdown
    cleaner_task.cancel()
    if llm_cache:
        await llm_cache.clear()
    session_store.clear()


def _prune_sessions(ttl_seconds: float) -> None:
    """Remove sessions that have been idle (disconnected) past their TTL."""
    now = time.time()
    stale = [
        cid for cid, data in session_store.items()
        if now - data.get("last_seen", data.get("created_at", 0)) > ttl_seconds
    ]
    for cid in stale:
        session_store.pop(cid, None)
    if stale:
        logger.info("Pruned %d stale session(s).", len(stale))


# Initialize FastAPI Application
app = FastAPI(
    title="AI Voice Assistant",
    description="Core backend with WebSockets, Jinja2, real-time audio pipeline.",
    version=__version__,
    lifespan=lifespan,
)

# Configure CORS Middleware — explicit origins (no wildcard + credentials).
_cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Assets & Template Engine
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class ConnectionManager:
    """Manages active client WebSocket connections and message broadcasting."""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []
        self.connection_sessions: Dict[WebSocket, str] = {}  # ws -> session_id

    async def connect(self, websocket: WebSocket, session_id: Optional[str] = None) -> str:
        await websocket.accept()
        self.active_connections.append(websocket)

        # Generate or reuse session ID for resumption
        if session_id and session_id in session_store:
            cid = session_id
            logger.info("Session resumed: %s", cid)
        else:
            cid = session_id or uuid.uuid4().hex[:12]

        self.connection_sessions[websocket] = cid
        set_correlation_id(cid)
        logger.info("New WebSocket client connected (session=%s). Active: %d",
                     cid, len(self.active_connections))
        return cid

    def disconnect(self, websocket: WebSocket) -> None:
        session_id = self.connection_sessions.pop(websocket, None)
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            # Persist state for potential session resumption
            if session_id and session_id in session_store:
                session_store[session_id]["last_seen"] = time.time()
            logger.info("WebSocket client disconnected (session=%s). Active: %d",
                         session_id, len(self.active_connections))

    def get_session_id(self, websocket: WebSocket) -> Optional[str]:
        return self.connection_sessions.get(websocket)


manager = ConnectionManager()


async def send_json(websocket: WebSocket, data: Dict[str, Any]) -> None:
    """Send a JSON frame, surfacing send failures distinctly from inference."""
    if "timestamp" not in data:
        data["timestamp"] = datetime.now(timezone.utc).isoformat()
    await websocket.send_text(json.dumps(data))


# Sentence & clause boundaries used to stream speech as the LLM generates (Alexa/Siri-style).
_FIRST_CLAUSE_RE = re.compile(r"[,;:!?…—\n]+(?=\s|$)")
_SENTENCE_RE = re.compile(r"[.!?…\n]+(?=\s|$)")


class VoicePipeline:
    """Wraps the state + service orchestration for a single connection."""

    def __init__(self, state: ConnectionState) -> None:
        self.state = state

    async def send(self, ws: WebSocket, data: Dict[str, Any]) -> None:
        await send_json(ws, data)

    async def set_stage(self, ws: WebSocket, stage: str, message: str = "", **extra: Any) -> None:
        self.state.stage = stage
        payload: Dict[str, Any] = {
            "type": "state_change",
            "status": stage.lower(),
            "stage": stage,
            "message": message,
        }
        payload.update(extra)
        await self.send(ws, payload)

    async def run_command(self, ws: WebSocket, text: str, stt_latency_ms: float = 0.0) -> None:
        """Start the text->LLM->TTS pipeline as a non-blocking background task.

        Unlike the old blocking flow, this returns immediately so the WebSocket
        receive loop stays responsive to pings, barge-in, and interrupt frames
        while the reply is being generated and spoken.
        """
        if self.state.busy or self.state.pipeline_lock.locked():
            await self.send(
                ws,
                {"type": "system_event", "status": "warning", "message": "Busy — please wait for the current reply to finish."},
            )
            return

        self.state.interrupt_event.clear()
        self.state.busy = True
        self.state.pipeline_task = asyncio.create_task(self._execute(ws, text, stt_latency_ms))

    async def _execute(self, ws: WebSocket, text: str, stt_latency_ms: float = 0.0) -> None:
        try:
            async with self.state.pipeline_lock:
                await self._run_command_locked(ws, text, stt_latency_ms)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline error: %s", exc)
            try:
                await self.set_stage(ws, "IDLE", "An internal error occurred. Please try again.")
            except Exception:  # noqa: BLE001
                pass
        finally:
            self.state.busy = False
            self.state.pipeline_task = None

    async def _run_command_locked(self, ws: WebSocket, text: str, stt_latency_ms: float = 0.0) -> None:
        if not text.strip():
            await self.set_stage(ws, "IDLE", "No speech recognized.")
            return

        t_cmd_start = time.perf_counter()
        await self.set_stage(ws, "PROCESSING", "Synthesizing intent and generating response...")

        # Build message history for the LLM on a working copy so we only commit
        # the user turn together with the assistant turn (keeps history balanced).
        history = list(self.state.history[-settings.history_limit * 2:])
        history.append({"role": "user", "content": text})

        state = self.state

        reply_parts: List[str] = []
        sentences: asyncio.Queue = asyncio.Queue()
        audio_started = False
        llm_error: str = ""
        t_llm_first: Optional[float] = None
        t_tts_first: Optional[float] = None
        cache_hit = False

        async def emit_audio_start() -> None:
            """Signal SPEAKING and send the opening audio frame exactly once."""
            nonlocal audio_started
            if audio_started:
                return
            audio_started = True
            await self.set_stage(ws, "SPEAKING", "Assistant responding...")
            self.state.speaking_started_at = self.state.now_ms()
            self.state.speaking_voice_streak = 0
            self.state.last_audio_sent_at = None
            # Register TTS output for echo cancellation (per-connection instance)
            ec = self.state.echo_canceler
            if ec is not None:
                ec.set_outputting(True)
            await ws.send_text(json.dumps({"type": "audio_start", "encoding": tts_service.encoding}))

        async def llm_producer() -> None:
            """Stream LLM tokens, split into spoken chunks immediately for rapid TTS."""
            nonlocal llm_error, t_llm_first, cache_hit
            buffer_ = ""
            window = ""
            first_chunk_emitted = False

            # Try cache first for ultra-fast known queries
            cached_reply = None
            if llm_cache is not None:
                cached_reply = await llm_cache.get(history, settings.gemini_model)
                if cached_reply:
                    cache_hit = True
                    t_llm_first = time.perf_counter()
                    logger.info("LLM cache hit — skipping API call.")
                    for token in cached_reply:
                        reply_parts.append(token)
                        window += token
                        buffer_ += token

                        while True:
                            regex = _FIRST_CLAUSE_RE if not first_chunk_emitted else _SENTENCE_RE
                            m = regex.search(buffer_)
                            if m:
                                chunk_text = buffer_[:m.end()].strip()
                                buffer_ = buffer_[m.end():]
                                if chunk_text:
                                    first_chunk_emitted = True
                                    await sentences.put(chunk_text)
                                    await self.send(ws, {"type": "transcript_partial", "content": chunk_text})
                                continue

                            words = buffer_.strip().split()
                            if not first_chunk_emitted and len(words) >= 5:
                                split_idx = buffer_.find(words[4]) + len(words[4])
                                chunk_text = buffer_[:split_idx].strip()
                                buffer_ = buffer_[split_idx:]
                                if chunk_text:
                                    first_chunk_emitted = True
                                    await sentences.put(chunk_text)
                                    await self.send(ws, {"type": "transcript_partial", "content": chunk_text})
                                continue
                            break

                        if len(window) >= 40:
                            await self.send(ws, {"type": "transcript_partial", "content": window})
                            window = ""

                    if window:
                        await self.send(ws, {"type": "transcript_partial", "content": window})
                    tail = buffer_.strip()
                    if tail:
                        await sentences.put(tail)
                    return

            try:
                async for token in llm_service.stream_reply(history):
                    if t_llm_first is None:
                        t_llm_first = time.perf_counter()
                    reply_parts.append(token)
                    window += token
                    buffer_ += token

                    # Emit the very first clause immediately so TTS can start speaking
                    # within ~300-500ms instead of waiting for a full 15-word sentence.
                    while True:
                        regex = _FIRST_CLAUSE_RE if not first_chunk_emitted else _SENTENCE_RE
                        m = regex.search(buffer_)
                        if m:
                            chunk_text = buffer_[:m.end()].strip()
                            buffer_ = buffer_[m.end():]
                            if chunk_text:
                                first_chunk_emitted = True
                                await sentences.put(chunk_text)
                                await self.send(ws, {"type": "transcript_partial", "content": chunk_text})
                            continue

                        # If no punctuation yet but first chunk has 5 words, emit early:
                        words = buffer_.strip().split()
                        if not first_chunk_emitted and len(words) >= 5:
                            split_idx = buffer_.find(words[4]) + len(words[4])
                            chunk_text = buffer_[:split_idx].strip()
                            buffer_ = buffer_[split_idx:]
                            if chunk_text:
                                first_chunk_emitted = True
                                await sentences.put(chunk_text)
                                await self.send(ws, {"type": "transcript_partial", "content": chunk_text})
                            continue
                        break

                    if len(window) >= 40:
                        await self.send(ws, {"type": "transcript_partial", "content": window})
                        window = ""

                if window:
                    await self.send(ws, {"type": "transcript_partial", "content": window})
                tail = buffer_.strip()
                if tail:
                    await sentences.put(tail)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error("LLM stream failed: %s", exc)
                llm_error = str(exc)
            finally:
                await sentences.put(None)

        async def tts_consumer() -> None:
            """Synthesize and stream audio segments for each clause/sentence."""
            nonlocal t_tts_first
            seg_idx = 0
            while True:
                sentence = await sentences.get()
                if sentence is None:
                    break
                if state.interrupt_event.is_set():
                    break
                try:
                    await emit_audio_start()
                    await self.send(ws, {
                        "type": "audio_segment_start",
                        "segment_index": seg_idx,
                        "text": sentence,
                    })
                    async for chunk in tts_service.stream_audio(sentence):
                        if state.interrupt_event.is_set():
                            break
                        if t_tts_first is None:
                            t_tts_first = time.perf_counter()
                        await ws.send_bytes(chunk)
                        # Track output for echo cancellation (container-aware:
                        # WAV headers stripped; MP3 tracked by timing only)
                        ec = state.echo_canceler
                        if ec is not None:
                            loop = asyncio.get_running_loop()
                            loop.call_soon(ec.register_container_output, chunk, tts_service.encoding)
                        # Remember when audio last flowed so barge-in only counts
                        # during genuine pauses, never while Nova is speaking.
                        state.last_audio_sent_at = state.now_ms()
                    await self.send(ws, {
                        "type": "audio_segment_end",
                        "segment_index": seg_idx,
                    })
                    seg_idx += 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.error("TTS stream failed: %s", exc)
                    break

        try:
            await asyncio.gather(llm_producer(), tts_consumer())
        except asyncio.CancelledError:
            raise

        # Cache the LLM response for future identical queries
        reply = "".join(reply_parts).strip()
        if llm_cache is not None and reply and not cache_hit and not llm_error:
            loop = asyncio.get_running_loop()
            loop.create_task(llm_cache.set(history, reply, settings.gemini_model))

        llm_ttft = (t_llm_first - t_cmd_start) * 1000 if t_llm_first else 0.0
        tts_first_latency = (t_tts_first - t_cmd_start) * 1000 if t_tts_first else 0.0
        total_roundtrip = stt_latency_ms + tts_first_latency

        metrics = {
            "stt_ms": round(stt_latency_ms, 1),
            "llm_ttft_ms": round(llm_ttft, 1),
            "tts_first_ms": round(tts_first_latency, 1),
            "total_roundtrip_ms": round(total_roundtrip, 1),
            "cache_hit": cache_hit,
        }

        if not reply:
            if audio_started:
                await ws.send_text(json.dumps({"type": "audio_end", "metrics": metrics}))
            if llm_error:
                detail = f" ({llm_error[:160]})" if llm_error else ""
                await self.set_stage(
                    ws, "IDLE",
                    f"I encountered an error while thinking.{detail}",
                )
            else:
                await self.set_stage(ws, "IDLE", "No response generated.")
            return

        if audio_started:
            await ws.send_text(json.dumps({"type": "audio_end", "metrics": metrics}))

        # Send final reply text with telemetry
        await self.send(ws, {
            "type": "chat_reply",
            "content": reply,
            "metrics": metrics,
        })

        # Commit both turns at once (bounded history).
        self.state.history.append({"role": "user", "content": text})
        self.state.history.append({"role": "assistant", "content": reply})
        if len(self.state.history) > settings.history_limit * 2 + 2:
            del self.state.history[: len(self.state.history) - (settings.history_limit * 2 + 2)]

        # Conversation summarization: compress old turns when history gets long
        if (summarizer is not None and
                settings.conversation_summary_enabled and
                len(self.state.history) >= settings.conversation_summary_turns * 2):
            loop = asyncio.get_running_loop()
            loop.create_task(self._summarize_history())

        if state.interrupt_event.is_set():
            await self.set_stage(ws, "IDLE", "Reply interrupted. Ready.")
        else:
            await self.set_stage(ws, "IDLE", "Ready for follow-up.")

    async def _summarize_history(self) -> None:
        """Summarize old conversation turns to maintain context within limits."""
        if not summarizer or not self.state.history:
            return
        turns = self.state.history
        if len(turns) < settings.conversation_summary_turns * 2:
            return

        old_turns = turns[: settings.conversation_summary_turns * 2]
        summary = await summarizer.summarize(old_turns)
        if summary:
            # Replace old turns with a single summary entry
            self.state.history = [{"role": "system", "content": f"Previous conversation summary: {summary}"}] + turns[settings.conversation_summary_turns * 2:]
            logger.info("Conversation summarized into %d chars.", len(summary))


@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request) -> HTMLResponse:
    """Serve the primary dashboard interface via Jinja2."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_name": "Nova AI Voice Assistant",
            "version": f"v{__version__}",
            "phase": "Phase 2 - Real-Time Audio Pipeline",
        },
    )


@app.get("/favicon.ico", response_class=Response)
async def get_favicon() -> Response:
    """Silence the favicon 404 from browser health probes."""
    return Response(
        content=(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
            '<circle cx="8" cy="8" r="7" fill="#4f8cff"/>'
            '<circle cx="8" cy="8" r="3" fill="#fff"/></svg>'
        ),
        media_type="image/svg+xml",
    )


@app.get("/api/health")
async def health_check() -> JSONResponse:
    """Health check endpoint providing server status and connection metrics."""
    return JSONResponse(
        content={
            "status": "healthy",
            "active_ws_connections": len(manager.active_connections),
            "active_sessions": len(session_store),
            "server_time": datetime.now(timezone.utc).isoformat(),
            "version": __version__,
            "wake_strategy": wake_service.strategy,
            "stt_model": settings.whisper_model,
            "llm_model": settings.gemini_model,
            "tts_engine": settings.tts_engine,
            "tts_voice": settings.tts_voice,
            "features": {
                "opus_codec": bool(codec and codec.available) if codec else False,
                "silero_vad": bool(vad and vad._vad.available) if vad else False,
                "echo_cancellation": echo_canceler is not None,
                "llm_cache": llm_cache is not None,
                "llm_cache_stats": llm_cache.get_stats() if llm_cache else None,
                "conversation_summarizer": summarizer is not None,
                "rate_limiting": rate_limiter is not None,
            },
        }
    )


@app.get("/api/metrics")
async def metrics() -> JSONResponse:
    """Prometheus-style metrics endpoint."""
    return JSONResponse(
        content={
            "active_connections": len(manager.active_connections),
            "active_sessions": len(session_store),
            "llm_cache": llm_cache.get_stats() if llm_cache else None,
        }
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Primary WebSocket endpoint supporting both text frames and binary audio."""
    session_id = websocket.query_params.get("session")
    cid = await manager.connect(websocket, session_id)
    set_correlation_id(cid)

    # Session resumption: restore state if client reconnects (unless the old
    # session went stale); otherwise start fresh with a working copy.
    restored = session_store.get(cid)
    if restored is not None:
        last_active = restored.get("last_seen", restored.get("created_at", 0))
        if time.time() - last_active > settings.session_ttl_seconds:
            session_store.pop(cid, None)
            restored = None
    if restored is not None and restored.get("state") is not None:
        state = restored["state"]
    else:
        state = ConnectionState()
    session_store[cid] = {"state": state, "created_at": time.time(), "last_seen": time.time()}

    state.wake_buffer = wake_service.new_buffer()
    # Reset any stale state left over from a crashed / disconnected session.
    state.busy = False
    state.pipeline_task = None
    state.interrupt_event.clear()
    # Per-connection echo canceller: the module-global instance only tracks
    # whether the feature is enabled; actual cancellation is per-client.
    if echo_canceler is not None:
        state.echo_canceler = EchoCanceler()
    else:
        state.echo_canceler = None
    # Barge-in tuning is per-connection too, so a client can override it via
    # config without affecting concurrent sessions.
    state.barge_in_grace_ms = settings.barge_in_grace_ms
    state.barge_in_gap_ms = settings.barge_in_gap_ms
    pipeline = VoicePipeline(state)

    # Dispatch welcome & initial state
    await send_json(
        websocket,
        {
            "type": "system_event",
            "status": "idle",
            "message": f"Connected to Voice Assistant Engine. Session: {cid}",
            "details": f"Wake strategy: {wake_service.strategy}. Ready for voice or text.",
        },
    )

    # Background watchdog that auto-finalizes capture on silence or max length.
    watchdog = asyncio.create_task(capture_watchdog(websocket, pipeline, state))

    # Per-connection TTS output tracking for echo cancellation
    conn_logger = ConnectionLogger(cid)

    try:
        while True:
            message = await websocket.receive()
            kind = message.get("type")

            # Peer closed the socket. Bail immediately instead of calling
            # receive() again (which raises "Cannot call receive once a
            # disconnect message has been received") and polluting the log.
            if kind == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))

            # Rate limiting for text/control messages
            if kind == "websocket.receive" and message.get("text") is not None:
                if rate_limiter and not rate_limiter.allow(cid):
                    await send_json(
                        websocket,
                        {
                            "type": "system_event",
                            "status": "warning",
                            "message": "Rate limit exceeded. Please slow down.",
                        },
                    )
                    continue

            # Binary audio chunk.
            if kind == "websocket.receive" and message.get("bytes") is not None:
                pcm = message["bytes"]
                if is_binary_frame(pcm):
                    # Echo cancellation: filter out assistant's own TTS audio
                    ec = state.echo_canceler
                    if ec is not None:
                        pcm = ec.cancel_echo(pcm)
                    await handle_audio_frame(websocket, pipeline, state, pcm)
                continue

            raw_text = message.get("text")
            if raw_text is None:
                # Includes websocket.disconnect frames (handled by outer except).
                continue

            try:
                message_data = json.loads(raw_text)
            except json.JSONDecodeError:
                message_data = {"type": "text", "content": raw_text}

            msg_type = message_data.get("type", "unknown")
            conn_logger.info("Received WebSocket frame: %s", msg_type)

            if msg_type == "ping":
                client_sent_at = message_data.get("client_timestamp")
                await send_json(
                    websocket,
                    {
                        "type": "pong",
                        "client_timestamp": client_sent_at,
                        "server_timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

            elif msg_type == "start_listening":
                state.stage = "LISTENING"
                state.interrupt_event.clear()
                state.command_buffer.clear()
                await send_json(
                    websocket,
                    {
                        "type": "state_change",
                        "status": "listening",
                        "stage": "LISTENING",
                        "message": "Assistant is actively listening for voice input...",
                    },
                )

            elif msg_type == "stop_listening":
                if state.pipeline_task and not state.pipeline_task.done():
                    # Let the current reply finish; just stop further capture.
                    await send_json(
                        websocket,
                        {
                            "type": "system_event",
                            "status": "info",
                            "message": "Listening stopped. Current reply will continue.",
                        },
                    )
                state.stage = "IDLE"
                state.reset_capture()
                await send_json(
                    websocket,
                    {
                        "type": "state_change",
                        "status": "idle",
                        "stage": "IDLE",
                        "message": "Assistant standby. Listening stopped.",
                    },
                )

            elif msg_type == "interrupt":
                # Barge-in: stop the current reply immediately (like Alexa/Siri).
                state.interrupt_event.set()
                ec = state.echo_canceler
                if ec is not None:
                    ec.set_outputting(False)
                await send_json(
                    websocket,
                    {
                        "type": "system_event",
                        "status": "info",
                        "message": "Reply interrupted.",
                    },
                )

            elif msg_type == "simulate_cycle":
                utterance = message_data.get("utterance", "Hello assistant, what is the weather today?")
                await pipeline.set_stage(
                    websocket, "LISTENING", f"Captured utterance: \"{utterance}\""
                )
                await asyncio.sleep(0.8)
                await pipeline.run_command(websocket, utterance)

            elif msg_type == "chat_message":
                content = message_data.get("content", "")
                if content.strip():
                    await pipeline.run_command(websocket, content)
                else:
                    await send_json(
                        websocket,
                        {
                            "type": "system_event",
                            "status": "warning",
                            "message": "Empty chat message ignored.",
                        },
                    )

            elif msg_type == "audio_start":
                state.stage = "LISTENING"
                state.capture_started_at = state.now_ms()
                await pipeline.set_stage(
                    websocket, "LISTENING", "Streaming audio capture began."
                )

            elif msg_type == "audio_end":
                # Cancel any pending silence auto-finalize then finalize now.
                await finalize_audio(websocket, pipeline, state)

            else:
                await send_json(
                    websocket,
                    {
                        "type": "system_event",
                        "status": "warning",
                        "message": f"Unrecognized frame type '{msg_type}'.",
                    },
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client cleanly disconnected from /ws (session=%s)", cid)
    except Exception as exc:
        logger.error("Error in websocket session: %s", exc)
        manager.disconnect(websocket)
    finally:
        watchdog.cancel()
        # Cancel any in-flight pipeline task so we don't leak it on disconnect.
        await cancel_pipeline(state)
        ec = state.echo_canceler
        if ec is not None:
            ec.set_outputting(False)
            ec.reset()
        clear_correlation_id()


def is_binary_frame(pcm: bytes) -> bool:
    """Validate an incoming binary frame before it enters the pipeline."""
    if len(pcm) < 2:
        # Ignore degenerate frames.
        return False
    if len(pcm) % 2 != 0:
        logger.warning("Dropping non-Int16-aligned binary frame (%d bytes).", len(pcm))
        return False
    if len(pcm) > settings.max_audio_chunk_bytes:
        logger.warning("Dropping oversized binary frame (%d bytes).", len(pcm))
        return False
    return True


async def handle_audio_frame(
    websocket: WebSocket,
    pipeline: VoicePipeline,
    state: ConnectionState,
    pcm: bytes,
) -> None:
    """Route a validated binary audio chunk through the wake/capture pipeline with instant VAD."""
    now = state.now_ms()
    ec = state.echo_canceler

    # Echo-aware speech detection: use Silero VAD when available, else RMS.
    # Skip speech detection entirely if this is likely assistant echo.
    is_echo = False
    if ec is not None:
        is_echo = ec.is_echo(pcm)
        if is_echo:
            logger.debug("Ignoring echo-fed mic chunk during capture.")
            return

    if vad is not None and vad._vad.available:
        is_voice = vad._vad.is_speech(pcm)
    else:
        is_voice = wake_service.is_speech(pcm)

    # Active speech while the assistant is talking -> barge in and stop it.
    # A short grace period after the reply starts lets Nova finish the very
    # first syllables even if its own TTS audio echoes into the mic. A debounce
    # streak is ALSO required so a lone echo/click blip doesn't cut us off.
    if state.stage == "SPEAKING":
        if not is_echo and is_voice:
            if state.barge_in_grace_ms > 0 and state.speaking_started_at is not None:
                since_speaking = now - state.speaking_started_at
                if since_speaking < state.barge_in_grace_ms:
                    state.speaking_voice_streak = 0
                    return
            # Only barge in during a genuine pause. While TTS audio is actively
            # flowing to the client, Nova's own voice is feeding the mic and a
            # continuous echo would otherwise count as a long user utterance and
            # cut the reply off mid-sentence. The gap lets real inter-clause
            # interruptions still work while making self-truncation impossible.
            if state.barge_in_gap_ms > 0 and state.last_audio_sent_at is not None:
                since_audio = now - state.last_audio_sent_at
                if since_audio < state.barge_in_gap_ms:
                    state.speaking_voice_streak = 0
                    return
            state.speaking_voice_streak += 1
            if state.speaking_voice_streak >= settings.barge_in_required_frames:
                state.speaking_voice_streak = 0
                if not state.interrupt_event.is_set():
                    logger.info("Barge-in: user speech detected during assistant reply.")
                    state.interrupt_event.set()
                    if ec is not None:
                        ec.set_outputting(False)
        elif not is_voice:
            # Sustained silence resets the barge-in counter so a stray blip
            # long after the fact can't accumulate towards an interrupt.
            state.speaking_voice_streak = 0

    if state.stage == "LISTENING":
        # Keep the rolling wake buffer fresh whether or not this chunk is voiced,
        # so we can seed the command buffer with pre-roll audio when capture starts.
        # We are already explicitly listening — do NOT run wake confirmation, so
        # faster-whisper stays free for the real command transcription.
        await wake_service.feed(pcm, state.wake_buffer, sniff_wake=False)

        if not is_voice:
            state.voice_streak = 0
            return

        # Debounce: require several consecutive voiced chunks (~20 ms each). A
        # lone noise burst (door slam, bump) must not start a capture that
        # transcribes to "".
        state.voice_streak += 1
        if state.voice_streak < settings.voice_debounce_frames:
            return

        # Sustained speech confirmed — start capture with a pre-roll so the
        # onset of the word isn't clipped.
        state.voice_streak = 0
        state.stage = "CAPTURING"
        state.capture_started_at = now
        state.last_voice_at = now
        state.has_speech = True
        state.command_buffer.clear()
        state.command_buffer.extend(state.wake_buffer.tail(settings.wake_tail_millis))
        await pipeline.set_stage(websocket, "CAPTURING", "Listening...")
        return

    if state.stage == "CAPTURING":
        state.command_buffer.extend(pcm)

        if is_voice:
            state.last_voice_at = now
            state.has_speech = True
        else:
            # Silence chunk while capturing!
            # Instant endpointing: check if user finished speaking and silence threshold met
            if state.has_speech and state.last_voice_at is not None:
                silence_dur = now - state.last_voice_at
                cmd_dur = now - (state.capture_started_at or now)
                if silence_dur >= settings.silence_timeout_ms and cmd_dur >= settings.min_command_ms:
                    logger.info("Speech endpoint detected (silence: %.0f ms). Finalizing command.", silence_dur)
                    await finalize_audio(websocket, pipeline, state)
                    return

        # Hard cap: never let the command buffer exceed max command length.
        max_bytes = int(settings.sample_rate * (settings.max_command_ms / 1000.0)) * 2
        if len(state.command_buffer) > max_bytes:
            logger.info("Max command length reached; finalizing capture.")
            await finalize_audio(websocket, pipeline, state)
            return

        return

    # STANDBY / IDLE: run wake-word detection on this connection's own buffer.
    result = await wake_service.feed(pcm, state.wake_buffer)
    if result == "wake":
        await pipeline.set_stage(
            websocket, "LISTENING", "Wake word detected. Listening for command..."
        )
        state.command_buffer.clear()
        state.command_buffer.extend(state.wake_buffer.tail(settings.wake_tail_millis))
        state.stage = "CAPTURING"
        state.capture_started_at = now
        state.last_voice_at = now
        state.has_speech = True


async def capture_watchdog(
    websocket: WebSocket,
    pipeline: VoicePipeline,
    state: ConnectionState,
) -> None:
    """Auto-finalize capture after silence or when the max command length hits."""
    try:
        while True:
            await asyncio.sleep(0.1)
            if state.stage not in ("LISTENING", "CAPTURING"):
                continue
            now = state.now_ms()

            # Max command length exceeded.
            if state.capture_started_at is not None and (
                now - state.capture_started_at > settings.max_command_ms
            ):
                logger.info("Max command length reached; finalizing capture.")
                await finalize_audio(websocket, pipeline, state)
                continue

            # Silence timeout since the last voice chunk.
            if state.has_speech and state.last_voice_at is not None and (
                now - state.last_voice_at > settings.silence_timeout_ms
            ):
                cmd_dur = now - (state.capture_started_at or now)
                if cmd_dur >= settings.min_command_ms:
                    logger.info("Watchdog silence timeout reached (%.0f ms); finalizing capture.", now - state.last_voice_at)
                    await finalize_audio(websocket, pipeline, state)
    except asyncio.CancelledError:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("Capture watchdog exiting: %s", exc)


async def finalize_audio(websocket: WebSocket, pipeline: VoicePipeline, state: ConnectionState) -> None:
    """End command capture, transcribe, and run the LLM+TTS pipeline."""
    if state.finalizing:
        return
    if state.pipeline_task and not state.pipeline_task.done():
        # The reply pipeline is still busy. Clear the capture so the overflow
        # / endpoint checks don't spin on a full buffer every 20 ms while we wait.
        state.reset_capture()
        logger.info("Skipping finalize: a pipeline is already running. Dropped captured audio.")
        return
    if state.stage not in ("LISTENING", "CAPTURING"):
        return

    state.finalizing = True
    try:
        captured = bytes(state.command_buffer)
        elapsed = state.now_ms() - (state.capture_started_at or state.now_ms())
        state.reset_capture()
        state.stage = "PROCESSING"

        min_bytes = int(settings.sample_rate * (settings.min_command_ms / 1000.0)) * 2
        if len(captured) < min_bytes:
            state.stage = "IDLE"
            await pipeline.set_stage(websocket, "IDLE", "Too short to transcribe. Try again.")
            return

        await pipeline.set_stage(
            websocket, "PROCESSING",
            f"Transcribing {elapsed:.0f} ms of audio...",
        )

        loop = asyncio.get_running_loop()
        text = ""
        stt_duration_ms = 0.0
        t_stt_0 = time.perf_counter()
        try:
            async with asyncio.timeout(settings.stt_timeout_seconds):
                async with stt_service.lock:
                    text = await loop.run_in_executor(
                        None, stt_service.transcribe, captured, settings.assistant_name
                    )
            stt_duration_ms = (time.perf_counter() - t_stt_0) * 1000
        except TimeoutError:
            logger.warning("STT timed out.")
            state.stage = "IDLE"
            await pipeline.set_stage(websocket, "IDLE", "Speech recognition timed out.")
            return
        except Exception as exc:  # noqa: BLE001
            logger.error("STT failed: %s", exc)
            state.stage = "IDLE"
            await pipeline.set_stage(websocket, "IDLE", "Speech recognition failed.")
            return

        logger.info("Transcribed in %.1f ms: %r", stt_duration_ms, text)
        await pipeline.send(websocket, {
            "type": "transcript_final",
            "content": text,
            "stt_ms": round(stt_duration_ms, 1),
        })
        await pipeline.run_command(websocket, text, stt_latency_ms=stt_duration_ms)
    finally:
        state.finalizing = False


async def cancel_pipeline(state: ConnectionState) -> None:
    """Cancel any in-flight pipeline task on disconnect."""
    task = state.pipeline_task
    if task and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
