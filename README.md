# Nova — Real-Time AI Voice Assistant

A low-latency, Alexa/Siri-style **voice agent** built on **FastAPI + WebSockets**. The
browser captures the microphone, streams 16 kHz PCM to the backend, and receives
clause-streamed synthesized speech back over the same socket — so the assistant
starts talking while the LLM is still finishing its reply.

```
version: 2.1.0  (see app/__init__.py — single source of truth)
```

---

## 1. What it does

- **Push-to-talk and hands-free modes** — tap the orb/button to talk, or say the wake
  phrase (`hey nova` / `nova`) out loud.
- **Instant endpointing** — chunk-level VAD (Silero, with RMS fallback) finalizes the
  utterance ~450 ms after you stop speaking; a 100 ms watchdog covers dropped frames.
- **Streaming speech** — the reply is split into clauses as Gemini writes them; each
  clause is synthesized and played gaplessly while the next one is still being generated.
- **Barge-in / interrupt** — talking over the assistant (or tapping while it speaks)
  stops it immediately, and it returns to standby for your next command. Echo
  cancellation prevents the assistant's own voice from re-triggering the mic.
- **Hands-free follow-up** — after answering, it keeps listening for the next turn.
- **Live telemetry** — STT / LLM first-token / first-spoken-clause / total round-trip
  times are streamed to the dashboard HUD.
- **LLM response caching** — identical queries answered from an in-memory LRU cache,
  eliminating repeat API latency.
- **Opus codec** — bandwidth-efficient audio streaming (falls back to raw PCM).
- **Connection resumption** — reconnecting mid-conversation restores your history.
- **Rate limiting** — per-client token bucket prevents abuse.
- **Structured logging** — correlation IDs trace each request end-to-end.

## 2. Architecture

```
Browser (app/static/js/app.js)              Backend (app/)
───────────────────────────────             ─────────────────────────────
AudioWorklet downsample → 16kHz PCM ──►     handle_audio_frame
WebSocket (binary PCM + JSON frames)        ├─ echo cancellation + VAD endpointing
WebAudio gapless playback ◄───── MP3/WAV ◄───►  ├─ wake word (whisper / OWW)
Siri/Alexa chimes, HUD, conversation        ├─ faster-whisper tiny (local)
                                            ├─ Gemini streaming + fallback + cache
                                            └─ XTTS-v2 local neural (or edge-tts)
```

| Layer      | Technology                                          |
|------------|-----------------------------------------------------|
| Transport  | FastAPI / ASGI + WebSockets (binary + text frames)  |
| Codec      | Opus (optional, auto-fallback to PCM)              |
| STT        | faster-whisper `tiny`, int8, CPU, VAD + Silero VAD  |
| LLM        | Google Gemini streaming + fallback + LRU cache     |
| TTS        | XTTS-v2 local neural (most human, WAV) or edge-tts MP3 / Piper |
| Wake word  | openWakeWord gate + Whisper confirm, or Whisper sniffing |
| ECHO/AEC   | In-process spectral echo cancellation + grace period |

### Speech pipeline stages

```
STANDBY → LISTENING → CAPTURING → PROCESSING → SPEAKING → IDLE
  (idle)   (mic on)   (recording)  (STT+LLM)   (talking)  (standby)
```

## 3. Quickstart

```powershell
# 1) Install dependencies (Python 3.11+)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) Set your Gemini API key
#    copy .env.example -> .env  (or set GOOGLE_API_KEY)

# 3) Run
python run.py                 # production
python run.py --dev           # hot reload (loads models twice on save)
```

Open **http://127.0.0.1:8000** — the dashboard auto-connects to `ws://127.0.0.1:8000/ws`.

> Microphone capture requires a secure context. `localhost`/`127.0.0.1` are treated as
> secure; for any other host, serve over HTTPS.

### Testing

```powershell
pip install -r requirements-dev.txt
pytest -q
```

## 4. Configuration (.env / app/config.py)

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Google Gemini API key (**required**) |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Primary LLM |
| `GEMINI_FALLBACK_MODELS` | `gemini-2.0-flash-lite, gemini-1.5-flash` | Tried if the primary fails pre-token |
| `WHISPER_MODEL` | `tiny` | Local STT size |
| `TTS_ENGINE` | `xtts` | `xtts` (local neural) / `edge` (cloud MP3) / `piper` (local WAV) |
| `TTS_VOICE` | `en-US-JennyNeural` | edge-tts voice |
| `TTS_RATE` | `+4%` | edge-tts speaking rate |
| `TTS_PITCH` | `+0Hz` | edge-tts pitch adjustment |
| `TTS_VOLUME` | `+0%` | edge-tts volume adjustment |
| `TTS_XTTS_SPEAKER` | `Daisy Studious` | XTTS-v2 bundled reference voice |
| `TTS_XTTS_LANGUAGE` | `en` | XTTS-v2 output language |
| `PIPER_MODEL_PATH` | `models/piper/en_US-lessac-medium.onnx` | Local Piper model |
| `WAKE_STRATEGY` | `auto` | `auto` / `whisper` / `openwakeword` |
| `WAKE_PHRASES` | `["hey nova", "nova"]` | Wake phrases (whisper mode) |
| `SILENCE_TIMEOUT_MS` | `450` | Silence that ends an utterance |
| `MAX_COMMAND_MS` | `12000` | Hard cap on a single command |
| `USE_OPUS_CODEC` | `true` | Opus codec for streaming (falls back to PCM) |
| `USE_SILERO_VAD` | `true` | Neural VAD endpointing (falls back to RMS) |
| `USE_ECHO_CANCELLATION` | `true` | Acoustic echo cancellation for barge-in |
| `LLM_CACHE_ENABLED` | `true` | Cache repeat LLM queries |
| `LLM_CACHE_MAX_SIZE` | `1000` | Max cached responses |
| `LLM_CACHE_TTL_SECONDS` | `3600` | Cache expiry (seconds) |
| `CONVERSATION_SUMMARY_ENABLED` | `false` | Summarize long conversations |
| `RATE_LIMIT_ENABLED` | `true` | Per-client request throttling |
| `RATE_LIMIT_MAX_REQUESTS` | `120` | Requests per window |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate-limit window |

### Local XTTS-v2 TTS (default — the most human voice, runs fully offline)

XTTS-v2 (Coqui) is a neural multi-speaker model that reproduces a warm human
voice without any cloud service. Because it runs on CPU here, each clause takes
**~3-9 s to synthesize** (accepted trade-off for the voice quality). The ~1.6 GB
model downloads automatically on first synthesis; set `COQUI_TOS_AGREED=1` first
to accept the non-commercial CPML (https://coqui.ai/cpml) on the model's first
import.

```powershell
# Dependencies are in requirements.txt (torch CPU, coqui-tts, transformers pinned).
COQUI_TOS_AGREED=1   # set once in your shell/`.env` after reading the license
TTS_ENGINE=xtts      # already the default
# Optional bundled reference voices: Claribel Dervla, Daisy Studious, Gracie Wise,
# Tammie Ema, Alison Dietlinde ... (TTS_XTTS_SPEAKER)
```

If the XTTS license flag or model are missing, TTS falls back to edge-tts per
sentence, so replies are never lost. Switch to `TTS_ENGINE=edge` for ~300 ms
cloud voices instead.

### Local Piper TTS (optional)

```powershell
pip install piper-tts
# Download a voice into models/piper/, e.g. https://huggingface.co/rhasspy/piper-voices
#   en_US-lessac-medium.onnx  +  en_US-lessac-medium.onnx.json
# Then set in .env:
TTS_ENGINE=piper
```

If Piper is unavailable (no package/model) it automatically falls back to edge-tts
per sentence, so responses are never lost.

## 5. WebSocket Frame Protocol

### Client → Server (`ws://host/ws`)

| `type` | Payload | Purpose |
|---|---|---|
| `ping` | `{client_timestamp: number}` | Latency probe (server replies `pong`) |
| `start_listening` | `{}` | Enter listening mode (VAD will start capture) |
| `stop_listening` | `{}` | Stop capture / return to standby |
| `interrupt` | `{}` | Barge-in: stop the current reply immediately |
| `chat_message` | `{content: str}` | Run the pipeline from text |
| `simulate_cycle` | `{utterance: str}` | Demo: simulate a voice turn |
| `audio_start` | `{}` | Client mic stream beginning |
| `audio_end` | `{}` | Client mic stream ending |
| `(binary)` | 16 kHz Int16 LE PCM, 20 ms frames | Live audio (opaque to JSON) |

### Server → Client

| `type` | Payload | Purpose |
|---|---|---|
| `pong` | `{client_timestamp, server_timestamp}` | Reply to `ping` |
| `state_change` | `{status, stage, message}` | Orb/state sync (`idle|listening|processing|speaking`) |
| `system_event` | `{status, message, details?}` | Info / warnings |
| `transcript_interim` | `{content}` | Live caption while you speak |
| `transcript_final` | `{content, stt_ms}` | Final STT result |
| `transcript_partial` | `{content}` | Live LLM generation caption |
| `chat_reply` | `{content, metrics?}` | Final assistant text |
| `audio_start` | `{encoding: "mp3"|"wav"}` | Beginning of TTS audio |
| `audio_segment_start` | `{segment_index, text}` | One spoken clause starts |
| `audio_segment_end` | `{segment_index}` | One spoken clause finished |
| `audio_end` | `{metrics?}` | End of the TTS stream |
| `pipeline_metrics` | `{metrics}` | Segment latency telemetry |
| `(binary)` | MP3 or WAV chunk | Synthesized audio |

### Latency telemetry

`metrics` objects carry `stt_ms`, `llm_ttft_ms`, `tts_first_ms` and
`total_roundtrip_ms` (the last = STT + time-to-first-spoken-clause), surfaced in the
dashboard HUD.

## 6. Project Layout

```
app/
├── main.py            # FastAPI app, WebSocket endpoint, VoicePipeline, watchdog
├── config.py          # pydantic-settings (reads .env)
├── llm.py             # Gemini streaming client + fallback models
├── stt.py             # faster-whisper local STT + rolling audio buffer + RMS
├── tts.py             # TTS engine selector (xtts / edge-tts / Piper)
├── xtts.py            # local XTTS-v2 neural voice (WAV, CPU)
├── piper_tts.py       # optional local Piper/onnx runner
├── wake.py            # wake-word detection (OWW gate + Whisper confirm / sniff)
├── state.py           # per-connection state machine
├── cache.py           # LLM response LRU cache + conversation summarizer
├── codec.py           # Opus audio codec (encode/decode)
├── vad.py             # Silero VAD + endpoint detector
├── echo_cancel.py     # acoustic echo cancellation
├── rate_limit.py      # per-client token bucket limiter
├── logging_config.py  # structured logging with correlation IDs
├── templates/index.html
└── static/            # dashboard UI (css/js)
run.py                 # launcher with port preflight + banner
Dockerfile             # containerized deployment
docker-compose.yml     # multi-service production config
tests/                 # pytest suite
```

## 7. Known limits (roadmap)

- One shared Whisper model serializes STT across concurrent connections.
- The built-in openWakeWord gate is `hey mycroft`, so hands-free wake uses Whisper
  sniffing until you provide a custom model trained for `nova` in `models/wake/`.
- XTTS-v2 is a **CPU-bound** ~3-9 s per clause on this machine — most human-sounding,
  but replies are delayed. `TTS_ENGINE=edge` trades voice quality for ~300 ms first
  voice; Piper is a middle ground (local, faster, more robotic).

## 8. Deployment

```powershell
# Docker
docker compose up --build
# Health check
curl http://localhost:8000/api/health
# Metrics
curl http://localhost:8000/api/metrics
```

CI/CD runs on push to `main` via GitHub Actions (`.github/workflows/ci.yml`):
lint (ruff) → type-check (mypy) → test (pytest) → Docker image build & push.