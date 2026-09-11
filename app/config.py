"""Application settings loaded from environment variables (.env)."""

from pydantic import Field
from pydantic.aliases import AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    app_name: str = "Nova AI Voice Assistant"

    # Google Gemini
    # Accept GEMINI_API_KEY (preferred) or GOOGLE_API_KEY (Google SDK convention)
    gemini_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    )
    gemini_model: str = "gemini-3.1-flash-lite"  # fastest Flash tier (~1.5s ttft) for snappy voice replies
    gemini_fallback_models: list[str] = ["gemini-3.6-flash", "gemini-flash-latest"]
    gemini_system_prompt: str = (
        "You are Nova, a warm, human-sounding voice assistant. Talk to the user "
        "the way Alexa or Siri does: friendly, casual, and natural, exactly as if "
        "you were speaking aloud. Use contractions (I'm, you're, it's, that's) and "
        "short, punchy, conversational sentences that flow when read aloud. "
        "Answer fully and completely — never cut a reply short. Never use bullet "
        "points, lists, markdown, numbers, or filler like 'As an AI'. Just answer "
        "like a helpful friend would out loud."
    )

    # LLM latency guards. The google-genai SDK's DEFAULT retry behavior (5 attempts,
    # exponential backoff up to 60 s) silently turns one transient 429/5xx into a
    # ~15 s stall before the first token — exactly what the old config produced.
    gemini_http_timeout_ms: int = 10000  # per-request HTTP timeout (milliseconds)
    gemini_retry_attempts: int = 2  # original + 1 fast retry (1 or 0 = no retries)
    gemini_retry_initial_delay: float = 0.2
    gemini_retry_max_delay: float = 0.5
    gemini_race_fallback: bool = False  # opt-in: race primary vs first fallback; measured SLOWER on free tier (concurrency throttling)
    llm_timeout_seconds: float = 15.0  # absolute cap on LLM streaming; fails with a clear error (split across model candidates)

    # Speech-to-Text (faster-whisper, local)
    whisper_model: str = "base"  # "tiny" is faster but drops quiet real-mic speech; "base" is far more robust
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"
    whisper_language: str = "en"
    whisper_normalize: bool = (
        True  # peak-normalize captured audio so quiet mics still transcribe
    )

    # Wake word
    assistant_name: str = "nova"
    wake_phrases: list[str] = ["hey nova", "nova"]
    wake_strategy: str = "auto"  # "whisper" | "openwakeword" | "auto"
    wake_rms_threshold: float = 250.0  # min RMS (16-bit int) to consider speech active
    wake_sniff_interval_ms: float = 500.0
    wake_window_millis: float = 3000.0
    wake_tail_millis: float = 800.0  # audio to seed command buffer after wake

    # Command capture & VAD endpointing
    sample_rate: int = 16000
    silence_timeout_ms: float = 450.0  # Instant endpointing after speech ends
    voice_debounce_frames: int = 3  # consecutive voiced chunks (~20 ms each) required before CAPTURING; rejects single noise bursts
    barge_in_required_frames: int = 5  # consecutive voiced chunks (~20 ms each) required while SPEAKING before barge-in; rejects TTS-echo blips
    barge_in_grace_ms: float = 600.0  # ignore barge-in for this long after a reply starts (Nova's own voice echoes into the mic)
    barge_in_gap_ms: float = 250.0  # while TTS audio is actively flowing, ignore mic voice until this silence gap; makes self-truncation impossible
    min_command_ms: float = 250.0
    max_command_ms: float = 12000.0
    max_audio_chunk_bytes: int = 65536

    # Speech-to-Text runtime guard (seconds)
    stt_timeout_seconds: float = 15.0

    # CORS: comma-separated allowlist (no wildcard with credentials)
    cors_origins: str = "http://127.0.0.1:8000,http://localhost:8000"

    # Conversation
    history_limit: int = 10
    # Maximum output tokens per reply. The assistant can hold whatever the model
    # reasoned through; this is just the API ceiling (0/None would disable it).
    llm_max_output_tokens: int = 2048
    # How long a disconnected session's history is kept for resume before it is
    # pruned, so `session_store` cannot grow unbounded over time.
    session_ttl_seconds: float = 3600.0

    # Text-to-Speech (edge-tts free cloud; XTTS-v2 or Piper via tts_engine)
    tts_engine: str = "edge"  # "edge" (cloud MP3, fast) | "xtts" (local, slow, human) | "piper" (local WAV)
    tts_voice: str = "en-US-JennyNeural"  # edge: warm, Siri/Alexa-like female voice
    tts_rate: str = "+4%"  # edge: calm, natural pace (avoid the rushed robotic feel)
    tts_pitch: str = "+0Hz"  # edge: slight lift sounds more engaged; try "+10Hz"
    tts_volume: str = "+0%"
    # XTTS-v2 (local neural, most human) — very slow on CPU; first run downloads ~1.6 GB
    tts_xtts_speaker: str = "Daisy Studious"  # bundled reference voice
    tts_xtts_language: str = "en"
    piper_model_path: str = "models/piper/en_US-lessac-medium.onnx"
    piper_config_path: str = "models/piper/en_US-lessac-medium.onnx.json"

    # OpenWakeWord (optional)
    ow_model_dir: str = "models/wake"

    # --- Advanced audio pipeline ---
    # Opus codec for bandwidth-efficient audio streaming (auto-falls back to PCM)
    use_opus_codec: bool = True
    # Silero VAD for precision endpointing (auto-falls back to RMS)
    use_silero_vad: bool = True
    # Acoustic echo cancellation for barge-in (auto-falls back to grace period)
    use_echo_cancellation: bool = True

    # LLM response caching
    llm_cache_enabled: bool = True
    llm_cache_max_size: int = 1000
    llm_cache_ttl_seconds: float = 3600.0

    # Conversation summarization
    conversation_summary_enabled: bool = False
    conversation_summary_turns: int = 8  # summarize after this many turns
    conversation_summary_max_chars: int = 400

    # Rate limiting (requests per client per time window)
    rate_limit_enabled: bool = True
    rate_limit_max_requests: int = 120
    rate_limit_window_seconds: float = 60.0
    rate_limit_burst_size: int = 10

    # Structured logging
    structured_logging: bool = True


def get_settings() -> Settings:
    return Settings()
