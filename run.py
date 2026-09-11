"""
Launcher for the AURA AI Voice Assistant server.

Usage:
    python run.py            # production mode  (reload=False)
    python run.py --dev      # development mode (reload=True, warns about double model loads)
    python run.py --port 8080
"""

import argparse
import io
import socket
import sys

# Force UTF-8 output so box-drawing chars render on Windows PowerShell / cmd.
# This must happen before any print() calls.
try:
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
    )
    _UTF8 = True
except AttributeError:
    # Already a text stream without .buffer (e.g. some IDEs / pytest captures)
    _UTF8 = False

import uvicorn

# ── Early config import (validates .env before uvicorn even starts) ──────────
try:
    from app import __version__
    from app.config import get_settings
    settings = get_settings()
except Exception as exc:  # noqa: BLE001
    print(f"\n[FATAL] Could not load application settings: {exc}")
    print("        Make sure a valid .env file exists in the project root.\n")
    sys.exit(1)

# ── CLI argument parsing ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="AURA AI Voice Assistant launcher",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument(
    "--dev",
    action="store_true",
    help="Enable hot-reload (development mode). "
         "NOTE: Whisper & wake-word models will be loaded twice on the first save.",
)
parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
parser.add_argument(
    "--log-level",
    default="info",
    choices=["critical", "error", "warning", "info", "debug", "trace"],
    help="Uvicorn log level (default: info)",
)
args = parser.parse_args()

# ── Port preflight check ─────────────────────────────────────────────────────
def _port_is_free(host: str, port: int) -> bool:
    """
    Return True if (host, port) is available to bind.

    On Windows, SO_REUSEADDR will succeed even when a previous process still
    holds an exclusive handle (phantom lock after a subprocess exits).
    SO_EXCLUSIVEADDRUSE gives the correct answer on Windows.
    On other platforms, a plain connect() probe is reliable.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if sys.platform == "win32":
            # Windows-specific: exclusive address use — refuses to bind if
            # anyone else holds the port, even with SO_REUSEADDR set.
            SO_EXCLUSIVEADDRUSE = 0x0004  # noqa: N806
            try:
                s.setsockopt(socket.SOL_SOCKET, SO_EXCLUSIVEADDRUSE, 1)
            except OSError:
                pass  # very old Python / unusual build — fall through
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False

# ── Startup validation ───────────────────────────────────────────────────────
_WARNINGS: list[str] = []

if not settings.gemini_api_key:
    _WARNINGS.append(
        "GEMINI_API_KEY / GOOGLE_API_KEY not found in .env — "
        "LLM responses will fail. Add it to your .env file."
    )

# ── Banner ───────────────────────────────────────────────────────────────────
_W = 55  # inner width of the box

if _UTF8:
    _TL, _TR, _BL, _BR = "╔", "╗", "╚", "╝"
    _VL, _LX, _RX      = "═", "╠", "╣"
    _SI                = "║"
    _WRN, _OK, _ERR    = "⚠", "✔", "✖"
else:
    _TL, _TR, _BL, _BR = "+", "+", "+", "+"
    _VL, _LX, _RX      = "-", "+", "+"
    _SI                = "|"
    _WRN, _OK, _ERR    = "!", "OK", "X"

_HR = _VL * _W

def _row(label: str, value: str) -> str:
    """One data row: ║  label<16> value<37>║"""
    cell = f"  {label:<16} {value}"
    return f"  {_SI}{cell:<{_W}}{_SI}"

def _warn_row(text: str) -> str:
    """Warning row with ⚠ prefix."""
    cell = f"  {_WRN}  {text}"
    return f"  {_SI}{cell:<{_W}}{_SI}"

def _banner() -> None:
    dashboard  = f"http://{args.host}:{args.port}"
    websocket  = f"ws://{args.host}:{args.port}/ws"
    health     = f"http://{args.host}:{args.port}/api/health"
    mode       = "DEVELOPMENT (hot-reload)" if args.dev else "PRODUCTION"

    lines = [
        f"  {_TL}{_HR}{_TR}",
        f"  {_SI}{'  NOVA AI Voice Assistant — Real-Time Agent':^{_W}}{_SI}",
        f"  {_LX}{_HR}{_RX}",
        _row("App",            settings.app_name),
        _row("Version",        __version__),
        _row("Mode",           mode),
        _row("LLM Model",      settings.gemini_model),
        _row("STT Model",      settings.whisper_model),
        _row("TTS Voice",      settings.tts_voice),
        _row("Wake Strategy",  settings.wake_strategy),
        _row("Assistant",      settings.assistant_name),
        f"  {_LX}{_HR}{_RX}",
        _row("Dashboard",      dashboard),
        _row("WebSocket",      websocket),
        _row("Health",         health),
    ]

    if _WARNINGS:
        lines.append(f"  {_LX}{_HR}{_RX}")
        for w in _WARNINGS:
            # word-wrap at (_W - 7) chars to fit inside the ⚠ row
            wrap = _W - 7
            words, buf = w.split(), ""
            for word in words:
                if len(buf) + len(word) + 1 > wrap:
                    lines.append(_warn_row(buf))
                    buf = word
                else:
                    buf = f"{buf} {word}".strip()
            if buf:
                lines.append(_warn_row(buf))

    lines += [
        f"  {_LX}{_HR}{_RX}",
        f"  {_SI}{'  Press Ctrl+C to stop the server.':<{_W}}{_SI}",
        f"  {_BL}{_HR}{_BR}",
        "",
    ]
    print("\n" + "\n".join(lines))


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _banner()

    # Port conflict: detect early and guide the user instead of crashing.
    if not _port_is_free(args.host, args.port):
        print(
            f"  {_ERR}  Port {args.port} is already in use on {args.host}.\n"
            f"\n"
            f"  To find and kill the blocking process, run:\n"
            f"      netstat -ano | findstr :{args.port}\n"
            f"  Then terminate it with:\n"
            f"      taskkill /PID <PID> /F\n"
            f"\n"
            f"  Or start on a different port:\n"
            f"      python run.py --port 8080\n"
        )
        sys.exit(1)

    if args.dev:
        print(
            "  ⚠  Hot-reload is ON — Whisper & wake-word models will reload on file changes.\n"
            "     Use production mode (no --dev flag) to avoid double model initialisation.\n"
        )

    try:
        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=args.port,
            reload=args.dev,
            log_level=args.log_level,
        )
    except KeyboardInterrupt:
        print(f"\n\n  {_OK}  Server stopped cleanly. Goodbye!\n")
