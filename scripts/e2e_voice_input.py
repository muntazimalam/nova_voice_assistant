"""Reproduce the browser voice-input path end-to-end against a real server.

It is impossible to run getUserMedia in this headless environment, so we build
the exact byte stream a tab would send: 16kHz Int16 LE PCM in ~640-byte frames,
after JSON start_listening + audio_start. The 'spoken' utterance is synthesized
locally with XTTS so whisper actually recognizes words.

Usage:
  python scripts/e2e_voice_input.py [utterance]
"""
import asyncio
import json
import os
import struct
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np


def resample_24k_to_16k(samples):
    """XTTS outputs 24k float32; downsample to 16k by linear interpolation."""
    src = np.asarray(samples, dtype=np.float32)
    n_out = int(round(len(src) * 16000 / 24000))
    x_old = np.linspace(0.0, 1.0, len(src), endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, src).astype(np.float32)


def synth_utterance(text: str) -> bytes:
    os.environ["COQUI_TOS_AGREED"] = "1"
    from TTS.api import TTS

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    wav24k = tts.tts(text=text, speaker="Daisy Studious", language="en")
    wav16k = resample_24k_to_16k(wav24k)
    pcm16 = np.clip(wav16k, -1.0, 1.0)
    return (pcm16 * 32767.0).astype("<i2").tobytes()


def chunk_pcm(pcm: bytes, frame_bytes: int = 640):
    return [pcm[i:i + frame_bytes] for i in range(0, len(pcm), frame_bytes)]


async def main(utterance: str):
    try:
        import websockets
    except ImportError:
        print("websockets not installed")
        sys.exit(1)

    utterance = utterance or "What is the weather like tomorrow?"
    print(f"[1/4] Synthesizing test utterance locally: {utterance!r}")
    pcm = synth_utterance(utterance)
    print(f"      -> {len(pcm)} bytes of 16k Int16 PCM")

    frames = chunk_pcm(pcm)
    print(f"[2/4] Connecting to ws://127.0.0.1:8000/ws")
    url = "ws://127.0.0.1:8000/ws"
    events = []

    async with websockets.connect(url, ping_interval=None) as ws:
        print("[3/4] Sending start_listening + audio_start, then streaming PCM...")
        await ws.send(json.dumps({"type": "start_listening"}))
        await ws.send(json.dumps({"type": "audio_start"}))

        for i, frame in enumerate(frames):
            await ws.send(frame)
            if i % 20 == 0:
                await asyncio.sleep(0.0)

        await ws.send(json.dumps({"type": "audio_end"}))
        print(f"      -> sent {len(frames)} frames")

        print("[4/4] Reading frames (up to 30s)...")
        deadline = time.monotonic() + 30
        got_transcript = False
        while time.monotonic() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
            except asyncio.TimeoutError:
                print("      (no frame in 5s; continuing)")
                continue
            if isinstance(raw, bytes):
                continue
            data = json.loads(raw)
            events.append(data["type"])
            print(f"      {data['type']}: {json.dumps(data)[:200]}")

            if data["type"] == "transcript_final":
                got_transcript = True
                text = data.get("content", "")
                print(f"\n  TRANSCRIPT: {text!r}")
                if "what" in text.lower() and ("weather" in text.lower() or "tomorrow" in text.lower()):
                    print("  RESULT: VOICE INPUT WORKED")
                    await ws.send(json.dumps({"type": "interrupt"}))
                    return 0
                print("  RESULT: audio captured but STT text didn't match (check mic level / STT)")
                await ws.send(json.dumps({"type": "interrupt"}))
                return 1
            if data["type"] in ("audio_start", "audio_segment_start"):
                pass
            if data["type"] == "system_event" and "Too short" in data.get("message", ""):
                print("  RESULT: captured audio was too short / too quiet")
                return 1

        print("  RESULT: no transcript_final received in time")
        return 1


if __name__ == "__main__":
    code = asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else ""))
    sys.exit(code)