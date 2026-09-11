"""Generate a speech WAV for Chrome's --use-file-for-fake-audio-capture."""
import os, sys, wave
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np

os.environ["COQUI_TOS_AGREED"] = "1"
from TTS.api import TTS

TEXT = "What time is it?"
OUT = os.path.join(os.environ.get("TEMP", "."), "nova_fake_mic.wav")

wav24k = TTS("tts_models/multilingual/multi-dataset/xtts_v2").tts(
    text=TEXT, speaker="Daisy Studious", language="en"
)
src = np.asarray(wav24k, dtype=np.float32)
n_out = int(round(len(src) * 44100 / 24000))
x_new = np.linspace(0, 1, n_out, endpoint=False)
x_old = np.linspace(0, 1, len(src), endpoint=False)
out441 = np.interp(x_new, x_old, src).astype(np.float32)

with wave.open(OUT, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(44100)
    w.writeframes((np.clip(out441, -1, 1) * 32767).astype("<i2").tobytes())

print(OUT, os.path.getsize(OUT), "bytes")