"""Synthesize demo questions with our own TTS so the demo works without a microphone."""
import wave
from pathlib import Path

from vora.config import ROOT, Settings
from vora.tts import Tts

SAMPLES = {
    "q_warranty_en": "How long is the warranty on the VORA X200?",
    "q_reset_en": "How do I factory reset the box?",
    "q_warranty_zh": "X200保修期多久？",
}

if __name__ == "__main__":
    tts = Tts(Settings())
    out = ROOT / "client" / "samples"
    out.mkdir(parents=True, exist_ok=True)
    for name, text in SAMPLES.items():
        pcm = b"".join(tts.synth(text))
        with wave.open(str(out / f"{name}.wav"), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm)
        print(name, len(pcm) / 32000, "s")
