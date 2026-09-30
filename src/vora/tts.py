import re
import threading
from typing import Iterator, Literal

import numpy as np
import sherpa_onnx
from scipy.signal import resample_poly

from vora.config import Settings

SR = 16000
CHUNK = int(0.2 * SR)  # 200 ms of samples
_CJK = re.compile(r"[一-鿿]")
_CJK_PUNCT = "，。！？；：、（）「」《》"
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")
_URL = re.compile(r"https?://\S+")
_MD = re.compile(r"[*_#>`~|]")
_CODE = re.compile(r"\b([A-Za-z]+)-?([A-Za-z]?)(\d+)\b")
_ZH_DIGITS = str.maketrans("0123456789.", "零一二三四五六七八九点")


def lang_of(text: str) -> Literal["zh", "en"]:
    # zh when CJK characters >= Latin words: "請問 VORA Box 支援 WiFi 嗎" is Chinese with English terms
    return "zh" if len(_CJK.findall(text)) >= max(1, len(re.findall(r"[A-Za-z]+", text))) else "en"


def _zh_digits(s: str) -> str:
    return re.sub(r"\d+(?:\.\d+)?", lambda m: m.group(0).translate(_ZH_DIGITS), s)


def sanitize_for_tts(text: str) -> str:
    t = _URL.sub(" ", text)
    t = _EMOJI.sub("", t)
    t = _MD.sub("", t)
    # model codes: VORA-X200 -> "VORA X 200" (letters and digits read separately)
    t = re.sub(r"\b([A-Za-z]{2,})-([A-Za-z])(\d+)\b", r"\1 \2 \3", t)
    t = re.sub(r"\b([A-Za-z])(\d+)\b", r"\1 \2", t)
    t = re.sub(r"(?<=[一-鿿])(\d+(?:\.\d+)?)(?=[一-鿿])", lambda m: _zh_digits(m.group(1)), t)
    return re.sub(r"[ \t]+", " ", t).strip()


def split_by_script(text: str) -> list[tuple[str, str]]:
    """zh/en runs; digits, spaces and ASCII punctuation stick to the current run."""
    runs: list[list[str]] = []
    pending = ""
    for ch in text:
        if _CJK.match(ch) or ch in _CJK_PUNCT:
            lang = "zh"
        elif ch.isascii() and ch.isalpha():
            lang = "en"
        else:
            if runs:
                runs[-1][1] += ch
            else:
                pending += ch
            continue
        if not runs or runs[-1][0] != lang:
            runs.append([lang, pending])
            pending = ""
        runs[-1][1] += ch
    return [(l, t) for l, t in runs]


def _voice(model: str, tokens: str, data_dir: str = "", lexicon: str = "", threads: int = 2):
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=model, tokens=tokens, data_dir=data_dir, lexicon=lexicon),
            num_threads=threads, provider="cpu"))
    return sherpa_onnx.OfflineTts(cfg)


class Tts:
    """Clause in, PCM16 16 kHz chunks (<=200 ms) out. One instance per process; call from one worker thread."""

    def __init__(self, settings: Settings):
        d = settings.models_dir
        en = d / "tts_en"
        zh = d / "tts_zh"
        self.voices = {
            "en": _voice(str(next(en.glob("*.onnx" if settings.tts_en_fp32 else "*.int8.onnx"))), str(en / "tokens.txt"), data_dir=str(en / "espeak-ng-data")),
            "zh": _voice(str(next(zh.glob("*.onnx"))), str(zh / "tokens.txt"), lexicon=str(zh / "lexicon.txt")),
        }
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def synth(self, text: str) -> Iterator[bytes]:
        self._cancel.clear()
        for lang, run in split_by_script(sanitize_for_tts(text)):
            if not re.search(r"\w", run):
                continue
            v = self.voices[lang]
            audio = v.generate(run.strip(), sid=0, speed=1.0, callback=lambda s, p: 1 if self._cancel.is_set() else 0)
            x = np.asarray(audio.samples, dtype=np.float32)
            if audio.sample_rate != SR:
                x = resample_poly(x, SR, audio.sample_rate).astype(np.float32)
            pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
            for i in range(0, len(pcm), CHUNK):
                if self._cancel.is_set():
                    return
                yield pcm[i:i + CHUNK].tobytes()
