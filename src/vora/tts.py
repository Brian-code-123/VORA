import logging
import re
import threading
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import sherpa_onnx
from scipy.signal import resample_poly

from vora.config import Settings

log = logging.getLogger("vora")
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


def zh_numbers(text: str) -> str:
    """Digits inside a Chinese run -> Chinese numerals (the voice has no digit tokens: they were skipped silently).
    Up to 4 integer digits read as a number (10 -> 十), longer strings digit by digit, 2.4 -> 二点四."""
    import cn2an

    def conv(m: re.Match) -> str:
        t = m.group(0)
        if "." in t:
            a, b = t.split(".", 1)
            return (cn2an.an2cn(a) if len(a) <= 4 else a.translate(_ZH_DIGITS)) + "点" + b.translate(_ZH_DIGITS)
        return cn2an.an2cn(t) if len(t) <= 4 else t.translate(_ZH_DIGITS)
    return re.sub(r"\d+(?:\.\d+)?", conv, text)


def _zh_digits(s: str) -> str:
    return re.sub(r"\d+(?:\.\d+)?", lambda m: m.group(0).translate(_ZH_DIGITS), s)


def sanitize_for_tts(text: str) -> str:
    t = _URL.sub(" ", text)
    t = _EMOJI.sub("", t)
    t = _MD.sub("", t)
    # model codes: VORA-X200 -> "VORA X 200" (letters and digits read separately)
    t = re.sub(r"\b([A-Za-z]{2,})-([A-Za-z])(\d+)\b", r"\1 \2 \3", t)
    t = re.sub(r"\b([A-Za-z])(\d+)\b", r"\1 \2", t)
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


def en_voice_dir(settings: Settings) -> Path:
    """ljspeech: public-domain voice (default). lessac: the older Blizzard-licensed voice, kept in its original folder."""
    return settings.models_dir / ("tts_en_ljspeech" if settings.tts_en_voice == "ljspeech" else "tts_en")


def pick_en_model(d: Path, fp32: bool) -> Path:
    """int8 voice unless fp32 requested; falls back to fp32 when no int8 file exists (fresh image / fresh checkout)."""
    plain = sorted(p for p in d.glob("*.onnx") if not p.name.endswith(".int8.onnx"))
    int8 = sorted(d.glob("*.int8.onnx"))
    if fp32 or not int8:
        if not fp32:
            log.warning("no int8 English voice in %s: using fp32 (run scripts/quantize_tts.py for the 30 MB gate)", d)
        return plain[0]
    return int8[0]


def ensure_patched_lexicon(zh_dir: Path) -> Path:
    """Derived `lexicon.patched.txt`: drop phoneme tokens absent from tokens.txt (only U+032A, a combining mark, in
    huayan x_low). Without it every z/c/s syllable (自 次 词 子 字 私 ...) was silently skipped. The original stays."""
    out = zh_dir / "lexicon.patched.txt"
    if out.exists():
        return out
    tokens = {l.split(" ")[0] for l in (zh_dir / "tokens.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
    lines = []
    for line in (zh_dir / "lexicon.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            parts = [parts[0]] + [t for t in parts[1:] if t in tokens]
        lines.append(" ".join(parts))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _usable_zh_chars(zh_dir: Path) -> set[str]:
    """Characters the zh voice can really say, from the patched lexicon (phonemes must all be in tokens.txt)."""
    tokens = {l.split(" ")[0] for l in (zh_dir / "tokens.txt").read_text(encoding="utf-8").splitlines() if l.strip()}
    out = set()
    for line in ensure_patched_lexicon(zh_dir).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and all(p in tokens for p in parts[1:]):
            out.add(parts[0])
    return out


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
        en = en_voice_dir(settings)
        zh = d / "tts_zh"
        self.voices = {
            "en": _voice(str(pick_en_model(en, settings.tts_en_fp32)), str(en / "tokens.txt"), data_dir=str(en / "espeak-ng-data")),
            "zh": _voice(str(next(zh.glob("*.onnx"))), str(zh / "tokens.txt"), lexicon=str(ensure_patched_lexicon(zh))),
        }
        self.zh_lexicon = _usable_zh_chars(zh)
        self.oov_total = 0   # zh characters the voice cannot say (counted, logged, exposed as metrics tts_oov)
        self._cancel = threading.Event()
        self._lock = threading.Lock()  # sherpa OfflineTts.generate is not documented thread-safe; held only while generating

    def cancel(self) -> None:
        self._cancel.set()

    def synth(self, text: str, cancel: threading.Event | None = None) -> Iterator[bytes]:
        if cancel is None:
            cancel = self._cancel
            cancel.clear()
        for lang, run in split_by_script(sanitize_for_tts(text)):
            if not re.search(r"\w", run):
                continue
            if lang == "zh":
                run = zh_numbers(run)
                oov = [c for c in run if _CJK.match(c) and c not in self.zh_lexicon]
                if oov:
                    self.oov_total += len(oov)
                    log.warning("TTS OOV: zh voice cannot say %s (skipped)", "".join(sorted(set(oov))))
            v = self.voices[lang]
            with self._lock:  # released before chunks are yielded: a slow client must not hold the voice
                audio = v.generate(run.strip(), sid=0, speed=1.0, callback=lambda s, p: 0 if cancel.is_set() else 1)  # sherpa 1.13: 1 = continue, 0 = stop (its docstring says the reverse)
            x = np.asarray(audio.samples, dtype=np.float32)
            if x.size == 0 or audio.sample_rate <= 0:   # all-OOV run: nothing to say (used to crash in resample)
                continue
            if audio.sample_rate != SR:
                x = resample_poly(x, SR, audio.sample_rate).astype(np.float32)
            pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
            for i in range(0, len(pcm), CHUNK):
                if cancel.is_set():
                    return
                yield pcm[i:i + CHUNK].tobytes()
