"""Text heuristics for endpointing: is the user still mid-sentence, and strip filler sounds."""
import re

_EN_OPEN = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with", "from", "by", "about", "is", "are",
    "was", "were", "does", "do", "did", "can", "could", "will", "would", "how", "what", "which", "who", "when", "where",
    "why", "if", "because", "that", "than", "then", "so", "my", "your", "its", "this", "these", "those", "as", "into",
}
_ZH_OPEN = ("的", "和", "与", "跟", "因为", "如果", "但是", "所以", "那", "然后", "还有", "或者", "在", "把", "被", "对", "给", "是", "能", "可以", "要", "会")
_EN_FILLER = re.compile(r"\b(?:uh+|um+|erm?|ah+|hmm+|mm+)\b", re.I)
_ZH_FILLER = re.compile(r"[嗯呃]+")   # not 额: it is part of 金额 / 额定 / 额外


def strip_disfluency(text: str, lang: str) -> str:
    out = _ZH_FILLER.sub("", text) if lang == "zh" else _EN_FILLER.sub(" ", text)
    return " ".join(out.split()) if lang != "zh" else out.strip()


def looks_incomplete(text: str, lang: str) -> bool:
    """True when the text ends where a sentence cannot end (article, conjunction, copula, 的/因为 ...)."""
    t = strip_disfluency(text, lang).strip()
    if not t:
        return False
    if lang == "zh":
        return t.endswith(_ZH_OPEN)
    last = re.sub(r"[^a-z0-9']+", "", t.lower().split()[-1])
    return last in _EN_OPEN
