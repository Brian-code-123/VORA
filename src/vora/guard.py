"""Cheap checks on a 0.5B model's answer against the retrieved chunk: polarity conflicts, refusals, best sentence.
No jieba: Chinese uses character windows, English whitespace words."""
import re

_CJK = re.compile(r"[一-鿿]")
_EN_AUX = ("is", "are", "was", "were", "does", "do", "did", "can", "could", "will", "would", "should", "has", "have")
_EN_NEG = {"not", "no", "never", "cannot", "can't", "doesn't", "isn't", "aren't", "don't", "won't", "without", "unsupported", "n't"}
_ZH_NEG = "不没无未非"
_EN_STOP = set(_EN_AUX) | {"the", "a", "an", "it", "its", "this", "that", "with", "for", "and", "what", "how", "which", "you", "your", "box", "about"}
_REFUSAL = re.compile(r"(does not provide|doesn't provide|not provide|no information|cannot find|can't find|not mentioned|"
                      r"not specified|don't have|do not have|in the context|provided in|没有提供|没有信息|没有相关|无法|不清楚|未提及|上下文)", re.I)
_SENT = re.compile(r"(?:[^。！？\n]*?(?:[。！？]|[.!?](?=\s|$)|$))")   # a "." inside an email / version is not a sentence end


def _is_zh(t: str) -> bool:
    return bool(_CJK.search(t))


def is_yes_no_question(q: str) -> bool:
    q = q.strip().lower()
    if _is_zh(q):
        return bool(re.search(r"(吗|嗎|是不是|能不能|可不可以|有没有|会不会|支不支持)", q))
    first = re.sub(r"[^a-z']+", " ", q).split()[:1]
    return bool(first) and first[0] in _EN_AUX


def _keywords(q: str) -> list[str]:
    q = q.lower()
    if _is_zh(q):
        zh = "".join(_CJK.findall(q))
        stop = set("吗嗎是不能可以有会么的了呢吧")
        return [zh[i:i + 2] for i in range(len(zh) - 1) if not (set(zh[i:i + 2]) & stop)]
    return [w for w in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", q) if w not in _EN_STOP]


def _neg_count_en(words: list[str]) -> int:
    return sum(1 for w in words if w in _EN_NEG or w.endswith("n't"))


def _chunk_negates_topic(question: str, chunk: str) -> bool:
    """Per sentence: a negation cue within 6 words / 6 chars of a question keyword; an even number of cues cancels."""
    kws = _keywords(question)
    for sent in (x.lower() for x in split_sentences(chunk)):
        if _is_zh(sent):
            for kw in kws:
                for m in re.finditer(re.escape(kw), sent):
                    win = sent[max(0, m.start() - 6): m.end() + 2]
                    if sum(win.count(ch) for ch in _ZH_NEG) % 2 == 1:
                        return True
            continue
        words = re.findall(r"[a-z0-9'\-]+", sent)
        for i, w in enumerate(words):
            if any(w.startswith(k) or (k.startswith(w) and len(w) > 3) for k in kws):
                if _neg_count_en(words[max(0, i - 6): i + 7]) % 2 == 1:
                    return True
    return False


def _answer_positive(head: str) -> bool:
    h = head.strip().lower()
    if not h:
        return False
    if _is_zh(h):
        return not any(ch in h for ch in _ZH_NEG)
    words = re.findall(r"[a-z']+", h)
    return _neg_count_en(words) == 0


def polarity_conflict(question: str, chunk: str, answer_head: str) -> bool:
    """Yes/no question, chunk says the topic is NOT so, answer says yes (0.5B models ignore negation)."""
    return is_yes_no_question(question) and _chunk_negates_topic(question, chunk) and _answer_positive(answer_head)


def is_refusal(answer_head: str) -> bool:
    return bool(_REFUSAL.search(answer_head or ""))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.findall(text) if s.strip()]


def best_sentence(chunk: str, question: str) -> str:
    sents = split_sentences(chunk)
    if len(sents) <= 1:
        return chunk.strip()
    kws = set(_keywords(question)) or set()
    if not kws:
        return sents[0]
    scores = [sum(1 for k in kws if k in s.lower()) for s in sents]
    return sents[scores.index(max(scores))]


def negating_sentence(question: str, chunk: str) -> str | None:
    """The sentence of `chunk` that negates the question's topic (what to say instead of a wrong "yes")."""
    for sent in split_sentences(chunk):
        if sent and _chunk_negates_topic(question, sent):
            return sent
    return None


_NUM = re.compile(r"(?<![A-Za-z\d])\d+(?:[.,]\d+)*")   # not the digits inside model codes like X200


def _numbers(text: str) -> set[float]:
    out = set()
    for m in _NUM.findall(text):
        t = m.replace(",", "") if re.fullmatch(r"\d{1,3}(?:,\d{3})+", m) else m.replace(",", ".")
        try:
            out.add(float(t))
        except ValueError:
            pass
    return out


_DIGITISH = re.compile(r"[\s$¥€]*[\d.,]+|[$¥€]")


def is_digitish(tok: str) -> bool:
    """A token that may be part of a number still being streamed ("$", "1", ",", "000")."""
    return bool(_DIGITISH.fullmatch(tok))


def numbers_mismatch(chunk: str, head: str) -> bool:
    """The answer states a figure that is not in the retrieved text (0.5B models invent prices and ranges).
    `head` must not end inside a number: the caller holds back digit-ish tokens until the number is complete."""
    h = _numbers(head)
    return bool(h) and not h <= _numbers(chunk)


def extractive_answer(question: str, chunks: list[str]) -> str:
    """Honest answer straight from the text. Yes/no question: the sentence that negates the topic if any, else the two
    best-matching sentences in text order (the answer is often the neighbour of the best match: "...erases settings.
    Firmware is kept."). Other questions: the single best sentence across the chunks."""
    yes_no = is_yes_no_question(question)
    if yes_no:
        for c in chunks:
            neg = negating_sentence(question, c)
            if neg:
                return neg
    kws = set(_keywords(question))
    best, best_score = None, -1
    for c in chunks:
        sents = split_sentences(c)
        scored = [(sum(1 for k in kws if k in x.lower()), i) for i, x in enumerate(sents)]
        if not scored:
            continue
        top = max(scored)[0]
        if top > best_score:
            if yes_no and len(sents) > 1:
                order = sorted(sorted(scored, key=lambda t: (-t[0], t[1]))[:2], key=lambda t: t[1])
                best = " ".join(sents[i] for _, i in order)
            else:
                best = sents[[i for sc, i in scored if sc == top][0]] if len(sents) > 1 else c.strip()
            best_score = top
    return best or (chunks[0] if chunks else "")


_EXPECTS_EN = re.compile(r"^\s*(how\s+(long|much|many|far|big|often|loud|tall|heavy|fast|old)|what(?:'s| is)?\s+(the\s+)?(temperature|power|size|price|cost|range|capacity|storage))\b", re.I)
_EXPECTS_ZH = re.compile(r"(多久|多少|多大|多远|多长|多高|多重|几[年个天月秒米瓦核次]|范围|距离|温度|功耗|价格|售价|容量)")
_NUMWORD = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|hundred|thousand)\b", re.I)


def expects_number(question: str) -> bool:
    return bool(_EXPECTS_ZH.search(question) if _is_zh(question) else _EXPECTS_EN.search(question))


def has_number(text: str) -> bool:
    return bool(_NUM.search(text) or _NUMWORD.search(text) or re.search(r"[一二三四五六七八九十百千万两]", text))
