"""RSS per component, each in a fresh interpreter. Brief: ASR+RAG+TTS <= 500 MB total; LLM reported separately."""
import json
import subprocess
import sys

import pytest

from vora.config import Settings

S = Settings()
pytestmark = [pytest.mark.perf, pytest.mark.skipif(not (S.index_dir / "faiss.index").exists(), reason="needs models+index")]

SNIPPET = """
import gc, json, os, psutil
p = psutil.Process()
base = p.memory_full_info().uss
from vora.config import Settings
S = Settings()
{load}
gc.collect()
print(json.dumps({{"delta_mb": (p.memory_full_info().uss - base) / 2**20}}))
"""
LOADS = {
    "asr_rag_tts": """
import numpy, sherpa_onnx, onnxruntime, faiss, jieba, fastembed
imp = p.memory_full_info().uss
from vora.asr import load_recognizers, AsrSession
r = load_recognizers(S)
for l in ('en', 'zh'): AsrSession(r, l).feed(b'\\x00\\x00' * 1600)
from vora.rag.retriever import Retriever
Retriever(S).search('warranty')
from vora.tts import Tts
t = Tts(S); list(t.synth('Hello there.')); list(t.synth('你好。'))
""",
    "llm": "from vora.llm import Llm\nfrom vora.rag.store import Hit\nl = Llm(S)\nlist(l.stream('hi',[Hit('w','Hello.',1.0)]))",
}


def measure(name: str) -> float:
    out = subprocess.run([sys.executable, "-c", SNIPPET.format(load=LOADS[name])], capture_output=True, text=True, timeout=300)
    return json.loads(out.stdout.strip().splitlines()[-1])["delta_mb"]


def test_uss_asr_rag_tts_le_500mb():
    """Median of 3 fresh processes, USS (unique set size). RSS over-counts shared/compressed pages on macOS."""
    runs = sorted(measure("asr_rag_tts") for _ in range(3))
    llm = measure("llm")
    m = {"asr_rag_tts": runs[1], "asr_rag_tts_runs": [round(r, 1) for r in runs], "llm": llm}
    print("USS MB:", json.dumps({k: (round(v, 1) if not isinstance(v, list) else v) for k, v in m.items()}))
    (S.models_dir.parent / "results").mkdir(exist_ok=True)
    (S.models_dir.parent / "results" / "memory.json").write_text(json.dumps({**{k: (round(v, 1) if not isinstance(v, list) else v) for k, v in m.items()}, "metric": "USS"}))
    assert runs[1] <= 500, m


def test_uss_stable_over_50_turns():
    """Leak guard. Warm-up = every eval question once (touches the longest prompts: llama.cpp keeps a logits row per
    prompt token, ~0.6 MB each, so USS grows only until the longest prompt was seen). Then 30 more turns must add <=30 MB.
    Profiled: retrieval 0 MB, TTS ~0 MB, LLM ~+8 MB plateau. Lengthening prompts by padding is NOT a leak."""
    import gc
    import statistics
    import time
    from vora.hostcheck import uss_mb
    from vora.llm import Llm
    from vora.rag.ingest import load_qa
    from vora.rag.retriever import Retriever
    from vora.tts import Tts
    ret, llm, tts = Retriever(S), Llm(S), Tts(S)
    qs = [q["q"] for q in load_qa(S.models_dir.parent / "eval" / "rag_qa.jsonl") if q["chunk_id"]]

    def turn(i):
        q = qs[i % len(qs)]
        b"".join(tts.synth("".join(llm.stream(q, ret.search(q)))))

    def uss():
        gc.collect()
        vals = []
        for _ in range(3):
            vals.append(uss_mb())
            time.sleep(0.2)
        return statistics.median(vals)

    for i in range(len(qs)):
        turn(i)
    base = uss()
    for i in range(30):
        turn(i)
    assert uss() - base <= 30, uss() - base
