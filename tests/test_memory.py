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
    from vora.hostcheck import write_result
    write_result(S.models_dir.parent / "results" / "memory.json", {**{k: (round(v, 1) if not isinstance(v, list) else v) for k, v in m.items()}, "metric": "USS"})
    assert runs[1] <= 500, m


LEAK_SCRIPT = """
import gc, json, statistics, time
from vora.config import Settings
from vora.hostcheck import uss_mb
from vora.llm import Llm
from vora.rag.ingest import load_qa
from vora.rag.retriever import Retriever
from vora.tts import Tts
S = Settings()
ret, llm, tts = Retriever(S), Llm(S), Tts(S)
qs = [q["q"] for q in load_qa(S.models_dir.parent / "eval" / "rag_qa.jsonl") if q["chunk_id"]]

def turn(i):
    q = qs[i % len(qs)]
    b"".join(tts.synth("".join(llm.stream(q, ret.search(q)))))

def uss():
    gc.collect()
    v = []
    for _ in range(3):
        v.append(uss_mb()); time.sleep(0.2)
    return statistics.median(v)

for i in range(len(qs)):
    turn(i)
base = uss()
for i in range(30):
    turn(i)
print(json.dumps({"drift_mb": uss() - base}))
"""


def test_uss_stable_over_50_turns():
    """Leak guard in a FRESH process (inside a pytest run other tests' leftovers add noise: +52 MB was measured there).
    Warm-up = every eval question once (llama.cpp keeps ~0.6 MB of logits per prompt token, so USS grows only until the
    longest prompt was seen); then 30 more turns must add <=150 MB. Profiled per component: retrieval 0, TTS +10..22 MB over 60 turns (ORT arenas for new
    text lengths), LLM swings -87..+8 MB (macOS compresses its pages), fresh-process total +94 MB measured. 150 MB catches a
    runaway (the removed RAM prompt cache grew by 1.4 GB) without flaking on allocator noise."""
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-c", LEAK_SCRIPT], capture_output=True, text=True, timeout=900,
                         env={**__import__("os").environ, "PYTHONPATH": str(S.models_dir.parent / "src")})
    drift = json.loads(out.stdout.strip().splitlines()[-1])["drift_mb"]
    print("USS drift MB:", round(drift, 1))
    assert drift <= 150, drift
