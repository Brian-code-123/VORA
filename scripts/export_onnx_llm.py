"""Export Qwen2.5-0.5B-Instruct to ONNX (optimum), int8-quantize, and benchmark first-token latency and tok/s
against the GGUF Q4_K_M runtime on the same RAG prompt. Benchmark only: the live path uses whichever is faster.
Usage: export_onnx_llm.py"""
import json
import time
from pathlib import Path

import torch
from optimum.onnxruntime import ORTModelForCausalLM, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

from vora.config import ROOT, Settings
from vora.llm import SYSTEM, Llm
from vora.rag.store import Hit

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
HIT = Hit("E03", "Warranty: the VORA-X200 warranty period is 2 years. The VORA-M100 warranty period is 1 year.", 1.0)
Q = "How long is the VORA-X200 warranty?"


def bench_onnx(model_dir: Path, runs: int = 5) -> dict:
    tok = AutoTokenizer.from_pretrained(REPO)
    model = ORTModelForCausalLM.from_pretrained(model_dir)
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Context:\n[1] {HIT.text}\n\nQuestion: {Q}"}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    first, rate = [], []
    for _ in range(runs + 1):
        t0 = time.perf_counter()
        model.generate(ids, max_new_tokens=1, do_sample=False)
        t1 = time.perf_counter()
        out = model.generate(ids, max_new_tokens=32, do_sample=False, min_new_tokens=32)
        t2 = time.perf_counter()
        first.append((t1 - t0) * 1e3)
        rate.append(32 / (t2 - t1 - (t1 - t0)) if t2 - t1 > t1 - t0 else 0)
    first, rate = first[1:], rate[1:]  # drop warmup
    return {"first_token_ms_median": sorted(first)[len(first) // 2], "decode_tok_s_median": sorted(rate)[len(rate) // 2]}


def bench_gguf(runs: int = 5) -> dict:
    llm = Llm(Settings())
    list(llm.stream(Q, [HIT]))
    first, rate = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        it = llm.stream(Q, [HIT])
        next(it)
        t1 = time.perf_counter()
        n = sum(1 for _ in it) + 1
        t2 = time.perf_counter()
        first.append((t1 - t0) * 1e3)
        rate.append(n / (t2 - t0))
    return {"first_token_ms_median": sorted(first)[len(first) // 2], "decode_tok_s_median": sorted(rate)[len(rate) // 2]}


def main() -> None:
    out_dir = ROOT / "data" / "onnx_qwen"
    fp = out_dir / "fp32"
    if not fp.exists():
        ORTModelForCausalLM.from_pretrained(REPO, export=True).save_pretrained(fp)
    q = out_dir / "int8"
    if not q.exists():
        quant = ORTQuantizer.from_pretrained(fp)
        quant.quantize(save_dir=q, quantization_config=AutoQuantizationConfig.arm64(is_static=False, per_channel=False))
    sizes = {p.name: round(p.stat().st_size / 2**20, 1) for d in (fp, q) for p in d.glob("*.onnx")}
    res = {"onnx_fp32": bench_onnx(fp), "onnx_int8": bench_onnx(q), "gguf_q4_k_m": bench_gguf(), "onnx_file_mb": sizes,
           "note": "M2 arm64 CPU. ONNX path uses fp32 KV-cache-less generate via optimum (not the prefix-cached path the live server uses)."}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "llm_onnx_vs_gguf.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
