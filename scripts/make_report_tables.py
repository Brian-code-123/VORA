"""Render docs/report.md tables from results/*.json (numbers are generated, never typed)."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"


def j(name: str) -> dict:
    return json.loads((R / name).read_text())


def ms(d: dict, key: str) -> str:
    return f"{d[key]['p50']:.0f} / {d[key]['p95']:.0f}"


def tables() -> str:
    b, base, asr, rag, mos, mem, onnx = (j(x) for x in ("bench.json", "baseline.json", "asr.json", "rag.json", "tts_mos.json", "memory.json", "llm_onnx_vs_gguf.json"))
    a, o = b["latency_ms_audio_mode"], b["latency_ms_oracle_text"]
    en = [x for x in b["rows"] if x["lang"] == "en"]
    zh = [x for x in b["rows"] if x["lang"] == "zh"]
    ent = sorted(x["total"] for x in en)
    en_p50, en_p95 = ent[len(ent) // 2], ent[max(0, int(len(ent) * 0.95) - 1)]
    zh_ok = sum(bool(x["context"]) for x in zh)
    distinct = len({x["q"] for x in b["rows"]})
    out = [f"*Host: {b['host']['system']}, {b['host']['cores']} cores, CPU-only, 1-min load average {b['host'].get('loadavg_1m', '?')} (other apps were running: numbers are noisy). Apple-Silicon Mac (arm64), not x86 and not a Raspberry Pi. N={b['n']} turns from {distinct} distinct questions (each repeated).*", "",
           "**Latency (ms, p50 / p95)**", "", "| Stage | Budget | Measured |", "|---|---|---|",
           f"| ASR endpoint wait (speech end → final text; per-chunk decode ≤300 is tested separately) | – | {ms(a, 'asr_final')} |",
           f"| Retrieval + LLM first token (oracle text) | ≤500 | {ms(o, 'rag_first_token')} |",
           f"| TTS first chunk (oracle text) | ≤200 | {ms(o, 'tts_first_chunk')} |",
           f"| RAG+LLM+TTS after final text (oracle) | | {ms(o, 'total')} |",
           f"| **End-to-end estimate** (ASR endpoint + oracle) | ≤1500 | **{b['estimated_total_ms']['p50']:.0f} / {b['estimated_total_ms']['p95']:.0f}** |",
           f"| End-to-end measured, English audio only (n={len(en)}) | ≤1500 | {en_p50:.0f} / {en_p95:.0f} |",
           f"| End-to-end measured, Chinese audio (n={len(zh)}) | ≤1500 | not meaningful: {zh_ok} of {len(zh)} turns retrieved anything (the ASR misheard the synthetic Chinese speech), so all took the fast 'not sure' path |",
           "", "**Streaming vs batch baseline (same models, same questions)**", "", "| | p50 | p95 |", "|---|---|---|",
           f"| Batch: endpoint + retrieve + full LLM + full TTS | {base['batch_total_ms']['p50']:.0f} | {base['batch_total_ms']['p95']:.0f} |",
           f"| Streaming (estimate) | {base['streaming_total_estimated_ms']['p50']:.0f} | {base['streaming_total_estimated_ms']['p95']:.0f} |",
           "", "**ASR accuracy (streaming wrapper, 50 clips per set)**", "", "| Set | Metric | Clean | 10 dB noise | RTF |", "|---|---|---|---|---|"]
    for k in ("en_librispeech_clean", "en_fleurs", "zh_fleurs"):
        c, n = asr[f"{k}/clean"], asr[f"{k}/noisy_10dB"]
        out.append(f"| {k} | {c['metric']} | {c['value']*100:.1f}% | {n['value']*100:.1f}% | {c['rtf']} |")
    out += ["", "**RAG and TTS**", "", "| Metric | Result | Target |", "|---|---|---|",
            f"| Top-3 retrieval, dev (n={rag['top3_dev']['n']}) | {rag['top3_dev']['acc']*100:.1f}% | ≥80% |",
            f"| Top-3 retrieval, held-out reworded (n={rag['top3_heldout']['n']}) | {rag['top3_heldout']['acc']*100:.1f}% | ≥80% |",
            f"| Off-topic queries rejected (n={rag['offtopic_rejected']['n']}) | {rag['offtopic_rejected']['rate']*100:.0f}% | – |",
            f"| Answer keyword faithfulness (n={rag['faithfulness']['n']}) | {rag['faithfulness']['acc']*100:.0f}% | – |",
            f"| TTS MOS proxy (UTMOS22) en / zh | {mos['en']['mean_mos_proxy']} / {mos['zh']['mean_mos_proxy']} | ≥3.5 |",
            "", "**Memory (RSS added by loading, M2)**", "", "| Component | MB |", "|---|---|",
            f"| ASR + RAG + TTS in one process (incl. shared library imports) | {mem['asr_rag_tts']:.0f} (target ≤500) |",
            f"| LLM Qwen2.5-0.5B Q4_K_M | {mem['llm']:.0f} |",
            "", "**LLM runtime: ONNX vs GGUF (same prompt, M2 CPU)**", "", "| Runtime | First token (ms) | Decode tok/s | Size |", "|---|---|---|---|",
            f"| ONNX fp32 (optimum) | {onnx['onnx_fp32']['first_token_ms_median']:.0f} | {onnx['onnx_fp32']['decode_tok_s_median']:.1f} | – |",
            f"| ONNX int8 (optimum) | {onnx['onnx_int8']['first_token_ms_median']:.0f} | {onnx['onnx_int8']['decode_tok_s_median']:.1f} | {onnx['onnx_file_mb']['model_quantized.onnx']:.0f} MB |",
            f"| GGUF Q4_K_M (llama.cpp, live path) | {onnx['gguf_q4_k_m']['first_token_ms_median']:.0f} | {onnx['gguf_q4_k_m']['decode_tok_s_median']:.1f}* | 469 MB |",
            "", "*GGUF tok/s includes prefill in its denominator, ONNX excludes first token: not like for like.*"]
    return "\n".join(out)


def main() -> None:
    p = ROOT / "docs" / "report.md"
    txt = p.read_text()
    new = re.sub(r"(<!-- TABLES:START -->).*?(<!-- TABLES:END -->)", lambda m: f"{m.group(1)}\n{tables()}\n{m.group(2)}", txt, flags=re.S)
    p.write_text(new)
    print("tables updated")


if __name__ == "__main__":
    main()
