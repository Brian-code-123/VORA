# Task 0 spikes (2026-09-30)
- `OnlineRecognizer.from_transducer` accepts `rule1_min_trailing_silence`, `rule2_min_trailing_silence`, `rule3_min_utterance_length`, `hotwords_file`, `rule_fsts` (sherpa-onnx 1.13.8). CONFIRMED.
- `OfflineTts.generate(text, sid, speed, callback)`; callback `(samples: float32 ndarray, progress: float) -> int`, non-zero return stops early. CONFIRMED → Task 4 uses callback.
- `fastembed` has NO `multilingual-e5-small`. Available: `BAAI/bge-small-zh-v1.5`, `BAAI/bge-small-en-v1.5`, `paraphrase-multilingual-MiniLM-L12-v2`, `multilingual-e5-large`. RULING: try bge-small-zh-v1.5 first (small), fall back to paraphrase-multilingual-MiniLM-L12-v2 if top-3 <80%. Query prefix "query: " dropped (bge-zh uses an instruction only for retrieval of short queries; not needed).
- Sizes (int8 runtime weights): asr_en 41.6 MB, asr_zh 24.1 MB pass ≤50. tts_zh 19.7 MB pass ≤30. tts_en fp32 60.3 MB FAIL → INT8 in Task 4. Qwen2.5-0.5B Q4_K_M 468.6 MB (LLM RSS reported separately).
- `llama-cpp-python` 0.3.35 built CPU-only (`CMAKE_ARGS=-DGGML_METAL=OFF`).
- `streaming-zipformer-small-bilingual-zh-en-2023-02-16` repo returns 401 (not public under that id) → dropped; two-model setup stays.
- ASR eval data: decided in Task 10 (LibriSpeech test-clean + FLEURS cmn_hans via HF datasets streaming).

# Remediation spikes (2026-10-02)
## English TTS candidates (M2 CPU, 20 unique clauses, load ~7, so absolute numbers inflated; ratios are what matter)
| Candidate | Size | p50 / p95 per clause | RTF | Verdict |
|---|---|---|---|---|
| Kitten nano v0.2 fp16 (Apache-2.0, sherpa `OfflineTtsKittenModelConfig`) | 23.8 MB | 1190 / 1795 ms | 0.375 | rejected: 3x slower than Piper int8 |
| Piper lessac-low int8 | 18.7 MB | 422 / 628 ms | 0.212 | kept (size gate) |
| Piper lessac-low fp32 | 63 MB | 212 / 448 ms | 0.105 | fails 30 MB gate |
| Piper fp16 (onnxconverter-common) | 32.0 MB | n/a | n/a | rejected: >30 MB and ORT refuses to load it (Cast type error) |
Latency scales with audio length (+ small fixed cost): 1 word ~100-170 ms, 2 words ~160-225 ms, 3 words ~240-330 ms, full clause ~430-540 ms (int8, threads=2). threads=1/4 no better.
=> G2 approach: 1-word first chunk, 3-word second chunk (chunker `first_words=1, second_words=3`); measured stall between chunks p95 0.12 s.
## zh TTS lexicon defect (huayan x_low)
`lexicon.txt` uses the phoneme U+032A (combining bridge) that `tokens.txt` lacks, so sherpa skipped every z/c/s syllable (自 次 词 子 字 私 …). `ensure_patched_lexicon` drops tokens missing from tokens.txt (derived `lexicon.patched.txt`). ASR round-trip on "这个词很有意思": before "这歌很有意有意", after "这个词汉有意思".
