# Task 0 spikes (2026-09-30)
- `OnlineRecognizer.from_transducer` accepts `rule1_min_trailing_silence`, `rule2_min_trailing_silence`, `rule3_min_utterance_length`, `hotwords_file`, `rule_fsts` (sherpa-onnx 1.13.8). CONFIRMED.
- `OfflineTts.generate(text, sid, speed, callback)`; callback `(samples: float32 ndarray, progress: float) -> int`, non-zero return stops early. CONFIRMED → Task 4 uses callback.
- `fastembed` has NO `multilingual-e5-small`. Available: `BAAI/bge-small-zh-v1.5`, `BAAI/bge-small-en-v1.5`, `paraphrase-multilingual-MiniLM-L12-v2`, `multilingual-e5-large`. RULING: try bge-small-zh-v1.5 first (small), fall back to paraphrase-multilingual-MiniLM-L12-v2 if top-3 <80%. Query prefix "query: " dropped (bge-zh uses an instruction only for retrieval of short queries; not needed).
- Sizes (int8 runtime weights): asr_en 41.6 MB, asr_zh 24.1 MB pass ≤50. tts_zh 19.7 MB pass ≤30. tts_en fp32 60.3 MB FAIL → INT8 in Task 4. Qwen2.5-0.5B Q4_K_M 468.6 MB (LLM RSS reported separately).
- `llama-cpp-python` 0.3.35 built CPU-only (`CMAKE_ARGS=-DGGML_METAL=OFF`).
- `streaming-zipformer-small-bilingual-zh-en-2023-02-16` repo returns 401 (not public under that id) → dropped; two-model setup stays.
- ASR eval data: decided in Task 10 (LibriSpeech test-clean + FLEURS cmn_hans via HF datasets streaming).
