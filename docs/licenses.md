# Licenses of components and data

Status as of the real-speech round (2026-10). "Permissive" = Apache-2.0 / MIT / BSD / CC-BY / public domain.

## Shipped components
| Component | License | Note |
|---|---|---|
| sherpa-onnx | Apache-2.0 | ASR + TTS runtime |
| Streaming zipformer en-20M, zh small CTC (k2-fsa/icefall) | Apache-2.0 | ASR models |
| Qwen2.5-0.5B-Instruct (+ GGUF Q4_K_M) | Apache-2.0 | LLM (Qwen3-0.6B, Apache-2.0, was A/B tested and not adopted) |
| llama-cpp-python / llama.cpp | MIT | |
| BAAI/bge-small-zh-v1.5, BAAI/bge-small-en-v1.5 via fastembed | MIT | one embedder per language |
| faiss-cpu, rank_bm25, jieba, rapidfuzz | MIT / Apache-2.0 / MIT / MIT | |
| FastAPI, uvicorn | MIT / BSD | |
| **English voice: Piper VITS `en_US-ljspeech-medium`** (csukuangfj/vits-piper-en_US-ljspeech-medium) | **public domain** (LJ Speech dataset, trained from scratch, model card) | default since this round. int8 19.3 MB |
| English voice (optional, `VORA_TTS_EN_VOICE=lessac`): Piper `en_US-lessac-low` | Blizzard 2013 Lessac licence (custom, not OSI / CC) | not permissive: kept only as a faster fallback, not downloaded by default |
| **Chinese voice: Piper VITS `zh_CN-huayan-x_low`** (csukuangfj/vits-piper-zh_CN-huayan-x_low) | dataset licence **unknown** (Piper model card) | **Documented exception, user ruling:** the only zh voice found that fits the 30 MB limit and the first-chunk latency. Replace before any redistribution. Swap path (untested): `vits-zh-aishell3` is Apache-2.0 but 40 MB, over the 30 MB limit |
| **espeak-ng / espeak-ng-data** (phonemizer bundled in the sherpa-onnx Piper packages) | **GPL-3.0-or-later** | used at runtime, unmodified. Redistributing the Docker image or the data directory needs legal review. This is separate from the voice licences above |

Exact repositories and pinned revisions are in `models/MANIFEST.json` / `scripts/fetch_models.py` (sherpa-onnx-streaming-zipformer-en-20M-2023-02-17,
sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01, sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23 (older zh model, still fetched),
vits-piper-en_US-ljspeech-medium, vits-piper-zh_CN-huayan-x_low, Qwen2.5-0.5B-Instruct-GGUF).

## Evaluation data and tools (not shipped, except the UI sample clips)
| Item | License | Note |
|---|---|---|
| MInDS-14 (PolyAI) en-US / en-GB / en-AU / zh-CN | CC-BY-4.0 | real telephone-style banking requests; attribution: Gerz et al., "Multilingual and Cross-Lingual Intent Detection from Spoken Data" (EMNLP 2021). Six short clips ship in `client/samples/real/` with `ATTRIBUTION.md` |
| LibriSpeech (clean, other) | CC-BY-4.0 | Panayotov et al. |
| FLEURS | CC-BY-4.0 | Conneau et al. |
| AISHELL-1 | Apache-2.0 | Beijing Shell Shell |
| DEMAND noise (via HF mirror `verbreb/demand_noise_subset_16k`) | CC-BY-4.0 at the source (Thiemann et al. 2013); the mirror card states no licence | internal evaluation only, never redistributed |
| silero VAD | MIT | dev-only reference for the latency cross-check |
| GTCRN speech enhancement (sherpa-onnx release) | Apache-2.0 | optional denoiser, off by default |
| UTMOS22 (SpeechMOS) | MIT | evaluation only |
| VORA Box KB, VORA Bank (demo) KB, glossary, eval questions | Ours (fictional products) | |
