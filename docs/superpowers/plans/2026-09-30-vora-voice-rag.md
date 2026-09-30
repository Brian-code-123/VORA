# VORA Streaming Voice-RAG Plan (v5, caveman rewrite)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use `- [ ]`.

**Goal:** Zero-cost CPU-only streaming Mic → ASR → RAG → LLM → TTS → Speaker. Web demo, benchmarks, edge Docker, 2-3 page report.

**Architecture:** One FastAPI WebSocket per user. Browser send 16 kHz PCM16 up. Server: sherpa-onnx streaming ASR → speculative retrieval on stable partials → Qwen tokens → sentence chunker → Piper TTS → PCM + JSON events down. Bounded queues = backpressure. New speech cancel old reply (barge-in).

**Spec:** Brief in chat. Copy to `docs/brief.md` in Task 0.

## Tech Stack (all free; checked with Context7 + HF file sizes)
| Layer | Pick | Why |
|---|---|---|
| ASR | `sherpa-onnx` `OnlineRecognizer.from_transducer` (`accept_waveform`/`is_ready`/`decode_stream`/`is_endpoint`/`get_result`/`reset`). Models: `streaming-zipformer-en-20M-2023-02-17` int8 ≈43.6 MB, `streaming-zipformer-zh-14M-2023-02-23` int8 ≈25.3 MB | Already ONNX+INT8, streaming. **Bilingual zh-en 2023-02-20 int8 encoder = 182 MB → FAIL 50 MB, rejected.** Per-session `lang` toggle, one recognizer active. |
| Embed | `fastembed` int8 `multilingual-e5-small` (fallback `bge-small-zh-v1.5`) | ONNX, bilingual |
| Vector | `faiss-cpu` IndexFlatIP + `rank_bm25` + `jieba` hybrid | KB tiny, exact search fast, BM25 catch product codes |
| LLM | Qwen2.5-0.5B-Instruct Q4_K_M GGUF, `llama-cpp-python` `create_chat_completion(stream=True)`. ONNX (optimum) benchmark only | Apache-2.0, ≤1B, fast |
| TTS | `sherpa-onnx` `OfflineTts` Piper VITS (en `lessac-low`, zh `huayan` x_low), `quantize_dynamic` INT8, per-clause chunks | Small VITS |
| Server | FastAPI WebSocket | Allowed, simple |
| Client | Static HTML + AudioWorklet, no build | Free |
| Eval | `jiwer`, UTMOS (dev extra), `pytest`, `psutil` | Free |
| Ship | Docker buildx arm64+amd64, GitHub Actions free | $0 |

## Decisions Locked (user)
- Mandarin + English. Cantonese out of scope → limitation.
- No Pi/Jetson. Pi numbers **emulated** (`docker --cpus=4 --memory=3g`, arm64). Pi LLM ≤500 ms marked unverified. Jetson out.
- KB = fictional "VORA Box" zh+en FAQ, we write it. 40-Q eval.
- LLM = Qwen2.5-0.5B + extractive fallback: answer shares no keyword with top-1 chunk → speak chunk first sentence.
- Single-turn only. No chat memory.

## Global Constraints
- In 16 kHz mono PCM16. TTS out 16 kHz mono PCM16.
- ASR ≤50 MB per model, TTS ≤30 MB, LLM ≤1B. (Two ASR models ≈69 MB total; read limit as per-model, say so in report. Small-bilingual candidate if strict total needed.)
- ASR chunk 320 ms, decode <300 ms. RAG+first token ≤500 ms. TTS first chunk ≤200 ms. End-of-speech→first audio ≤1.5 s.
- ASR/TTS RTF ≤1.0 on Pi 4 (emulated). RAM ≤500 MB (ASR+RAG+TTS). KB ≤1 GB. ≥2 users.
- WER/CER ≤15%. MOS ≥3.5. Top-3 ≥80%.
- Live path streaming only. CPU only.
- `speech_end` = last voiced input frame (endpoint wait counts).
- Bind `127.0.0.1` default. Max 4 sessions, 64 KB frame, 60 s idle. Mic needs HTTPS or localhost.
- `uv.lock` pin deps. Models pinned sha256 + license in `models/MANIFEST.json`.
- Zero paid service. Licenses Apache/MIT/CC-BY only → `docs/licenses.md`.

## Latency Budget (end-of-speech → first audio ≤1.5 s)
| Stage | Budget | How |
|---|---|---|
| Endpoint | ≤0.5 s | rule2 silence 0.4 s |
| Retrieval | ~0 prefetched / ≤50 ms cold | speculative RAG |
| LLM first clause | ≤0.5 s | short prompt, KV prefix cache, 8-CJK/12-Latin first-clause split |
| TTS first chunk | ≤0.2 s | clause chunk, low voices |
| Net+playback | ≤0.1 s | local/LAN |

Sum ≈1.3-1.4 s. Report two numbers: **TTFA** (with optional filler "好的，"/"Sure,", `filler=false` default) and **first content audio**. Never mix.

## Honest Risks (go in report)
0. ASR 50 MB gate: default bilingual fail. Two-model per-model reading stated.
0b. 14M/20M models may miss WER/CER 15% on hard/noisy sets. Report clean + 10 dB noisy. No cherry-pick.
0c. Eval circular (we write KB+Qs). Keep held-out paraphrase split, report separate.
1. Pi 4 LLM first token ≤500 ms unrealistic (~20-50 tok/s prompt eval). Mitigate: prompt ≤120 tok, prefix KV, prefetch. Claim ≤500 ms on x86 only; Pi emulated + labelled.
2. RAM ≤500 MB tight. Qwen Q4_K_M ≈398 MB alone. Task 0 measure. Fallback Q4_0/Q3_K_M or report LLM RSS separate.
3. LLM live path = GGUF not ONNX. ONNX benchmarked, faster ship.
4. Apple Silicon Mac = arm64, not x86. Label every row by host.

## Review Focus (each has test in owning task)
- Silence/empty transcript → no LLM call. (T7)
- Mixed zh+en → text forwarded verbatim; TTS voice per script run. (T4, T7)
- Barge-in → cancel within one chunk, drain queues. (T7)
- Slow client → bounded queue, other user not starved. (T8)
- Own TTS echo → no self-barge-in. (T7, T9)
- No retrieval hit → fixed "don't know", no hallucination. (T5, T7)

---

### Task 0: Scaffold + feasibility gates + model fetch
**Files:** Create `pyproject.toml` (uv, `[eval]` extra), `tests/__init__.py`, `scripts/fetch_models.py`, `scripts/check_gates.py`, `models/.gitignore`, `docs/brief.md`, `docs/superpowers/plans/2026-09-30-vora-voice-rag.md` (copy of plan)
**Interfaces:** Produces `models/{asr_en,asr_zh,tts_en,tts_zh,llm,embed}/`; `check_gates.evaluate(sizes: dict[str,int]) -> dict[str,dict]`; script exit 0/1 with gate table.
- [ ] Step 1: `tests/test_gates.py::test_gate_table_flags_oversize` — fake 60 MB asr → `evaluate(...)["asr"]["ok"] is False`.
- [ ] Step 2: `pytest tests/test_gates.py -v` → FAIL.
- [ ] Step 3: Implement `evaluate` (ASR 50 MB, TTS 30 MB) + `fetch_models.py` (sha256 in MANIFEST). Spike, record in `docs/spikes.md`: Python `OfflineTts.generate(..., callback=)` exists? `from_transducer` accepts `rule1_min_trailing_silence`/`rule2_min_trailing_silence`? `fastembed` lists `multilingual-e5-small`? Licenses (Qwen Apache-2.0, Piper voices, e5 MIT). Measure `streaming-zipformer-small-bilingual-zh-en-2023-02-16` int8 size. Measure Piper huayan medium vs x_low, fp32 vs int8. Pick eval data: LibriSpeech test-clean (en) + AISHELL-1/FLEURS cmn_hans (zh) via HF `datasets` streaming (no 15 GB download). Build `llama-cpp-python` with `CMAKE_ARGS="-DGGML_METAL=OFF"`, `n_gpu_layers=0`.
- [ ] Step 4: Test PASS. `python scripts/fetch_models.py && python scripts/check_gates.py`. Gate fail → stop, choose fallback.
- [ ] Step 5: Commit `chore: scaffold, model fetch, size gates`.

### Task 1: Config + metrics
**Files:** Create `src/vora/config.py`, `src/vora/metrics.py`, `tests/test_metrics.py`
**Interfaces:** `Settings` (pydantic-settings: `asr_chunk_ms=320`, `top_k=3`, `min_score`, `max_ctx_tokens=120`, `queue_max=8`, `filler=False`); `LatencyTrace.mark(name: str)`, `.report() -> dict[str,float]` ms keys `asr_final`,`rag_first_token`,`tts_first_chunk`,`total`.
- [ ] Step 1: `test_trace_report_orders_marks` (fake clock: `speech_end`,`first_token`,`first_audio` → deltas).
- [ ] Step 2: FAIL. Step 3: implement, injected `time.perf_counter`. Step 4: PASS. Step 5: commit.

### Task 2: Streaming ASR wrapper
**Files:** Create `src/vora/asr.py`, `tests/test_asr.py`; test wavs = model repo `test_wavs/`.
**Interfaces:** `AsrSession(lang: Literal["zh","en"])`: `feed(pcm16: bytes) -> list[AsrEvent]`, `reset()`. `AsrEvent(kind: Literal["partial","final"], text: str, stable: bool)`. `load_recognizers(settings) -> dict[str, OnlineRecognizer]` (`enable_endpoint_detection=True`, `rule1_min_trailing_silence=0.6`, `rule2_min_trailing_silence=0.4`, `greedy_search`, `num_threads=2`). `feed` sync → call via ASR `ThreadPoolExecutor(2)`, never event loop.
- [ ] Step 1 tests: `test_en_wav_streams_partials_then_final` (320 ms slices, ≥2 growing partials, 1 final, WER ≤0.15); `test_zh_wav_final_cer_le_15pct`; `test_silence_emits_no_final`; `test_30s_monologue_force_final`; `test_text_lowercased_and_stripped`; `test_chunk_decode_under_300ms` (p95); `test_endpoint_within_700ms_of_speech_end`.
- [ ] Step 2: FAIL. Step 3: int16→float32/32768; 0.5 s tail only on endpoint; `reset` after final; `stable=True` when text same 2 chunks. Step 4: PASS. Step 5: commit.

### Task 3: Sentence chunker
**Files:** Create `src/vora/chunker.py`, `tests/test_chunker.py`
**Interfaces:** `SentenceChunker.push(token: str) -> list[str]`, `.flush() -> list[str]`. Split `。！？!?.\n`; first chunk also split on `，,；;` at ≥12 Latin / ≥8 CJK chars; force flush at 60 CJK / 120 Latin with no punctuation; no split inside decimals or abbreviations.
- [ ] Tests: `test_first_clause_emitted_early`, `test_decimal_not_split` (`"版本3.5很好。"`→1), `test_flush_returns_tail`, `test_long_unpunctuated_run_force_flushed`, `test_empty_and_whitespace_tokens`.
- [ ] Steps 2-5: FAIL → implement → PASS → commit.

### Task 4: Streaming TTS + INT8 quantize
**Files:** Create `src/vora/tts.py`, `scripts/quantize_tts.py`, `tests/test_tts.py`
**Interfaces:** `Tts.synth(text: str) -> Iterator[bytes]` (PCM16 16 kHz, chunk ≤200 ms; use `generate(callback=)` if spike says yes, else per-clause). `lang_of(text) -> Literal["zh","en"]` (CJK ratio >0.3). `sanitize_for_tts(text: str) -> str` (strip markdown, emoji, URLs; spell codes `X200` letter by letter; digits per language). `split_by_script(text: str) -> list[tuple[str,str]]` — zh Piper cannot say Latin words, so each script run uses own voice, audio concatenated.
- [ ] Tests: `test_sanitize_strips_markdown_emoji_url`; `test_split_by_script_mixed_sentence` (`"請問 VORA Box 支援 WiFi 嗎"`→zh/en/zh/en/zh); `test_lang_of_mixed`; `test_first_chunk_under_200ms_short_sentence`; `test_output_is_16k_mono_pcm16`; `test_quantized_models_under_30mb`.
- [ ] Steps 2-5: FAIL → implement (lessac-low native 16 kHz; else `scipy.signal.resample_poly`) → PASS → commit.

### Task 5: RAG ingest + hybrid retrieval
**Files:** Create `src/vora/rag/{ingest.py,store.py,retriever.py}`, `kb/*.md` (≥30 chunks zh+en), `kb/glossary.json` (product names, codes, ASR-style variants), `eval/rag_qa.jsonl` (≥40 answerable, `split: dev|heldout` heldout reworded; ≥10 off-topic `chunk_id: null`), `tests/test_rag.py`
**Interfaces:** `ingest(kb_dir, index_dir)`; `Retriever.search(query: str, k: int=3) -> list[Hit]`; `Hit(chunk_id, text, score)`; `[]` if best < `min_score`. `normalize_query(text) -> str` fuzzy-maps ASR-mangled terms via `rapidfuzz` (`"vora x 200"`→`"VORA-X200"`). Hybrid `0.7*dense + 0.3*bm25_norm`, jieba, e5 `"query: "` prefix. `min_score` grid-searched on dev (max top-3, reject ≥90% off-topic) → config.
- [ ] Tests: `test_top3_accuracy_ge_80pct`; `test_heldout_top3_reported`; `test_exact_product_term_ranks_first` (`"VORA-X200 保養期"`); `test_asr_mangled_term_normalised`; `test_offtopic_returns_empty`; `test_search_under_20ms` (warm).
- [ ] Steps 2-5: FAIL → implement (chunk 200-300 chars, overlap 40, `faiss.write_index` + `chunks.json`) → PASS → commit. <80% → tune before moving on.

### Task 6: Streaming LLM (+ ONNX compare)
**Files:** Create `src/vora/llm.py`, `scripts/export_onnx_llm.py`, `tests/test_llm.py`
**Interfaces:** `Llm.stream(question: str, hits: list[Hit]) -> Iterator[str]`; `cancel()`. Fixed system prompt first (prefix KV). Answer only from context, ≤2 sentences, reply in question language. Empty hits → fixed "我不確定"/"I don't know", no model call. Extractive fallback per Decisions.
- [ ] Tests: `test_streams_multiple_tokens_incrementally`; `test_cancel_stops_within_one_token`; `test_empty_hits_no_model_call`; `test_cold_retrieval_plus_first_token_under_500ms` (`perf`); `test_warm_first_token_under_300ms`; `test_answer_uses_context_term`; `test_extractive_fallback_when_no_keyword_overlap`; `test_lock_timeout_returns_busy`; `test_two_users_alternating_keeps_prefix_cache_hit`.
- [ ] Steps 2-5: implement `n_ctx=1024, n_threads=4, n_batch=128, temperature=0.2, max_tokens=80, n_gpu_layers=0`. `LlamaRAMCache(capacity_bytes=64<<20)` — llama-cpp-python reuses KV prefix of last prompt only, two alternating users thrash. Cancel = `close()` stream generator. Llama not thread-safe → one `threading.Lock`, 3 s acquire timeout → "System busy". Then `export_onnx_llm.py` (optimum int8), record tok/s + first token vs GGUF in `docs/bench.md`, ship faster. Commit.

### Task 7: Pipeline orchestrator
**Files:** Create `src/vora/pipeline.py`, `tests/test_pipeline.py`
**Interfaces:** Consumes Tasks 2-6 + `LatencyTrace`. `Pipeline.on_audio(pcm: bytes)` async; output `asyncio.Queue[OutMsg]` (maxsize `queue_max`), `OutMsg = ("json", dict) | ("audio", bytes)`. Events: `cancel` (client flush audio now), `partial`, `final`, `context`, `token`, `audio_start`, `metrics`. Prefetch search on stable partials ≥6 chars, ≤1 per 500 ms, single-thread executor (ASR keep cores); cache by text; drop if final differs >30% edit distance. Non-empty final → LLM→chunker→TTS. Partial persisting ≥2 chunks during reply → cancel+drain (never raw energy). New final while old turn running → cancel old first. Executors: ASR, LLM, TTS each own; LLM/TTS single worker + lock.
- [ ] Tests (fakes): `test_empty_final_no_llm_call`; `test_prefetch_reused_when_final_matches_partial`; `test_prefetch_rate_limited`; `test_barge_in_cancels_and_drains`; `test_single_chunk_blip_does_not_barge_in`; `test_new_final_cancels_previous_turn`; `test_event_loop_lag_under_50ms_during_decode`; `test_mixed_zh_en_question_forwarded_verbatim`; `test_no_hits_speaks_fallback`; `test_filler_off_by_default_and_reported_separately`; `test_total_latency_recorded`.
- [ ] Steps 2-5: FAIL → implement → PASS → commit. Doc: two users serialize on LLM.

### Task 8: WebSocket server + backpressure
**Files:** Create `src/vora/server.py`, `tests/test_server.py`
**Interfaces:** `GET /health` → 503 until models loaded **and** one warmup inference each. `WS /ws`: up = binary PCM16 + JSON `{"type":"config"|"stop"}` (config carries `lang`); down = JSON events + binary PCM16. One `Pipeline` per connection, models in `lifespan`. Chain LLM → bounded sentence queue → TTS worker → bounded audio queue → sender. **LLM lock released when token generation ends, never held while waiting on client queue** → slow client cannot starve other user. One JSON log line per turn with all stage timings.
- [ ] Tests (Starlette `TestClient.websocket_connect`): `test_ws_roundtrip_en_question_returns_audio`; `test_two_concurrent_sessions_independent_transcripts`; `test_slow_client_queue_bounded`; `test_slow_client_does_not_starve_other_session` (B first-audio ≤2× baseline); `test_disconnect_cancels_work`; `test_fifth_session_rejected`; `test_oversize_frame_closes_1009`; `test_idle_60s_closes`; `test_health_503_until_warm`.
- [ ] Steps 2-5: FAIL → implement (`asyncio.gather` receive + sender, catch `WebSocketDisconnect`) → PASS → commit.

### Task 9: Web client
**Files:** Create `client/index.html`, `client/worklet.js`; `StaticFiles` mount in `server.py`.
**Interfaces:** AudioWorklet → 16 kHz PCM16, 100 ms frames. Playback: scheduled `AudioBufferSource`, ~80 ms jitter lead. On `cancel`: stop all sources, reset clock. UI: live partial, context, reply, latency panel ("Response time: X.Xs", TTFA + content audio). `getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,channelCount:1}})`. Secure context: localhost ok; Pi on LAN → mkcert HTTPS (`scripts/deploy_pi.sh --https`). Language toggle zh/en.
- [ ] Step 1: Playwright `tests/e2e/test_client.py::test_page_loads_and_connects_ws` with Chromium `--use-file-for-fake-audio-capture=tests/data/en.wav` → partial text appears, audio bytes received.
- [ ] Steps 2-5: FAIL → implement → PASS → commit. Manual screenshot via built-in browser.

### Task 10: Evaluation + benchmarks
**Files:** Create `scripts/{eval_asr.py,eval_rag.py,eval_tts.py,bench.py,baseline_batch.py}`, `docs/bench.md`
**Interfaces:** Scripts write `results/*.json` incl. CPU model/cores/OS. `bench.py`: latency table (ASR/RAG/TTS/total, p50/p95, N=30), RTF ASR/TTS, RSS via `psutil`. `baseline_batch.py`: wait full utterance → full LLM → full TTS. Torch/UTMOS/jiwer in `[eval]` extra, not runtime image. Label rows by host (Mac arm64 vs x86 CI vs emulated Pi).
- [ ] Step 1 tests: `test_rtf_and_percentile`; `tests/test_memory.py::test_asr_rag_tts_rss_under_500mb` (`perf`, LLM RSS separate); `eval_rag.py` also 20 answer-keyword faithfulness checks.
- [ ] Steps 2-5: FAIL → implement → PASS. Run: 50 zh + 50 en ASR (WER en, CER zh; clean + 10 dB noisy); TTS UTMOS 20 sentences (proxy, human survey if time); streaming vs batch latency. Commit results.

### Task 11: Edge deploy
**Files:** Create `docker/Dockerfile`, `docker/compose.yml`, `scripts/deploy_pi.sh`, `.github/workflows/ci.yml`
**Interfaces:** Multi-stage `python:3.11-slim`, models baked by `fetch_models.py`, `buildx --platform linux/arm64,linux/amd64`, healthcheck `/health`, `--https` mkcert mode. `llama-cpp-python` no arm64 wheel → source build 10-20 min under qemu → GitHub arm64 runner or buildx cache. Emulated Pi: `--cpus=4 --memory=3g`. CI: unit tests, skip `perf`.
- [ ] Step 1: `tests/test_docker_smoke.sh` — `docker run` → `curl /health` 200 within 30 s.
- [ ] Steps 2-5: FAIL → implement → PASS → commit. Record image size, idle RSS, 2-user RSS.

### Task 12: Report + demo script
**Files:** Create `docs/report.md` (2-3 pages), `docs/demo-script.md`, `docs/licenses.md`, `scripts/make_report_tables.py`
**Interfaces:** Report mirrors brief: Mermaid diagram, model rationale, streaming details, latency/RTF/WER/MOS/top-3 tables generated from `results/`, deployment guide, **Honest Limitations** (Pi LLM latency, memory, noise, long-form, Cantonese, 2-user LLM serialization, MOS proxy, eval circularity, ASR per-model size reading). Demo script: 2-min video beats, 10-min live outline, trade-off talking points, OBS recording checklist (user records; we script).
- [ ] Step 1: `make_report_tables.py` renders tables from `results/*.json` (generated, not typed). Step 2: run, proofread, commit.

## Review Log
- P1: endpoint latency, secure-context mic, barge-in echo, prefetch contention, cold latency test, WS hardening, memory test, JSON logs.
- P2 (Context7): sherpa streaming API + llama-cpp-python prefix cache/lock checked. TTS sanitise, script-run split, 30 s force-final, cancel event, health warmup, cache thrash fix, lock timeout, extractive fallback, arm64 build.
- P3 (fact check): default ASR fail 50 MB → en-20M + zh-14M. Latency budget, TTFA vs content audio, scaffold first, ASR off event loop, turn-overlap cancel, LLM-lock starvation fix, glossary fuzzy, off-topic calibration, held-out split, Metal off, host labels, 100 ms frames.
- P4: rewrite terse. No scope change.

## Self-Review
- Coverage: ASR, RAG, TTS, pipeline, WS, client, ONNX/quant, Docker, report, demo, baseline, innovation (speculative RAG, first-clause TTS, prefix KV) all mapped.
- Open (Task-0 spikes, docs unconfirmed): Python TTS callback, endpoint kwargs, small-bilingual size, eval dataset choice.
