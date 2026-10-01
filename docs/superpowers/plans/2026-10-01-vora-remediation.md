# VORA Remediation Plan: close unmet targets (v7, caveman, 3 review passes)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use `- [ ]`.

**Goal:** Fix every brief target VORA miss. Latency, TTS first chunk, memory, ASR accuracy, faithfulness, 2 users, deploy proof. Test first. Gate cannot be met → report unmet + evidence. Never edit gate to pass.

**Architecture:** Same pipeline (`src/vora/`). Add: quiet-host guard + gate report, lazy model load, slim retriever, TTS backend abstraction, better zh ASR, optional denoiser / second pass, endpoint hold inside ASR stream, speculative shadow turn, answer guard, shared admission control.

**Spec:** `docs/brief.md`, old plan `docs/superpowers/plans/2026-09-30-vora-voice-rag.md`, state `docs/report.md`, `docs/rulings.md`, `results/*.json`.

## Current state (M2 arm64, busy host, 2026-09-30/10-01)
| Gate | Target | Now |
|---|---|---|
| G1 e2e (speech_end = last voiced frame → first content audio) | p50 ≤1.5 s, p90 ≤1.8 s | est p50 ~1.36 s. en audio measured p50 1.58 s. p95 2-3 s |
| G2 TTS first chunk | ≤200 ms p95, voice ≤30 MB | en int8 230-260 ms. zh ~110 ms |
| G3 ASR+RAG+TTS memory | ≤500 MB USS | 437-638 MB RSS (noisy) |
| G4 ASR | WER/CER ≤15% | LibriSpeech clean 9.7% ok. FLEURS en 28.7%, zh 25.4%. Noisy worse |
| G5 RAG | top-3 ≥80%, faithfulness ≥95% | 92.9/85.7% ok. Faithfulness 90% |
| G6 2 users | both served, bounded | never measured on real models |
| G7 deploy | Docker + Pi proof | Docker never built. Pi never measured |
| G8 licences | permissive | both Piper voices non-permissive / unknown |

## Decisions (defaults)
- Bars: G1 p50 ≤1.5 s and p90 ≤1.8 s, quiet host, all 42 answerable eval questions once (no repeats), per-turn measured e2e (never sum of percentiles). G2 p95 ≤200 ms. G3 USS median of 3 ≤500 MB. G4 en LibriSpeech-clean ≤15%, zh AISHELL-1 ≤15%. FLEURS + noisy reported. G5 faithfulness ≥95% on 40 Q incl. 10 blind negation Q.
- zh TTS licence stay "documented exception". No permissive ≤30 MB zh voice found.
- No Pi → G7 "unverified" unless user run `scripts/pi_bench.sh`.
- Feature default ON only if A/B gate pass. Else ship OFF, document.
- Stop-loss: after Tasks 0-3 re-run gates. Skip optional Tasks 4, 6, 10 when gate already pass.

## Facts checked (2026-10-01; sherpa-onnx 1.13.8, psutil, HF API)
- `psutil.Process().memory_full_info().uss` work on macOS (own process / children).
- `OnlineRecognizer.from_zipformer2_ctc` take `model, tokens, decoding_method, rule1/2/3` endpoint kwargs (greedy only). `OfflineRecognizer.from_transducer` exist (second pass).
- `OfflineTtsKittenModelConfig(model, voices, tokens, data_dir, length_scale)`. `OnlineSpeechDenoiser` + `OnlineSpeechDenoiserConfig(model=...gtcrn)` exist.
- `Xenova/bge-small-zh-v1.5`: `onnx/model_int8.onnx` 23.9 MB + `tokenizer.json`. `AISHELL/AISHELL-1` Apache-2.0, per-speaker tarballs.
- Not verifiable now: GTCRN download URL, Kitten latency on ARM, ctc-small CER. Spike steps cover.

## Downloads (ask user ONCE at start of Task 0. Nothing else download)
| What | Size | Source | Task |
|---|---|---|---|
| `kitten-nano-en-v0_2-fp16` (+espeak data) | ~24 MB | HF `csukuangfj/kitten-nano-en-v0_2-fp16`, Apache-2.0 | 2 |
| `sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01` | 26 MB | HF csukuangfj | 3 |
| AISHELL-1 test speakers S0764-S0916 (3 tar shards) + `aishell_transcript_v0.8.txt` | ~110 MB | HF `AISHELL/AISHELL-1`, Apache-2.0 | 3 |
| GTCRN `gtcrn_simple.onnx` | <1 MB | sherpa-onnx GitHub release asset `speech-enhancement-models/gtcrn_simple.onnx` (not on HF. Verify URL at spike, pin sha256) | 4 |
| `sherpa-onnx-zipformer-small-en-2023-06-26` int8 | ~30 MB | HF csukuangfj | 4 |
| bge-small-zh-v1.5 `onnx/model_int8.onnx` (23.9 MB) + `tokenizer.json` (0.4 MB) | ~25 MB | HF `Xenova/bge-small-zh-v1.5`, pin revision | 1 |

Each pinned revision + sha256 + licence → `models/MANIFEST.json`.

## Global Constraints
- Keep: 16 kHz mono PCM16 I/O. ASR ≤50 MB per model. TTS ≤30 MB. LLM ≤1B. Live path streaming. CPU only. Bind 127.0.0.1. ≥2 users.
- Bench honesty: quiet host only (`hostcheck.is_quiet`). USS not RSS. Real `speech_end`. Host + load recorded. Rows labelled by host.
- Markers registered in `pyproject.toml`: `perf`, `eval`, `integration`. CI run `-m "not perf and not eval and not integration"`. Perf / eval tests skip (with reason) on busy host. Never flake.
- Every fix: failing test first. Watch it fail. Then implement.
- Metrics event + per-turn JSON log gain: `hold_ms`, `denoise_ms`, `secondpass_ms`, `speculated`, `tts_oov`.
- No gate edits. Unmet = evidence in `docs/rulings.md`.

## Review Focus
- Shadow turn: user keep talking / final differ → cancelled, zero audio leaked, LLM lock freed, shadow text NOT in `_spoken`. (T6)
- Lazy load: 16 concurrent first-use calls load once. Language switch mid-session. Unload never hit live stream. Lease released on disconnect. (T1)
- Second pass / denoiser: timeout fall back. Clean WER no regress. Silence stay silent. Second pass never block other sessions' frames. (T4)
- Endpoint hold: complete sentence never delayed. Rambling "and… and…" capped. "uh/um/嗯" ignored. Stream never reset during hold. (T5)
- zh TTS OOV chars: counted + logged, never silent. KB fully covered. (T2)
- Busy host: perf tests skip, bench refuse, results record `quiet`. (T0)

---

### Task 0: Gate report + quiet-host guard + USS + downloads approval
**Files:** Create `src/vora/hostcheck.py`, `scripts/report_gates.py`, `tests/conftest.py`, `tests/test_hostcheck.py`, `tests/test_report_gates.py`. Modify `scripts/bench.py`, `tests/test_memory.py`, `pyproject.toml` (markers).
**Interfaces:** `hostcheck.is_quiet(load1: float | None = None, cores: int | None = None, max_ratio: float = 0.5) -> bool`. `hostcheck.uss_mb(pid: int | None = None) -> float` (psutil `memory_full_info().uss`, fall back to RSS + flag on AccessDenied). `report_gates.evaluate(results_dir: Path) -> list[Gate]`. `Gate(id, name, target, measured, ok: bool | None, quiet: bool)` (None = unverified).
- [ ] Tests: `test_is_quiet_false_when_load_high` (load1=6, cores=8), `test_is_quiet_true_when_idle`, `test_perf_marker_skips_on_busy_host` (conftest hook, monkeypatched), `test_bench_refuses_busy_host_without_force`, `test_gate_fail_and_unverified` (fixture JSONs: latency 1.6 s → G1 False. No docker result → G7 None), `test_report_gates_reads_current_results`.
- [ ] Bench: loop all 42 answerable questions once. Record per-turn e2e. `--force` mark `quiet:false`. Test `test_bench_questions_unique`.
- [ ] Ask user approve Downloads table. Run → FAIL → implement → PASS.
- [ ] `python scripts/report_gates.py --write results/gates_before.json` (baseline). Commit.

### Task 1: Memory ≤500 MB USS
**Files:** Create `src/vora/rag/embed.py`, `scripts/mem_profile.py`, `scripts/quantize_embedder.py`, `tests/test_lazy_models.py`, `tests/test_embed.py`. Modify `src/vora/asr.py`, `src/vora/tts.py`, `src/vora/rag/{store,retriever,ingest}.py`, `src/vora/server.py`, `pyproject.toml`.
**Interfaces:** `Recognizers(settings).get(lang) -> OnlineRecognizer` (lazy, locked, load once). `.lease(lang)` context manager (refcount). `.unload(lang)` refuse while leased. `AsrSession(recognizers: Recognizers, lang)` + `.close()` release lease. `Tts._voice(lang)` lazy same way. `Embedder(model_dir).embed(texts: list[str]) -> np.ndarray` (ORT `enable_cpu_mem_arena=False`, 1 thread, `tokenizers`, CLS pooling, L2 normalise, truncate 512). Ingest + query both use it. Source `Xenova/bge-small-zh-v1.5` `onnx/model_int8.onnx`. `Settings.default_lang="en"`.
- [ ] `mem_profile.py`: USS per step in fresh subprocesses (imports, asr en/zh, rag, tts en/zh, llm) → `results/mem_profile.json`. Cut in order of measured gain.
- [ ] Keep `load_recognizers(settings)` name (now return `Recognizers`). Update `tests/test_asr.py` fixtures + `server.Models`.
- [ ] Embedder edge tests: `test_embed_empty_string_ok`, `test_embed_over_512_tokens_truncated`, `test_embed_batch_equals_single`, `test_embed_zh_en_mixed`.
- [ ] Tests: `test_get_loads_once_under_16_threads`, `test_second_language_loads_on_first_use`, `test_unload_refused_while_leased`, `test_language_switch_keeps_old_until_released`, `test_serve_session_releases_lease_on_disconnect_and_switch`, `test_warmup_loads_only_default_lang`, `test_embedder_int8_cosine_vs_fp32_ge_0_98` (fastembed dev-only oracle), `test_retriever_top3_unchanged` (dev ≥92%, heldout ≥85%, off-topic ≥90% rejected), `test_retriever_import_is_light` (subprocess: `faiss` / `fastembed` / `jieba` not imported if replaced), `test_uss_asr_rag_tts_le_500mb` (perf, quiet, median of 3), `test_uss_stable_over_50_turns` (drift ≤30 MB), `test_first_use_load_of_second_language_under_1s` (perf, reported).
- [ ] Implement: lazy ASR/TTS → int8 Embedder (pinned source) → numpy matmul replace FAISS → char-bigram BM25 if jieba cost >30 MB. After each step: retrieval regression + USS.
- [ ] Still >500 MB → per-component USS into `docs/rulings.md`, G3 unmet. Commit.

### Task 2: English TTS ≤200 ms, licence, zh OOV
**Files:** Modify `src/vora/tts.py`, `src/vora/config.py`, `scripts/fetch_models.py`. Create `tests/test_tts_backends.py`.
**Interfaces:** `class Voice(Protocol): sample_rate: int; def generate(self, text: str, cancel: threading.Event) -> np.ndarray`. `VitsVoice`, `KittenVoice(sid=0)` (resample 24 kHz → 16 kHz inside). `Settings.tts_en_backend: Literal["kitten","piper_int8","piper_fp32"]`.
- [ ] Tests: `test_backend_size_le_30mb[kitten|piper_int8]`, `test_en_first_chunk_p95_le_200ms[backend]` (perf, 20 unique clauses), `test_backend_output_16k_mono`, `test_kitten_reads_model_codes` (synth "VORA X 200" → ASR text has "200" / "two hundred"), `test_mos_proxy_ge_3_5[backend]` (eval), `test_kb_zh_chars_all_in_lexicon`, `test_oov_char_logged_and_counted` (`tts_oov_total`), `test_all_oov_text_no_audio_no_crash`.
- [ ] Rule: pick backend meeting size + p95 + MOS. None → keep piper_int8 + 3-word first chunk, re-measure. Still >200 ms → G2 unmet (en).
- [ ] Reword KB zh sentences with OOV chars (次, 词 …). OOV report → `results/tts_oov.md` (never in `kb/`, `parse_kb` glob it). Update `docs/licenses.md`. Commit.

### Task 3: zh ASR accuracy
**Files:** Modify `src/vora/asr.py`, `src/vora/config.py`, `scripts/fetch_models.py`, `scripts/eval_asr.py`. Create `tests/test_asr_zh.py`. Add `docs/spikes.md` entries.
**Interfaces:** `Settings.asr_zh_model: Literal["zipformer14m","ctc_small"]`. Loader pick layout by files (`model.int8.onnx` = CTC via `from_zipformer2_ctc`, else transducer). `eval_asr.norm_zh(text) -> str` (Chinese numerals → Arabic, punctuation/space removed, scoring only).
- [ ] Spike first: print `from_zipformer2_ctc` signature, confirm endpoint kwargs + decoding options (greedy only). Record in `docs/spikes.md`.
- [ ] Dataset: AISHELL-1 test speakers S0764-S0916 (3 shards + transcript) cached in `data/eval_cache`.
- [ ] Diagnostic: run Piper-huayan synthetic zh questions through BOTH zh models. If ctc-small understand them, zh audio bench (T10 fallback) become meaningful. Record in `docs/spikes.md`.
- [ ] Tests: `test_ctc_small_size_le_50mb`, `test_ctc_streams_partials_then_final` (bundled wavs), `test_norm_zh_numbers` (十五米 == 15米, 二零一七年 == 2017年), `test_zh_cer_aishell_le_15pct` (eval), `test_chunk_decode_p95_le_300ms_ctc` (perf), `test_endpoint_within_700ms_zh`.
- [ ] Compare both zh models on AISHELL + FLEURS (CER, decode p95, USS). Pick lowest CER meeting size / latency. Re-sweep pre-roll (0 / 0.3 / 0.8 s) + rule1 / rule2 for winner. HK Traditional-input limitation stay documented. Commit.

### Task 4: en accuracy + noise (denoiser, second pass) [optional by stop-loss]
**Files:** Create `src/vora/denoise.py`, `src/vora/secondpass.py`, `tests/test_denoise.py`, `tests/test_secondpass.py`. Modify `src/vora/asr.py`, `src/vora/pipeline.py` (`Executors.second`), `src/vora/config.py`.
**Interfaces:** `Denoiser().process(pcm16: bytes) -> bytes` (GTCRN `sherpa_onnx.OnlineSpeechDenoiser`, buffer to `frame_shift_in_samples`, `flush()`, `reset()`), called INSIDE `AsrSession.feed` (same thread). Final `AsrEvent` gain `pcm: np.ndarray | None` (utterance audio, ≤30 s buffer, dropped after use). `SecondPass(lang).rescore(pcm: np.ndarray) -> str | None`. Pipeline await it on separate `Executors.second` (1 worker) with `secondpass_timeout_s`.
- [ ] Denoiser tests: `test_preserves_dtype_and_length_within_one_frame`, `test_chunking_independent` (100 ms vs 320 ms), `test_silence_stays_silent` (peak <100), `test_clean_wer_not_worse_than_0_5pt` (eval), `test_noisy_10db_wer_improves` (eval), `test_added_latency_le_30ms`.
- [ ] Second-pass tests: `test_replaces_text_when_ready`, `test_timeout_falls_back_to_first_pass`, `test_skipped_when_under_0_6s_or_over_30s`, `test_does_not_block_other_session_feeds` (slow fake second pass, other session frames still processed), `test_memory_delta_le_60mb`, `test_wer_gain_ge_3pt_on_fleurs_en` (eval), `test_adds_le_150ms_p95` (perf).
- [ ] Ship each feature ON only if: lift noisy / FLEURS WER, no clean-WER regress, ≤150 ms p95 added, G3 intact. Else default OFF, document. Decide before Task 6 (speculation depend on first-pass text). Commit.

### Task 5: Endpoint hold (inside ASR stream) + endpoint latency
**Files:** Create `src/vora/endpoint.py`, `tests/test_endpoint.py`. Modify `src/vora/asr.py`, `src/vora/config.py`.
**Interfaces:** `looks_incomplete(text: str, lang: str) -> bool`. `strip_disfluency(text: str, lang: str) -> str`. `Settings.hold_ms=500`, `hold_total_cap_ms=1200`, `endpoint_rule2_s=0.4`.
- [ ] Design: endpoint fire + `looks_incomplete(text)` → do NOT emit final, do NOT `reset` (reset hurt next words: measured 31-37% WER on second utterance). Keep decoding same stream. Audio-time counter `hold_samples`. Text grow → keep going, hold counter reset (capped by `hold_total_cap_ms` per utterance). No growth by `hold_ms` → emit final. No merge logic. `hold_ms=0` disable.
- [ ] Tests (scripted fake recognizer, audio-time driven): `test_looks_incomplete` table ("how long is the" T, "what about the m100 and" T, "how long is the warranty" F, "" F, "保修期是" T, "因为" T, "保修期是几年" F, "uh" F after strip), `test_incomplete_endpoint_no_final_no_reset`, `test_continuation_extends_same_utterance`, `test_complete_sentence_final_not_delayed`, `test_hold_released_after_hold_ms`, `test_rambling_capped_at_total_cap`, `test_disfluency_stripped_before_final`, `test_zh_no_space_join`, `test_hold_ms_reported_in_metrics`.
- [ ] Endpoint speed: measure `endpoint_rule2_s` 0.4 vs 0.3 on 50 LibriSpeech + 50 FLEURS clips: wait p95 + extra-final rate. Test `test_extra_final_rate_le_3pct`. Keep 0.3 only if pass. Commit.

### Task 6: Speculative shadow turn [optional by stop-loss]
**Files:** Modify `src/vora/pipeline.py`, `src/vora/config.py`, `src/vora/server.py`. Create `tests/test_shadow.py`.
**Interfaces:** `Pipeline(..., active_sessions: Callable[[], int] = lambda: 1)`. `Settings.speculate: bool`, `speculate_min_cores=6`. `class ShadowTurn(text, task, ev, buffer: list[bytes], hits)`.
- [ ] Behaviour: on stable partial (≥3 words or ≥4 CJK chars), no active turn, `active_sessions()==1`, `os.cpu_count() >= speculate_min_cores`, `not secondpass` unless measured hit-rate ≥60% → start shadow (retrieve + LLM + pre-render first 2 TTS chunks into bounded buffer. Nothing sent, nothing in `_spoken`). Final exactly equal (normalised) → promote: flush buffer, add to `_spoken`, continue. Different, or new unstable partial → cancel, free LLM lock, drop buffer. Held-incomplete state cancel shadow.
- [ ] Tests (fakes): `test_promoted_first_audio_within_50ms_of_final`, `test_cancelled_when_final_differs_no_audio_leaked`, `test_cancelled_when_user_keeps_talking`, `test_not_started_for_short_text`, `test_not_started_when_other_session_active`, `test_not_started_when_cores_below_min`, `test_buffer_bounded_2_chunks`, `test_cancel_releases_llm_lock`, `test_shadow_text_not_in_spoken_until_promoted`, `test_barge_in_discards_shadow`, `test_speculate_off_setting`, `test_shadow_cancelled_when_second_session_joins`, `test_shadow_gives_up_when_admission_busy` (shadow use `try_acquire`, never queue behind real turn).
- [ ] Bench A/B (`--speculate on/off`, same 42 prompts, quiet host): report p50 / p90, hit-rate, CPU waste (cancelled / total). Default ON only if p50 gain ≥150 ms and p90 not worse. Commit.

### Task 7: LLM first token + answer guard
**Files:** Create `src/vora/guard.py`, `tests/test_guard.py`, `eval/faithfulness.jsonl`. Modify `src/vora/llm.py`, `scripts/eval_rag.py`.
**Interfaces (no jieba import: zh use character windows, en use whitespace words; T1 may remove jieba):** `polarity_conflict(question: str, chunk: str, answer_head: str) -> bool`. `is_refusal(answer_head: str) -> bool`. `best_sentence(chunk: str, question: str) -> str`.
- [ ] Guard tests: `test_conflict_cantonese_en`, `test_conflict_cantonese_zh` ("支持粤语吗" / "不支持粤语"), `test_no_conflict_when_negation_unrelated`, `test_no_conflict_non_yes_no_question`, `test_refusal_phrases_en_zh`, `test_best_sentence_prefers_overlap`, `test_double_negation_not_flagged`.
- [ ] LLM tests: `test_guard_replaces_answer_with_extractive_sentence` (fake LLM), `test_prompt_tokens_le_170`, `test_decide_buffer_6_tokens_keeps_fallback`, `test_first_token_p95_le_400ms_unique_prompts` (perf).
- [ ] Eval: 40 Q, 10 negation Q written blind in different wording (held-out). Gate ≥95%. Tune `_DECIDE_TOKENS` 10→6, system prompt ≤40 tokens, context ≤100 tokens. Re-run faithfulness + RAG after each. Commit.

### Task 8: 2 users on real models + shared admission
**Files:** Create `scripts/bench_concurrent.py`, `tests/test_admission.py`. Modify `src/vora/pipeline.py` (`Executors.admission`), `src/vora/server.py`.
**Interfaces:** `Admission(max_waiting: int)` with `try_acquire() -> bool`, `release()`. Turn take slot BEFORE submitting `llm_worker`, release in `finally`. Over cap → immediate busy reply. `Settings.max_waiting_turns=2`.
- [ ] Tests: `test_third_waiting_turn_gets_busy_immediately`, `test_slot_released_on_cancel_disconnect_and_error`, `test_two_sessions_different_languages_lazy_load_once`, `test_two_session_p50_le_2x_single` (perf, real), `test_two_sessions_uss_delta_le_150mb` (perf).
- [ ] `bench_concurrent.py` (2 sessions, 42 unique prompts each) → `results/concurrent.json`. Commit.

### Task 9: Deployment proof
**Files:** Create `tests/test_dockerfile_static.py`, `scripts/pi_bench.sh`. Modify `docker/Dockerfile`.
- [ ] Static tests (no Docker): `test_dockerfile_copy_sources_exist_and_not_dockerignored`, `test_dockerfile_installs_onnx_and_quantize_script`, `test_deploy_script_bash_n`, `test_compose_ports_loopback_only`, `test_fetch_models_dry_run_lists_pinned_revisions`.
- [ ] Optional (approval + Docker Desktop running): `bash tests/test_docker_smoke.sh` arm64. Record image size + container USS. Emulated Pi `--cpus=4 --memory=3g` labelled "emulated, fast M2 cores". Skipped → G7 "unverified".
- [ ] `scripts/pi_bench.sh`: bench + evals on real Pi (user-run). Commit.

### Task 10: Real-speech bench audio (optional, user records)
**Files:** Create `scripts/prepare_eval_audio.py`, `tests/test_bench_audio.py`. Modify `scripts/bench.py`.
- [ ] User record 12 en + 12 zh questions (QuickTime / phone) into `eval/audio_raw/`. Script convert to 16 kHz mono via ffmpeg into `eval/audio/`. Recordings = user's voice: `eval/audio_raw/` + `eval/audio/` git-ignored unless user opt in.
- [ ] Tests: `test_bench_prefers_recorded_audio`, `test_bench_marks_zh_na_without_real_audio`, `test_prepare_converts_to_16k_mono`. Commit.

### Task 11: Rebench, gates, report
- [ ] Quiet host: bench (speculate on/off), evals (asr, rag, tts), concurrent, mem profile. `python scripts/report_gates.py --write results/gates_after.json`.
- [ ] `make_report_tables.py` add gate table (PASS / FAIL / UNVERIFIED, before → after). Update `docs/report.md`, `docs/rulings.md`, `docs/licenses.md`, `docs/demo-script.md` numbers.
- [ ] Fold-in cleanups (each owned by its task): per-turn-event cancel test, real-lock starvation test, pins check (`scripts/check_pins.py`, marker `integration`).
- [ ] Full: `pytest -m "not perf and not eval and not integration"` green. Perf / eval green on quiet host. Commit.

## Risks / unmet fallbacks
- zh CER ≤15% may stay unmet on FLEURS (read text, digits). AISHELL = gate. FLEURS reported.
- Kitten may be slower on ARM fp16 → fallback chain in T2.
- Second pass + denoiser may break G3 / G1 → ship OFF.
- Speculation waste CPU under load → gated by sessions, cores, A/B.
- Host never quiet → gates recorded `quiet:false`, not claimed.
- 2-user p50 ≤2× single may fail (LLM serialised) → report measured. Admission keep it bounded.

## Review log
- Pass 1 (staff review, code read): endpoint hold moved into ASR stream (no merge, no reset). Admission moved out of `Llm`. Lease release path. Embedder source pin. Second pass own executor. Shadow `_spoken` bookkeeping + core guard. CTC loader by layout. G1 = 42 unique prompts, per-turn e2e. Markers, renamed gate script, OOV report path, observability, one downloads table.
- Pass 2 (grill + real APIs): checked psutil USS, sherpa CTC / Kitten / denoiser / offline recognizer signatures, Xenova int8 bge, AISHELL licence. Added embedder edge tests, CLS pooling, GTCRN URL caveat, stop-loss, shadow vs second-session / admission.
- Pass 3 (fresh eye): `load_recognizers` rename fallout, final `AsrEvent.pcm` for second pass, recordings git-ignored, guard without jieba.
- Pass 4: caveman rewrite. No scope change.
