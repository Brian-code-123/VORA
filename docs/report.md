# VORA: streaming voice Q&A on CPU (ASR, RAG, LLM, TTS)

Free, open-source models, CPU only, English and Mandarin. Every number below comes from `results/*.json` (final run 2026-10-06, Apple M2, 8 cores, test splits read once with `--final`). Full design notes and every earlier table: `docs/appendix.md`.

## 1. Architecture

The browser microphone sends 16 kHz PCM16 audio in 100 ms frames (AudioWorklet) over a WebSocket to the streaming ASR, which is sherpa-onnx with a zipformer-20M for English (int8, 41.6 MB) and a small CTC model for Chinese (int8, 25 MB). Partial text goes to the UI, and stable partials start retrieval early. Retrieval is hybrid: int8 bge-small embedders per language, FAISS with sentence passages, BM25 and a calibrated refusal threshold. Qwen2.5-0.5B Q4_K_M (llama.cpp) streams tokens into an answer guard (negation, figures, refusals answered from the retrieved text). A chunker cuts the guarded text into 1 word, then 3 words, then sentences, and Piper VITS TTS (en LJSpeech int8, zh huayan) sends 16 kHz audio back on the same socket.

Each user gets one socket; ASR, retrieval, LLM and TTS run on separate executors with a bounded output queue, so a slow client only blocks its own TTS. Speaking over the answer cancels it (barge-in). The browser client is plain ES modules with a state machine covering every state (loading, busy, mic errors, close codes, iOS interruption) and was checked at 10 screen sizes, light and dark.

## 2. Gates (brief targets)

| Gate | Target | Result | Measured |
|---|---|---|---|
| G1 latency, synthetic en audio | p50 ≤1500, p90 ≤1800 ms | PASS | p50 1006 / p90 1438 ms |
| G1r latency, real voices (MInDS-14) | answered p50 ≤1500, p90 ≤1800 ms | PASS | p90: en-US 1337, en-GB 1422, en-AU 1725, zh 1507 ms (§3) |
| G2 TTS first chunk | p95 ≤200 ms | PASS | en 70 / zh 89 ms |
| G3 ASR+RAG+TTS memory | ≤500 MB USS | PASS | 305 MB (macOS); 465 MB in the Linux container |
| G4 ASR accuracy, clean read speech | WER/CER ≤15% | PASS | en LibriSpeech 8.4%, zh AISHELL 5.8% |
| G5 retrieval + faithfulness | top-3 ≥80%, faithful ≥95% | FAIL | top-3 96%, faithful 89% of 100 (blind 81.7%) |
| G6 two users at once | each p50 ≤2× single | PASS | ×1.36 (1034 / 1215 vs 891 ms) |
| G7 Docker + Pi-class proof | arm64, amd64, limited cores, Pi-class CPU | UNVERIFIED | 3 of 4 done; Pi-class run refused by AWS free plan |
| G8 permissive licences | no unknown licences | FAIL | zh voice huayan: licence unknown (kept by owner ruling) |

Latency = speech end to first audio of the answer, in-process through the real models. LLM memory is reported separately: 653 MB.

## 3. Real human speech (third-party, not self-authored)

Questions and audio come from MInDS-14 (PolyAI, CC-BY-4.0): real callers asking a bank about 14 intents, recorded over the phone at 8 kHz. They are answered from a separate demo bank knowledge base written from the intent names only, before any utterance was read. Accuracy is on 100 test clips per cell; latency on 42 fixed clips per language (14 per English accent).

| Scene | ASR error | Top-3 (ASR text / exact text) | Faithful (ASR / exact text) | Answered p50 / p90 ms (n) |
|---|---|---|---|---|
| en-US phone, 8 kHz | WER 44.0% | 90% / 98% | 52% / 67% | 1061 / 1337 (13 answered, 1 refused) |
| en-GB phone | WER 49.1% | 74% / 93% | 42% / 57% | 1178 / 1422 (10, 3) |
| en-AU phone | WER 42.0% | 76% / 96% | 43% / 56% | 1034 / 1725 (11, 3) |
| zh-CN phone | CER 21.5% | 83% / 92% | 59% / 60% | 1006 / 1507 (28, 14) |

Latency counts answered turns, as the gate defines it; refused turns get a fixed short reply and are 7 to 33% of the latency clips (zh highest). Speech end is cross-checked against the Silero VAD (median gap 68 ms on en-US).

Noise and room effects on the same speakers (top-3 on ASR text):

| Scene | en-US WER / top-3 | zh-CN CER / top-3 |
|---|---|---|
| clean phone | 44.0% / 90% | 21.5% / 83% |
| ambient noise 10 dB | 45.4% / 87% | 22.9% / 83% |
| ambient noise 5 dB | 53.8% / 74% | 25.6% / 77% |
| babble 10 dB | 77.5% / 70% | 56.2% / 81% |
| reverb 0.6 s | 92.3% / 8% | 39.5% / 56% |
| quiet (−30 dB) / loud (+12 dB, clipped) | 41.9% / 88%, 44.7% / 80% | 21.7% / 84%, 21.9% / 82% |

Read speech (WER/CER, clean to ambient noise at 10 dB): LibriSpeech clean 5.9% to 7.4% (phone codec 8.3%), LibriSpeech other 12.8% to 16.0%, FLEURS en 25.9% to 28.7%, AISHELL 5.2% to 5.8%, FLEURS zh 12.3% to 14.6%.

Phone speech is hard for 16 kHz models at the 50 MB size limit (WER 42 to 49%). Retrieval still finds the right answer most of the time because a few key words are enough, and the exact-transcript column shows what ASR costs. Reverberation is the failure case: the English model collapses (8% top-3), so the system refuses (77% refused) instead of guessing. Noise reduction (GTCRN) was tested and made accuracy worse, so it is not in the live path.

## 4. What changed in this round

- Retrieval: a word-boundary bug dropped synonyms for Latin words written next to Chinese ("5g的wifi"); fixed. Added per-language int8 embedders, sentence-level passages, normalised BM25 and per-language refusal thresholds tuned on dev only. Blind sets are frozen by sha256 and used once.
- English voice: LJSpeech (public domain, trained from scratch) replaced lessac; first chunk p95 70 ms.
- Memory in Linux: int8 embedders without fastembed, numpy resampler: 560 to 465 MB USS.
- Automatic gain control before ASR fixes quiet speakers (LibriSpeech at −30 dB: WER 42.9% to 5.6%, dev split) and costs at most 1.3 points elsewhere.
- Deployment: Docker images for arm64 and amd64, a demo on AWS (below), and an access key for the WebSocket.

## 5. Deployment

`docker build -f docker/Dockerfile -t vora .` fetches pinned models and builds the index. Then `docker run -p 8000:8000 vora` and open `http://localhost:8000`. For other devices, use HTTPS (`scripts/make_cert.sh`), because browsers allow the microphone only on secure pages. Set `VORA_ACCESS_KEY` when the port is reachable from outside.

| Proof | Result |
|---|---|
| arm64 image (native) | built, ready in 84 s, WebSocket smoke test passes in English, Chinese and with a real voice |
| amd64 image (emulated) | built 1.06 GB, boots, smoke test passes; latency under emulation not meaningful |
| Pi 4 shape (`--cpus=4 --memory=1g`, fast M2 cores) | first audio p50 979 / p95 2811 ms; Linux USS 465 MB |
| AWS t4g.small demo (2 Graviton2 vCPU, 2 GB) | HTTPS on 443, access key required; 3.8 to 5.6 s per answer: works, does not meet G1 |
| Pi-class CPU (a1.xlarge, Cortex-A72) | not run: AWS free plan refused the instance type |
| Jetson | not run: no hardware. Same arm64 image, CPU only |

## 6. Limitations

- G5 faithfulness is 89%, below the 95% target. The misses are retrieval near-misses and the 0.5B model paraphrasing wrongly; off-topic refusal on the blind set is 60% (6 of 10). On real phone speech faithfulness is far lower, because the ASR text is wrong before retrieval starts.
- Raspberry Pi and Jetson are unmeasured. The closest evidence is the 4-core Docker run on M2 cores, which is optimistic. A Pi 4 core is roughly 1.5× slower than the A72 in a1 and far slower than an M2.
- The Chinese voice (huayan) has an unknown licence. An Apache-2.0 swap exists (aishell3, 40 MB), but it is over the 30 MB TTS limit.
- Phone speech and reverb: no ≤50 MB streaming English model was found that does better, and a second pass model (27.6 MB) would break the per-model size limit together with the first.
- Real microphones, Firefox and physical phones were not tested automatically; a manual checklist is in `docs/demo-script.md`. The VORA Box question sets are self-authored and labelled as such; the real-voice results above are not.
- Cantonese is not supported.
