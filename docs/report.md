# VORA: streaming voice Q&A on CPU (ASR, RAG, LLM, TTS)

Free, open-source models, CPU only, English and Mandarin. Numbers come from `results/*.json` (final run 2026-10-06, Apple M2, 8 cores, test splits read once with `--final`), except design-choice figures that cite `docs/spikes.md`. Full design notes and every earlier table: `docs/appendix.md`.

## 1. Architecture

The browser microphone sends 16 kHz PCM16 audio in 100 ms frames over a WebSocket to the streaming ASR. Partial text goes to the UI, and stable partials start retrieval early. Retrieval is hybrid (vectors plus BM25, with a calibrated refusal threshold). The LLM streams tokens into an answer guard (negation, figures, refusals answered from the retrieved text). A chunker cuts the guarded text into 1 word, then 3 words, then sentences, and the TTS sends 16 kHz audio back on the same socket.

Each user gets one socket; ASR, retrieval, LLM and TTS run on separate executors with a bounded output queue, so a slow client only blocks its own TTS. Speaking over the answer cancels it (barge-in).

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

Latency = speech end to first audio of the answer, through the real models. LLM memory is separate: 653 MB.

## 3. Real human speech (third-party, not self-authored)

Questions and audio come from MInDS-14 (PolyAI, CC-BY-4.0): real callers asking a bank about 14 intents, recorded over the phone at 8 kHz. They are answered from a separate demo bank knowledge base written from the intent names before any utterance was read. Accuracy is on 100 test clips per cell; latency on 42 fixed clips per language (14 per English accent).

| Scene | ASR error | Top-3 (ASR text / exact text) | Faithful (ASR / exact text) | Answered p50 / p90 ms (n) |
|---|---|---|---|---|
| en-US phone, 8 kHz | WER 44.0% | 90% / 98% | 52% / 67% | 1061 / 1337 (13 answered, 1 refused) |
| en-GB phone | WER 49.1% | 74% / 93% | 42% / 57% | 1178 / 1422 (10, 3) |
| en-AU phone | WER 42.0% | 76% / 96% | 43% / 56% | 1034 / 1725 (11, 3) |
| zh-CN phone | CER 21.5% | 83% / 92% | 59% / 60% | 1006 / 1507 (28, 14) |

Latency counts answered turns, as the gate defines it; refused turns get a fixed short reply and are 7 to 33% of the latency clips.

Phone speech is hard for 16 kHz models at the 50 MB size limit, yet retrieval still finds the right answer most of the time because a few key words are enough. Babble and reverb are far worse (en-US WER 77.5% and 92.3%, top-3 70% and 8%; full noise table in the appendix), and under reverberation the system refuses (77% refused) instead of guessing.

## 4. Latency stages, RTF and MOS

English audio, final run (26 turns, quiet host, the same turns as G1):

| Stage | p50 ms | p90 ms |
|---|---|---|
| ASR (speech end to final text) | 910 | 1123 |
| RAG + LLM first token (after final text) | 0 | 147 |
| TTS first chunk (after first token) | 316 | 676 |
| Total (speech end to first audio) | 1006 | 1438 |

The rows overlap, so they do not add up: a shadow turn starts the answer on a stable partial and hides most retrieval and LLM time inside the endpoint wait (faster on 22 of 25 questions in a paired test, median gain 153 ms). The LLM alone needs 438 ms to its first token; the cost without the shadow turn was not measured.

Real-time factor (processing time over audio length) on the M2: ASR English 0.053 to 0.076 and Chinese 0.028 to 0.034, at least 13 times faster than real time. English TTS has a clause RTF of 0.08 (`docs/spikes.md`); Chinese TTS RTF was not recorded, and nothing was measured on a Raspberry Pi 4.

MOS proxy (UTMOS22, 10 clips per voice): English 4.22, Chinese 3.44, against the 3.5 target. UTMOS is an English-trained predictor, not a listening test.

## 5. Stack choices

| Part | Choice | Why |
|---|---|---|
| ASR | sherpa-onnx zipformer en-20M int8 (41.6 MB), small CTC zh int8 (25 MB) | Both fit 50 MB. The CTC model beat the 14M transducer on the same 60 AISHELL-1 clips (CER 5.2% vs 16.7%); the bilingual model's 182 MB encoder is too big. |
| Retrieval | per-language int8 bge-small, FAISS plus BM25 | BM25 keeps codes such as VORA-X200 retrievable. Int8 embedders and a numpy resampler cut Linux memory from 560 to 465 MB. |
| LLM | Qwen2.5-0.5B Q4_K_M via llama.cpp (469 MB) | Under 1B, Apache-2.0. GGUF has the lowest first token (438 ms vs 628 ms for ONNX int8). ONNX int8 decodes faster but is 603 MB, so it was benchmarked, not shipped. |
| TTS | Piper VITS: en LJSpeech (22.3 MB, int8 with fp32 decoder), zh huayan (20 MB) | Kitten was 3x slower; fp32 Piper is 63 MB; all-int8 LJSpeech had a 145 / 179 ms first chunk against 57 / 79 ms with the fp32 decoder. |
| Transport | FastAPI WebSocket, one socket per user | One connection carries audio both ways, with per-user backpressure; simpler than gRPC or WebRTC. |

## 6. Trade-offs

- Model size against accuracy: the 50 MB ASR limit gives models that are strong on read speech (WER 8.4% en, CER 5.8% zh) and weak on phone speech.
- Latency against naturalness: the first audio chunk is one word, then three, then sentences. That keeps English first audio at p95 70 ms and costs prosody at the start.
- Endpoint delay against cut-off sentences: 0.4 s of trailing silence gives about half the extra finals of 0.3 s.
- Refusal threshold against recall: a higher threshold refuses more off-topic questions and also some real ones with ASR errors.
- Speculation against CPU: the shadow turn needs 6 or more cores and one session, so a 4-core Pi runs without it.
- Gain control against clean-speech accuracy: it fixes quiet speakers (WER 42.9% to 5.6% at −30 dB) and costs at most 1.3 points elsewhere.

## 7. Deployment

`docker build -f docker/Dockerfile -t vora .` fetches pinned models and builds the index; `docker run -p 8000:8000 vora` serves `http://localhost:8000`. Other devices need HTTPS (`scripts/make_cert.sh`); set `VORA_ACCESS_KEY` when the port is reachable from outside. It needs a 64-bit x86 or arm64 CPU, about 1.2 GB of disk for models, and 465 MB of memory for ASR, RAG and TTS plus 653 MB for the LLM. Main dependencies: sherpa-onnx, onnxruntime, llama-cpp-python, FAISS, FastAPI.

| Proof | Result |
|---|---|
| arm64 image (native) | built, ready in 84 s, WebSocket smoke test passes in English, Chinese and with a real voice |
| amd64 image (emulated) | built 1.06 GB, boots, smoke test passes |
| Pi 4 shape (`--cpus=4 --memory=1g`, fast M2 cores) | first audio p50 979 / p95 2811 ms; Linux USS 465 MB |
| AWS t4g.small demo (2 Graviton2 vCPU, 2 GB) | HTTPS on 443, access key required; 3.8 to 5.6 s per answer: works, does not meet G1 |
| Pi-class CPU (a1.xlarge) and Jetson | not run: AWS free plan refused the instance type, and there is no Jetson hardware |

## 8. Limitations

- G5 faithfulness is 89%, below the 95% target. The misses are retrieval near-misses and the 0.5B model paraphrasing wrongly; off-topic refusal on the blind set is 60%. On real phone speech it is far lower, because the ASR text is wrong before retrieval starts.
- Raspberry Pi and Jetson are unmeasured. The closest evidence is the 4-core Docker run on M2 cores, which is optimistic.
- The Chinese voice has an unknown licence and a MOS proxy of 3.44, below the target. An Apache-2.0 swap exists (aishell3, 40 MB) but is over the 30 MB TTS limit.
- Real microphones, Firefox and physical phones were not tested automatically. The VORA Box question sets are self-authored, unlike the real-voice results.
- Cantonese is not supported.
