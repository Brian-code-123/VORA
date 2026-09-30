# Demo script

## 2-minute video (record with QuickTime or OBS, screen + audio)
| Time | Show | Say |
|---|---|---|
| 0:00 | Terminal: `python -m vora.server`, `curl :8000/health` → `{"ready":true}` | "Everything runs locally on CPU, no cloud." |
| 0:10 | Browser `http://127.0.0.1:8000`, English selected, press Start mic | "Streaming ASR sends partial text while I talk." |
| 0:20 | Ask: "How long is the warranty on the VORA X200?" | Point at the live transcript growing, then the retrieved chunk ids. |
| 0:40 | Reply plays; show latency card (say the real number) | "Response time: X seconds. First audio starts before the sentence is finished." |
| 0:55 | Switch to 中文, ask "怎么恢复出厂设置" | "Mandarin uses a separate small streaming model." |
| 1:15 | Interrupt the reply by speaking | "Barge-in cancels the LLM and TTS and flushes the audio." |
| 1:30 | Ask something off-topic ("what is the weather") | "No relevant document, so it says it is not sure instead of making something up." |
| 1:45 | Show `results` table: streaming vs batch baseline | "Same models: batch waits for the whole answer and the whole audio, so the median delay is higher." |

If the microphone is unavailable, use the "Sample question" dropdown (synthetic speech) and say so.

## 10-minute live demo + Q&A
1. (1 min) Architecture diagram from the report.
2. (3 min) Live demo as above, English then Chinese, show partial text and context.
3. (2 min) Latency and baseline table; explain endpoint 0.4 s, speculative retrieval, first-clause chunking.
4. (2 min) Honest limitations: no Pi measured, memory over 500 MB, FLEURS accuracy, 0.5B faithfulness, Cantonese.
5. (2 min) Q&A.

**Trade-off talking points**
- Model size vs accuracy: the only ≤50 MB streaming ASRs are weak outside clean read speech. A larger model would fit accuracy, not the size limit.
- Latency vs naturalness: int8 TTS fits 30 MB but is slower on ARM than fp32; splitting the first chunk early lowers latency and costs prosody.
- Endpoint delay vs mid-sentence cuts: 0.4 s silence is fast and splits slow speakers.
- GGUF vs ONNX: GGUF has the lower first-token time and a prefix cache; ONNX int8 decodes faster per token but is larger.
- Retrieval threshold vs recall: a higher threshold blocks off-topic questions and also some real ones with ASR errors.
