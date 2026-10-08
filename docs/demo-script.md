# Demo script

## Where to run it
- Local, for the best latency: `docker run -p 8000:8000 vora` or `python -m vora.server`, open `http://localhost:8000`.
- AWS demo, works from any network but is slower: in AWS CloudShell run `cat ~/vora-demo-link.txt` and open the `https://<IP>/#key=…` link. The certificate is self-signed, so in Chrome choose "Advanced", then "Proceed". The page removes the key from the address bar, so screenshots are safe. Expect 4 to 6 s per answer on the 2-vCPU box, and say so.
- No microphone (or a noisy room): use the "Sample question" menu. Synthetic samples ask VORA Box questions; "Real callers" are MInDS-14 recordings of bank questions, which are off-topic for the VORA Box knowledge base and show real-voice ASR plus the "not sure" refusal.
- Room: use a quiet room without echo, with a headset or a mic close to the mouth. Reverberation is the measured failure case (en-US top-3 drops to 8%), and the system then says it is not sure. If an answer says "not sure", repeat the question closer to the mic.

## 2-minute video (recorded automatically)
`python scripts/make_demo_video.py` (server running, Chrome and ffmpeg installed) records `docs/demo/vora-demo.mp4` from the real system and writes the script it follows to `docs/demo-video-script.md`. The microphone is a recorded question played through Chrome's fake microphone; to use your own voice, pass `--mic english=your.wav` (one file per scene id). Close other apps first, otherwise the latencies in the video are inflated.

## 2-minute video, live narration version
| Time | Show | Say |
|---|---|---|
| 0:00 | Terminal: `curl localhost:8000/health` returns `{"ready":true,...}` | "Everything runs on CPU, no cloud API." |
| 0:10 | Press Start; point at the level meter and red recording dot | "Streaming ASR shows partial text while I talk." |
| 0:20 | Ask "How long is the warranty on the VORA X200?" | Point at the live transcript, then the context chips (E03). |
| 0:40 | Answer plays; open the latency card | Read the real number and the stage split (ASR / retrieve / LLM / TTS). |
| 0:55 | Switch to 中文, ask "怎么恢复出厂设置" | "Mandarin has its own small streaming model." |
| 1:15 | Talk over the answer | "Barge-in cancels the LLM and TTS and stops playback." |
| 1:30 | Ask "what is the weather" | "Nothing relevant was retrieved, so it says it is not sure." |
| 1:45 | Pick a "Real callers" sample (EN-GB) | "Real phone speech: the transcript is rough; this question is outside the knowledge base, so it refuses." |

## 10-minute live demo + Q&A
1. (1 min) Architecture line from the report.
2. (3 min) Live demo as above, English then Chinese.
3. (2 min) Gate table and real-voice table: what passes, what does not (G5, G7 Pi, G8 zh voice).
4. (2 min) Limitations and trade-offs below.
5. (2 min) Q&A.

## What we do not claim
- Not measured on a Raspberry Pi or a Jetson; Pi-class numbers are a 4-core Docker run on Apple M2 cores (optimistic).
- Faithfulness is 89%, not the 95% target; phone-quality speech is much worse.
- The Chinese voice (huayan) has an unknown licence.
- The AWS demo box does not meet the 1.5 s latency target.

## Manual checklist (not covered by automated tests)
- [ ] Real microphone in Chrome, Safari (macOS), Safari (physical iPhone), Chrome (Android), Firefox, Edge: permission prompt, level meter moves, partial text appears.
- [ ] Deny the mic permission: error banner names the fix.
- [ ] Unplug or revoke the mic mid-session: "microphone ended" state, no hang.
- [ ] iPhone: phone call or app switch during an answer: "tap to resume" banner, then resume works.
- [ ] Wrong or missing `#key=` on the AWS link: "access key missing or wrong" error, no retry loop.
- [ ] Two people at once on the AWS link: both get answers.

## Trade-off talking points
- Model size vs accuracy: the only ≤50 MB streaming ASRs are weak outside clean read speech (phone WER 42 to 49%).
- Latency vs naturalness: the first audio chunk is one word, which keeps first audio fast and costs prosody.
- Endpoint delay vs mid-sentence cuts: 0.4 s silence is fast; unfinished sentences are held up to 500 ms.
- Retrieval threshold vs recall: a higher threshold refuses more off-topic questions and also some real ones with ASR errors.
- GGUF vs ONNX: GGUF has the lower first-token time; ONNX int8 decodes faster per token but is larger.
