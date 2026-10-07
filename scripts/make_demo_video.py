"""Record the 2-minute demo video (brief, deliverable 3) from the REAL system: real server, real models, real web page.

What the brief asks the video to show: the user speaks into a microphone; ASR shows partial text in real time; RAG retrieves
context and generates a response; TTS plays it; latency is highlighted. Each scene below is recorded in Chrome (headless)
over the DevTools protocol:
  - the "microphone" is a recorded question played through Chrome's fake microphone (--use-fake-device-for-media-stream), so the
    page runs its real getUserMedia -> AudioWorklet -> WebSocket path; the video says so on screen;
  - the step captions are driven by what the page actually does (partial text appears, context chips appear, ...), not by timers;
  - the TTS sound is tapped from the page's own WebAudio graph, so the audio is exactly what the page played.
Usage: python scripts/make_demo_video.py [--server http://127.0.0.1:8000] [--only english,mandarin] [--takes 3] [--out docs/demo]
Needs: a running server (python -m vora.server), Google Chrome, ffmpeg, macOS `say` (for the recorded questions)."""
import argparse
import asyncio
import base64
import html
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
W, H = 1280, 720
TRIGGERS = ("start", "partial", "context", "tokens", "speaking", "barge_in", "latency")
MIC_NOTE_SHORT = "Microphone: a recorded question played through Chrome's fake microphone (the page's real getUserMedia → AudioWorklet → WebSocket path)."
MIC_NOTE = ("Microphone: a recorded question played through Chrome's fake microphone. The page runs its real "
            "getUserMedia → AudioWorklet → WebSocket path; models and server are the real ones.")


@dataclass(frozen=True)
class Step:
    trigger: str        # when the page does this, the caption lights up
    caption: str        # {ids} and {latency} are filled with what the page shows


@dataclass(frozen=True)
class Utterance:
    voice: str          # macOS `say` voice
    text: str
    at_s: float         # when the question starts on the microphone timeline


@dataclass(frozen=True)
class Scene:
    id: str
    title: str
    lang: str
    budget_s: float     # planned length (also the recording timeout)
    steps: tuple
    mic: tuple
    expect_context: bool = False   # True: the scene claims VORA answered, so a take where retrieval found nothing is rejected


SCENES = (
    Scene("english", "English question", "en", 14, (
        Step("start", "User speaks into the microphone: 16 kHz mono PCM, 100 ms frames, one WebSocket."),
        Step("partial", "Streaming ASR: the partial transcript updates while the user is still speaking."),
        Step("context", "End of speech: retrieval (FAISS + BM25) finds the knowledge-base context {ids}."),
        Step("tokens", "Streaming LLM (Qwen2.5-0.5B, CPU only): tokens appear as they are generated."),
        Step("speaking", "Streaming TTS synthesizes the answer in small chunks and plays it as soon as the first chunk is ready."),
        Step("latency", "Response time: {latency}, from the end of the user's speech to the first audio (target < 1.5 s)."),
    ), (Utterance("Samantha", "How long is the warranty on the VORA X200?", 1.6),), expect_context=True),
    Scene("mandarin", "Mandarin question", "zh", 13, (
        Step("start", "Mandarin: its own streaming ASR model and Chinese voice, the same pipeline."),
        Step("partial", "The partial transcript (Simplified Chinese) streams in while the user speaks."),
        Step("context", "Retrieval over the Chinese knowledge base finds {ids}."),
        Step("tokens", "The answer streams token by token."),
        Step("speaking", "The Chinese voice synthesizes and plays the reply."),
        Step("latency", "Response time: {latency}."),
    ), (Utterance("Tingting", "如何恢复出厂设置？", 1.6),), expect_context=True),
    Scene("barge_in", "Barge-in: interrupting the answer", "en", 22, (
        Step("start", "The user asks a question…"),
        Step("speaking", "…and VORA starts speaking the answer."),
        Step("barge_in", "Barge-in: the user talks over the answer. Playback stops at once, and the LLM and TTS are cancelled if they are still running."),
        Step("latency", "The new question is answered. Response time for the new question: {latency}."),
    ), (Utterance("Samantha", "How long is the warranty on the VORA X200?", 1.6),
        Utterance("Samantha", "What is the wake word?", 7.6)), expect_context=True),
    Scene("refusal", "Off-topic question", "en", 11, (
        Step("start", "An off-topic question: nothing about it is in the knowledge base."),
        Step("partial", "The transcript still streams in."),
        Step("tokens", "No relevant context, so VORA says it is not sure instead of making something up."),
        Step("latency", "Spoken refusal. Response time: {latency}."),
    ), (Utterance("Samantha", "What is the capital of France?", 1.6),)),
    Scene("real_voice", "Real human voice (phone quality)", "en", 14, (
        Step("start", "Real human speech: a phone recording from the public MInDS-14 dataset (CC-BY-4.0), 8 kHz."),
        Step("partial", "Phone-quality speech is hard: the transcript is rough (WER 42–49% on this set)."),
        Step("tokens", "A banking question is outside the VORA Box knowledge base: VORA {outcome}."),
        Step("latency", "Spoken reply. Response time: {latency}."),
    ), (Utterance("file", "client/samples/real/en_gb_freeze.wav", 1.6),)),
)
CARDS = {"title": 7, "latency": 9, "gates": 11, "limits": 9}      # seconds on screen


# ---- pure helpers (unit-tested) ---------------------------------------------------------------------------------
TAIL_S = 4.5          # the scene keeps recording this long after its last step, so the spoken answer is heard


def latency_due(metrics: int, base: int) -> bool:
    """True once the server has sent a NEW `metrics` message since `base`. Comparing the displayed number is wrong: two
    questions can have the same latency, and then the first question's number would be shown as the second's."""
    return metrics > base


def response_time_text(ms) -> str:
    """The page's own format (client/js/ui.js `sec`)."""
    return "–" if ms is None else f"{ms / 1000:.2f} s"


def check_scene_audio(scene_id: str, mean_db: float, floor_db: float = -55.0) -> str | None:
    """None if there is sound after the scene's last step, else a message."""
    return None if mean_db > floor_db else f"{scene_id}: no sound after the last step (mean {mean_db:.0f} dB): the answer is not audible"


def window_mean_volume(path, start: float, dur: float) -> float:
    """Mean volume in dB of a window of an audio or video file (ffmpeg volumedetect); -91 if it is digital silence."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-ss", str(start), "-t", str(dur), "-i", str(path), "-vn", "-af", "volumedetect",
                        "-f", "null", "-"], capture_output=True, text=True)
    m = re.search(r"mean_volume:\s*(-?[\d.]+|-inf) dB", r.stderr)
    return -91.0 if not m or m.group(1) == "-inf" else float(m.group(1))


def take_ok(fired: int, steps: int, ctx: list, expect_context: bool, fits: bool = True) -> bool:
    """A take is usable when every step lit up, the script panel shows all of them (a cut-off caption is invisible), and, for a
    scene that shows an answer, retrieval really found context."""
    return fired == steps and fits and (bool(ctx) or not expect_context)


def steps_to_fire(steps, fired: set, due: dict) -> list[int]:
    """Indices of the steps to light up in this poll: in script order, up to the first one that is not due yet. After a
    `barge_in` step the poll ends: the later steps depend on state that changes when it fires (the metrics baseline), so they
    are judged on the next poll. (Without this, the response time of the FIRST question was shown for the new question.)"""
    done, out = set(fired), []
    for i, st in enumerate(steps):
        if i in done:
            continue
        if not due.get(st.trigger):
            break
        out.append(i)
        done.add(i)
        if st.trigger == "barge_in":
            break
    return out


def outcome_text(ctx_ids: list[str]) -> str:
    """What the caption says VORA did: taken from what the page showed, never from what the script hoped for."""
    return "says it is not sure" if not ctx_ids else "still answers, from loosely related text (a known weakness)"


def total_budget_s() -> float:
    return sum(s.budget_s for s in SCENES) + sum(CARDS.values())


def frame_durations(stamps: list[float], end: float, min_dt: float = 1 / 60) -> list[float]:
    """Screencast frames arrive only when the page repaints: each frame is shown until the next one (the last until `end`)."""
    out = [max(min_dt, b - a) for a, b in zip(stamps, stamps[1:])]
    out.append(max(min_dt, end - stamps[-1]))
    return out


def concat_list(paths: list[str], durations: list[float]) -> str:
    """ffmpeg concat-demuxer list; the last file is listed twice so its duration is honoured."""
    lines = []
    for p, d in zip(paths, durations):
        lines += [f"file '{p}'", f"duration {round(d, 4)}"]
    lines.append(f"file '{paths[-1]}'")
    return "\n".join(lines)


def mix_filter(user_offset_ms: int, tts_offset_ms: int) -> str:
    """Input 1 = what the page played (TTS), input 2 = the recorded question; both placed on the video timeline."""
    t, u = max(0, int(tts_offset_ms)), max(0, int(user_offset_ms))
    return (f"[1:a]adelay={t}:all=1[t];[2:a]adelay={u}:all=1[u];"
            f"[t][u]amix=inputs=2:normalize=0:duration=longest,alimiter=limit=0.95[a]")


def pick_take(takes: list[dict]) -> dict:
    """The median-latency take among those that showed every step: no cherry-picking the fastest."""
    ok = sorted((t for t in takes if t.get("ok")), key=lambda t: t["latency_ms"])
    if not ok:
        raise ValueError("no take showed every step")
    return ok[(len(ok) - 1) // 2]


def gate_rows(gates: list[dict]) -> list[tuple[str, str, str, str]]:
    word = {True: "PASS", False: "FAIL", None: "UNVERIFIED"}
    return [(g["id"], g["name"], word[g["ok"]], g["measured"]) for g in gates]


def script_markdown(scenes, cards) -> str:
    total = sum(s.budget_s for s in scenes) + sum(cards.values())
    out = ["# Demo video script", "",
           "Generated by `scripts/make_demo_video.py` from the same scene definitions that drive the recording, so the on-screen "
           "script and this file cannot drift apart. Planned length: " f"{total:.0f} s (limit: 120 s).", "",
           "**Microphone.** " + MIC_NOTE + " To use your own voice, record the questions and pass them with `--mic scene=file.wav`.", "",
           "## What the brief asks the video to show, and where", "",
           "| Brief item | Scene · step |", "|---|---|",
           "| User speaks into a microphone | English · step 1 |",
           "| ASR transcribes in real time (partial text) | English · step 2, Mandarin · step 2 |",
           "| RAG retrieves context | English · step 3 |",
           "| Streaming LLM generates the response | English · step 4 |",
           "| TTS synthesizes and plays the audio | English · step 5 |",
           "| Latency highlighted (\"Response time\") | English · step 6 (blue box on the Latency panel), Mandarin · step 6, the latency card |",
           "| Streaming extras: barge-in, refusal, Mandarin | Scenes 2–4 |",
           "| Streaming vs batch baseline | latency card |", "",
           "## Timeline", "", f"1. **Title card** ({cards['title']} s): what the system is and what the video checks."]
    for i, sc in enumerate(scenes, start=2):
        out += [f"{i}. **{sc.title}** (up to {sc.budget_s:.0f} s)", ""]
        out += [f"   {j}. {s.caption}" for j, s in enumerate(sc.steps, start=1)]
        out += [""]
    n = len(scenes) + 2
    out += [f"{n}. **Latency card** ({cards['latency']} s): the takes in this video, the benchmark on real recordings, streaming vs batch.",
            f"{n + 1}. **Gates card** ({cards['gates']} s): every brief target with its real PASS / FAIL / UNVERIFIED.",
            f"{n + 2}. **Limits card** ({cards['limits']} s): what is not met or not measured.", ""]
    return "\n".join(out)


# ---- cards -----------------------------------------------------------------------------------------------------
CSS = """body{margin:0;width:1280px;height:720px;background:#0d1017;color:#e8ecf4;font:500 22px/1.4 -apple-system,'PingFang SC','Helvetica Neue',sans-serif;
padding:48px 64px;box-sizing:border-box}h1{font-size:46px;margin:0 0 6px}h2{font-size:30px;margin:0 0 18px;color:#9db8ff}
p{margin:8px 0}.dim{color:#9aa6c4}table{border-collapse:collapse;width:100%;font-size:19px}td,th{padding:6px 10px;border-bottom:1px solid #263047;text-align:left;vertical-align:top}
.PASS{color:#4fd18b;font-weight:700}.FAIL{color:#ff7a7a;font-weight:700}.UNVERIFIED{color:#ffcf5c;font-weight:700}
.box{background:#161b27;border:1px solid #2a3550;border-radius:14px;padding:16px 22px;margin:10px 0}.big{font-size:34px;font-weight:700;color:#fff}"""


def page(body: str) -> str:
    return f"<!doctype html><meta charset=utf-8><style>{CSS}</style>{body}"


def title_card() -> str:
    items = ["Streaming ASR → RAG retrieval → streaming LLM → streaming TTS, over one WebSocket · English and Mandarin · CPU only",
             "Target: first spoken answer within 1.5 s of the end of the user's speech"]
    return page("<h1>VORA</h1><h2>Streaming voice Q&amp;A on CPU</h2>"
                + "".join(f"<p>• {html.escape(i)}</p>" for i in items)
                + "<h2 style='margin-top:26px'>What you will see</h2>"
                + "".join(f"<p>{html.escape(i)}</p>" for i in
                          ["1 · The user speaks into the microphone; streaming ASR shows partial text in real time",
                           "2 · RAG retrieves context from the knowledge base; the LLM streams its answer token by token",
                           "3 · Streaming TTS speaks it; the Response time is highlighted for every question",
                           "4 · Mandarin, barge-in, an off-topic question and a real human phone voice",
                           "5 · Benchmarks, streaming vs batch, every brief target (pass and fail) and the limits"])
                + f"<div class=box><p class=dim>{html.escape(MIC_NOTE)} Every number in this video is measured; "
                  "the benchmark numbers come from results/*.json.</p></div>")


def latency_card(takes: dict[str, dict], gates: list[dict], baseline: dict | None) -> str:
    def shown(t):
        sec_ = f"{t['latency_ms'] / 1000:.2f}"
        return sec_ if t["ok"] else f"<s>{sec_}</s>"
    rows = "".join(f"<tr><td>{html.escape(sc.title)}</td><td>{takes[sc.id]['latency_ms'] / 1000:.2f} s</td>"
                   f"<td class=dim>{', '.join(shown(t) for t in takes[sc.id]['all'])} "
                   f"({sum(t['ok'] for t in takes[sc.id]['all'])} of {takes[sc.id]['n']} takes usable)</td></tr>"
                   for sc in SCENES if sc.id in takes)
    g = {x["id"]: x for x in gates}
    base = ""
    if baseline:
        base = (f"<div class=box><b>Streaming vs batch baseline</b> (same models and questions, earlier measurement round): "
                f"batch p50 <b>{baseline['batch_total_ms']['p50'] / 1000:.2f} s</b> → streaming p50 "
                f"<b>{baseline['streaming_total_estimated_ms']['p50'] / 1000:.2f} s</b> "
                f"(p95 {baseline['batch_total_ms']['p95'] / 1000:.1f} → {baseline['streaming_total_estimated_ms']['p95'] / 1000:.1f} s).</div>")
    return page("<h2>Response time (speech end → first audio)</h2>"
                "<table><tr><th>Scene in this video</th><th>Shown in the video (median of the usable takes)</th><th>Every take (s); struck through = rejected (question misheard, no context found, or a step missing)</th></tr>" + rows + "</table>"
                "<p class=dim>Each scene was preceded by one warm-up question. Recorded on this Mac while Chrome and the recorder were also running, so a little slower than the benchmark.</p>"
                f"<div class=box><b>Benchmark, real recordings, test split, quiet host</b> (answered turns, p50/p90, target ≤ 1500 / ≤ 1800 ms):"
                f"<p>{html.escape(g['G1r']['measured'])}</p><p>Synthetic questions: {html.escape(g['G1']['measured'])}</p></div>" + base)


def gates_card(gates: list[dict]) -> str:
    rows = "".join(f"<tr><td>{html.escape(i)}</td><td>{html.escape(n)}</td><td class={s}>{s}</td><td class=dim>{html.escape(m)}</td></tr>"
                   for i, n, s, m in gate_rows(gates))
    return page("<h2>Brief targets, final run (results/gates_final.json)</h2>"
                f"<table><tr><th></th><th>Gate</th><th>Result</th><th>Measured</th></tr>{rows}</table>")


def limits_card(takes: dict | None = None) -> str:
    items = ["Answer faithfulness is 89% on 100 questions (target 95%).",
             "Off-topic questions are not always refused: 60% on the blind set. In a spot check of the six sample callers, "
             "4 were refused and 2 were answered from loosely related text.",
             "Raspberry Pi and Jetson were not measured (no hardware, AWS refused the Pi-class instance).",
             "The Chinese voice has an unknown licence; Mandarin MOS proxy 3.44 (target 3.5, English-trained predictor).",
             "Real phone-quality speech is hard: WER 42–49% (English), CER 21.5% (Chinese); reverberation is the failure case.",
             "The LLM ships as GGUF (ONNX export was benchmarked, not shipped)."]
    if takes and "barge_in" in takes:
        t = takes["barge_in"]["all"]
        items.insert(1, f"Interrupting speech is understood less reliably than a normal question: in this recording only "
                        f"{sum(x['ok'] for x in t)} of {len(t)} takes found an answer to the interrupting question.")
    return page("<h2>Honest limitations</h2>" + "".join(f"<p>• {html.escape(i)}</p>" for i in items))


# ---- recording (needs Chrome, ffmpeg, the server) ---------------------------------------------------------------
INIT_JS = r"""
(() => {
  const D = (window.__demo = { mic: 0, audioStart: 0, metrics: 0, lastMs: null });
  // The page's bargeIn flag is true only between the first partial of the new speech and its final (about 1 s): latch it, so a
  // slow poll from outside cannot miss it.
  setInterval(() => { try { if (window.__vora && window.__vora.state.bargeIn) D.bargeIn = true; } catch (e) { /* page not ready */ } }, 20);
  const WS = window.WebSocket;
  window.WebSocket = class extends WS {
    constructor(...a) {
      super(...a);
      this.addEventListener("message", (e) => {
        if (typeof e.data !== "string" || !e.data.includes('"metrics"')) return;
        try { const m = JSON.parse(e.data); if (m.type === "metrics") { D.metrics++; D.lastMs = m.first_content_audio_ms; } } catch (err) { /* not json */ }
      });
    }
  };
  const gum = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async (c) => { const s = await gum(c); D.mic = Date.now(); return s; };
  const orig = AudioNode.prototype.connect; let dest = null, rec = null;
  AudioNode.prototype.connect = function (t, ...r) {
    const res = orig.call(this, t, ...r);
    try {
      if (t instanceof AudioDestinationNode) {
        D.ctx = t.context;
        if (!dest) {
          dest = t.context.createMediaStreamDestination();
          rec = new MediaRecorder(dest.stream, { mimeType: "audio/webm;codecs=opus" });
          rec.ondataavailable = async (e) => {
            if (!e.data.size) return;
            const u = new Uint8Array(await e.data.arrayBuffer()); let s = "";
            for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000));
            window.demoAudio(btoa(s));
          };
          rec.start(250); D.audioStart = Date.now();
        }
        orig.call(this, dest);
      }
    } catch (e) { /* tap is best effort */ }
    return res;
  };
  window.__demoStopTap = () => { try { rec && rec.state !== "inactive" && rec.stop(); } catch (e) {} };
  const mk = (tag, css, txt) => { const e = document.createElement(tag); Object.assign(e.style, css); if (txt != null) e.textContent = txt; return e; };
  window.__demoSetup = (title, steps, note) => {
    const log = document.querySelector(".card-log"); if (log) log.style.display = "none";
    const p = mk("div", { position: "fixed", left: "16px", top: "262px", width: "787px", bottom: "70px", background: "rgba(18,22,32,.97)",
      border: "1px solid #3a4666", borderRadius: "14px", padding: "16px 20px", zIndex: 2147483646, boxSizing: "border-box",
      font: '500 17px/1.3 -apple-system,"PingFang SC","Helvetica Neue",sans-serif', color: "#9aa6c4", overflow: "hidden" });
    p.id = "demo-script";
    p.append(mk("div", { font: "700 13px/1 -apple-system,sans-serif", letterSpacing: ".12em", color: "#7fa1ff", marginBottom: "10px" }, "SCRIPT · " + title.toUpperCase()));
    steps.forEach((s, i) => {
      const row = mk("div", { display: "flex", gap: "12px", padding: "4px 8px", margin: "1px 0", borderLeft: "4px solid transparent", opacity: ".42" });
      row.id = "demo-step-" + i; row.append(mk("span", { minWidth: "26px", fontWeight: 700 }, (i + 1) + "."), mk("span", {}, s.replace("{ids}", "…").replace("{latency}", "…").replace("{outcome}", "…"))); p.append(row);
    });
    document.body.append(p);
    const n = mk("div", { position: "fixed", right: "16px", top: "578px", width: "448px", background: "rgba(18,22,32,.97)", border: "1px solid #3a4666",
      borderRadius: "12px", padding: "10px 14px", zIndex: 2147483646, boxSizing: "border-box", font: '500 13px/1.4 -apple-system,sans-serif', color: "#9aa6c4" }, note);
    document.body.append(n);
  };
  window.__demoStep = (i, text) => {
    for (let k = 0; k < i; k++) { const r = document.getElementById("demo-step-" + k); if (r) { r.style.opacity = ".8"; r.style.borderLeft = "4px solid transparent"; r.style.background = "none"; r.style.color = "#c4cde4"; } }
    const r = document.getElementById("demo-step-" + i); if (!r) return;
    r.style.opacity = "1"; r.style.color = "#fff"; r.style.background = "rgba(91,140,255,.18)"; r.style.borderLeft = "4px solid #5b8cff";
    r.lastChild.textContent = text;
  };
  window.__demoFits = () => { const p = document.getElementById("demo-script"); return !!p && p.scrollHeight <= p.clientHeight + 1; };
  window.__demoHighlight = (id) => {
    const el = document.getElementById(id); const c = el && (el.closest(".card") || el); if (!c) return;
    c.style.outline = "4px solid #5b8cff"; c.style.boxShadow = "0 0 24px rgba(91,140,255,.65)";
  };
})();
"""
SNAPSHOT_JS = r"""(() => { const t = (id) => (document.getElementById(id) || {}).textContent || "";
  return { phase: window.__vora.state.phase, error: window.__vora.state.error, bargeIn: !!(window.__demo || {}).bargeIn || !!window.__vora.state.bargeIn,
    live: t("live"), reply: t("reply"), lat: t("lat"), lvl: (document.getElementById("lvl") || { style: {} }).style.width || "", ctx_state: ((window.__demo || {}).ctx || {}).state || "none", stages: [...document.querySelectorAll("#stages .stage")].map((e) => e.textContent), ctx: [...document.querySelectorAll("#ctx .chip")].map((e) => e.textContent),
    mic: (window.__demo || {}).mic || 0, audioStart: (window.__demo || {}).audioStart || 0,
    metrics: (window.__demo || {}).metrics || 0, lastMs: (window.__demo || {}).lastMs,
    fits: window.__demoFits ? window.__demoFits() : false }; })()"""


class Cdp:
    """Minimal Chrome DevTools Protocol client over a websocket."""

    def __init__(self, ws):
        self.ws, self.n, self.pending, self.handlers = ws, 0, {}, {}
        self.task = asyncio.create_task(self._read())

    async def _read(self):
        async for raw in self.ws:
            m = json.loads(raw)
            if "id" in m:
                fut = self.pending.pop(m["id"], None)
                if fut and not fut.done():
                    fut.set_result(m)
            elif (h := self.handlers.get(m.get("method"))):
                h(m["params"])

    async def send(self, method: str, **params):
        self.n += 1
        fut = asyncio.get_running_loop().create_future()
        self.pending[self.n] = fut
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        r = await fut
        if "error" in r:
            raise RuntimeError(f"{method}: {r['error']}")
        return r.get("result", {})

    async def js(self, expr: str):
        r = await self.send("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"].get("text", "js error") + " " + str(r["exceptionDetails"].get("exception", ""))[:200])
        return r["result"].get("value")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def sh(*cmd, **kw):
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def clean_wav(src: Path, dst: Path) -> Path:
    """Chrome's fake microphone wants a plain PCM wav: rewrite it with a bare 44-byte header (ffmpeg adds a LIST chunk)."""
    import wave
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(src), "-f", "s16le", "-ar", "48000", "-ac", "1", "-"],
                         capture_output=True, check=True).stdout
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(raw)
    return dst


def make_mic_wav(scene: Scene, work: Path, override: Path | None = None) -> Path:
    """The recorded questions placed on one timeline (48 kHz mono 16-bit). Chrome starts playing the file when the page opens
    the microphone, so `at_s` is seconds after the user presses Start."""
    out = work / f"mic_{scene.id}.wav"
    tmp = work / f"mic_{scene.id}_raw.wav"
    total = scene.budget_s + 4
    if override:
        sh("ffmpeg", "-loglevel", "error", "-y", "-i", override, "-ar", 48000, "-ac", 1, "-af", f"apad=whole_dur={total}", tmp)
        return clean_wav(tmp, out)
    inputs, filt = [], []
    for i, u in enumerate(scene.mic):
        if u.voice == "file":                                       # a real recording instead of a synthetic voice
            inputs += ["-i", ROOT / u.text]
        else:
            aiff = work / f"say_{scene.id}_{i}.aiff"
            sh("say", "-v", u.voice, "-o", aiff, u.text)
            inputs += ["-i", aiff]
        filt.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=mono,adelay={int(u.at_s * 1000)}:all=1[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(scene.mic))) + f"amix=inputs={len(scene.mic)}:normalize=0,apad=whole_dur={total}[m]"
    sh("ffmpeg", "-loglevel", "error", "-y", *inputs, "-filter_complex", ";".join(filt + [mix]), "-map", "[m]", "-ar", 48000, "-ac", 1, "-sample_fmt", "s16", tmp)
    return clean_wav(tmp, out)


async def record_take(scene: Scene, server: str, mic_wav: Path, work: Path, take: int) -> dict:
    import websockets
    tdir = work / f"{scene.id}_take{take}"
    (tdir / "frames").mkdir(parents=True, exist_ok=True)
    port = free_port()
    proc = subprocess.Popen(["nice", "-n", "10", CHROME, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={tdir / 'profile'}",
                             "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                             "--disable-features=AudioServiceSandbox",      # else the sandboxed audio service cannot read the wav: silence
                             f"--use-file-for-fake-audio-capture={mic_wav}%noloop", "--autoplay-policy=no-user-gesture-required",
                             f"--window-size={W},{H}", "--hide-scrollbars", "--no-first-run", "--no-default-browser-check", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = {"scene": scene.id, "take": take, "ok": False, "latency_ms": 0, "fired": []}
    try:
        for _ in range(60):
            try:
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=1))
                page_t = next(t for t in targets if t["type"] == "page")
                break
            except Exception:
                await asyncio.sleep(0.25)
        else:
            raise RuntimeError("Chrome did not start")
        async with websockets.connect(page_t["webSocketDebuggerUrl"], max_size=None) as ws:
            cdp = Cdp(ws)
            frames, stamps, audio, acks = [], [], bytearray(), set()

            def on_frame(p):
                path = tdir / "frames" / f"{len(frames):06d}.jpg"
                path.write_bytes(base64.b64decode(p["data"]))
                frames.append(str(path))
                stamps.append(p["metadata"]["timestamp"])
                task = asyncio.create_task(cdp.send("Page.screencastFrameAck", sessionId=p["sessionId"]))
                acks.add(task)
                task.add_done_callback(acks.discard)

            def on_binding(p):
                if p.get("name") == "demoAudio":
                    audio.extend(base64.b64decode(p["payload"]))

            cdp.handlers["Page.screencastFrame"], cdp.handlers["Runtime.bindingCalled"] = on_frame, on_binding
            if os.environ.get("VORA_DEMO_DEBUG"):
                cdp.handlers["Runtime.consoleAPICalled"] = lambda p: print("  console:", p["type"], [a.get("value", a.get("description")) for a in p["args"]][:3], flush=True)
                cdp.handlers["Runtime.exceptionThrown"] = lambda p: print("  exception:", p["exceptionDetails"].get("text"), str(p["exceptionDetails"].get("exception", {}).get("description", ""))[:200], flush=True)
            await cdp.send("Page.enable")
            await cdp.send("Runtime.enable")
            await cdp.send("Runtime.addBinding", name="demoAudio")
            await cdp.send("Page.setBypassCSP", enabled=True)       # for the overlay only; the real CSP was verified separately
            await cdp.send("Page.addScriptToEvaluateOnNewDocument", source=INIT_JS)
            await cdp.send("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
            await cdp.send("Page.navigate", url=server.rstrip("/") + "/")
            for _ in range(80):
                if await cdp.js("!!window.__vora && !!document.getElementById('go') && !!window.__demoSetup"):
                    break
                await asyncio.sleep(0.25)
            await cdp.send("Page.startScreencast", format="jpeg", quality=85, maxWidth=W, maxHeight=H, everyNthFrame=1)
            if scene.lang == "zh":
                await cdp.js("(()=>{const s=document.getElementById('lang');s.value='zh';s.dispatchEvent(new Event('change'));})()")
            await cdp.js(f"window.__demoSetup({json.dumps(scene.title)}, {json.dumps([s.caption for s in scene.steps])}, {json.dumps(MIC_NOTE_SHORT)})")
            await asyncio.sleep(1.4)                                   # the script is on screen before anything happens
            await cdp.js("document.getElementById('go').click()")
            t_click, fired, metrics_base, last = time.time(), {}, 0, time.time()
            while time.time() - t_click < scene.budget_s - 1.4 + TAIL_S:
                snap = await cdp.js(SNAPSHOT_JS)
                if os.environ.get("VORA_DEMO_DEBUG"):
                    print(f"  {time.time() - t_click:5.1f}s", {k: v for k, v in snap.items() if k not in ("mic", "audioStart")}, flush=True)
                due = {
                    "start": True,
                    "partial": snap["live"].strip() not in ("", "…"),
                    "context": bool(snap["ctx"]),
                    "tokens": snap["reply"].strip() not in ("", "…"),
                    "speaking": snap["phase"] == "speaking" or latency_due(snap["metrics"], metrics_base),
                    "barge_in": snap["bargeIn"],
                    "latency": latency_due(snap["metrics"], metrics_base),
                }
                for i in steps_to_fire(scene.steps, set(fired), due):
                    st = scene.steps[i]
                    text = st.caption.replace("{ids}", ", ".join(snap["ctx"]) or "–").replace("{latency}", response_time_text(snap["lastMs"])).replace("{outcome}", outcome_text(snap["ctx"]))
                    await cdp.js(f"window.__demoStep({i}, {json.dumps(text)})")
                    if st.trigger == "latency":
                        await cdp.js("window.__demoHighlight('lat')")
                        result["latency_ms"] = float(snap["lastMs"] or 0)
                        result["stages"] = snap["stages"]
                        result["ctx"] = snap["ctx"]
                        result["heard"] = snap["live"].strip()
                    fired[i] = time.time()
                    if st.trigger == "barge_in":
                        metrics_base = snap["metrics"]                   # the next metrics message belongs to the new question
                    last = time.time()
                if len(fired) == len(scene.steps) and time.time() - last > TAIL_S:
                    break
                await asyncio.sleep(0.1)
            snap = await cdp.js(SNAPSHOT_JS)
            await cdp.js("window.__demoStopTap()")
            await asyncio.sleep(0.8)
            await cdp.send("Page.stopScreencast")
            result.update(ok=take_ok(len(fired), len(scene.steps), result.get("ctx", []), scene.expect_context, bool(snap["fits"])), fired=sorted(fired), fired_t=[fired[i] for i in sorted(fired)], mic_epoch=snap["mic"] / 1000,
                          audio_epoch=snap["audioStart"] / 1000, end_epoch=time.time(), frames=frames, stamps=stamps)
            (tdir / "tts.webm").write_bytes(bytes(audio))
            result["tts"] = str(tdir / "tts.webm") if audio else None
            if acks:
                await asyncio.wait(acks, timeout=1.0)
                for t in list(acks):
                    t.cancel()
            cdp.task.cancel()
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
    return result


def render_take(take: dict, mic_wav: Path, out: Path, work: Path) -> float:
    """frames + the page's own audio + the recorded question -> one scene mp4. Returns its duration in seconds."""
    stamps, frames = take["stamps"], take["frames"]
    t0 = stamps[0]
    end = min(take["end_epoch"], stamps[-1] + 3.5)
    durs = frame_durations(stamps, end)
    dur = end - t0
    lst = work / f"{out.stem}_frames.txt"
    lst.write_text(concat_list(frames, durs))
    silent = work / f"{out.stem}_silent.mp4"
    sh("ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", 0, "-i", lst, "-vf", "fps=30,format=yuv420p,pad=ceil(iw/2)*2:ceil(ih/2)*2",
       "-c:v", "libx264", "-preset", "veryfast", "-crf", 20, silent)
    tts = take["tts"] or str(work / "silence.wav")
    if not take["tts"]:
        sh("ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", 1, tts)
    user_ms = int((take["mic_epoch"] - t0) * 1000) if take["mic_epoch"] else 0
    tts_ms = int((take["audio_epoch"] - t0) * 1000) if take["audio_epoch"] else 0
    sh("ffmpeg", "-loglevel", "error", "-y", "-i", silent, "-i", tts, "-i", mic_wav, "-filter_complex", mix_filter(user_ms, tts_ms),
       "-map", "0:v", "-map", "[a]", "-t", round(dur, 3), "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-ar", 48000, "-ac", 2, out)
    return dur


def still_clip(html_text: str, seconds: float, out: Path, work: Path) -> None:
    page_file = work / f"{out.stem}.html"
    page_file.write_text(html_text, encoding="utf-8")
    png = work / f"{out.stem}.png"
    sh(CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={W},{H}", f"--screenshot={png}", f"file://{page_file}",
       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sh("ffmpeg", "-loglevel", "error", "-y", "-loop", 1, "-framerate", 30, "-t", seconds, "-i", png, "-f", "lavfi", "-t", seconds,
       "-i", "anullsrc=r=48000:cl=stereo", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", out)


async def warm_up(server: str, lang: str) -> None:
    """One throw-away question first (a demo is not the first thing a cold server ever answers). Said on the latency card."""
    sys.path.insert(0, str(ROOT))
    from scripts import ws_smoke
    wav = ROOT / "client" / "samples" / ("q_warranty_zh.wav" if lang == "zh" else "q_warranty_en.wav")
    try:
        await ws_smoke.run(server.replace("http", "ws", 1).rstrip("/") + "/ws", str(wav), lang, 20.0, False)
    except Exception as e:                                          # noqa: BLE001 - a failed warm-up only costs latency
        print(f"warm-up skipped: {e}", file=sys.stderr)


async def main_async(a) -> int:
    only = [x for x in a.only.split(",") if x] or [s.id for s in SCENES]
    scenes = [s for s in SCENES if s.id in only]
    overrides = {k: Path(v) for k, v in (kv.split("=", 1) for kv in a.mic)}
    try:
        urllib.request.urlopen(a.server.rstrip("/") + "/health", timeout=3)
    except Exception:
        print(f"no server at {a.server}: start it with `python -m vora.server`", file=sys.stderr)
        return 2
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="vora-demo-"))
    gates = json.loads((ROOT / "results" / "gates_final.json").read_text())
    baseline_p = ROOT / "results" / "baseline.json"
    baseline = json.loads(baseline_p.read_text()) if baseline_p.exists() else None
    os.nice(5)                                                      # the recorder must not out-compete the server it films
    from vora.hostcheck import perf_skip_reason
    if (why := perf_skip_reason()):
        print(f"WARNING: {why}. Close other apps (browsers, Word, ...) first, or the latencies in the video are inflated.", file=sys.stderr)
    chosen, clips, problems = {}, [], []
    still_clip(title_card(), CARDS["title"], work / "card_title.mp4", work)
    for sc in scenes:
        mic = make_mic_wav(sc, work, overrides.get(sc.id))
        await warm_up(a.server, sc.lang)
        takes = []
        for n in range(1, a.takes + 1):
            t = await record_take(sc, a.server, mic, work, n)
            print(f"{sc.id} take {n}: ok={t['ok']} latency={t['latency_ms']:.0f} ms steps={len(t['fired'])}/{len(sc.steps)} stages={t.get('stages')}", flush=True)
            takes.append(t)
        best = pick_take(takes)
        chosen[sc.id] = {**best, "n": len(takes), "all": [{"ok": t["ok"], "latency_ms": t["latency_ms"], "heard": t.get("heard", "")} for t in takes]}
        clip = work / f"scene_{sc.id}.mp4"
        dur = render_take(best, mic, clip, work)
        last_step = best["fired_t"][-1] - best["stamps"][0]                # seconds into the clip when the last step lit up
        mean_db = window_mean_volume(clip, last_step + 0.5, 2.0)
        chosen[sc.id]["audio_after_last_step_db"] = round(mean_db, 1)
        if (problem := check_scene_audio(sc.id, mean_db)):
            problems.append(problem)
        print(f"{sc.id}: chosen take {best['take']} ({best['latency_ms']:.0f} ms), {dur:.1f} s, "
              f"sound after the last step {mean_db:.0f} dB", flush=True)
        clips.append(clip)
    still_clip(latency_card(chosen, gates, baseline), CARDS["latency"], work / "card_latency.mp4", work)
    still_clip(gates_card(gates), CARDS["gates"], work / "card_gates.mp4", work)
    still_clip(limits_card(chosen), CARDS["limits"], work / "card_limits.mp4", work)
    order = [work / "card_title.mp4", *clips, work / "card_latency.mp4", work / "card_gates.mp4", work / "card_limits.mp4"]
    listing = work / "all.txt"
    listing.write_text("\n".join(f"file '{p}'" for p in order))
    final = out_dir / "vora-demo.mp4"
    sh("ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", 0, "-i", listing, "-c:v", "libx264", "-preset", "medium", "-crf", 22,
       "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", 48000, "-ac", 2, "-movflags", "+faststart", final)
    (out_dir / "takes.json").write_text(json.dumps({k: {"latency_ms": v["latency_ms"], "takes_ok": sum(t["ok"] for t in v["all"]), "takes": v["n"], "all": v["all"],
                                                           "audio_after_last_step_db": v["audio_after_last_step_db"]} for k, v in chosen.items()}, indent=1))
    (ROOT / "docs" / "demo-video-script.md").write_text(script_markdown(SCENES, CARDS), encoding="utf-8")
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", final], capture_output=True, text=True).stdout)
    print(f"wrote {final} ({dur:.0f} s, {final.stat().st_size / 1e6:.1f} MB); work dir kept at {work}")
    for p_ in problems:
        print("PROBLEM:", p_, file=sys.stderr)
    return 0 if dur <= 120 and not problems else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--only", default="", help="comma-separated scene ids (default: all)")
    ap.add_argument("--takes", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "docs" / "demo"))
    ap.add_argument("--mic", action="append", default=[], metavar="SCENE=FILE.wav", help="your own recording instead of the synthetic voice")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
