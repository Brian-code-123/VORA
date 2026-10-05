// Controller: wires the state machine, the socket, the audio layer and the view.
import { initial, reduce } from "./state.js";
import { History } from "./stats.js";
import { backoff, closeInfo, wsUrl } from "./net.js";
import { detect } from "./env.js";
import { Audio } from "./audio.js";
import * as ui from "./ui.js";

const $ = (id) => document.getElementById(id);
const REAL_SAMPLES = [   // CC-BY-4.0 MInDS-14 clips, see samples/real/ATTRIBUTION.md
  ["real/en_us_balance", "en", "EN-US · balance"], ["real/en_gb_freeze", "en", "EN-GB · freeze card"], ["real/en_au_atm", "en", "EN-AU · ATM limit"],
  ["real/zh_balance", "zh", "中文 · 余额"], ["real/zh_abroad", "zh", "中文 · 境外用卡"], ["real/zh_bill", "zh", "中文 · 缴费"],
];

let s = initial(), ws = null, lang = localStorageGet("vora.lang") || (navigator.language?.startsWith("zh") ? "zh" : "en");
let retries = 0, silentSince = null, wake = null, intentional = false;
const hist = new History(12);
const env = detect(window);

const audio = new Audio({
  onFrame: (buf) => { if (ws?.readyState === 1) ws.send(buf); },
  onLevel: (v) => { ui.level(v); noSoundWatch(v); },
  onPlaybackEnd: () => dispatch({ type: "playback_end" }),
  onStateChange: (st) => dispatch({ type: st === "running" ? "resumed" : "interrupted" }),
  onMicEnded: () => { dispatch({ type: "mic_ended" }); teardown(); },
});

function localStorageGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function localStorageSet(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } }

function dispatch(ev) {
  const prev = s;
  s = reduce(s, ev);
  if (s !== prev) render(prev);
}

function render(prev) {
  ui.renderStatus(lang, s);
  if (s.phase === "error") ui.banner(lang, "error", `err_${s.error}`, s.error === "network" ? { key: "reconnect", run: start } : null);
  else if (s.phase === "unsupported") ui.banner(lang, "error", `err_${s.error}`);
  else if (s.interrupted && ["listening", "hearing", "thinking", "speaking"].includes(s.phase)) ui.banner(lang, "warn", "resume", { key: "resume", run: () => audio.ctx?.resume() });
  else if (prev.phase === "error" || prev.interrupted || prev.phase === "unsupported") ui.banner(lang, null);
  if (["listening", "hearing"].includes(s.phase) && !wake && navigator.wakeLock) navigator.wakeLock.request("screen").then((w) => (wake = w), () => {});
}

function noSoundWatch(v) {
  if (s.phase !== "listening" || !audio.stream) { silentSince = null; return; }
  if (v > 0.002) { silentSince = null; if (!$("banner").hidden && $("banner").dataset.kind === "warn" && !s.interrupted) ui.banner(lang, null); return; }
  silentSince ??= performance.now();
  if (performance.now() - silentSince > 5000) { ui.banner(lang, "warn", "no_sound"); silentSince = performance.now() + 1e9; }
}

async function serverLoading() {
  try { const r = await fetch("health", { cache: "no-store" }); return r.status === 503; } catch { return false; }
}

function connect() {
  return new Promise((resolve, reject) => {
    const sock = new WebSocket(wsUrl(location));
    sock.binaryType = "arraybuffer";
    sock.onopen = () => { sock.send(JSON.stringify({ type: "config", lang })); resolve(sock); };
    sock.onerror = () => reject(new Error("websocket error"));
    sock.onmessage = (m) => (typeof m.data === "string" ? onServer(m.data) : onAudio(m.data));
    sock.onclose = (e) => onClose(sock, e.code);
  });
}

function onServer(raw) {
  let m;
  try { m = JSON.parse(raw); } catch { return; }            // malformed frame: ignored
  if (!m || typeof m !== "object") return;
  const was = s.phase;
  dispatch({ type: "server", msg: m });
  if (!["listening", "hearing", "thinking", "speaking"].includes(s.phase)) return;
  if (m.type === "partial") { if (was === "speaking") audio.stopPlayback(); ui.partial(m.text); }
  else if (m.type === "final") ui.finalText(lang, m.text);
  else if (m.type === "context") ui.sources(lang, m.ids || []);
  else if (m.type === "token") ui.token(m.text);
  else if (m.type === "cancel") { audio.stopPlayback(); ui.event(lang, "cancelled"); }
  else if (m.type === "echo_ignored") ui.event(lang, "echo_ignored", m.text);
  else if (m.type === "metrics") { hist.push(m); ui.latency(lang, m, hist); ui.logLine(lang, "a", $("reply").textContent); }
}

function onAudio(buf) {
  dispatch({ type: "audio" });
  if (!s.dropAudio && ["speaking", "hearing"].includes(s.phase)) audio.play(buf);   // frames of a cancelled turn are dropped
}

async function onClose(sock, code) {
  if (sock !== ws) return;
  ws = null;
  audio.stopPlayback();
  if (intentional) return;
  const loading = code === 1013 && (await serverLoading());
  dispatch({ type: "close", code, loading });
  const info = closeInfo(code, { loading });
  teardown();
  if (loading) setTimeout(() => s.phase === "loading" && start(), 2000);
  else if (info.retry) {
    const d = backoff(retries++);
    if (d != null) setTimeout(() => s.phase === "error" && s.error === "network" && start(), d);
  }
}

function teardown() {
  audio.stopInput();
  ui.level(0);
  wake?.release?.().catch(() => {}); wake = null;
}

async function start(file) {
  if (!env.ok && env.problems.some((p) => p !== "no_mic_api")) return;
  const ready = audio.unlock();                              // synchronously, inside the gesture (Safari)
  dispatch({ type: "start" });
  if (s.phase !== "connecting") return;
  intentional = false;
  try {
    await ready;
    if (await serverLoading()) { dispatch({ type: "close", code: 1013, loading: true }); setTimeout(() => s.phase === "loading" && start(file), 2000); return; }
    const sock = await connect();
    if (s.phase !== "connecting") { sock.close(); return; }  // Stop was pressed while connecting
    ws = sock;
    dispatch({ type: "ws_open" });
    retries = 0;
    if (file) { ui.renderStatus(lang, s, ""); await audio.startFile(file, () => {}); }
    else await audio.startMic();
  } catch (e) {
    if (e?.name && /Error$/.test(e.name) && e.name !== "Error") dispatch({ type: "mic_error", name: e.name });
    else dispatch({ type: "close", code: 1006 });
    stop(true);
  }
}

function stop(keepState = false) {
  intentional = true;
  if (ws?.readyState <= 1) { try { ws.send(JSON.stringify({ type: "stop" })); } catch { /* closing */ } ws.close(); }
  ws = null;
  audio.stopPlayback();
  teardown();
  if (!keepState) dispatch({ type: "stop" });
}

const live = () => ["connecting", "listening", "hearing", "thinking", "speaking"].includes(s.phase);

// ---- wiring ----
const dock = document.querySelector(".dock");    // the dock wraps to 2-3 rows on phones: pad the page by its real height
const setDock = () => document.documentElement.style.setProperty("--dock-h", `${dock.offsetHeight}px`);
setDock();
new ResizeObserver(setDock).observe(dock);
addEventListener("resize", setDock);
for (const [file, l, label] of REAL_SAMPLES) {
  const o = document.createElement("option"); o.value = file; o.dataset.lang = l; o.textContent = label; $("real-samples").append(o);
}
$("lang").value = lang;
ui.applyStrings(lang);
ui.renderStatus(lang, s);
if (env.problems.includes("insecure") || env.problems.includes("no_worklet")) dispatch({ type: "unsupported", problems: env.problems });
else if (env.problems.includes("no_mic_api")) ui.banner(lang, "warn", "err_no_mic_api");

$("go").onclick = () => (live() ? stop() : start());
$("lang").onchange = (e) => {
  lang = e.target.value; localStorageSet("vora.lang", lang); ui.applyStrings(lang); ui.renderStatus(lang, s);
  if (ws?.readyState === 1) ws.send(JSON.stringify({ type: "config", lang }));
};
$("sample").onchange = async (e) => {
  const opt = e.target.selectedOptions[0];
  if (!opt?.value) return;
  const l = opt.dataset.lang || (opt.value.endsWith("_zh") ? "zh" : "en");
  if (l !== lang) { $("lang").value = l; $("lang").onchange({ target: $("lang") }); }
  if (live()) stop();
  const buf = fetch(`samples/${opt.value}.wav`).then((r) => r.arrayBuffer());
  audio.unlock();
  start(await buf);
  e.target.value = "";
};
$("file").onchange = async (e) => { const f = e.target.files?.[0]; if (!f) return; if (live()) stop(); audio.unlock(); start(await f.arrayBuffer()); e.target.value = ""; };
$("mute").onclick = () => {
  audio.setMuted(!audio.muted);
  $("mute").setAttribute("aria-pressed", String(audio.muted));
  $("mute").dataset.i18n = audio.muted ? "unmute_out" : "mute_out"; ui.applyStrings(lang);
};
$("clear").onclick = () => $("log").replaceChildren();
document.addEventListener("keydown", (e) => {
  if (e.target.closest?.("input, select, textarea, button")) return;
  if (e.code === "Space") { e.preventDefault(); live() ? stop() : start(); }
  if (e.key === "Escape" && live()) stop();
});
window.__vora = { get state() { return s; } };   // read-only hook for the layout check and debugging
