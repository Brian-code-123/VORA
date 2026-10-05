// DOM rendering. Reads state, never changes it.
import { t } from "./i18n.js";
import { stages } from "./stats.js";

const $ = (id) => document.getElementById(id);
const sec = (ms) => (ms == null ? "–" : `${(ms / 1000).toFixed(2)} s`);
const LOG_CAP = 100;

export function applyStrings(lang) {
  document.documentElement.lang = lang === "zh" ? "zh-Hans" : "en";
  for (const el of document.querySelectorAll("[data-i18n]")) el.textContent = t(lang, el.dataset.i18n);
}

const PHASE_KEY = { idle: "st_idle", connecting: "st_connecting", loading: "st_loading", listening: "st_listening", hearing: "st_hearing",
  thinking: "st_thinking", speaking: "st_speaking", busy: "st_busy", unsupported: "st_unsupported" };

export function renderStatus(lang, s, extra = "") {
  const box = $("status");
  box.dataset.phase = s.phase;
  let key = PHASE_KEY[s.phase];
  if (s.phase === "ended") key = "st_ended_idle";
  if (s.phase === "error") key = `err_${s.error}`;
  $("st").textContent = extra || t(lang, key);
  const live = ["connecting", "listening", "hearing", "thinking", "speaking"].includes(s.phase);
  $("go").textContent = t(lang, live ? "stop" : "start");
  $("go").setAttribute("aria-pressed", String(live));
  $("go").disabled = s.phase === "unsupported";
}

export function banner(lang, kind, key, action) {
  const b = $("banner");
  if (!kind) { b.hidden = true; b.replaceChildren(); return; }
  b.hidden = false; b.dataset.kind = kind;
  const p = document.createElement("div");
  p.textContent = t(lang, key);
  b.replaceChildren(p);
  if (action) {
    const btn = document.createElement("button");
    btn.type = "button"; btn.textContent = t(lang, action.key); btn.onclick = action.run;
    b.append(btn);
  }
}

export function level(v) {
  const pct = Math.min(100, Math.round(Math.sqrt(v) * 160));          // perceptual-ish scale
  $("lvl").style.width = `${pct}%`;
  $("lvl").parentElement.setAttribute("aria-valuenow", v.toFixed(3));
}

export function partial(text) { const el = $("live"); el.textContent = text; el.className = "text partial"; }
export function finalText(lang, text) {
  const el = $("live"); el.textContent = text; el.className = "text";
  $("reply").textContent = "…";
  logLine(lang, "u", text);
}
export function token(text) { const el = $("reply"); el.textContent = el.textContent === "…" ? text : el.textContent + text; }

export function logLine(lang, kind, text) {
  const log = $("log"), p = document.createElement("p");
  p.className = kind;
  if (kind !== "note") p.dataset.who = t(lang, kind === "u" ? "you" : "vora");
  p.textContent = text;
  if (/[一-鿿]/.test(text)) p.lang = "zh-Hans";
  log.append(p);
  while (log.children.length > LOG_CAP) log.firstChild.remove();
  log.scrollTop = log.scrollHeight;
}

export function sources(lang, ids) {
  const box = $("ctx");
  if (!ids.length) { const s = document.createElement("span"); s.className = "muted"; s.textContent = t(lang, "no_sources"); box.replaceChildren(s); return; }
  box.replaceChildren(...ids.map((id) => { const c = document.createElement("span"); c.className = "chip"; c.textContent = id; return c; }));
}

export function latency(lang, m, hist) {
  const ms = m.first_content_audio_ms;
  $("lat").textContent = sec(ms);
  $("lat").className = `lat-big ${ms == null ? "" : ms <= 1500 ? "good" : "bad"}`;
  const sum = hist.summary();
  $("lat-sub").textContent = sum.n ? `${t(lang, "lat_p50", { ms: sec(sum.p50) })} · ${t(lang, "lat_target")}` : t(lang, "lat_none");
  drawChart(hist.items.map((x) => x.first_content_audio_ms).filter((x) => typeof x === "number"));
  const st = stages(m), max = Math.max(1, ...st.map((x) => x.ms));
  $("stages").replaceChildren(...st.map(({ key, ms: v }) => {
    const row = document.createElement("div"); row.className = "stage";
    const name = document.createElement("span"); name.textContent = t(lang, `stage_${key}`);
    const bar = document.createElement("span"); bar.className = "b"; bar.style.width = `${Math.max(2, (v / max) * 100)}%`;
    const val = document.createElement("span"); val.className = "ms"; val.textContent = `${Math.round(v)} ms`;
    row.append(name, bar, val); return row;
  }));
  const dl = $("details"), rows = [["d_speculated", m.speculated ? "✓" : "–"], ["d_hold", m.hold_ms ? `${Math.round(m.hold_ms)} ms` : "–"], ["d_oov", String(m.tts_oov ?? 0)]];
  dl.replaceChildren(...rows.flatMap(([k, v]) => { const dt = document.createElement("dt"); dt.textContent = t(lang, k); const dd = document.createElement("dd"); dd.textContent = v; return [dt, dd]; }));
}

function drawChart(xs) {
  const svg = $("chart"), W = 240, H = 88, NS = "http://www.w3.org/2000/svg";
  const top = Math.max(2000, ...xs) * 1.05, bw = W / 12;
  const els = xs.map((v, i) => {
    const r = document.createElementNS(NS, "rect"), h = (v / top) * H;
    r.setAttribute("x", String(i * bw + 1)); r.setAttribute("width", String(bw - 2)); r.setAttribute("y", String(H - h)); r.setAttribute("height", String(h));
    if (v > 1500) r.setAttribute("class", "over");
    const tt = document.createElementNS(NS, "title"); tt.textContent = sec(v); r.append(tt);
    return r;
  });
  const line = document.createElementNS(NS, "line"), y = H - (1500 / top) * H;
  line.setAttribute("x1", "0"); line.setAttribute("x2", String(W)); line.setAttribute("y1", String(y)); line.setAttribute("y2", String(y));
  svg.replaceChildren(...els, line);
}

export function event(lang, key, text = "") {
  const el = $("events"), p = document.createElement("div");
  p.textContent = `${t(lang, key)}${text ? `: ${text}` : ""}`;
  el.prepend(p);
  while (el.children.length > 5) el.lastChild.remove();
}
