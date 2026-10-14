// DOM rendering. Reads state, never changes it.
import { t } from "./i18n.js";
import { stages } from "./stats.js";
import { appendToken, cutText, overflow, PLACEHOLDER } from "./turns.js";

const $ = (id) => document.getElementById(id);
const sec = (ms) => (ms == null ? "–" : `${(ms / 1000).toFixed(2)} s`);

export function applyStrings(lang) {
  document.documentElement.lang = lang === "zh" ? "zh-Hans" : "en";
  for (const el of document.querySelectorAll("[data-i18n]")) el.textContent = t(lang, el.dataset.i18n);
}

const PHASE_KEY = { idle: "st_idle", connecting: "st_connecting", loading: "st_loading", listening: "st_listening", hearing: "st_hearing",
  thinking: "st_thinking", speaking: "st_speaking", busy: "st_busy", unsupported: "st_unsupported" };

export function renderStatus(lang, s, extra = "") {
  const box = $("status");
  box.dataset.phase = s.phase;
  document.body.dataset.phase = s.phase;      // drives the orb animation
  let key = PHASE_KEY[s.phase];
  if (s.phase === "ended") key = "st_ended_idle";
  if (s.phase === "error") key = `err_${s.error}`;
  $("st").textContent = extra || t(lang, key);
  const live = ["connecting", "listening", "hearing", "thinking", "speaking"].includes(s.phase);
  $("go-label").textContent = t(lang, live ? "stop" : "start");
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
  $("go").style.setProperty("--lvl", (pct / 100).toFixed(2));
  $("lvl").parentElement.setAttribute("aria-valuenow", v.toFixed(3));
}

let curMe = null, curReply = null;      // the bubbles of the turn in progress

function bubble(lang, kind, whoKey) {
  const box = document.createElement("div"), who = document.createElement("span"), text = document.createElement("div");
  box.className = `bubble ${kind}`; who.className = "who"; who.dataset.i18n = whoKey; who.textContent = t(lang, whoKey);
  text.className = "text";
  box.append(who, text);
  const chat = $("chat");
  chat.append(box);
  while (overflow(chat.children.length) > 0) chat.firstChild.remove();
  document.body.classList.add("chatting");
  box.scrollIntoView?.({ block: "nearest" });
  return { box, text };
}

export function partial(lang, text) {
  curMe ??= bubble(lang, "me", "you");
  curMe.text.textContent = text; curMe.text.className = "text partial";
  curMe.box.scrollIntoView?.({ block: "nearest" });
}
export function finalText(lang, text) {
  curMe ??= bubble(lang, "me", "you");
  curMe.text.textContent = text; curMe.text.className = "text";
  curMe = null;
  curReply = bubble(lang, "vora", "vora");
  curReply.text.textContent = PLACEHOLDER;
}
export function token(text, full) {
  if (!curReply) return;
  curReply.text.textContent = typeof full === "string" ? full : appendToken(curReply.text.textContent, text);   // `full` survives dropped tokens
  curReply.box.scrollIntoView?.({ block: "nearest" });
}
export function setReply(text) { if (curReply) curReply.text.textContent = text; }   // the complete answer replaces the live tokens
export function markCut() {
  if (!curReply) return;
  curReply.text.textContent = cutText(curReply.text.textContent);
  curReply.box.classList.add("cut");
}
export function turnDone() { curReply = null; }
export function note(lang, key) {
  const p = document.createElement("p");
  p.className = "note"; p.dataset.i18n = key; p.textContent = t(lang, key);
  $("chat").append(p);
  p.scrollIntoView?.({ block: "nearest" });
}
export function resetChat() {
  $("chat").replaceChildren();
  curMe = curReply = null;
  document.body.classList.remove("chatting");
}

const EXAMPLES = {
  en: [["q_warranty_en", "How long is the X200 warranty?"], ["q_reset_en", "How do I factory reset?"]],
  zh: [["q_warranty_zh", "X200保修期多久？"]],
};
export function setExamplesDisabled(off) { for (const b of document.querySelectorAll(".suggest")) b.disabled = off; }
export function examples(lang, pick) {
  $("examples").replaceChildren(...EXAMPLES[lang].map(([id, text]) => {
    const b = document.createElement("button");
    b.type = "button"; b.className = "suggest"; b.textContent = text; b.onclick = () => pick(id);
    return b;
  }));
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
