// Latency history and helpers. Pure.
export function percentile(xs, p) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b), k = (s.length - 1) * p / 100, lo = Math.floor(k), hi = Math.ceil(k);
  return s[lo] + (s[hi] - s[lo]) * (k - lo);
}

export class History {
  constructor(cap = 12) { this.cap = cap; this.items = []; }
  push(m) { this.items.push(m); if (this.items.length > this.cap) this.items.shift(); }
  summary() {
    const xs = this.items.map((m) => m.first_content_audio_ms).filter((x) => typeof x === "number");
    return { n: xs.length, p50: percentile(xs, 50), p90: percentile(xs, 90) };
  }
}

const STAGES = [["asr", "asr_final"], ["llm", "rag_first_token"], ["tts", "tts_first_chunk"]];
export const stages = (m) => STAGES.filter(([, k]) => typeof m[k] === "number").map(([key, k]) => ({ key, ms: m[k] }));
