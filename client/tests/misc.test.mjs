import test from "node:test";
import assert from "node:assert/strict";
import { percentile, History, stages } from "../js/stats.js";
import { closeInfo, backoff, wsUrl } from "../js/net.js";
import { detect } from "../js/env.js";
import { STRINGS, t } from "../js/i18n.js";

test("percentile edges", () => {
  assert.equal(percentile([], 50), null);
  assert.equal(percentile([7], 90), 7);
  assert.equal(percentile([1000, 1200, 1400], 90), 1360);
});

test("history keeps the last 12 turns and summarises first-content latency", () => {
  const h = new History(12);
  for (let i = 1; i <= 15; i++) h.push({ first_content_audio_ms: i * 100 });
  assert.equal(h.items.length, 12);
  assert.equal(h.items[0].first_content_audio_ms, 400);
  assert.deepEqual(h.summary(), { n: 12, p50: 950, p90: 1390 });
  h.push({ total: 5 });                                   // a turn without content audio is kept but not summarised
  assert.equal(h.summary().n, 11);
});

test("stage breakdown ignores unknown and missing keys", () => {
  assert.deepEqual(stages({ asr_final: 300, rag_first_token: 400, tts_first_chunk: 60, mystery: 1 }).map((x) => x.key), ["asr", "llm", "tts"]);
  assert.deepEqual(stages({}), []);
});

test("close codes -> message key and retry policy", () => {
  assert.deepEqual(closeInfo(1006), { key: "err_network", retry: true });
  assert.deepEqual(closeInfo(1013, { loading: true }), { key: "st_loading", retry: false });
  assert.deepEqual(closeInfo(1013), { key: "st_busy", retry: false });
  assert.deepEqual(closeInfo(1008), { key: "err_origin", retry: false });
  assert.deepEqual(closeInfo(1000), { key: "st_ended_idle", retry: false });
  assert.deepEqual(closeInfo(4999), { key: "err_network", retry: false });
});

test("backoff is capped and stops after 3 tries", () => {
  assert.deepEqual([0, 1, 2, 3].map(backoff), [500, 1500, 4000, null]);
});

test("ws url follows the page scheme", () => {
  assert.equal(wsUrl({ protocol: "https:", host: "a:8000" }), "wss://a:8000/ws");
  assert.equal(wsUrl({ protocol: "http:", host: "localhost:8000" }), "ws://localhost:8000/ws");
});

test("feature detection", () => {
  const good = { isSecureContext: true, AudioWorkletNode: function () {}, navigator: { mediaDevices: { getUserMedia() {} }, wakeLock: {} }, location: { hostname: "x" } };
  assert.deepEqual(detect(good), { ok: true, problems: [], wakeLock: true });
  const insecure = { ...good, isSecureContext: false, location: { hostname: "192.168.1.5" } };
  assert.deepEqual(detect(insecure).problems, ["insecure"]);
  const old = { isSecureContext: true, navigator: {}, location: { hostname: "x" } };
  assert.deepEqual(detect(old).problems.sort(), ["no_mic_api", "no_worklet"]);
  assert.equal(detect({ ...good, navigator: { mediaDevices: { getUserMedia() {} } } }).wakeLock, false);   // Safari < 16.4
});

test("i18n: en and zh have the same keys and no empty strings; t() fills vars and falls back to en", () => {
  assert.deepEqual(Object.keys(STRINGS.en).sort(), Object.keys(STRINGS.zh).sort());
  for (const lang of ["en", "zh"]) for (const [k, v] of Object.entries(STRINGS[lang])) assert.ok(v.trim(), `${lang}.${k} empty`);
  assert.equal(t("en", "lat_p50", { ms: "1.10 s" }).includes("1.10 s"), true);
  assert.equal(t("fr", "st_idle"), STRINGS.en.st_idle);
  assert.equal(t("en", "no_such_key"), "no_such_key");
});
