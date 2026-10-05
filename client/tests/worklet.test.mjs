import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

// Runs the real worklet.js in a sandbox with stand-ins for the AudioWorklet globals.
function load(sampleRate) {
  const posted = [];
  let Proc = null;
  const sandbox = {
    sampleRate,
    AudioWorkletProcessor: class { constructor() { this.port = { postMessage: (m) => posted.push(m) }; } },
    registerProcessor: (name, cls) => { assert.equal(name, "pcm16"); Proc = cls; },
  };
  vm.runInNewContext(readFileSync(new URL("../worklet.js", import.meta.url), "utf8"), sandbox);
  return { proc: new Proc(), posted };
}

function feed(proc, seconds, sr, freq = 440, amp = 0.5) {
  const n = Math.round(seconds * sr);
  for (let i = 0; i < n; i += 128) {
    const block = new Float32Array(Math.min(128, n - i));
    for (let j = 0; j < block.length; j++) block[j] = amp * Math.sin(2 * Math.PI * freq * (i + j) / sr);
    assert.equal(proc.process([[block]]), true);
  }
}

// the sandbox has its own ArrayBuffer constructor, so instanceof would fail across realms
const isBuf = (m) => Object.prototype.toString.call(m) === "[object ArrayBuffer]";
const frames = (posted) => posted.filter(isBuf).map((b) => new Int16Array(b));

for (const sr of [48000, 44100, 16000]) {
  test(`${sr} Hz in -> 1600-sample 16 kHz PCM16 frames out`, () => {
    const { proc, posted } = load(sr);
    feed(proc, 1.0, sr);
    const fs = frames(posted);
    assert.ok(fs.length >= 9 && fs.length <= 10, `${fs.length} frames`);
    assert.ok(fs.every((f) => f.length === 1600));
    // frequency preserved: count zero crossings of a 440 Hz tone over 0.5 s of 16 kHz output
    const x = Int16Array.from(fs.flatMap((f) => [...f])).slice(800, 8800);
    let zc = 0;
    for (let i = 1; i < x.length; i++) if ((x[i - 1] < 0) !== (x[i] < 0)) zc++;
    assert.ok(Math.abs(zc - 440) <= 6, `zero crossings ${zc}`);
  });
}

test("level messages: about every 50 ms, RMS of the input", () => {
  const { proc, posted } = load(48000);
  feed(proc, 1.0, 48000, 440, 0.5);
  const levels = posted.filter((m) => m && typeof m === "object" && "level" in m);
  assert.ok(levels.length >= 18 && levels.length <= 22, `${levels.length} level messages`);
  assert.ok(Math.abs(levels.at(-1).level - 0.5 / Math.SQRT2) < 0.02);
});

test("silence gives level ~0 and missing input does not throw", () => {
  const { proc, posted } = load(48000);
  feed(proc, 0.2, 48000, 440, 0);
  assert.ok(posted.filter((m) => "level" in (m || {})).every((m) => m.level < 1e-6));
  assert.equal(proc.process([[]]), true);
  assert.equal(proc.process([]), true);
});

test("no unbounded growth: internal carry-over stays tiny over 10 s", () => {
  const { proc } = load(44100);
  feed(proc, 10, 44100);
  assert.ok(proc.carry.length <= 8, `carry ${proc.carry.length}`);
});
