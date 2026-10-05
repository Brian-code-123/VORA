import test from "node:test";
import assert from "node:assert/strict";
import { initial, reduce } from "../js/state.js";

const run = (events, s = initial()) => events.reduce(reduce, s);
const srv = (msg) => ({ type: "server", msg });

test("happy path: start, connect, hear, think, speak, back to listening", () => {
  let s = run([{ type: "start" }]);
  assert.equal(s.phase, "connecting");
  s = run([{ type: "ws_open" }], s);
  assert.equal(s.phase, "listening");
  s = run([srv({ type: "partial", text: "how long" })], s);
  assert.equal(s.phase, "hearing");
  s = run([srv({ type: "final", text: "how long is the warranty" })], s);
  assert.equal(s.phase, "thinking");
  s = run([srv({ type: "token", text: "Two" }), { type: "audio" }], s);
  assert.equal(s.phase, "speaking");
  s = run([{ type: "playback_end" }], s);
  assert.equal(s.phase, "listening");
});

test("barge-in: speaking + partial -> hearing, flagged", () => {
  const s = run([{ type: "start" }, { type: "ws_open" }, srv({ type: "final", text: "q" }), { type: "audio" }, srv({ type: "partial", text: "wait" })]);
  assert.equal(s.phase, "hearing");
  assert.equal(s.bargeIn, true);
});

test("server cancel stops playback and drops in-flight audio of that turn", () => {
  let s = run([{ type: "start" }, { type: "ws_open" }, srv({ type: "final", text: "q" }), { type: "audio" }, srv({ type: "cancel" })]);
  assert.equal(s.phase, "listening");
  assert.equal(s.dropAudio, true);
  s = run([{ type: "audio" }], s);
  assert.equal(s.phase, "listening");          // late frame of the cancelled turn: not played
  s = run([srv({ type: "final", text: "next" })], s);
  assert.equal(s.dropAudio, false);
});

test("close codes", () => {
  const live = run([{ type: "start" }, { type: "ws_open" }]);
  assert.deepEqual([reduce(live, { type: "close", code: 1000 }).phase, reduce(live, { type: "close", code: 1000 }).reason], ["ended", "idle"]);
  assert.equal(reduce(live, { type: "close", code: 1013, loading: true }).phase, "loading");
  assert.equal(reduce(live, { type: "close", code: 1013, loading: false }).phase, "busy");
  assert.deepEqual([reduce(live, { type: "close", code: 1008 }).phase, reduce(live, { type: "close", code: 1008 }).error], ["error", "origin"]);
  assert.equal(reduce(live, { type: "close", code: 1009 }).error, "frame");
  assert.equal(reduce(live, { type: "close", code: 1011 }).error, "stalled");
  assert.equal(reduce(live, { type: "close", code: 1006 }).error, "network");
});

test("ws close while speaking ends playback state", () => {
  const s = run([{ type: "start" }, { type: "ws_open" }, srv({ type: "final", text: "q" }), { type: "audio" }, { type: "close", code: 1006 }]);
  assert.equal(s.phase, "error");
  assert.equal(s.speaking, false);
});

test("events after ended / stop are ignored; start restarts", () => {
  let s = run([{ type: "start" }, { type: "ws_open" }, { type: "close", code: 1000 }]);
  for (const ev of [srv({ type: "partial", text: "x" }), srv({ type: "final", text: "x" }), { type: "audio" }, { type: "ws_open" }]) s = reduce(s, ev);
  assert.equal(s.phase, "ended");
  s = run([{ type: "stop" }], run([{ type: "start" }, { type: "ws_open" }]));
  assert.equal(reduce(s, srv({ type: "partial", text: "late" })).phase, "idle");
  assert.equal(reduce(s, { type: "start" }).phase, "connecting");
});

test("double start is idempotent, stop during connecting cleans up", () => {
  const s = run([{ type: "start" }, { type: "start" }]);
  assert.equal(s.phase, "connecting");
  const t = reduce(s, { type: "stop" });
  assert.equal(t.phase, "idle");
  assert.equal(reduce(t, { type: "ws_open" }).phase, "idle");   // a socket that opens after Stop does not resurrect the session
});

test("mic errors map to keys", () => {
  const c = run([{ type: "start" }]);
  assert.equal(reduce(c, { type: "mic_error", name: "NotAllowedError" }).error, "mic_denied");
  assert.equal(reduce(c, { type: "mic_error", name: "NotFoundError" }).error, "mic_missing");
  assert.equal(reduce(c, { type: "mic_error", name: "NotReadableError" }).error, "mic_busy");
  assert.equal(reduce(c, { type: "mic_error", name: "Weird" }).error, "mic_other");
  assert.equal(reduce(run([{ type: "start" }, { type: "ws_open" }]), { type: "mic_ended" }).error, "mic_revoked");
});

test("unsupported is terminal; interrupted / resumed toggle a banner flag", () => {
  const u = reduce(initial(), { type: "unsupported", problems: ["insecure"] });
  assert.equal(u.phase, "unsupported");
  assert.equal(reduce(u, { type: "start" }).phase, "unsupported");
  let s = run([{ type: "start" }, { type: "ws_open" }, { type: "interrupted" }]);
  assert.equal(s.interrupted, true);
  s = reduce(s, { type: "resumed" });
  assert.equal(s.interrupted, false);
});

test("unknown events and unknown server messages leave state unchanged", () => {
  const s = run([{ type: "start" }, { type: "ws_open" }]);
  assert.equal(reduce(s, { type: "nope" }), s);
  assert.equal(reduce(s, srv({ type: "mystery" })), s);
  assert.equal(reduce(s, srv(null)), s);
});
