// Feature detection against an injected window (testable in Node).
const LOOPBACK = new Set(["localhost", "127.0.0.1", "[::1]", "::1"]);

export function detect(win) {
  const problems = [];
  if (!win.isSecureContext && !LOOPBACK.has(win.location?.hostname)) problems.push("insecure");   // getUserMedia needs HTTPS off-localhost
  if (!win.navigator?.mediaDevices?.getUserMedia) problems.push("no_mic_api");
  if (!win.AudioWorkletNode) problems.push("no_worklet");
  return { ok: problems.length === 0, problems, wakeLock: Boolean(win.navigator?.wakeLock) };
}
