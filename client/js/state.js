// Session state machine. Pure: reduce(state, event) -> new state. No DOM, importable in Node for tests.
// Phases: unsupported | idle | connecting | loading | listening | hearing | thinking | speaking | busy | error | ended
export const initial = () => ({ phase: "idle", error: null, reason: null, speaking: false, bargeIn: false, dropAudio: false, interrupted: false });

const LIVE = new Set(["listening", "hearing", "thinking", "speaking"]);
const MIC = { NotAllowedError: "mic_denied", SecurityError: "mic_denied", NotFoundError: "mic_missing", OverconstrainedError: "mic_missing", NotReadableError: "mic_busy", AbortError: "mic_busy" };
const CLOSE = { 1008: "origin", 1009: "frame", 1011: "stalled" };

export function reduce(s, ev) {
  if (!ev || s.phase === "unsupported") return s;
  const to = (phase, extra = {}) => ({ ...s, phase, ...extra });
  switch (ev.type) {
    case "unsupported": return to("unsupported", { error: ev.problems?.[0] || "unsupported" });
    case "start": return ["idle", "ended", "error", "busy", "loading"].includes(s.phase) ? to("connecting", { ...initial(), phase: "connecting" }) : s;
    case "stop": return { ...initial() };
    case "ws_open": return s.phase === "connecting" ? to("listening") : s;
    case "mic_error": return to("error", { error: MIC[ev.name] || "mic_other", speaking: false });
    case "mic_ended": return LIVE.has(s.phase) ? to("error", { error: "mic_revoked", speaking: false }) : s;
    case "interrupted": return { ...s, interrupted: true };
    case "resumed": return { ...s, interrupted: false };
    case "playback_end": return s.phase === "speaking" ? to("listening", { speaking: false }) : { ...s, speaking: false };
    case "audio":
      if (!LIVE.has(s.phase) || s.dropAudio) return s;
      return s.phase === "hearing" ? { ...s, speaking: true } : to("speaking", { speaking: true });
    case "close": {
      if (!LIVE.has(s.phase) && s.phase !== "connecting") return s;
      if (ev.code === 1000) return to("ended", { reason: "idle", speaking: false });
      if (ev.code === 1013) return to(ev.loading ? "loading" : "busy", { speaking: false });
      if (ev.code === 1008 && /attempts/i.test(ev.reason || "")) return to("error", { error: "throttled", speaking: false });
      if (ev.code === 1008 && /key/i.test(ev.reason || "")) return to("error", { error: "key", speaking: false });
      return to("error", { error: CLOSE[ev.code] || "network", speaking: false });
    }
    case "server": {
      const m = ev.msg;
      if (!m || !LIVE.has(s.phase)) return s;
      if (m.type === "partial") return to("hearing", { bargeIn: s.phase === "speaking" || s.bargeIn });
      if (m.type === "final") return to("thinking", { dropAudio: false, bargeIn: false });
      if (m.type === "cancel") return to("listening", { speaking: false, dropAudio: true });
      return s;     // token / context / metrics / echo_ignored: the view shows them, the phase does not change
    }
    default: return s;
  }
}
