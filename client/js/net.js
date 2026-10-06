// WebSocket helpers. Pure.
const KEYS = { 1000: "st_ended_idle", 1008: "err_origin", 1009: "err_frame", 1011: "err_stalled" };

export function closeInfo(code, { loading = false, reason = "" } = {}) {
  if (code === 1008 && /attempts/i.test(reason)) return { key: "err_throttled", retry: false };   // before the key test: keep the reason free of "key"
  if (code === 1008 && /key/i.test(reason)) return { key: "err_key", retry: false };
  if (code === 1013) return { key: loading ? "st_loading" : "st_busy", retry: false };
  if (code === 1006) return { key: "err_network", retry: true };          // dropped connection: worth a retry
  return { key: KEYS[code] || "err_network", retry: false };
}

const DELAYS = [500, 1500, 4000];
export const backoff = (attempt) => DELAYS[attempt] ?? null;
// The access key never goes into a URL: uvicorn logs request paths with their query strings. It is sent as the first message.
export const wsUrl = (loc) => `${loc.protocol === "https:" ? "wss" : "ws"}://${loc.host}/ws`;

// The link looks like https://host/#key=<urlencoded>. A fragment is never sent to the server (not even in a Referer).
export function parseKeyFromHash(hash = "") {
  for (const part of String(hash ?? "").replace(/^#/, "").split("&")) {
    const i = part.indexOf("=");
    if (i > 0 && part.slice(0, i) === "key") {
      try { return decodeURIComponent(part.slice(i + 1)); } catch { return ""; }
    }
  }
  return "";
}
