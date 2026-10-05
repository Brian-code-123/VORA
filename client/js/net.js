// WebSocket helpers. Pure.
const KEYS = { 1000: "st_ended_idle", 1008: "err_origin", 1009: "err_frame", 1011: "err_stalled" };

export function closeInfo(code, { loading = false } = {}) {
  if (code === 1013) return { key: loading ? "st_loading" : "st_busy", retry: false };
  if (code === 1006) return { key: "err_network", retry: true };          // dropped connection: worth a retry
  return { key: KEYS[code] || "err_network", retry: false };
}

const DELAYS = [500, 1500, 4000];
export const backoff = (attempt) => DELAYS[attempt] ?? null;
export const wsUrl = (loc) => `${loc.protocol === "https:" ? "wss" : "ws"}://${loc.host}/ws`;
