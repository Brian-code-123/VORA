# VORA client (UI)

Plain HTML + ES modules, no build step, no npm dependency. Served by the FastAPI server from `client/` with
`Cache-Control: no-cache` (ETag revalidation, so an upgrade never runs stale JavaScript).

| File | Role |
|---|---|
| `index.html`, `css/app.css` | layout, colour tokens (light/dark), responsive rules |
| `js/state.js` | session state machine (pure, tested in Node) |
| `js/stats.js`, `js/net.js`, `js/env.js`, `js/i18n.js` | latency history, close codes / retries, feature detection, en + 简体中文 strings (pure, tested) |
| `js/audio.js` | AudioContext, microphone through the worklet, reply playback |
| `js/ui.js`, `js/main.js` | rendering and wiring |
| `worklet.js` | mic → 16 kHz PCM16 100 ms frames + input level every ~50 ms (tested in a Node `vm` sandbox) |
| `samples/real/` | six MInDS-14 clips (CC-BY-4.0, see `ATTRIBUTION.md`) |

## What the user sees
- One **status pill** with a coloured dot for every state: ready, connecting, models loading (auto-retry), listening
  (red dot), hearing you (pulsing), thinking, speaking (green; talk to interrupt), busy, session ended, error.
- **Microphone level meter**; "no sound for 5 s" hint when the mic is silent.
- Live transcript, streaming answer, **source chips** (or "no matching document").
- **Latency**: last response (green ≤ 1.5 s, amber above), median of the last 12 turns, a bar chart with the 1.5 s
  target line, and a per-stage breakdown (speech end → text, retrieve + first word, first voice chunk).
- **Details**: speculative start, sentence-end hold, unspeakable characters, interrupted / echo-ignored events.
- Conversation log (capped at 100 lines), Clear, Mute voice, sample questions (synthetic + real voices), audio file.
- Errors in plain words with a fix: mic permission denied / missing / busy / revoked, HTTPS needed off localhost,
  browser without AudioWorklet, origin refused, frame too large, client stalled, connection lost (3 automatic retries
  with back-off, then a Reconnect button), server busy, models loading.
- Keyboard: Space = start/stop, Esc = stop. Focus ring on every control. `aria-live` transcript and status.

## Layouts
Mobile first. Bottom control dock (thumb reach, safe-area aware; the page is padded by its measured height).
≥720 px: conversation | side panel. ≥1024 px: wider side panel. Phone landscape (height ≤ 500 px): no subtitle,
two columns. Text in `rem`, lines ≤ 75 ch, touch targets ≥ 44 px (48 px on coarse pointers), light/dark from the OS,
`prefers-reduced-motion`, `forced-colors`.

## Verified (2026-10-05)
- `node --test client/tests` (also run by `pytest tests/test_client_js.py`): state machine, stats, close codes,
  feature detection, i18n key parity, WCAG AA contrast of every text token pair (light + dark), worklet resampling.
- Built-in browser (Chromium): `client/tests/layout_check.js` → zero violations at 320×568, 375×812, 412×915,
  812×375, 768×1024, 1024×768, 1280×720, 1440×900, 1920×1080, 2560×1080, and at 200 % text on a phone; axe-core 0
  violations (light and dark). Real samples played end to end.
- iOS Simulator, iPhone 17 Pro (iOS 26.3), Mobile Safari at `http://localhost:8000`: page loads, layout and light theme correct, no
  insecure-context notice, status ready (`docs/img/ui-ios-safari-iphone17pro.jpg`). Sample playback not driven: the native
  select picker does not open from the automation, so WebKit audio stays on the manual list.
- Screenshots: `docs/img/ui-phone-375-dark.jpg`, `ui-desktop-1440-light.jpg`, `ui-desktop-1440-dark.jpg`.

## Not verified here (manual checklist in docs/demo-script.md)
A real microphone, Firefox, Edge, a physical iPhone / Android phone, Safari on macOS. The code follows their rules
(AudioContext created and resumed inside the click, `webkitAudioContext` fallback, feature detection, wake lock only
when available, "interrupted" audio state shows a tap-to-resume banner), but that is not the same as testing.

## Phones and tablets on the LAN
Browsers allow the microphone only on HTTPS or localhost. The server binds 127.0.0.1 by default. For a LAN demo:

```bash
scripts/make_cert.sh
```
```bash
VORA_HOST=0.0.0.0 VORA_SSL_CERT=certs/vora.crt VORA_SSL_KEY=certs/vora.key .venv/bin/python -m vora.server
```
Open `https://<LAN-IP>:8000` and accept the self-signed certificate once. Only on a network you trust.
