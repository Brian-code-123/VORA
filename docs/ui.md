# VORA client (UI)

Plain HTML + ES modules, no build step, no npm dependency. Served by the FastAPI server from `client/` with
`Cache-Control: no-cache` (ETag revalidation, so an upgrade never runs stale JavaScript).

| File | Role |
|---|---|
| `index.html`, `css/app.css` | layout, colour tokens (light/dark), responsive rules |
| `js/state.js` | session state machine (pure, tested in Node) |
| `js/stats.js`, `js/net.js`, `js/env.js`, `js/i18n.js` | latency history, close codes / retries, feature detection, en + 简体中文 strings (pure, tested) |
| `js/audio.js` | AudioContext, microphone through the worklet, reply playback |
| `js/turns.js` | chat helpers: token append, interrupted text, turn cap (pure, tested) |
| `js/ui.js`, `js/main.js` | rendering and wiring |
| `worklet.js` | mic → 16 kHz PCM16 100 ms frames + input level every ~50 ms (tested in a Node `vm` sandbox) |
| `samples/real/` | six MInDS-14 clips (CC-BY-4.0, see `ATTRIBUTION.md`) |

## What the user sees
- A large microphone button. Press it to start and press it again to stop. Its ring shows what is happening: connecting
  (spinning dashes), listening (red), hearing you (grows with your voice), thinking (turning dashes), speaking (green).
- A greeting, one line on what to do, and two example questions. Tapping an example plays that question, so a new visitor
  can try the page without allowing the microphone. When the example is done the page goes back to idle.
- The conversation as chat bubbles: you on the right, VORA on the left, newest at the bottom (last 20 turns). A reply that
  was interrupted ends with "…" and a dashed edge. "New chat" clears the bubbles and brings the greeting back.
- A status line at the bottom with the microphone level. Errors appear in a banner in plain words with a fix: microphone
  permission denied, missing, busy or removed, HTTPS needed off localhost, browser without AudioWorklet, connection lost
  (3 automatic retries, then a Reconnect button), server busy, models loading.
- Everything for engineers sits in one collapsed "Show details" panel: source documents, latency chart and stage times,
  the speculative-start and hold figures, example and file inputs, mute. `?debug=1` opens it on load.
- Keyboard: Space = start/stop, Esc = stop. Focus ring on every control. The conversation is an `aria-live` log.

## Layouts
Mobile first, one centred column (max 44 rem). Bottom status bar, safe-area aware; the page is padded by its measured
height. Phone landscape (height ≤ 500 px): smaller orb, no greeting. Text in `rem`, lines ≤ 75 ch, touch targets ≥ 44 px
(48 px on coarse pointers), light/dark from the OS, `prefers-reduced-motion`, `forced-colors`.

## Verified (2026-10-05, before the first-screen redesign)
- `node --test client/tests` (also run by `pytest tests/test_client_js.py`): state machine, stats, close codes,
  feature detection, i18n key parity, WCAG AA contrast of every text token pair (light + dark), worklet resampling.
- Built-in browser (Chromium): `client/tests/layout_check.js` → zero violations at 320×568, 375×812, 412×915,
  812×375, 768×1024, 1024×768, 1280×720, 1440×900, 1920×1080, 2560×1080, and at 200 % text on a phone; axe-core 0
  violations (light and dark). Real samples played end to end.
- iOS Simulator, iPhone 17 Pro (iOS 26.3), Mobile Safari at `http://localhost:8000`: page loads, layout and light theme correct, no
  insecure-context notice, status ready (`docs/img/ui-ios-safari-iphone17pro.jpg`). Sample playback not driven: the native
  select picker does not open from the automation, so WebKit audio stays on the manual list.
- Screenshots: `docs/img/ui-phone-375-dark.jpg`, `ui-desktop-1440-light.jpg`, `ui-desktop-1440-dark.jpg`.

## Verified after the redesign (2026-10-14)
- `node --test client/tests` and `tests/test_client_js.py` pass. In the built-in browser: idle, example flow back to idle,
  second turn, New chat, `?debug=0` and `?debug=1`, light and dark, 375 px, 812x375, and a layout check at 320 and 1440 px
  with no violations; no console errors.
- Not run after the redesign: axe-core, and the full-size sweep above.

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
