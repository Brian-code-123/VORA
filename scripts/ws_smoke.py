"""End-to-end smoke over the real WebSocket: stream a wav at real-time pace, expect final, context, audio and metrics.
Usage: python scripts/ws_smoke.py ws://127.0.0.1:8000/ws client/samples/q_warranty_en.wav [--lang en] [--timeout 30] [-v]
Access key (servers with VORA_ACCESS_KEY): --key K or env VORA_KEY; it is sent as the first message, never in the URL.
Exit code 0 = all seen; 1 = something missing (printed); 2 = could not connect; 3 = server closed the socket (e.g. no access key)."""
import argparse
import asyncio
import json
import os
import sys
import time
from math import gcd

import wave

import numpy as np
from scipy.signal import resample_poly


def load_pcm16(path: str) -> bytes:
    """PCM16 wav -> 16 kHz mono PCM16 (stdlib wave + numpy/scipy only, so it runs inside the server image)."""
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2, "16-bit PCM wav expected"
        sr, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, ch)[:, 0].astype(np.float32) / 32768
    if sr != 16000:
        g = gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()


async def run(url: str, wav: str, lang: str, timeout: float, verbose: bool, insecure: bool = False, key: str = "") -> dict:
    import ssl
    import websockets
    pcm = load_pcm16(wav) + bytes(int(1.6 * 16000) * 2)          # trailing silence so the endpoint fires
    seen: dict = {"events": [], "audio_bytes": 0}
    ctx = None
    if url.startswith("wss://"):
        ctx = ssl.create_default_context()
        if insecure:                                   # the demo box uses a self-signed certificate
            ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    async with websockets.connect(url, max_size=None, ssl=ctx) as ws:
        if key:
            await ws.send(json.dumps({"type": "auth", "key": key}))      # must be the first message
        await ws.send(json.dumps({"type": "config", "lang": lang}))

        async def reader():
            async for m in ws:
                if isinstance(m, bytes):
                    seen["audio_bytes"] += len(m)
                    continue
                d = json.loads(m)
                seen["events"].append(d)
                if verbose and d.get("type") != "token":
                    print("  <-", json.dumps(d, ensure_ascii=False)[:160])
                if d.get("type") == "metrics":
                    return
        task = asyncio.create_task(reader())
        t0 = time.perf_counter()
        for i in range(0, len(pcm), 3200):                     # 100 ms frames, real-time pace
            await ws.send(pcm[i:i + 3200])
            await asyncio.sleep(max(0.0, t0 + (i + 3200) / 32000 - time.perf_counter()))
        try:
            await asyncio.wait_for(task, timeout)
        except asyncio.TimeoutError:
            task.cancel()
    return seen


def check(seen: dict) -> list[str]:
    types = [e.get("type") for e in seen["events"]]
    missing = [t for t in ("final", "context", "metrics") if t not in types]
    if seen["audio_bytes"] == 0:
        missing.append("audio")
    return missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("wav")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--key", default=os.environ.get("VORA_KEY", ""), help="access key (default: env VORA_KEY)")
    ap.add_argument("--insecure", action="store_true", help="accept a self-signed certificate (wss://)")
    a = ap.parse_args(argv)
    import websockets
    try:
        seen = asyncio.run(run(a.url, a.wav, a.lang, a.timeout, a.verbose, a.insecure, a.key))
    except websockets.exceptions.ConnectionClosed as e:     # e.g. 1008 "access key missing or wrong"
        rc = e.rcvd or e.sent
        print(json.dumps({"ok": False, "closed": getattr(rc, "code", None), "reason": getattr(rc, "reason", "")}))
        return 3
    except (OSError, asyncio.TimeoutError, websockets.exceptions.InvalidHandshake) as e:
        print("could not connect:", e)
        return 2
    missing = check(seen)
    m = next((e for e in seen["events"] if e.get("type") == "metrics"), {})
    print(json.dumps({"ok": not missing, "missing": missing, "audio_bytes": seen["audio_bytes"],
                      "first_content_audio_ms": m.get("first_content_audio_ms"),
                      "finals": [e["text"] for e in seen["events"] if e.get("type") == "final"]}, ensure_ascii=False))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
