// Mic (native rate, float32) -> 16 kHz mono PCM16 frames of 100 ms (1600 samples), plus {level} (input RMS) every ~50 ms.
// Runs on the audio thread: no per-sample array push/splice (GC pauses glitch audio on weak phones); a tiny Float32Array
// carry-over holds the samples the interpolator still needs from the previous block.
class Pcm16Worklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.pos = 0;                         // fractional read position inside [carry, block]
    this.carry = new Float32Array(0);
    this.out = new Int16Array(1600);
    this.n = 0;
    this.levelEvery = Math.round(sampleRate * 0.05);
    this.sumSq = 0;
    this.count = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch || !ch.length) return true;
    for (let i = 0; i < ch.length; i++) {
      this.sumSq += ch[i] * ch[i];
      if (++this.count >= this.levelEvery) {
        this.port.postMessage({ level: Math.sqrt(this.sumSq / this.count) });
        this.sumSq = 0;
        this.count = 0;
      }
    }
    const buf = new Float32Array(this.carry.length + ch.length);
    buf.set(this.carry);
    buf.set(ch, this.carry.length);
    while (this.pos + 1 < buf.length) {            // linear-interpolation resample
      const i = Math.floor(this.pos), f = this.pos - i;
      const v = buf[i] * (1 - f) + buf[i + 1] * f;
      this.out[this.n++] = Math.max(-1, Math.min(1, v)) * 32767;
      this.pos += this.ratio;
      if (this.n === 1600) {
        const frame = this.out.buffer.slice(0);
        this.port.postMessage(frame, [frame]);
        this.n = 0;
      }
    }
    const drop = Math.min(Math.floor(this.pos), buf.length);
    this.carry = buf.slice(drop);
    this.pos -= drop;
    return true;
  }
}
registerProcessor("pcm16", Pcm16Worklet);
