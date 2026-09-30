// Mic (native rate, float32) -> 16 kHz mono PCM16 frames of 100 ms (1600 samples).
class Pcm16Worklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.pos = 0;            // fractional read position in the pending buffer
    this.pending = [];
    this.out = new Int16Array(1600);
    this.n = 0;
  }
  process(inputs) {
    const ch = inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) this.pending.push(ch[i]);
    // linear-interpolation downsample
    while (this.pos + 1 < this.pending.length) {
      const i = Math.floor(this.pos), f = this.pos - i;
      const v = this.pending[i] * (1 - f) + this.pending[i + 1] * f;
      this.out[this.n++] = Math.max(-1, Math.min(1, v)) * 32767;
      this.pos += this.ratio;
      if (this.n === 1600) {
        const buf = this.out.buffer.slice(0);
        this.port.postMessage(buf, [buf]);
        this.n = 0;
      }
    }
    const drop = Math.floor(this.pos);
    this.pending.splice(0, drop);
    this.pos -= drop;
    return true;
  }
}
registerProcessor("pcm16", Pcm16Worklet);
