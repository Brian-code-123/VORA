// Browser audio: AudioContext, microphone through the PCM16 worklet, reply playback. DOM-free but browser-only.
export class Audio {
  constructor({ onFrame, onLevel, onPlaybackEnd, onStateChange, onMicEnded }) {
    Object.assign(this, { onFrame, onLevel, onPlaybackEnd, onStateChange, onMicEnded });
    this.ctx = null; this.node = null; this.stream = null; this.src = null;
    this.sources = new Set(); this.playAt = 0; this.muted = false;
  }

  // Must run synchronously inside the click handler: Safari/iOS only unlock audio during a user gesture.
  unlock() {
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AC();
      this.ctx.onstatechange = () => this.onStateChange?.(this.ctx.state);
      this.out = this.ctx.createGain();
      this.out.connect(this.ctx.destination);
      this.ready = this.ctx.audioWorklet.addModule("worklet.js");
    }
    if (this.ctx.state !== "running") this.ctx.resume();   // not awaited: keeps the gesture
    return this.ready;
  }

  async _attach(sourceNode) {
    await this.ready;
    this.node = new AudioWorkletNode(this.ctx, "pcm16");
    this.node.port.onmessage = (e) => (e.data instanceof ArrayBuffer ? this.onFrame(e.data) : this.onLevel(e.data.level));
    const silent = this.ctx.createGain();
    silent.gain.value = 0;                                   // the worklet must be pulled, but the mic must not be heard
    sourceNode.connect(this.node); this.node.connect(silent); silent.connect(this.ctx.destination);
    this.src = sourceNode;
  }

  async startMic() {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
    for (const tr of this.stream.getAudioTracks()) tr.onended = () => this.onMicEnded?.();
    await this._attach(this.ctx.createMediaStreamSource(this.stream));
  }

  async startFile(arrayBuffer, onEnded) {
    const audio = await this.ctx.decodeAudioData(arrayBuffer);
    const pad = this.ctx.createBuffer(1, audio.length + Math.floor(audio.sampleRate * 2), audio.sampleRate);   // 2 s of silence: the endpoint fires
    pad.copyToChannel(audio.getChannelData(0), 0);
    const src = this.ctx.createBufferSource();
    src.buffer = pad;
    src.onended = () => onEnded?.();
    await this._attach(src);
    src.start();
  }

  play(buf) {
    if (!this.ctx) return;
    const i16 = new Int16Array(buf), f32 = new Float32Array(i16.length);
    for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
    const ab = this.ctx.createBuffer(1, f32.length, 16000);
    ab.copyToChannel(f32, 0);
    const s = this.ctx.createBufferSource();
    s.buffer = ab; s.connect(this.out);
    this.playAt = Math.max(this.playAt, this.ctx.currentTime + 0.08);   // ~80 ms jitter lead
    s.start(this.playAt); this.playAt += ab.duration;
    this.sources.add(s);
    s.onended = () => { this.sources.delete(s); if (!this.sources.size) this.onPlaybackEnd?.(); };
  }

  stopPlayback() {
    for (const s of this.sources) { s.onended = null; try { s.stop(); } catch { /* already stopped */ } }
    this.sources.clear(); this.playAt = 0;
  }

  setMuted(m) { this.muted = m; if (this.out) this.out.gain.value = m ? 0 : 1; }

  stopInput() {
    this.stream?.getTracks().forEach((t) => { t.onended = null; t.stop(); });
    try { this.src?.stop?.(); } catch { /* buffer source already ended */ }
    this.node?.disconnect(); this.src?.disconnect?.();
    this.stream = null; this.node = null; this.src = null;
  }
}
