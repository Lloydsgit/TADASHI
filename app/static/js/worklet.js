/*
 * TADASHI real-time voice engine (AudioWorklet).
 *
 * Runs the same transformation chain as the Python DSP, but inside the audio
 * thread so local mode adds no network round trip. Parameters arrive over the
 * message port as a plain object; the processor rebuilds its filter
 * coefficients whenever they change.
 *
 * Chain: pitch shift (2-tap crossfade + jitter/vibrato) -> formant emphasis
 * -> body/presence/brightness EQ -> ring mod -> drive -> breath noise
 * -> chorus -> echo -> limiter -> output gain
 */

const EPS = 1e-12;
const TWO_PI = Math.PI * 2;

function dbToLin(db) { return Math.pow(10, db / 20); }
function clamp(x, lo, hi) { return x < lo ? lo : x > hi ? hi : x; }

/* --- biquad, transposed direct form II ------------------------------- */
class Biquad {
  constructor() { this.reset(); this.setIdentity(); }
  setIdentity() { this.b0 = 1; this.b1 = 0; this.b2 = 0; this.a1 = 0; this.a2 = 0; }
  reset() { this.z1 = 0; this.z2 = 0; }

  setCoeffs(b0, b1, b2, a0, a1, a2) {
    this.b0 = b0 / a0; this.b1 = b1 / a0; this.b2 = b2 / a0;
    this.a1 = a1 / a0; this.a2 = a2 / a0;
  }

  peaking(sr, f0, q, gainDb) {
    if (Math.abs(gainDb) < 1e-4) { this.setIdentity(); return; }
    const w0 = TWO_PI * clamp(f0, 5, 0.45 * sr) / sr;
    const alpha = Math.sin(w0) / (2 * Math.max(q, 0.05));
    const cw = Math.cos(w0);
    const A = Math.pow(10, gainDb / 40);
    this.setCoeffs(1 + alpha * A, -2 * cw, 1 - alpha * A,
                   1 + alpha / A, -2 * cw, 1 - alpha / A);
  }

  lowShelf(sr, f0, gainDb, slope = 1) {
    if (Math.abs(gainDb) < 1e-4) { this.setIdentity(); return; }
    const w0 = TWO_PI * clamp(f0, 5, 0.45 * sr) / sr;
    const cw = Math.cos(w0);
    const A = Math.pow(10, gainDb / 40);
    const alpha = Math.sin(w0) / 2 * Math.sqrt((A + 1 / A) * (1 / slope - 1) + 2);
    const sA = Math.sqrt(A);
    this.setCoeffs(
      A * ((A + 1) - (A - 1) * cw + 2 * sA * alpha),
      2 * A * ((A - 1) - (A + 1) * cw),
      A * ((A + 1) - (A - 1) * cw - 2 * sA * alpha),
      (A + 1) + (A - 1) * cw + 2 * sA * alpha,
      -2 * ((A - 1) + (A + 1) * cw),
      (A + 1) + (A - 1) * cw - 2 * sA * alpha);
  }

  highShelf(sr, f0, gainDb, slope = 1) {
    if (Math.abs(gainDb) < 1e-4) { this.setIdentity(); return; }
    const w0 = TWO_PI * clamp(f0, 5, 0.45 * sr) / sr;
    const cw = Math.cos(w0);
    const A = Math.pow(10, gainDb / 40);
    const alpha = Math.sin(w0) / 2 * Math.sqrt((A + 1 / A) * (1 / slope - 1) + 2);
    const sA = Math.sqrt(A);
    this.setCoeffs(
      A * ((A + 1) + (A - 1) * cw + 2 * sA * alpha),
      -2 * A * ((A - 1) + (A + 1) * cw),
      A * ((A + 1) + (A - 1) * cw - 2 * sA * alpha),
      (A + 1) - (A - 1) * cw + 2 * sA * alpha,
      2 * ((A - 1) - (A + 1) * cw),
      (A + 1) - (A - 1) * cw - 2 * sA * alpha);
  }

  bandpass(sr, f0, q) {
    const w0 = TWO_PI * clamp(f0, 5, 0.45 * sr) / sr;
    const alpha = Math.sin(w0) / (2 * Math.max(q, 0.05));
    const cw = Math.cos(w0);
    this.setCoeffs(alpha, 0, -alpha, 1 + alpha, -2 * cw, 1 - alpha);
  }

  process(x) {
    const y = this.b0 * x + this.z1;
    this.z1 = this.b1 * x - this.a1 * y + this.z2;
    this.z2 = this.b2 * x - this.a2 * y;
    return y;
  }
}

/* --- pitch shifter --------------------------------------------------- */
class PitchShifter {
  constructor(sr, windowMs = 44) {
    this.window = Math.max(64, Math.round(sr * windowMs / 1000));
    this.size = 2 * this.window + 8;
    this.buf = new Float32Array(this.size);
    this.wp = 0;
    this.phase = 0;
    this.cosTable = new Float32Array(this.window);
    for (let i = 0; i < this.window; i++) {
      this.cosTable[i] = 0.5 * (1 - Math.cos(TWO_PI * i / this.window));
    }
  }

  read(d) {
    let idx = (this.wp - d) % this.size;
    if (idx < 0) idx += this.size;
    const i0 = Math.floor(idx);
    const frac = idx - i0;
    const i1 = (i0 + 1) % this.size;
    return this.buf[i0] * (1 - frac) + this.buf[i1] * frac;
  }

  process(x, ratio, n) {
    for (let i = 0; i < n; i++) {
      this.buf[this.wp] = x[i];
      const d1 = this.phase;
      const d2 = (this.phase + this.window * 0.5) % this.window;
      const w = this.cosTable[Math.floor(d1) % this.window];
      const t1 = this.read(d1);
      const t2 = this.read(d2);
      x[i] = t1 * w + t2 * (1 - w);
      this.phase += 1 - ratio[i];
      if (this.phase >= this.window) this.phase -= this.window;
      else if (this.phase < 0) this.phase += this.window;
      this.wp = (this.wp + 1) % this.size;
    }
  }
}

/* --- delay line ------------------------------------------------------ */
class DelayLine {
  constructor(sr, maxMs) {
    this.size = Math.max(4, Math.round(sr * maxMs / 1000) + 4);
    this.buf = new Float32Array(this.size);
    this.pos = 0;
  }
  read(d) {
    d = clamp(d, 0, this.size - 2);
    let idx = this.pos - d;
    while (idx < 0) idx += this.size;
    const i0 = Math.floor(idx);
    const frac = idx - i0;
    const i1 = (i0 + 1) % this.size;
    return this.buf[i0] * (1 - frac) + this.buf[i1] * frac;
  }
  write(x) { this.buf[this.pos] = x; this.pos = (this.pos + 1) % this.size; }
}

/* --- noise source ---------------------------------------------------- */
class NoiseSource {
  constructor(sr) {
    this.sr = sr;
    this.bp = new Biquad();
    this.env = 0;
    this.tone = -1;
  }
  process(sample, amount, toneHz, gate) {
    if (amount <= 0) return 0;
    if (Math.abs(toneHz - this.tone) > 1) { this.bp.bandpass(this.sr, toneHz, 0.8); this.tone = toneHz; }
    this.env = Math.max(Math.abs(sample), this.env * 0.9995);
    const gateAmp = clamp(gate, 0, 1);
    return this.bp.process(Math.random() * 2 - 1) * gateAmp * amount * 0.5;
  }
}

class TadashiProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() { return []; }

  constructor() {
    super();
    this.sr = sampleRate;
    this.params = null;
    this.t = 0;
    this.jitter = 0;
    this.vibPhase = 0;
    this.ringPhase = 0;
    this.chorusPhase = 0;
    this.held = 0;
    this.limGain = 1;
    this.bypassed = true;

    this.dc = new Biquad(); this.dc.setIdentity();
    this.pitch = new PitchShifter(this.sr);
    this.formants = [new Biquad(), new Biquad(), new Biquad(), new Biquad()];
    this.body = new Biquad(); this.presence = new Biquad(); this.bright = new Biquad();
    this.noise = new NoiseSource(this.sr);
    this.chorusDl = new DelayLine(this.sr, 40);
    this.echoDl = new DelayLine(this.sr, 1200);
    this.echoDamp = new Biquad();
    this.ratioBuf = new Float32Array(256);

    this.port.onmessage = (e) => {
      if (e.data && e.data.type === 'params') this.configure(e.data.params);
    };
    this.port.postMessage({ type: 'worklet_ready', sampleRate: this.sr });
  }

  configure(p) {
    this.params = p;
    const g = (k, d) => (p && typeof p[k] === 'number' ? p[k] : d);

    // formant emphasis
    const shift = clamp(g('formant_shift', 1), 0.5, 2);
    const fg = clamp(Math.abs(shift - 1) * 34, 0, 7) * (Math.abs(shift - 1) > 1e-3 ? 1 : 0);
    const hz = [520, 1180, 2600, 3400];
    const q = [1.1, 1.3, 1.6, 1.8];
    for (let i = 0; i < 4; i++) this.formants[i].peaking(this.sr, hz[i] * shift, q[i], fg);

    this.body.lowShelf(this.sr, g('body_freq', 180), g('body_gain_db', 0));
    this.presence.peaking(this.sr, g('presence_freq', 2800), g('presence_q', 0.9), g('presence_gain_db', 0));
    this.bright.highShelf(this.sr, g('brightness_freq', 5000), g('brightness_gain_db', 0));
    this.echoDamp.peaking(this.sr, 6500, 0.7, -8);

    this.bypassed = this.isNeutral();
  }

  isNeutral() {
    const p = this.params;
    if (!p) return true;
    const keys = ['pitch_semitones', 'pitch_jitter', 'vibrato_depth_cents', 'body_gain_db',
      'presence_gain_db', 'brightness_gain_db', 'ring_mod_mix', 'drive_mix', 'noise_mix',
      'chorus_mix', 'echo_mix', 'stutter_hz'];
    for (const k of keys) if (Math.abs(p[k] || 0) > 1e-6) return false;
    return Math.abs((p.formant_shift || 1) - 1) < 1e-3 && Math.abs(p.output_gain_db || 0) < 1e-3;
  }

  process(inputs, outputs) {
    const input = inputs[0];
    const output = outputs[0];
    if (!output || output.length === 0) return true;

    const inCh = input && input.length ? input[0] : null;
    const n = output[0].length;

    if (!inCh) {
      for (const ch of output) ch.fill(0);
      return true;
    }

    if (this.bypassed || !this.params) {
      for (const ch of output) ch.set(inCh.subarray(0, n));
      this.t += n;
      return true;
    }

    const p = this.params;
    const g = (k, d) => (typeof p[k] === 'number' ? p[k] : d);
    const work = new Float32Array(n);

    // pitch ratio, with optional jitter + vibrato
    const semi = clamp(g('pitch_semitones', 0), -24, 24);
    const jitterAmt = g('pitch_jitter', 0);
    const vibCents = g('vibrato_depth_cents', 0);
    const vibHz = g('vibrato_hz', 0);
    const needPitch = Math.abs(semi) > 1e-3 || jitterAmt > 1e-4 || vibCents > 1e-3;

    if (needPitch) {
      if (this.ratioBuf.length < n) this.ratioBuf = new Float32Array(n);
      const ratio = this.ratioBuf;
      for (let i = 0; i < n; i++) {
        let inst = semi;
        if (jitterAmt > 0) {
          this.jitter = 0.995 * this.jitter + 0.005 * (Math.random() * 2 - 1) * 6;
          inst += this.jitter * jitterAmt;
        }
        if (vibCents > 0 && vibHz > 0) {
          inst += (vibCents / 100) * Math.sin(this.vibPhase + TWO_PI * vibHz * i / this.sr);
        }
        ratio[i] = Math.pow(2, clamp(inst, -24, 24) / 12);
      }
      if (vibCents > 0 && vibHz > 0) {
        this.vibPhase = (this.vibPhase + TWO_PI * vibHz * n / this.sr) % TWO_PI;
      }
      work.set(inCh.subarray(0, n));
      this.pitch.process(work, ratio, n);
    } else {
      work.set(inCh.subarray(0, n));
    }

    // formant bank
    if (Math.abs(g('formant_shift', 1) - 1) > 1e-3) {
      for (let i = 0; i < n; i++) {
        let s = work[i];
        for (const f of this.formants) s = f.process(s);
        work[i] = s;
      }
    }

    const ringMix = g('ring_mod_mix', 0);
    const ringHz = g('ring_mod_hz', 0);
    const drive = g('drive', 0);
    const driveMix = g('drive_mix', 0);
    const noiseMix = g('noise_mix', 0);
    const noiseTone = g('noise_tone_hz', 3200);
    const chorusMix = g('chorus_mix', 0);
    const chorusHz = Math.max(g('chorus_hz', 0), 0.01);
    const chorusDepth = g('chorus_depth_ms', 0);
    const echoMix = g('echo_mix', 0);
    const echoMs = g('echo_ms', 0);
    const echoFb = clamp(g('echo_feedback', 0), 0, 0.92);
    const stutterHz = g('stutter_hz', 0);
    const outGain = dbToLin(g('output_gain_db', 0));

    for (let i = 0; i < n; i++) {
      let s = work[i];

      s = this.body.process(s);
      s = this.presence.process(s);
      s = this.bright.process(s);

      if (ringMix > 0 && ringHz > 0) {
        const carrier = Math.sin(this.ringPhase + TWO_PI * ringHz * i / this.sr);
        s = s * (1 - ringMix) + s * carrier * ringMix;
      }
      if (drive > 0 && driveMix > 0) {
        const shaped = Math.tanh(s * (1 + drive)) / Math.max(1, 1 + drive * 0.35);
        s = s * (1 - driveMix) + shaped * driveMix;
      }
      if (noiseMix > 0) {
        s += this.noise.process(s, noiseMix, noiseTone, Math.min(1, Math.abs(s) * 3));
      }
      if (chorusMix > 0) {
        const lfo = Math.sin(this.chorusPhase + TWO_PI * chorusHz * i / this.sr);
        const mod = this.sr * 0.012 + this.sr * (chorusDepth / 1000) * (0.5 + 0.5 * lfo);
        s = s * (1 - chorusMix) + this.chorusDl.read(mod) * chorusMix;
        this.chorusDl.write(s);
      }
      if (echoMix > 0) {
        const d = this.sr * echoMs / 1000;
        const delayed = this.echoDl.read(d);
        const damped = this.echoDamp.process(delayed);
        const wet = delayed * 0.6 + damped * 0.4;
        this.echoDl.write(s + wet * echoFb);
        s = s + wet * echoMix;
      }
      if (stutterHz > 0) {
        const period = Math.max(2, Math.round(this.sr / stutterHz));
        if ((this.t + i) % period === 0) this.held = s;
        s = this.held;
      }

      // limiter: output gain first so the ceiling is a hard guarantee
      s *= outGain;
      const target = Math.min(1, 0.98 / (Math.abs(s) + EPS));
      const coeff = target < this.limGain ? 0.35 : 0.0006;
      this.limGain += (target - this.limGain) * coeff;

      work[i] = clamp(s * this.limGain, -0.98, 0.98);
    }

    this.ringPhase = (this.ringPhase + TWO_PI * ringHz * n / this.sr) % TWO_PI;
    this.chorusPhase = (this.chorusPhase + TWO_PI * chorusHz * n / this.sr) % TWO_PI;
    this.t += n;

    for (const ch of output) ch.set(work.subarray(0, n));
    return true;
  }
}

registerProcessor('tadashi-voice', TadashiProcessor);
