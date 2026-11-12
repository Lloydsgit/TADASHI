/* ==========================================================================
   Voice-character analysis, in the browser.

   A port of the server's `estimate_f0` / `spectral_features` /
   `preset_from_analysis` so the mimic feature works on a static host, where
   there is no Python process to call. Same measurements, same mapping.
   ========================================================================== */

const EPS = 1e-12;

function clamp(x, lo, hi) { return x < lo ? lo : x > hi ? hi : x; }

/** In-place iterative radix-2 Cooley-Tukey FFT. `n` must be a power of two. */
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wr = Math.cos(ang); const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1; let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k]; const ui = im[i + k];
        const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

function nextPow2(n) { let p = 1; while (p < n) p <<= 1; return p; }

function hann(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
  return w;
}

/** Split `x` into overlapping frames. Returns an array of Float64Array. */
function frames(x, frame, hop) {
  const out = [];
  if (x.length < frame) {
    const pad = new Float64Array(frame);
    pad.set(x);
    out.push(pad);
    return out;
  }
  const count = 1 + Math.floor((x.length - frame) / hop);
  for (let i = 0; i < count; i++) {
    out.push(x.subarray(i * hop, i * hop + frame));
  }
  return out;
}

function median(a) {
  const s = Array.from(a).sort((p, q) => p - q);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function percentile(a, p) {
  const s = Array.from(a).sort((x, y) => x - y);
  if (!s.length) return 0;
  const idx = (p / 100) * (s.length - 1);
  const lo = Math.floor(idx); const hi = Math.ceil(idx);
  return lo === hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (idx - lo);
}

/** Autocorrelation pitch tracking over the whole signal. */
export function estimateF0(x, sr) {
  const frame = Math.round(0.045 * sr);
  const hop = Math.round(0.020 * sr);
  const fr = frames(x, frame, hop);
  const empty = { f0_median: 0, f0_p10: 0, f0_p90: 0, voiced_ratio: 0 };
  if (!fr.length) return empty;

  const n = nextPow2(2 * frame);
  const win = hann(frame);
  const lo = Math.max(2, Math.round(sr / 400));
  const hi = Math.min(frame - 2, Math.round(sr / 60));

  const f0s = [];
  let voicedCount = 0;

  for (const f of fr) {
    let mean = 0;
    for (let i = 0; i < frame; i++) mean += f[i];
    mean /= frame;

    const re = new Float64Array(n);
    const im = new Float64Array(n);
    for (let i = 0; i < frame; i++) re[i] = (f[i] - mean) * win[i];
    fft(re, im);

    // power spectrum -> autocorrelation via inverse FFT
    for (let i = 0; i < n; i++) {
      re[i] = re[i] * re[i] + im[i] * im[i];
      im[i] = 0;
    }
    for (let i = 0; i < n; i++) im[i] = -im[i];
    fft(re, im);

    const ac0 = re[0] + EPS;
    let best = 0; let bestLag = lo;
    for (let lag = lo; lag < hi; lag++) {
      const v = re[lag] / ac0;
      if (v > best) { best = v; bestLag = lag; }
    }
    if (best > 0.35) { voicedCount++; f0s.push(sr / bestLag); }
  }

  if (!f0s.length) return { ...empty, voiced_ratio: voicedCount / fr.length };
  return {
    f0_median: median(f0s),
    f0_p10: percentile(f0s, 10),
    f0_p90: percentile(f0s, 90),
    voiced_ratio: voicedCount / fr.length,
  };
}

/** Spectral centroid, rolloff, flatness, tilt and peak locations. */
export function spectralFeatures(x, sr) {
  const frame = Math.round(0.040 * sr);
  const hop = Math.round(0.020 * sr);
  const fr = frames(x, frame, hop);
  if (!fr.length) return {};

  const n = nextPow2(frame);
  const win = hann(frame);
  const bins = n / 2 + 1;
  const freqs = new Float64Array(bins);
  for (let i = 0; i < bins; i++) freqs[i] = (i * sr) / n;

  const spectra = [];
  const energies = [];
  for (const f of fr) {
    let mean = 0;
    for (let i = 0; i < frame; i++) mean += f[i];
    mean /= frame;
    const re = new Float64Array(n);
    const im = new Float64Array(n);
    for (let i = 0; i < frame; i++) re[i] = (f[i] - mean) * win[i];
    fft(re, im);
    const mag = new Float64Array(bins);
    let e = 0;
    for (let i = 0; i < bins; i++) {
      const m = Math.hypot(re[i], im[i]) + EPS;
      mag[i] = m;
      e += m;
    }
    spectra.push(mag);
    energies.push(e);
  }

  // keep the frames that actually carry energy
  const threshold = energies.length > 2 ? percentile(energies, 60) : 0;
  const keep = energies.map((e) => e > threshold);
  const used = keep.some(Boolean) ? spectra.filter((_, i) => keep[i]) : spectra;

  const meanSpec = new Float64Array(bins);
  for (const s of used) for (let i = 0; i < bins; i++) meanSpec[i] += s[i] / used.length;

  let total = 0; let wsum = 0;
  for (let i = 0; i < bins; i++) { total += meanSpec[i]; wsum += freqs[i] * meanSpec[i]; }
  total += EPS;
  const centroid = wsum / total;

  let cum = 0; let rolloff = 0;
  const target = 0.85 * (total - EPS);
  for (let i = 0; i < bins; i++) { cum += meanSpec[i]; if (cum >= target) { rolloff = freqs[i]; break; } }

  let logSum = 0; let linSum = 0;
  for (let i = 0; i < bins; i++) { logSum += Math.log(meanSpec[i]); linSum += meanSpec[i]; }
  const flatness = Math.exp(logSum / bins) / (linSum / bins + EPS);

  // least-squares slope of the log spectrum over the speech band
  const bandLo = 120; const bandHi = 0.4 * sr;
  let sx = 0; let sy = 0; let sxx = 0; let sxy = 0; let cnt = 0;
  for (let i = 0; i < bins; i++) {
    const fq = freqs[i];
    if (fq <= bandLo || fq >= bandHi) continue;
    const yv = 20 * Math.log10(meanSpec[i] + EPS);
    sx += fq; sy += yv; sxx += fq * fq; sxy += fq * yv; cnt++;
  }
  const slope = cnt > 8 ? (cnt * sxy - sx * sy) / (cnt * sxx - sx * sx + EPS) : 0;

  // formant-ish peaks from a smoothed envelope
  const k = 9;
  const smooth = new Float64Array(bins);
  const hw = hann(k);
  let hwSum = 0;
  for (let i = 0; i < k; i++) hwSum += hw[i];
  for (let i = 0; i < bins; i++) {
    let acc = 0;
    for (let j = 0; j < k; j++) {
      const idx = clamp(i + j - (k >> 1), 0, bins - 1);
      acc += meanSpec[idx] * hw[j];
    }
    smooth[i] = acc / hwSum;
  }
  const peaks = [];
  for (let i = 2; i < bins - 2; i++) {
    if (smooth[i] > smooth[i - 1] && smooth[i] >= smooth[i + 1] && smooth[i] > 0 && freqs[i] > 150) {
      peaks.push(freqs[i]);
    }
  }

  return {
    centroid,
    rolloff,
    flatness,
    tilt_db_per_khz: slope * 1000,
    peaks: peaks.slice(0, 6),
  };
}

/** Full voice-character analysis used by the mimic feature. */
export function analyzeVoice(x, sr) {
  let peak = 0;
  for (let i = 0; i < x.length; i++) peak = Math.max(peak, Math.abs(x[i]));
  let sig = x;
  if (peak > 0) {
    sig = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) sig[i] = (x[i] / peak) * 0.9;
  }
  const pitch = estimateF0(sig, sr);
  const spectral = spectralFeatures(sig, sr);
  return { pitch, spectral, duration_s: x.length / sr };
}

/** Turn measured voice character into DSP parameters (a matched voice). */
export function presetFromAnalysis(features) {
  const f0 = features.pitch || {};
  const sf = features.spectral || {};
  const m = Number(f0.f0_median) || 0;
  const reference = 165;

  let semitones = m > 40 ? 12 * Math.log2(m / reference) : 0;
  semitones = clamp(semitones, -12, 12);

  const centroid = Number(sf.centroid) || 1800;
  const tilt = Number(sf.tilt_db_per_khz) || 0;
  const flatness = Number(sf.flatness) || 0.1;

  const formant = clamp(0.75 + 0.35 * (centroid / 1800), 0.7, 1.45);
  const brightness = clamp(tilt / 12, -9, 9);
  const noise = clamp(flatness * 0.9, 0, 0.06);

  return {
    pitch_semitones: Math.round(semitones * 100) / 100,
    formant_shift: Math.round(formant * 1000) / 1000,
    brightness_gain_db: Math.round(brightness * 100) / 100,
    presence_gain_db: Math.round(clamp(brightness * 0.4, -4, 4) * 100) / 100,
    noise_mix: Math.round(noise * 10000) / 10000,
    output_gain_db: 0,
  };
}

/** Human-readable name for a captured voice. */
export function nameFromFeatures(features) {
  const f0 = Number((features.pitch || {}).f0_median) || 0;
  const centroid = Number((features.spectral || {}).centroid) || 1800;
  if (f0 <= 0) return 'CAPTURED VOICE';
  const base = f0 < 110 ? 'DEEP' : f0 < 150 ? 'LOW' : f0 < 200 ? 'MID' : f0 < 260 ? 'HIGH' : 'BRIGHT';
  const tone = centroid < 1500 ? 'SMOOTH' : centroid < 2600 ? 'CLEAR' : 'SHARP';
  return `${base} ${tone}`;
}
