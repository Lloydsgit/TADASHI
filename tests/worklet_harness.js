/*
 * Node-side harness that loads the AudioWorklet DSP so it can be tested
 * without a browser.
 *
 * The worklet is an ES module that registers itself only when the audio-thread
 * globals exist. Here we simply import it and construct the processor with an
 * explicit sample rate, which exercises the same code the browser runs.
 */
const path = require('path');

const WORKLET = path.join(__dirname, '..', 'app', 'static', 'js', 'worklet.js');
const SR = 24000;

async function loadProcessor() {
  const mod = await import(`file://${WORKLET}`);
  return mod.TadashiProcessor;
}

function sine(freq, seconds, amp = 0.5, sr = SR) {
  const n = Math.round(sr * seconds);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = amp * Math.sin((2 * Math.PI * freq * i) / sr);
  return out;
}

/** Push `input` through the processor in `blockSize` chunks. */
function render(processor, input, params, blockSize) {
  processor.configure(params);
  const out = new Float32Array(input.length);
  for (let s = 0; s < input.length; s += blockSize) {
    const n = Math.min(blockSize, input.length - s);
    const inp = new Float32Array(n);
    inp.set(input.subarray(s, s + n));
    const outCh = new Float32Array(n);
    processor.process([[inp]], [[outCh]]);
    out.set(outCh.subarray(0, n), s);
  }
  return out;
}

/** Coarse magnitude scan for the strongest sinusoid. */
function dominantFreq(x, sr = SR) {
  const n = x.length;
  let best = 0;
  let bestMag = -1;
  for (let f = 40; f < 2000; f += 0.5) {
    let re = 0;
    let im = 0;
    for (let i = 0; i < n; i++) {
      const a = (2 * Math.PI * f * i) / sr;
      re += x[i] * Math.cos(a);
      im -= x[i] * Math.sin(a);
    }
    const mag = re * re + im * im;
    if (mag > bestMag) { bestMag = mag; best = f; }
  }
  return best;
}

/** Magnitude at a single frequency. */
function magAt(x, freq, sr = SR) {
  let re = 0;
  let im = 0;
  for (let i = 0; i < x.length; i++) {
    const a = (2 * Math.PI * freq * i) / sr;
    re += x[i] * Math.cos(a);
    im -= x[i] * Math.sin(a);
  }
  return Math.sqrt(re * re + im * im) / x.length;
}

module.exports = { loadProcessor, sine, render, dominantFreq, magAt, SR };
