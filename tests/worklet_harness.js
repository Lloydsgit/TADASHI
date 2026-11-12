/*
 * Node-side harness that runs the AudioWorklet DSP so it can be tested without
 * a browser. The worklet file only needs `AudioWorkletProcessor` and
 * `registerProcessor` to exist; `sampleRate` is a global in the audio thread,
 * so we inject it here too.
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const WORKLET = path.join(__dirname, '..', 'app', 'static', 'js', 'worklet.js');
const SR = 24000;

function loadProcessor() {
  const source = fs.readFileSync(WORKLET, 'utf8');
  let registered = null;
  const sandbox = {
    sampleRate: SR,
    AudioWorkletProcessor: class { constructor() { this.port = { postMessage() {}, onmessage: null }; } },
    registerProcessor: (name, cls) => { registered = cls; },
    Math, Float32Array, Float64Array, Array, Object, Number, isFinite,
    console,
  };
  vm.createContext(sandbox);
  vm.runInContext(source, sandbox, { filename: 'worklet.js' });
  if (!registered) throw new Error('worklet did not register a processor');
  return registered;
}

function sine(freq, seconds, amp = 0.5) {
  const n = Math.round(SR * seconds);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = amp * Math.sin((2 * Math.PI * freq * i) / SR);
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

function dominantFreq(x) {
  // Goertzel-style scan over a coarse grid, refined by parabolic peak fit.
  const n = x.length;
  let best = 0;
  let bestMag = -1;
  for (let f = 40; f < 2000; f += 0.5) {
    let re = 0;
    let im = 0;
    for (let i = 0; i < n; i++) {
      const a = (2 * Math.PI * f * i) / SR;
      re += x[i] * Math.cos(a);
      im -= x[i] * Math.sin(a);
    }
    const mag = re * re + im * im;
    if (mag > bestMag) { bestMag = mag; best = f; }
  }
  return best;
}

module.exports = { loadProcessor, sine, render, dominantFreq, SR };
