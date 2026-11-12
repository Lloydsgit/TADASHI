/*
 * Tests for the browser AudioWorklet DSP, run under Node.
 *
 * These mirror the Python DSP tests: the worklet is the engine that runs in
 * "Local" mode, so a bug here is a bug the user hears immediately.
 *
 * Run: node tests/worklet_test.js
 */
const assert = require('assert');
const { loadProcessor, sine, render, dominantFreq } = require('./worklet_harness');

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  ok   ${name}`);
  } catch (err) {
    failed++;
    console.log(`  FAIL ${name}\n       ${err.message}`);
  }
}

const Processor = loadProcessor();

function rms(x) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return Math.sqrt(s / x.length);
}

console.log('worklet DSP');

test('natural passthrough is bit-exact', () => {
  const p = new Processor();
  const x = sine(300, 0.2);
  const y = render(p, x, { output_gain_db: 0 }, 128);
  for (let i = 0; i < x.length; i++) assert.strictEqual(y[i], x[i]);
});

test('pitch shift up raises the fundamental', () => {
  const x = sine(200, 1.0);
  const up = render(new Processor(), x, { pitch_semitones: 7 }, 128);
  const got = dominantFreq(up.subarray(8000));
  assert.ok(Math.abs(got - 299.3) < 12, `expected ~299 Hz, got ${got}`);
});

test('pitch shift down lowers the fundamental', () => {
  const x = sine(200, 1.0);
  const down = render(new Processor(), x, { pitch_semitones: -7 }, 128);
  const got = dominantFreq(down.subarray(8000));
  assert.ok(Math.abs(got - 133.6) < 12, `expected ~134 Hz, got ${got}`);
});

test('output never exceeds the ceiling', () => {
  const x = sine(200, 0.6, 1.4);
  const y = render(new Processor(), x, {
    output_gain_db: 18, drive: 8, drive_mix: 1.0, ring_mod_hz: 60, ring_mod_mix: 0.5,
  }, 128);
  let peak = 0;
  for (let i = 0; i < y.length; i++) peak = Math.max(peak, Math.abs(y[i]));
  assert.ok(peak <= 0.98 + 1e-6, `peak ${peak} exceeded ceiling`);
});

test('all output samples are finite', () => {
  const x = sine(180, 0.4);
  const y = render(new Processor(), x, {
    pitch_semitones: -5, formant_shift: 0.8, drive: 4, drive_mix: 0.5,
    ring_mod_hz: 70, ring_mod_mix: 0.3, chorus_hz: 2, chorus_mix: 0.4,
    echo_ms: 90, echo_feedback: 0.5, echo_mix: 0.3, noise_mix: 0.05,
    stutter_hz: 12, output_gain_db: 6,
  }, 128);
  for (let i = 0; i < y.length; i++) assert.ok(Number.isFinite(y[i]), `sample ${i} not finite`);
});

test('result is independent of render block size', () => {
  const params = {
    pitch_semitones: 3, vibrato_hz: 5.5, vibrato_depth_cents: 40,
    ring_mod_hz: 77, ring_mod_mix: 0.4, chorus_hz: 2.5, chorus_mix: 0.4,
    echo_ms: 80, echo_feedback: 0.4, echo_mix: 0.4, stutter_hz: 11,
  };
  const x = sine(150, 0.5);
  const reference = render(new Processor(), x, params, 1024);
  for (const block of [128, 256, 512]) {
    const got = render(new Processor(), x, params, block);
    let maxDiff = 0;
    for (let i = 0; i < reference.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(reference[i] - got[i]));
    }
    assert.ok(maxDiff < 1e-5, `block ${block} diverged by ${maxDiff}`);
  }
});

test('ring modulation creates sidebands at +/- the carrier', () => {
  const x = sine(300, 0.6);
  const y = render(new Processor(), x, { ring_mod_hz: 60, ring_mod_mix: 0.6 }, 1024);
  const band = (f) => {
    let re = 0; let im = 0;
    for (let i = 0; i < y.length; i++) {
      const a = (2 * Math.PI * f * i) / 24000;
      re += y[i] * Math.cos(a); im -= y[i] * Math.sin(a);
    }
    return Math.sqrt(re * re + im * im) / y.length;
  };
  const carrier = band(300);
  assert.ok(band(240) > carrier * 0.2, 'missing lower sideband');
  assert.ok(band(360) > carrier * 0.2, 'missing upper sideband');
});

test('stutter holds samples within each period', () => {
  const x = sine(200, 0.4);
  const y = render(new Processor(), x, { stutter_hz: 20 }, 1024);
  const period = Math.round(24000 / 20);
  let held = 0;
  for (let i = 100; i < 200; i++) {
    if (y[i] === y[i + 1]) held++;
  }
  assert.ok(held > 80, `stutter did not hold samples (${held}/99)`);
  assert.ok(period > 0);
});

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
