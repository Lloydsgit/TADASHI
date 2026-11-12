/*
 * Tests for the browser-side DSP and voice analysis, run under Node.
 *
 * These mirror the Python tests: the worklet is the engine that runs in
 * "Local" mode and on static hosts such as Vercel, so a bug here is a bug
 * the user hears immediately.
 *
 * Run: node tests/worklet_test.js
 */
const assert = require('assert');
const path = require('path');
const { pathToFileURL } = require('url');
const { loadProcessor, sine, render, dominantFreq, magAt, SR } = require('./worklet_harness');

let passed = 0;
let failed = 0;

async function test(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`  ok   ${name}`);
  } catch (err) {
    failed++;
    console.log(`  FAIL ${name}\n       ${err.message}`);
  }
}

const allParams = {
  pitch_semitones: -5, formant_shift: 0.8, drive: 4, drive_mix: 0.5,
  ring_mod_hz: 70, ring_mod_mix: 0.3, chorus_hz: 2, chorus_mix: 0.4,
  echo_ms: 90, echo_feedback: 0.5, echo_mix: 0.3, noise_mix: 0.05,
  stutter_hz: 12, output_gain_db: 6,
};

(async () => {
  const Processor = await loadProcessor();
  const analysis = await import(pathToFileURL(
    path.join(__dirname, '..', 'app', 'static', 'js', 'analysis.js'),
  ).href);

  console.log('worklet DSP');

  await test('natural passthrough is bit-exact', () => {
    const x = sine(300, 0.2);
    const y = render(new Processor(), x, { output_gain_db: 0 }, 128);
    for (let i = 0; i < x.length; i++) assert.strictEqual(y[i], x[i]);
  });

  await test('pitch shift up raises the fundamental', () => {
    const x = sine(200, 1.0);
    const y = render(new Processor(), x, { pitch_semitones: 7 }, 128);
    const got = dominantFreq(y.subarray(8000));
    assert.ok(Math.abs(got - 299.3) < 12, `expected ~299 Hz, got ${got}`);
  });

  await test('pitch shift down lowers the fundamental', () => {
    const x = sine(200, 1.0);
    const y = render(new Processor(), x, { pitch_semitones: -7 }, 128);
    const got = dominantFreq(y.subarray(8000));
    assert.ok(Math.abs(got - 133.6) < 12, `expected ~134 Hz, got ${got}`);
  });

  await test('output never exceeds the ceiling', () => {
    const x = sine(200, 0.6, 1.4);
    const y = render(new Processor(), x, {
      output_gain_db: 18, drive: 8, drive_mix: 1.0, ring_mod_hz: 60, ring_mod_mix: 0.5,
    }, 128);
    let peak = 0;
    for (let i = 0; i < y.length; i++) peak = Math.max(peak, Math.abs(y[i]));
    assert.ok(peak <= 0.98 + 1e-6, `peak ${peak} exceeded ceiling`);
  });

  await test('all output samples are finite', () => {
    const y = render(new Processor(), sine(180, 0.4), allParams, 128);
    for (let i = 0; i < y.length; i++) assert.ok(Number.isFinite(y[i]), `sample ${i} not finite`);
  });

  await test('result is independent of render block size', () => {
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

  await test('ring modulation creates sidebands at +/- the carrier', () => {
    const y = render(new Processor(), sine(300, 0.6), { ring_mod_hz: 60, ring_mod_mix: 0.6 }, 1024);
    const carrier = magAt(y, 300);
    assert.ok(magAt(y, 240) > carrier * 0.2, 'missing lower sideband');
    assert.ok(magAt(y, 360) > carrier * 0.2, 'missing upper sideband');
  });

  await test('stutter holds samples within each period', () => {
    const y = render(new Processor(), sine(200, 0.4), { stutter_hz: 20 }, 1024);
    let held = 0;
    for (let i = 100; i < 200; i++) if (y[i] === y[i + 1]) held++;
    assert.ok(held > 80, `stutter did not hold samples (${held}/99)`);
  });

  console.log('\nvoice analysis');

  await test('estimates the pitch of a pure tone', () => {
    const x = Float32Array.from(sine(150, 1.0, 0.5));
    const f0 = analysis.estimateF0(x, SR);
    assert.ok(Math.abs(f0.f0_median - 150) < 8, `expected ~150 Hz, got ${f0.f0_median}`);
    assert.ok(f0.voiced_ratio > 0.5, `voiced ratio too low: ${f0.voiced_ratio}`);
  });

  await test('tracks pitch across a range of tones', () => {
    for (const hz of [90, 120, 180, 240, 300]) {
      const f0 = analysis.estimateF0(Float32Array.from(sine(hz, 0.8, 0.5)), SR);
      assert.ok(Math.abs(f0.f0_median - hz) < hz * 0.08,
        `${hz} Hz tracked as ${f0.f0_median}`);
    }
  });

  await test('spectral centroid rises with brightness', () => {
    const dark = analysis.spectralFeatures(Float32Array.from(sine(120, 0.5, 0.5)), SR);
    // a bright signal: fundamental plus strong high harmonics
    const n = SR * 0.5;
    const bright = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      bright[i] = 0.3 * (Math.sin(2 * Math.PI * 3000 * i / SR) + Math.sin(2 * Math.PI * 6000 * i / SR));
    }
    const b = analysis.spectralFeatures(bright, SR);
    assert.ok(b.centroid > dark.centroid,
      `bright centroid ${b.centroid} not above dark ${dark.centroid}`);
  });

  await test('preset maps a low voice to a downward pitch shift', () => {
    const features = analysis.analyzeVoice(Float32Array.from(sine(100, 1.0, 0.5)), SR);
    const preset = analysis.presetFromAnalysis(features);
    assert.ok(preset.pitch_semitones < -3,
      `100 Hz should map below the 165 Hz reference, got ${preset.pitch_semitones}`);
    assert.ok(preset.pitch_semitones >= -12, 'pitch shift must stay in range');
    assert.ok(preset.formant_shift >= 0.7 && preset.formant_shift <= 1.45,
      `formant ${preset.formant_shift} out of range`);
  });

  await test('preset maps a high voice to an upward pitch shift', () => {
    const features = analysis.analyzeVoice(Float32Array.from(sine(280, 1.0, 0.5)), SR);
    const preset = analysis.presetFromAnalysis(features);
    assert.ok(preset.pitch_semitones > 3,
      `280 Hz should map above the 165 Hz reference, got ${preset.pitch_semitones}`);
  });

  await test('preset applies cleanly to the engine', () => {
    const features = analysis.analyzeVoice(Float32Array.from(sine(140, 1.0, 0.5)), SR);
    const preset = analysis.presetFromAnalysis(features);
    const y = render(new Processor(), sine(140, 0.3), preset, 1024);
    for (let i = 0; i < y.length; i++) assert.ok(Number.isFinite(y[i]), 'non-finite output');
  });

  await test('names a captured voice', () => {
    const features = analysis.analyzeVoice(Float32Array.from(sine(100, 1.0, 0.5)), SR);
    const name = analysis.nameFromFeatures(features);
    assert.ok(/^(DEEP|LOW|MID|HIGH|BRIGHT) (SMOOTH|CLEAR|SHARP)$/.test(name),
      `unexpected name: ${name}`);
  });

  console.log(`\n${passed} passed, ${failed} failed`);
  process.exit(failed === 0 ? 0 : 1);
})();
