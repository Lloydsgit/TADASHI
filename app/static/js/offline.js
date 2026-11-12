/* ==========================================================================
   Browser-side audio plumbing for offline work.

   Everything here runs locally, so the app keeps working on a static host
   (Vercel) where there is no Python process: decoding an uploaded file,
   running the voice engine over a buffer, writing a WAV, and playing it back.
   ========================================================================== */

import { renderOffline } from './worklet.js';

const SR = 24000;

let playCtx = null;

/** Lazily create one shared context for all playback. */
function context() {
  if (!playCtx) {
    playCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (playCtx.state === 'suspended') playCtx.resume();
  return playCtx;
}

/**
 * Decode any browser-supported audio file to mono float samples at SR.
 * `OfflineAudioContext.decodeAudioData` resamples to the context rate, so
 * asking for a context at SR is all the resampling we need.
 */
export async function decodeToMono(arrayBuffer, targetSr = SR) {
  const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const ctx = new OfflineCtx(1, targetSr, targetSr);
  const buf = await ctx.decodeAudioData(arrayBuffer.slice(0));
  const out = new Float32Array(buf.length);
  for (let ch = 0; ch < buf.numberOfChannels; ch++) {
    const data = buf.getChannelData(ch);
    for (let i = 0; i < buf.length; i++) out[i] += data[i] / buf.numberOfChannels;
  }
  return out;
}

/** Write mono float samples as a 16-bit PCM WAV blob. */
export function encodeWav(samples, sr = SR) {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const view = new DataView(buf);
  const str = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };

  str(0, 'RIFF');
  view.setUint32(4, 36 + n * 2, true);
  str(8, 'WAVE');
  str(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);            // PCM
  view.setUint16(22, 1, true);            // mono
  view.setUint32(24, sr, true);
  view.setUint32(28, sr * 2, true);       // byte rate
  view.setUint16(32, 2, true);            // block align
  view.setUint16(34, 16, true);           // bits
  str(36, 'data');
  view.setUint32(40, n * 2, true);

  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

/** Run a buffer through the voice engine. */
export function render(input, params, sr = SR) {
  return renderOffline(input, params, sr);
}

/** Play mono float samples; returns the AudioBufferSourceNode. */
export function play(samples, sr = SR) {
  const ctx = context();
  const buf = ctx.createBuffer(1, samples.length, sr);
  buf.copyToChannel(samples instanceof Float32Array ? samples : Float32Array.from(samples), 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.start();
  return src;
}

/** Decode, transform and return a downloadable WAV blob. */
export async function convertFile(file, params) {
  const raw = await file.arrayBuffer();
  const decoded = await decodeToMono(raw, SR);
  const morphed = render(decoded, params, SR);
  return encodeWav(morphed, SR);
}

export { SR };
