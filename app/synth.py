"""A tiny formant voice synthesizer.

Purpose: let a user audition every voice in the bank without speaking into a
microphone. It produces a vowel sequence ("ah-eh-ee-oh-oo") from a glottal pulse
train shaped by three time-varying formant resonators, which is enough for the
ear to judge pitch, brightness and character.

This is a demo source, not a text-to-speech system - browser TTS handles
speaking arbitrary text (see the frontend's "Agent" panel).
"""

from __future__ import annotations

import numpy as np

from .dsp import Biquad

# F1, F2, F3 for a typical adult male vocal tract, per vowel.
VOWELS: dict[str, tuple[float, float, float]] = {
    "ah": (730.0, 1090.0, 2440.0),
    "eh": (530.0, 1840.0, 2480.0),
    "ee": (270.0, 2290.0, 3010.0),
    "oh": (570.0, 840.0, 2410.0),
    "oo": (300.0, 870.0, 2240.0),
}
VOWEL_Q = (12.0, 16.0, 18.0)


def _glottal_source(n: int, sr: int, f0: float, phase: float = 0.0) -> tuple[np.ndarray, float]:
    """Bandlimited-ish pulse train: a shaped sawtooth with a soft knee."""
    t = (np.arange(n) + phase) / sr
    frac = (t * f0) % 1.0
    # Rosenberg-style glottal pulse
    pulse = np.where(frac < 0.4, 0.5 * (1 - np.cos(np.pi * frac / 0.4)),
                     np.where(frac < 0.6, np.cos(np.pi * (frac - 0.4) / 0.2), 0.0))
    pulse = pulse - pulse.mean()
    new_phase = float((phase + n) % max(sr / max(f0, 1e-6), 1.0))
    return pulse, new_phase


def synth_vowels(sr: int = 24000, f0: float = 165.0, order: tuple[str, ...] = ("ah", "eh", "ee", "oh", "oo"),
                 vowel_ms: float = 260.0, gap_ms: float = 60.0) -> np.ndarray:
    """Render a vowel sequence at a fixed F0. Returns float audio in [-1, 1]."""
    block = max(32, int(sr * 0.010))
    per_vowel = int(sr * vowel_ms / 1000.0)
    gap = int(sr * gap_ms / 1000.0)
    total = len(order) * (per_vowel + gap)

    out = np.zeros(total)
    filt = [Biquad(sr) for _ in range(3)]
    phase = 0.0
    pos = 0
    current = VOWELS[order[0]]

    while pos < total:
        n = min(block, total - pos)
        src, phase = _glottal_source(n, sr, f0, phase)

        # which vowel are we in, and how far through it
        seg = pos // (per_vowel + gap)
        within = pos % (per_vowel + gap)
        idx = min(seg, len(order) - 1)
        target = VOWELS[order[idx]]

        if within < per_vowel:
            # glide formants over the first 40 ms for a natural transition
            glide = min(1.0, within / max(sr * 0.040, 1.0))
            freqs = tuple(current[i] + (target[i] - current[i]) * glide for i in range(3))
            amp = min(1.0, within / max(sr * 0.012, 1.0)) * min(
                1.0, (per_vowel - within) / max(sr * 0.025, 1.0))
        else:
            freqs = target
            amp = 0.0
            current = target

        for f, hz, q in zip(filt, freqs, VOWEL_Q):
            f.bandpass(hz, q)
        shaped = src.copy()
        for f in filt:
            shaped = f.process(shaped)

        out[pos:pos + n] = shaped * max(0.0, amp)
        pos += n

    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.85
    return out
