"""Real-time voice transformation DSP.

Everything here is stateful and chunk-streamable: you feed audio in blocks of
any size and get transformed audio out, with the internal filter/phase state
carried across calls. That is what makes true real-time (low latency) voice
changing possible on CPU.

Signal chain, in order:

    DC/rumble block -> pitch shift (w/ jitter + vibrato) -> formant resonators
    -> body EQ -> presence EQ -> brightness EQ -> ring modulation -> drive
    -> breath noise -> chorus -> echo -> stutter -> limiter -> output gain

Stages whose mix/amount is zero are bypassed entirely, which keeps the
"natural" passthrough both transparent and latency-free.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
from scipy.signal import lfilter

EPS = 1e-12
MAX_SEMITONES = 24.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def lin_to_db(x: float) -> float:
    return float(20.0 * math.log10(max(x, EPS)))


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _one_pole(x: np.ndarray, coeff: float, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Streaming one-pole lowpass: y[i] = a*y[i-1] + (1-a)*x[i]."""
    b = [1.0 - coeff]
    a = [1.0, -coeff]
    y, zf = lfilter(b, a, x, zi=state)
    return y, zf


# --------------------------------------------------------------------------- #
# biquad (RBJ cookbook), streaming via scipy.signal.lfilter
# --------------------------------------------------------------------------- #
class Biquad:
    def __init__(self, sr: int):
        self.sr = sr
        self.b = np.array([1.0, 0.0, 0.0])
        self.a = np.array([1.0, 0.0, 0.0])
        self.zi = np.zeros(2)

    def reset(self) -> None:
        self.zi = np.zeros(2)

    def _set(self, b: Iterable[float], a: Iterable[float]) -> None:
        b = np.asarray(b, float)
        a = np.asarray(a, float)
        self.b = b / a[0]
        self.a = a / a[0]

    def _w0(self, f0: float) -> float:
        f0 = _clamp(f0, 5.0, 0.45 * self.sr)
        return 2.0 * math.pi * f0 / self.sr

    def highpass(self, f0: float, q: float = 0.707) -> "Biquad":
        w0 = self._w0(f0)
        alpha = math.sin(w0) / (2 * q)
        cw = math.cos(w0)
        self._set([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2], [1 + alpha, -2 * cw, 1 - alpha])
        return self

    def lowpass(self, f0: float, q: float = 0.707) -> "Biquad":
        w0 = self._w0(f0)
        alpha = math.sin(w0) / (2 * q)
        cw = math.cos(w0)
        self._set([(1 - cw) / 2, 1 - cw, (1 - cw) / 2], [1 + alpha, -2 * cw, 1 - alpha])
        return self

    def bandpass(self, f0: float, q: float = 1.0) -> "Biquad":
        w0 = self._w0(f0)
        alpha = math.sin(w0) / (2 * q)
        cw = math.cos(w0)
        self._set([alpha, 0.0, -alpha], [1 + alpha, -2 * cw, 1 - alpha])
        return self

    def peaking(self, f0: float, q: float, gain_db: float) -> "Biquad":
        if abs(gain_db) < 1e-4:
            self._set([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
            return self
        w0 = self._w0(f0)
        alpha = math.sin(w0) / (2 * max(q, 0.05))
        cw = math.cos(w0)
        A = 10.0 ** (gain_db / 40.0)
        self._set(
            [1 + alpha * A, -2 * cw, 1 - alpha * A],
            [1 + alpha / A, -2 * cw, 1 - alpha / A],
        )
        return self

    def low_shelf(self, f0: float, gain_db: float, slope: float = 1.0) -> "Biquad":
        if abs(gain_db) < 1e-4:
            self._set([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
            return self
        w0 = self._w0(f0)
        cw = math.cos(w0)
        A = 10.0 ** (gain_db / 40.0)
        alpha = math.sin(w0) / 2 * math.sqrt((A + 1 / A) * (1 / slope - 1) + 2)
        sA = math.sqrt(A)
        self._set(
            [
                A * ((A + 1) - (A - 1) * cw + 2 * sA * alpha),
                2 * A * ((A - 1) - (A + 1) * cw),
                A * ((A + 1) - (A - 1) * cw - 2 * sA * alpha),
            ],
            [
                (A + 1) + (A - 1) * cw + 2 * sA * alpha,
                -2 * ((A - 1) + (A + 1) * cw),
                (A + 1) + (A - 1) * cw - 2 * sA * alpha,
            ],
        )
        return self

    def high_shelf(self, f0: float, gain_db: float, slope: float = 1.0) -> "Biquad":
        if abs(gain_db) < 1e-4:
            self._set([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
            return self
        w0 = self._w0(f0)
        cw = math.cos(w0)
        A = 10.0 ** (gain_db / 40.0)
        alpha = math.sin(w0) / 2 * math.sqrt((A + 1 / A) * (1 / slope - 1) + 2)
        sA = math.sqrt(A)
        self._set(
            [
                A * ((A + 1) + (A - 1) * cw + 2 * sA * alpha),
                -2 * A * ((A - 1) + (A + 1) * cw),
                A * ((A + 1) + (A - 1) * cw - 2 * sA * alpha),
            ],
            [
                (A + 1) - (A - 1) * cw + 2 * sA * alpha,
                2 * ((A - 1) - (A + 1) * cw),
                (A + 1) - (A - 1) * cw - 2 * sA * alpha,
            ],
        )
        return self

    def process(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return x
        y, self.zi = lfilter(self.b, self.a, x, zi=self.zi)
        return y


# --------------------------------------------------------------------------- #
# delay line
# --------------------------------------------------------------------------- #
class DelayLine:
    def __init__(self, sr: int, max_ms: float):
        self.sr = sr
        self.size = max(4, int(sr * max_ms / 1000.0) + 4)
        self.buf = np.zeros(self.size)
        self.pos = 0

    def reset(self) -> None:
        self.buf.fill(0.0)
        self.pos = 0

    def read(self, n: int, delay_samples: float | np.ndarray) -> np.ndarray:
        """Read `n` samples delayed by `delay_samples`, without advancing."""
        if n == 0:
            return np.zeros(0)
        d = np.clip(delay_samples, 0.0, self.size - 2)
        idx = (self.pos + np.arange(n) - d) % self.size
        i0 = np.floor(idx).astype(np.int64)
        frac = idx - i0
        i1 = (i0 + 1) % self.size
        return self.buf[i0] * (1.0 - frac) + self.buf[i1] * frac

    def write(self, x: np.ndarray) -> None:
        n = len(x)
        if n == 0:
            return
        wpos = (self.pos + np.arange(n)) % self.size
        self.buf[wpos] = x
        self.pos = int((self.pos + n) % self.size)

    def process(self, x: np.ndarray, delay_samples: float | np.ndarray) -> np.ndarray:
        """Read the delayed signal, then write the input in.

        Read happens before write, so a delay of 0 samples is a true
        passthrough. This is exact only while the delay is at least as long as
        the block: for shorter delays a later sample in the block would read a
        value written earlier in the same block. Callers with short delays
        should either split into sub-blocks (see ``_echo``) or use
        :meth:`process_fractional`.
        """
        out = self.read(len(x), delay_samples)
        self.write(x)
        return out

    def process_fractional(self, x: np.ndarray, delay_samples: float | np.ndarray) -> np.ndarray:
        """Write first, then read.

        Exact for *any* delay >= 0, including delays shorter than the block,
        because every sample the read can reach has already been written. This
        is the mode to use for modulated delays such as the chorus, where the
        delay routinely drops below the block length.

        The block is split internally so a block longer than the ring buffer can
        never clobber samples it still needs to read.
        """
        n = len(x)
        if n == 0:
            return x
        if np.isscalar(delay_samples):
            delay_samples = np.full(n, float(delay_samples))
        delay_samples = np.asarray(delay_samples, float).ravel()

        out = np.empty(n)
        limit = max(1, self.size // 2)
        for s in range(0, n, limit):
            e = min(n, s + limit)
            seg = x[s:e]
            d = np.clip(delay_samples[s:e], 0.0, self.size - 2)
            self.write(seg)
            idx = (self.pos - len(seg) + np.arange(len(seg)) - d) % self.size
            i0 = np.floor(idx).astype(np.int64)
            frac = idx - i0
            i1 = (i0 + 1) % self.size
            out[s:e] = self.buf[i0] * (1.0 - frac) + self.buf[i1] * frac
        return out


# --------------------------------------------------------------------------- #
# pitch shifter: two crossfaded delay taps (granular / "sine crossfade")
# --------------------------------------------------------------------------- #
class PitchShifter:
    """Streaming pitch shifter.

    Two read pointers sweep the ring buffer at a rate set by the pitch ratio,
    offset by half a window and crossfaded with a raised-cosine so the wrap
    point of one tap is always hidden behind the peak of the other.

    Latency is one half-window, so a 44 ms window costs ~22 ms of delay.
    """

    def __init__(self, sr: int, window_ms: float = 44.0):
        self.sr = sr
        self.window = max(64, int(sr * window_ms / 1000.0))
        self.buf = np.zeros(2 * self.window + 8)
        self.size = len(self.buf)
        self.wp = 0
        self.phase = 0.0

    def reset(self) -> None:
        self.buf.fill(0.0)
        self.wp = 0
        self.phase = 0.0

    def process(self, x: np.ndarray, ratio: np.ndarray | float) -> np.ndarray:
        n = len(x)
        if n == 0:
            return x
        if np.isscalar(ratio):
            ratio = np.full(n, float(ratio))
        ratio = np.asarray(ratio, float)

        out = np.empty(n)
        # Sub-block so the ring buffer only has to span one window + block.
        step = max(1, self.window // 2)
        for s in range(0, n, step):
            e = min(n, s + step)
            m = e - s
            wpos = (self.wp + np.arange(m)) % self.size
            self.buf[wpos] = x[s:e]

            # read pointer advance = 1 - ratio; ratio>1 -> phase decreases.
            delta = 1.0 - ratio[s:e]
            phases = (self.phase + np.cumsum(delta)) % self.window
            self.phase = float((self.phase + np.sum(delta)) % self.window)

            d1 = phases
            d2 = (phases + self.window * 0.5) % self.window
            tap1 = self._tap(wpos, d1)
            tap2 = self._tap(wpos, d2)

            theta = 2.0 * np.pi * d1 / self.window
            w1 = 0.5 * (1.0 - np.cos(theta))
            out[s:e] = tap1 * w1 + tap2 * (1.0 - w1)

            self.wp = int((self.wp + m) % self.size)
        return out

    def _tap(self, wpos: np.ndarray, d: np.ndarray) -> np.ndarray:
        idx = (wpos - d) % self.size
        i0 = np.floor(idx).astype(np.int64)
        frac = idx - i0
        i1 = (i0 + 1) % self.size
        return self.buf[i0] * (1.0 - frac) + self.buf[i1] * frac


# --------------------------------------------------------------------------- #
# formant emphasis bank
# --------------------------------------------------------------------------- #
FORMANT_HZ = (520.0, 1180.0, 2600.0, 3400.0)
FORMANT_Q = (1.1, 1.3, 1.6, 1.8)


class FormantBank:
    """Approximates formant shifting with a bank of peaking resonators whose
    centre frequencies scale with the requested shift.

    This is a spectral-envelope emphasis (cheap, streaming-safe), not an
    LPC/PSOLA formant warp - it moves vowel resonances enough to read as a
    different vocal tract without needing pitch-synchronous analysis.
    """

    def __init__(self, sr: int):
        self.sr = sr
        self.filters = [Biquad(sr) for _ in FORMANT_HZ]
        self.shift = 1.0

    def reset(self) -> None:
        for f in self.filters:
            f.reset()

    def configure(self, shift: float) -> None:
        shift = _clamp(shift, 0.5, 2.0)
        if abs(shift - self.shift) < 1e-4:
            return
        self.shift = shift
        gain = _clamp(abs(shift - 1.0) * 34.0, 0.0, 7.0)
        for f, hz, q in zip(self.filters, FORMANT_HZ, FORMANT_Q):
            f.peaking(hz * shift, q, gain if abs(shift - 1.0) > 1e-3 else 0.0)

    @property
    def active(self) -> bool:
        return abs(self.shift - 1.0) > 1e-3

    def process(self, x: np.ndarray) -> np.ndarray:
        if not self.active:
            return x
        for f in self.filters:
            x = f.process(x)
        return x


# --------------------------------------------------------------------------- #
# noise source (breath / grit), gated by the input envelope
# --------------------------------------------------------------------------- #
class NoiseSource:
    """Breath/grit noise, gated by the input envelope.

    The noise is read sequentially from a fixed seeded buffer and the gate
    filters carry their state across calls, so the output does not depend on how
    the stream happens to be split into blocks.
    """

    def __init__(self, sr: int, seed: int = 0xBEEF):
        self.sr = sr
        self.bp = Biquad(sr)
        self.tone = -1.0
        self.buf = np.random.default_rng(seed).standard_normal(1 << 20)
        self.pos = 0
        self.env_fast = np.zeros(1)
        self.env_slow = np.zeros(1)
        self.primed = False

    def reset(self) -> None:
        self.bp.reset()
        self.pos = 0
        self.env_fast = np.zeros(1)
        self.env_slow = np.zeros(1)
        self.primed = False

    def _noise(self, n: int) -> np.ndarray:
        out = np.empty(n)
        i = 0
        while i < n:
            take = min(n - i, len(self.buf) - self.pos)
            out[i:i + take] = self.buf[self.pos:self.pos + take]
            self.pos = (self.pos + take) % len(self.buf)
            i += take
        return out

    def process(self, x: np.ndarray, amount: float, tone_hz: float) -> np.ndarray:
        if amount <= 0.0 or len(x) == 0:
            return np.zeros_like(x)
        if abs(tone_hz - self.tone) > 1.0:
            self.bp.bandpass(tone_hz, 0.8)
            self.tone = tone_hz

        noise = self.bp.process(self._noise(len(x)))

        mag = np.abs(x)
        a_att = math.exp(-1.0 / max(self.sr * 0.006, 1.0))
        a_rel = math.exp(-1.0 / max(self.sr * 0.090, 1.0))
        if not self.primed:
            self.env_fast = np.array([a_att * mag[0]])
            self.env_slow = np.array([a_rel * mag[0]])
            self.primed = True
        fast, self.env_fast = lfilter([1 - a_att], [1, -a_att], mag, zi=self.env_fast)
        slow, self.env_slow = lfilter([1 - a_rel], [1, -a_rel], mag, zi=self.env_slow)
        env = np.maximum(fast, slow)

        gate = np.tanh(env * 8.0)
        return noise * gate * amount * 0.5


# --------------------------------------------------------------------------- #
# the full voice processor
# --------------------------------------------------------------------------- #
class VoiceProcessor:
    """Applies one voice preset to a stream of audio."""

    def __init__(self, sr: int = 24000, params: dict[str, Any] | None = None,
                 seed: int = 0xBEEF):
        self.sr = sr
        self.p = dict(params or {})
        self.t = 0
        self.vib_phase = 0.0
        self.ring_phase = 0.0
        self.chorus_phase = 0.0
        self.jitter_state = np.zeros(1)
        self.held_value = 0.0
        self._rng = np.random.default_rng(seed)

        self.dc = Biquad(sr).highpass(70.0)
        self.pitch = PitchShifter(sr)
        self.formants = FormantBank(sr)
        self.body = Biquad(sr)
        self.presence = Biquad(sr)
        self.bright = Biquad(sr)
        self.noise = NoiseSource(sr, seed)
        self.chorus_delay = DelayLine(sr, 90.0)
        self.echo_delay = DelayLine(sr, 1200.0)
        self.echo_damp = Biquad(sr).lowpass(6500.0)
        self.lim_zi_fast = np.zeros(1)
        self.lim_zi_slow = np.zeros(1)
        self.lim_primed = False
        self.configure(self.p)

    # -- configuration ----------------------------------------------------- #
    def configure(self, params: dict[str, Any]) -> None:
        self.p = dict(params)
        g = self._get
        self.formants.configure(g("formant_shift", 1.0))
        self.body.low_shelf(g("body_freq", 180.0), g("body_gain_db", 0.0))
        self.presence.peaking(g("presence_freq", 2800.0), g("presence_q", 0.9),
                              g("presence_gain_db", 0.0))
        self.bright.high_shelf(g("brightness_freq", 5000.0), g("brightness_gain_db", 0.0))
        self.echo_max = int(self.sr * 1.2)

    def _get(self, key: str, default: float) -> float:
        try:
            return float(self.p.get(key, default))
        except (TypeError, ValueError):
            return default

    @property
    def needs_pitch(self) -> bool:
        return (abs(self._get("pitch_semitones", 0.0)) > 1e-3
                or abs(self._get("pitch_jitter", 0.0)) > 1e-4
                or abs(self._get("vibrato_depth_cents", 0.0)) > 1e-3)

    # -- processing -------------------------------------------------------- #
    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return x
        n = x.size

        y = self.dc.process(x)

        if self.needs_pitch:
            ratio = self._pitch_ratio(n)
            y = self.pitch.process(y, ratio)

        y = self.formants.process(y)
        y = self.body.process(y)
        y = self.presence.process(y)
        y = self.bright.process(y)

        ring_mix = self._get("ring_mod_mix", 0.0)
        if ring_mix > 0.0:
            hz = self._get("ring_mod_hz", 0.0)
            if hz > 0.0:
                # Phase is advanced by a running accumulator, not by absolute
                # time, so the carrier frequency is independent of block size.
                carrier = np.sin(self.ring_phase + 2.0 * np.pi * hz * np.arange(n) / self.sr)
                self.ring_phase = float((self.ring_phase + 2.0 * np.pi * hz * n / self.sr)
                                        % (2.0 * np.pi))
                y = y * (1.0 - ring_mix) + (y * carrier) * ring_mix

        drive = self._get("drive", 0.0)
        drive_mix = self._get("drive_mix", 0.0)
        if drive > 0.0 and drive_mix > 0.0:
            shaped = np.tanh(y * (1.0 + drive))
            y = y * (1.0 - drive_mix) + (shaped / max(1.0, 1.0 + drive * 0.35)) * drive_mix

        noise_mix = self._get("noise_mix", 0.0)
        if noise_mix > 0.0:
            y = y + self.noise.process(y, noise_mix, self._get("noise_tone_hz", 3200.0))

        chorus_mix = self._get("chorus_mix", 0.0)
        if chorus_mix > 0.0:
            hz = max(self._get("chorus_hz", 0.0), 0.01)
            depth = self._get("chorus_depth_ms", 0.0)
            base = self.sr * 0.012
            # LFO driven by an accumulator so the rate is block-size independent
            lfo = np.sin(self.chorus_phase + 2.0 * np.pi * hz * np.arange(n) / self.sr)
            self.chorus_phase = float((self.chorus_phase + 2.0 * np.pi * hz * n / self.sr)
                                      % (2.0 * np.pi))
            mod = base + self.sr * (depth / 1000.0) * (0.5 + 0.5 * lfo)
            wet = self.chorus_delay.process_fractional(y, mod)
            y = y * (1.0 - chorus_mix) + wet * chorus_mix

        echo_mix = self._get("echo_mix", 0.0)
        if echo_mix > 0.0:
            fb = _clamp(self._get("echo_feedback", 0.0), 0.0, 0.92)
            d = self.sr * self._get("echo_ms", 0.0) / 1000.0
            # Feedback path must be sample-by-sample; block it for speed.
            y = self._echo(y, d, fb, echo_mix)

        stutter = self._get("stutter_hz", 0.0)
        if stutter > 0.0:
            y = self._stutter(y, stutter)

        # Output gain first, then limit, so the ceiling is a hard guarantee.
        y = y * db_to_lin(self._get("output_gain_db", 0.0))
        y = self._limit(y)

        self.t += n
        return y

    def _pitch_ratio(self, n: int) -> np.ndarray:
        semi = self._get("pitch_semitones", 0.0)
        jitter = self._get("pitch_jitter", 0.0)
        vib_hz = self._get("vibrato_hz", 0.0)
        vib_cents = self._get("vibrato_depth_cents", 0.0)

        if jitter <= 0.0 and vib_cents <= 0.0:
            return np.full(n, 2.0 ** (_clamp(semi, -MAX_SEMITONES, MAX_SEMITONES) / 12.0))

        inst = np.full(n, semi)

        if jitter > 0.0:
            # slow random drift: lowpassed white noise, drawn from one RNG so
            # the character is stable across the whole session
            w = self._rng.standard_normal(n)
            drift, self.jitter_state = _one_pole(w, 0.995, self.jitter_state)
            inst = inst + drift * jitter * 2.0

        if vib_cents > 0.0 and vib_hz > 0.0:
            # accumulated phase keeps the vibrato rate block-size independent
            inst = inst + (vib_cents / 100.0) * np.sin(
                self.vib_phase + 2.0 * np.pi * vib_hz * np.arange(n) / self.sr)
            self.vib_phase = float((self.vib_phase + 2.0 * np.pi * vib_hz * n / self.sr)
                                   % (2.0 * np.pi))

        inst = np.clip(inst, -MAX_SEMITONES, MAX_SEMITONES)
        return 2.0 ** (inst / 12.0)

    def _echo(self, y: np.ndarray, delay: float, fb: float, mix: float) -> np.ndarray:
        """Feedback comb with damped repeats.

        The delay line's read-then-write contract requires the delay to be at
        least as long as the block, so the sub-block is capped below the delay
        time. Feedback still works because each sub-block writes ``seg + wet*fb``
        back into the line.
        """
        sub = int(_clamp(delay, 1.0, 64.0))
        out = np.empty_like(y)
        for s in range(0, len(y), sub):
            e = min(len(y), s + sub)
            seg = y[s:e]
            delayed = self.echo_delay.read(len(seg), delay)
            wet = delayed * 0.6 + self.echo_damp.process(delayed) * 0.4
            out[s:e] = seg + wet * mix
            self.echo_delay.write(seg + wet * fb)
        return out

    def _stutter(self, y: np.ndarray, hz: float) -> np.ndarray:
        """Sample-and-hold at `hz`, vectorised, with state across blocks."""
        period = max(2, int(self.sr / max(hz, 0.1)))
        idx = self.t + np.arange(len(y))
        out = np.full(len(y), self.held_value)
        starts = np.flatnonzero(idx % period == 0)
        if len(starts):
            vals = y[starts]
            which = np.searchsorted(starts, np.arange(len(y)), side="right") - 1
            mask = which >= 0
            out[mask] = vals[which[mask]]
        self.held_value = float(out[-1])
        return out

    def _limit(self, y: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
        """Brickwall-ish limiter.

        The required gain ``target`` is smoothed by two one-pole trackers and
        the *lower* of the two is used: the fast tracker wins when the signal
        jumps up (instant attack) while the slow one holds the gain down
        (smooth release). Filter state is carried across calls, and the state is
        seeded at the steady-state value so the limiter starts at unity gain
        instead of ramping up from silence.
        """
        mag = np.abs(y)
        target = np.minimum(1.0, ceiling / (mag + EPS))
        a_att = math.exp(-1.0 / max(self.sr * 0.0015, 1.0))
        a_rel = math.exp(-1.0 / max(self.sr * 0.12, 1.0))

        if not self.lim_primed:
            self.lim_zi_fast = np.array([a_att * target[0]])
            self.lim_zi_slow = np.array([a_rel * target[0]])
            self.lim_primed = True

        fast, self.lim_zi_fast = lfilter([1 - a_att], [1, -a_att], target,
                                         zi=self.lim_zi_fast)
        slow, self.lim_zi_slow = lfilter([1 - a_rel], [1, -a_rel], target,
                                         zi=self.lim_zi_slow)
        # The smoothed gain cannot react to a step within one sample, so a hard
        # clip at the ceiling guarantees the output bound.
        return np.clip(y * np.minimum(fast, slow), -ceiling, ceiling)


# --------------------------------------------------------------------------- #
# analysis: estimate voice character from a sample (for the "mimic" feature)
# --------------------------------------------------------------------------- #
def _frames(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    count = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(count)[:, None]
    return x[idx]


def estimate_f0(x: np.ndarray, sr: int) -> dict[str, float]:
    """Autocorrelation pitch tracking over the whole signal."""
    frame = int(0.045 * sr)
    hop = int(0.020 * sr)
    fr = _frames(x, frame, hop)
    if len(fr) == 0:
        return {"f0_median": 0.0, "f0_p10": 0.0, "f0_p90": 0.0, "voiced_ratio": 0.0}

    fr = fr - fr.mean(axis=1, keepdims=True)
    win = np.hanning(frame)[None, :]
    spec = np.fft.rfft(fr * win, n=2 * frame, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :frame]
    ac = ac / (ac[:, :1] + EPS)

    lo = max(2, int(sr / 400.0))
    hi = min(frame - 2, int(sr / 60.0))
    seg = ac[:, lo:hi]
    lag = np.argmax(seg, axis=1) + lo
    strength = seg[np.arange(len(seg)), np.argmax(seg, axis=1)]

    voiced = strength > 0.35
    if not voiced.any():
        return {"f0_median": 0.0, "f0_p10": 0.0, "f0_p90": 0.0,
                "voiced_ratio": float(voiced.mean())}
    f0 = sr / lag[voiced]
    return {
        "f0_median": float(np.median(f0)),
        "f0_p10": float(np.percentile(f0, 10)),
        "f0_p90": float(np.percentile(f0, 90)),
        "voiced_ratio": float(voiced.mean()),
    }


def spectral_features(x: np.ndarray, sr: int) -> dict[str, float]:
    frame = int(0.040 * sr)
    hop = int(0.020 * sr)
    fr = _frames(x, frame, hop)
    if len(fr) == 0:
        return {}
    fr = fr - fr.mean(axis=1, keepdims=True)
    win = np.hanning(frame)[None, :]
    spec = np.abs(np.fft.rfft(fr * win, axis=1)) + EPS
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)

    # use only frames with real energy
    energy = spec.sum(axis=1)
    keep = energy > (np.percentile(energy, 60) if len(energy) > 2 else 0)
    spec = spec[keep] if keep.any() else spec

    mean_spec = spec.mean(axis=0)
    total = mean_spec.sum() + EPS
    centroid = float((freqs * mean_spec).sum() / total)
    cumsum = np.cumsum(mean_spec)
    rolloff = float(freqs[np.searchsorted(cumsum, 0.85 * cumsum[-1])])
    flatness = float(np.exp(np.log(mean_spec).mean()) / (mean_spec.mean() + EPS))

    logm = 20.0 * np.log10(mean_spec + EPS)
    band = (freqs > 120) & (freqs < 0.4 * sr)
    if band.sum() > 8:
        slope = float(np.polyfit(freqs[band], logm[band], 1)[0])
    else:
        slope = 0.0

    # formant-ish peaks from a smoothed envelope
    smooth = np.convolve(mean_spec, np.hanning(9) / np.hanning(9).sum(), mode="same")
    peaks: list[float] = []
    for i in range(2, len(smooth) - 2):
        if smooth[i] > smooth[i - 1] and smooth[i] >= smooth[i + 1] and smooth[i] > 0:
            peaks.append(float(freqs[i]))
    peaks = [p for p in peaks if p > 150]

    return {
        "centroid": centroid,
        "rolloff": rolloff,
        "flatness": flatness,
        "tilt_db_per_khz": slope * 1000.0,
        "peaks": peaks[:6],
    }


def analyze_voice(x: np.ndarray, sr: int) -> dict[str, Any]:
    """Full voice-character analysis used by the mimic feature."""
    x = np.asarray(x, float).ravel()
    if x.size == 0:
        raise ValueError("empty audio")
    peak = float(np.max(np.abs(x)))
    if peak > 0:
        x = x / peak * 0.9
    f0 = estimate_f0(x, sr)
    sf = spectral_features(x, sr)
    return {"pitch": f0, "spectral": sf, "duration_s": len(x) / sr}


def preset_from_analysis(features: dict[str, Any], sr: int = 24000) -> dict[str, Any]:
    """Turn measured voice character into DSP parameters (a matched voice)."""
    f0 = features.get("pitch", {}) or {}
    sf = features.get("spectral", {}) or {}
    median = float(f0.get("f0_median", 0.0) or 0.0)
    reference = 165.0  # neutral reference F0 used to define "0 semitones"

    if median > 40.0:
        semitones = 12.0 * math.log2(median / reference)
    else:
        semitones = 0.0
    semitones = _clamp(semitones, -12.0, 12.0)

    centroid = float(sf.get("centroid", 1800.0) or 1800.0)
    tilt = float(sf.get("tilt_db_per_khz", 0.0) or 0.0)
    flatness = float(sf.get("flatness", 0.1) or 0.1)

    # brighter voices -> higher formants; darker -> lower
    formant = _clamp(0.75 + 0.35 * (centroid / 1800.0), 0.7, 1.45)
    brightness = _clamp(tilt / 12.0, -9.0, 9.0)
    noise = _clamp(flatness * 0.9, 0.0, 0.06)

    return {
        "pitch_semitones": round(semitones, 2),
        "formant_shift": round(formant, 3),
        "brightness_gain_db": round(brightness, 2),
        "presence_gain_db": round(_clamp(brightness * 0.4, -4.0, 4.0), 2),
        "noise_mix": round(noise, 4),
        "output_gain_db": 0.0,
    }


def process_array(x: np.ndarray, sr: int, params: dict[str, Any],
                  block: int = 1024, seed: int = 0xBEEF) -> np.ndarray:
    """Offline convenience: run the full chain over an array in blocks."""
    vp = VoiceProcessor(sr=sr, params=params, seed=seed)
    parts = [vp.process(x[s:s + block]) for s in range(0, len(x), block)]
    return np.concatenate(parts) if parts else np.zeros(0)
