import math

import numpy as np
import pytest

from app import audio, dsp
from app.voices import bank

SR = 24000


def sine(freq: float, seconds: float, sr: int = SR, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return amp * np.sin(2 * math.pi * freq * t)


def dominant_freq(x: np.ndarray, sr: int = SR) -> float:
    """Peak frequency of the spectrum, ignoring DC."""
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    spec[:3] = 0
    return float(np.fft.rfftfreq(len(x), 1 / sr)[np.argmax(spec)])


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def hf_fraction(y: np.ndarray, sr: int = SR, lo: float = 1000.0, hi: float = 8000.0) -> float:
    """Share of spectral energy above `lo` Hz - a proxy for spectral tilt."""
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    return float(spec[(freqs > lo) & (freqs < hi)].sum() / (spec.sum() + 1e-12))


def rich_source(seconds: float = 0.5, sr: int = SR) -> np.ndarray:
    """A harmonic-rich test tone: buzz at 120 Hz plus a 300 Hz partial."""
    t = np.arange(int(sr * seconds)) / sr
    buzz = 0.3 * np.sign(np.sin(2 * math.pi * 120 * t))
    return buzz + 0.3 * np.sin(2 * math.pi * 300 * t)


# --------------------------------------------------------------------------- #
# pitch shifting
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("semitones,expected", [(-12, 110.0), (-5, 164.8), (0, 220.0), (7, 329.6), (12, 440.0)])
def test_pitch_shift_hits_target(semitones, expected):
    x = sine(220.0, 1.5)
    y = dsp.process_array(x, SR, {"pitch_semitones": semitones})
    # ignore the first window while the shifter fills
    got = dominant_freq(y[SR // 2:])
    assert abs(got - expected) < 12.0, f"{got} Hz != {expected} Hz"


def test_pitch_shift_up_and_down_differ():
    x = sine(200.0, 1.0)
    up = dominant_freq(dsp.process_array(x, SR, {"pitch_semitones": 7})[SR // 2:])
    down = dominant_freq(dsp.process_array(x, SR, {"pitch_semitones": -7})[SR // 2:])
    assert up > down * 1.8


def test_pitch_shift_preserves_energy_order_of_magnitude():
    x = sine(220.0, 1.0)
    y = dsp.process_array(x, SR, {"pitch_semitones": 5})
    rms_in = audio.rms(x)
    rms_out = audio.rms(y[SR // 2:])
    assert 0.15 * rms_in < rms_out < 3.0 * rms_in


def test_chunked_matches_single_block():
    """Streaming in small chunks must not change the result."""
    x = sine(180.0, 0.6)
    params = {"pitch_semitones": 4, "drive": 2.0, "drive_mix": 0.4}
    one_shot = dsp.process_array(x, SR, params, block=len(x))
    chunked = dsp.process_array(x, SR, params, block=256)
    assert np.allclose(one_shot, chunked, atol=1e-9)


def test_stateful_processor_is_continuous_across_calls():
    vp = dsp.VoiceProcessor(sr=SR, params={"pitch_semitones": 3})
    x = sine(200.0, 0.5)
    a = np.concatenate([vp.process(x[:1000]), vp.process(x[1000:])])
    b = dsp.process_array(x, SR, {"pitch_semitones": 3}, block=len(x))
    assert np.allclose(a, b, atol=1e-9)


@pytest.mark.parametrize("voice_id", [v["id"] for v in bank().list_dicts()])
def test_output_is_independent_of_block_size(voice_id):
    """The whole chain must be block-size invariant.

    This is the core streaming guarantee: any oscillator, LFO or jitter source
    that keys off absolute block position instead of a running phase will break
    it, and the failure is silent until you stream at a different frame size.
    """
    params = bank().resolve(voice_id)
    x = rich_source(0.5)
    reference = dsp.process_array(x, SR, params, block=len(x))
    for block in (128, 333, 1024, 4096):
        got = dsp.process_array(x, SR, params, block=block)
        assert np.allclose(reference, got, atol=1e-9), f"{voice_id} @ block {block}"


@pytest.mark.parametrize("mod", ["ring_mod_hz", "vibrato_hz", "chorus_hz", "stutter_hz"])
def test_modulation_sources_are_block_size_independent(mod):
    params = {"ring_mod_mix": 0.4, "vibrato_depth_cents": 40.0, "chorus_mix": 0.4}
    params[mod] = {"ring_mod_hz": 77.0, "vibrato_hz": 5.5,
                   "chorus_hz": 2.5, "stutter_hz": 11.0}[mod]
    x = rich_source(0.4)
    reference = dsp.process_array(x, SR, params, block=len(x))
    for block in (256, 1000, 2048):
        assert np.allclose(reference, dsp.process_array(x, SR, params, block=block),
                           atol=1e-9), f"{mod} @ block {block}"


# --------------------------------------------------------------------------- #
# transparency / fallback
# --------------------------------------------------------------------------- #
def test_natural_preset_is_effectively_transparent():
    """The DSP chain must not change loudness or pitch for the natural preset."""
    x = sine(300.0, 0.5)
    y = dsp.process_array(x, SR, bank().resolve("natural"))
    assert abs(rms(y) - rms(x)) / rms(x) < 0.10
    assert abs(dominant_freq(y) - dominant_freq(x)) < 2.0


def test_natural_engine_is_bit_exact_passthrough():
    """The natural *engine* must be a pure identity, with zero added latency."""
    from app.engines import EngineSession
    x = sine(300.0, 0.3)
    session = EngineSession(SR, bank(), engine_id="natural", voice_id="titan")
    assert np.array_equal(session.process(x), x)


def test_limiter_prevents_clipping():
    x = sine(200.0, 0.5, amp=1.4)  # deliberately over unity
    y = dsp.process_array(x, SR, {"output_gain_db": 18, "drive": 8, "drive_mix": 1.0})
    assert np.max(np.abs(y)) <= 1.0 + 1e-6
    assert np.all(np.isfinite(y))


def test_all_voices_produce_finite_audio():
    for voice in bank().list_dicts():
        vid = voice["id"]
        x = sine(180.0, 0.35, amp=0.4)
        y = dsp.process_array(x, SR, bank().resolve(vid))
        assert y.size == x.size, vid
        assert np.all(np.isfinite(y)), vid
        assert np.max(np.abs(y)) <= 1.0001, vid


def test_voices_are_distinguishable():
    """A pitch-shifting voice must actually move the fundamental."""
    x = sine(200.0, 1.0)
    neutral = dominant_freq(dsp.process_array(x, SR, bank().resolve("natural"))[SR // 2:])
    deep = dominant_freq(dsp.process_array(x, SR, bank().resolve("titan"))[SR // 2:])
    high = dominant_freq(dsp.process_array(x, SR, bank().resolve("pixie"))[SR // 2:])
    assert deep < neutral * 0.8
    assert high > neutral * 1.4


# --------------------------------------------------------------------------- #
# effects
# --------------------------------------------------------------------------- #
def test_ring_modulation_adds_sidebands():
    x = sine(300.0, 0.6)
    y = dsp.process_array(x, SR, {"ring_mod_hz": 60.0, "ring_mod_mix": 0.6})
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    def energy_at(f, bw=8.0):
        return spec[(freqs > f - bw) & (freqs < f + bw)].sum()
    assert energy_at(240.0) > 0.02 * spec.sum()   # 300 - 60
    assert energy_at(360.0) > 0.02 * spec.sum()   # 300 + 60


def test_echo_creates_a_late_repeat():
    impulse = np.zeros(SR // 2)
    impulse[0] = 1.0
    y = dsp.process_array(impulse, SR, {"echo_ms": 100.0, "echo_feedback": 0.5, "echo_mix": 0.8})
    delay = int(0.1 * SR)
    window = y[delay - 40:delay + 40]
    assert np.max(np.abs(window)) > 0.05, "no echo repeat found"


def test_formant_shift_changes_spectral_tilt():
    """Scaling formant resonances up must move energy toward higher frequencies."""
    x = rich_source(0.5)
    low = dsp.process_array(x, SR, {"formant_shift": 0.75})
    high = dsp.process_array(x, SR, {"formant_shift": 1.4})
    assert hf_fraction(high) > hf_fraction(low) * 1.5


def test_chorus_is_deterministic_for_same_seed():
    x = sine(220.0, 0.4)
    p = {"chorus_hz": 2.0, "chorus_depth_ms": 4.0, "chorus_mix": 0.5}
    a = dsp.process_array(x, SR, p, seed=7)
    b = dsp.process_array(x, SR, p, seed=7)
    assert np.allclose(a, b)


# --------------------------------------------------------------------------- #
# analysis / mimicry
# --------------------------------------------------------------------------- #
def test_estimate_f0_recovers_known_pitch():
    x = sine(150.0, 1.0)
    est = dsp.estimate_f0(x, SR)
    assert abs(est["f0_median"] - 150.0) < 6.0


def test_analysis_preset_raises_pitch_for_a_high_voice():
    low = dsp.preset_from_analysis(dsp.analyze_voice(sine(110.0, 1.0), SR))
    high = dsp.preset_from_analysis(dsp.analyze_voice(sine(240.0, 1.0), SR))
    assert low["pitch_semitones"] < high["pitch_semitones"]


def test_analyze_voice_reports_duration_and_spectral_centroid():
    x = sine(300.0, 0.5)
    feats = dsp.analyze_voice(x, SR)
    assert abs(feats["duration_s"] - 0.5) < 0.01
    assert abs(feats["spectral"]["centroid"] - 300.0) < 40.0


def test_analyze_voice_rejects_empty_audio():
    with pytest.raises(ValueError):
        dsp.analyze_voice(np.zeros(0), SR)


# --------------------------------------------------------------------------- #
# audio codec
# --------------------------------------------------------------------------- #
def test_pcm16_roundtrip_is_accurate():
    x = sine(440.0, 0.1, amp=0.7)
    y = audio.pcm16_to_float(audio.float_to_pcm16(x))
    assert np.max(np.abs(x - y)) < 1.0 / 32768


def test_resample_preserves_pitch_and_duration():
    x = sine(200.0, 1.0, sr=24000)
    up = audio.resample(x, 24000, 48000)
    assert abs(len(up) - 48000) <= 1
    assert abs(dominant_freq(up, 48000) - 200.0) < 4.0


def test_level_db_tracks_amplitude():
    quiet = audio.level_db(sine(200.0, 0.2, amp=0.01))
    loud = audio.level_db(sine(200.0, 0.2, amp=0.5))
    assert loud > quiet + 25
