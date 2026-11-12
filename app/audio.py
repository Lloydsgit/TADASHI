"""Audio encoding helpers: PCM <-> numpy, resampling, WAV I/O."""

from __future__ import annotations

import io
import math

import numpy as np
import soundfile as sf

INT16_SCALE = 32768.0


def pcm16_to_float(data: bytes) -> np.ndarray:
    """Little-endian signed 16-bit PCM bytes -> float64 in [-1, 1)."""
    if not data:
        return np.zeros(0)
    if len(data) % 2:
        data = data[:-1]
    x = np.frombuffer(data, dtype="<i2").astype(np.float64)
    return x / INT16_SCALE


def float_to_pcm16(x: np.ndarray) -> bytes:
    """float array -> little-endian signed 16-bit PCM bytes, hard-clipped."""
    if x.size == 0:
        return b""
    y = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0 - 1.0 / INT16_SCALE)
    return (y * INT16_SCALE).astype("<i2").tobytes()


def float_to_pcm32(x: np.ndarray) -> bytes:
    """float array -> little-endian signed 32-bit PCM (more headroom for tests)."""
    y = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    return (y * 2147483647.0).astype("<i4").tobytes()


def resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Band-limited-enough resampling via FFT (offline; not for streaming)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if src_sr == dst_sr or x.size == 0:
        return x
    n_out = max(1, int(round(x.size * dst_sr / src_sr)))
    spec = np.fft.rfft(x)
    n_spec_out = n_out // 2 + 1
    out = np.zeros(n_spec_out, dtype=complex)
    m = min(len(spec), n_spec_out)
    out[:m] = spec[:m]
    # scale preserves amplitude across the length change
    y = np.fft.irfft(out, n=n_out) * (n_out / x.size)
    return y


def decode_audio(data: bytes, target_sr: int) -> tuple[np.ndarray, int]:
    """Decode WAV/FLAC/OGG bytes to mono float at ``target_sr``."""
    buf = io.BytesIO(data)
    audio, sr = sf.read(buf, dtype="float64", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != target_sr:
        mono = resample(mono, sr, target_sr)
    return mono, sr


def encode_wav(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def peak(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.max(np.abs(x)))


def level_db(x: np.ndarray) -> float:
    r = rms(x)
    return -120.0 if r < 1e-9 else float(20.0 * math.log10(r))
