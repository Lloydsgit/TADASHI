"""Text-to-speech for the agent, with a fallback chain.

Tiers, tried in order:

  1. ``piper``   - neural TTS (offline ONNX). Natural, fast on CPU.
  2. ``espeak``  - formant TTS, if the ``espeak-ng`` binary is on PATH.
  3. ``formant`` - the built-in vowel synthesiser in :mod:`app.synth`.
                   Never says the actual words; it is the last-resort tone
                   source so the UI always produces sound.

Whatever tier is used, the audio is then run through the voice engine, so the
agent's speech comes out in the selected voice.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import audio, synth

log = logging.getLogger("tadashi.tts")

MODELS_DIR = Path(__file__).parent / "models" / "piper"

# base TTS voices -> friendly label. Keys match the model file stems.
PIPER_VOICES: dict[str, str] = {
    "en_US-lessac-medium": "Lessac (neutral female)",
    "en_US-ryan-high": "Ryan (warm male)",
    "en_US-amy-medium": "Amy (bright female)",
    "en_US-joe-medium": "Joe (deep male)",
    "en_GB-alba-medium": "Alba (Scottish female)",
}


@dataclass(frozen=True)
class TtsVoice:
    id: str
    label: str
    available: bool


@dataclass
class _Cache:
    voices: dict[str, Any]
    lock: threading.Lock


_cache = _Cache(voices={}, lock=threading.Lock())


def piper_available() -> bool:
    try:
        import piper  # noqa: F401
    except Exception:
        return False
    return any(MODELS_DIR.glob("*.onnx"))


def espeak_available() -> str | None:
    for name in ("espeak-ng", "espeak"):
        if (path := shutil.which(name)):
            return path
    return None


def list_tts_voices() -> list[TtsVoice]:
    out: list[TtsVoice] = []
    for vid, label in PIPER_VOICES.items():
        out.append(TtsVoice(vid, label, (MODELS_DIR / f"{vid}.onnx").is_file()))
    return out


def active_tier() -> str:
    if piper_available():
        return "piper"
    if espeak_available():
        return "espeak"
    return "formant"


def _load_piper(model_id: str):
    with _cache.lock:
        if model_id in _cache.voices:
            return _cache.voices[model_id]
        from piper import PiperVoice
        onnx = MODELS_DIR / f"{model_id}.onnx"
        cfg = MODELS_DIR / f"{model_id}.onnx.json"
        if not onnx.is_file():
            raise FileNotFoundError(f"no piper model {model_id}")
        voice = PiperVoice.load(str(onnx), config_path=str(cfg) if cfg.is_file() else None)
        _cache.voices[model_id] = voice
        return voice


def _first_available_model(preferred: str | None) -> str:
    if preferred and (MODELS_DIR / f"{preferred}.onnx").is_file():
        return preferred
    for vid in PIPER_VOICES:
        if (MODELS_DIR / f"{vid}.onnx").is_file():
            return vid
    raise FileNotFoundError("no piper models installed")


def _synth_piper(text: str, model_id: str, target_sr: int) -> np.ndarray:
    voice = _load_piper(model_id)
    parts: list[np.ndarray] = []
    src_sr = target_sr
    for chunk in voice.synthesize(text):
        src_sr = chunk.sample_rate
        parts.append(np.frombuffer(chunk.audio_int16_bytes, dtype="<i2"))
    if not parts:
        return np.zeros(0)
    x = np.concatenate(parts).astype(np.float64) / 32768.0
    if src_sr != target_sr:
        x = audio.resample(x, src_sr, target_sr)
    return x


def _synth_espeak(text: str, exe: str, target_sr: int) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "out.wav"
        subprocess.run(
            [exe, "-v", "en-us", "-s", "165", "-p", "48", "-w", str(wav), text],
            check=True, capture_output=True, timeout=30,
        )
        x, _ = audio.decode_audio(wav.read_bytes(), target_sr)
    return x


def synthesize(text: str, target_sr: int, base_voice: str | None = None,
               max_chars: int = 600) -> tuple[np.ndarray, str, str]:
    """Render ``text`` to mono float audio.

    Returns ``(audio, tier_used, note)``. Never raises for ordinary input: it
    walks the fallback chain and, in the worst case, returns the formant tone
    sequence so the caller always has something to play.
    """
    text = (text or "").strip()[:max_chars]
    if not text:
        return np.zeros(0), "none", "empty text"

    if piper_available():
        try:
            model_id = _first_available_model(base_voice)
            x = _synth_piper(text, model_id, target_sr)
            if x.size:
                return x, "piper", model_id
        except Exception as exc:  # pragma: no cover - depends on local models
            log.warning("piper synthesis failed: %s", exc)

    if (exe := espeak_available()) is not None:
        try:
            x = _synth_espeak(text, exe, target_sr)
            if x.size:
                return x, "espeak", Path(exe).name
        except Exception as exc:  # pragma: no cover - depends on local binary
            log.warning("espeak synthesis failed: %s", exc)

    # last resort: vowel sequence shaped to the text length
    n_vowels = max(3, min(12, len(text) // 8 + 3))
    order = tuple(list(synth.VOWELS)[i % len(synth.VOWELS)] for i in range(n_vowels))
    x = synth.synth_vowels(target_sr, order=order)
    return x, "formant", "vowel fallback (no TTS engine installed)"


def synthesize_and_morph(text: str, target_sr: int, session, base_voice: str | None = None
                         ) -> tuple[np.ndarray, dict[str, Any]]:
    """Synthesise text then run it through a voice session's engine."""
    x, tier, note = synthesize(text, target_sr, base_voice)
    if x.size == 0:
        return x, {"tts_tier": tier, "note": note}
    y = session.process(x)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-6:
        y = y / peak * 0.95
    return y, {"tts_tier": tier, "tts_voice": note,
               "engine": getattr(session, "engine_id", "?"),
               "voice": getattr(session, "voice_id", "?")}
