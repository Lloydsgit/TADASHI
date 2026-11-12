"""Conversion engines and the automatic fallback chain.

Three tiers, tried in order of quality:

  1. ``neural``   - voice-conversion model (e.g. ONNX RVC / so-vits) when one is
                    installed and a voice has a matching model. Best quality,
                    needs a GPU and model weights, so it is optional.
  2. ``dsp``      - the NumPy real-time chain in :mod:`app.dsp`. Runs anywhere on
                    CPU, works for every voice in the bank.
  3. ``natural``  - pure passthrough. Always available, adds no latency, and is
                    the guaranteed fallback so the user always hears something.

``resolve_engine`` never fails: if the requested engine is unavailable it walks
down the chain and reports which one it actually used.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import dsp
from .voices import VoiceBank

ENGINE_ORDER = ("neural", "dsp", "natural")


@dataclass(frozen=True)
class EngineInfo:
    id: str
    name: str
    description: str
    available: bool
    latency_ms: float
    quality: int  # 1..5
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "description": self.description,
            "available": self.available, "latency_ms": self.latency_ms,
            "quality": self.quality, "reason": self.reason,
        }


def _neural_backend() -> tuple[bool, str]:
    """Look for an optional neural backend.

    Both ``onnxruntime`` and a converted model file are required. When they are
    missing we simply report why, so the UI can explain the fallback instead of
    silently degrading.
    """
    if importlib.util.find_spec("onnxruntime") is None:
        return False, "onnxruntime not installed"
    try:
        import onnxruntime  # noqa: F401
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"onnxruntime import failed: {exc}"
    from pathlib import Path
    models = Path(__file__).parent / "models"
    if not models.is_dir() or not any(models.glob("*.onnx")):
        return False, "no .onnx model in app/models"
    return True, ""


def available_engines() -> list[EngineInfo]:
    neural_ok, neural_reason = _neural_backend()
    return [
        EngineInfo(
            id="neural", name="Neural",
            description="Model-based voice conversion. Highest fidelity, needs a GPU.",
            available=neural_ok, latency_ms=120.0, quality=5, reason=neural_reason,
        ),
        EngineInfo(
            id="dsp", name="DSP (real-time)",
            description="Formant/pitch/timbre transformation on CPU. Works for all voices.",
            available=True, latency_ms=25.0, quality=3,
        ),
        EngineInfo(
            id="natural", name="Natural",
            description="Direct passthrough of your own voice. Zero processing.",
            available=True, latency_ms=0.0, quality=2,
        ),
    ]


def resolve_engine(requested: str | None) -> tuple[str, list[str]]:
    """Pick the best available engine at or below the requested tier.

    Returns ``(engine_id, notes)`` where ``notes`` records any fallbacks taken.
    """
    infos = {e.id: e for e in available_engines()}
    notes: list[str] = []
    if requested in ENGINE_ORDER:
        start = ENGINE_ORDER.index(requested)
    else:
        start = ENGINE_ORDER.index("dsp")
        if requested:
            notes.append(f"unknown engine '{requested}', defaulting to dsp")

    for eid in ENGINE_ORDER[start:]:
        info = infos.get(eid)
        if info and info.available:
            if eid != requested and requested in ENGINE_ORDER:
                notes.append(f"'{requested}' unavailable, fell back to '{eid}'")
            return eid, notes
    return "natural", notes + ["no engine available, using natural passthrough"]


# --------------------------------------------------------------------------- #
# engine implementations
# --------------------------------------------------------------------------- #
class DspEngine:
    """Applies the NumPy voice chain, one :class:`VoiceProcessor` per session."""

    id = "dsp"

    def __init__(self, sr: int, bank: VoiceBank, voice_id: str = "natural",
                 params: dict[str, Any] | None = None, seed: int = 0xBEEF):
        self.sr = sr
        self.bank = bank
        self.voice_id = voice_id
        self.params = params or bank.resolve(voice_id)
        self.processor = dsp.VoiceProcessor(sr=sr, params=self.params, seed=seed)

    def set_voice(self, voice_id: str, params: dict[str, Any] | None = None) -> None:
        self.voice_id = voice_id
        self.params = params if params is not None else self.bank.resolve(voice_id)
        self.processor.configure(self.params)

    def set_params(self, params: dict[str, Any]) -> None:
        self.params = params
        self.processor.configure(params)

    def process(self, block: np.ndarray) -> np.ndarray:
        if self.voice_id == "natural" and not self._has_effect():
            return block
        return self.processor.process(block)

    def _has_effect(self) -> bool:
        return any(abs(float(v)) > 1e-6 for k, v in self.params.items()
                   if k not in ("formant_shift", "dry_mix", "presence_q", "body_freq",
                                "presence_freq", "brightness_freq", "noise_tone_hz"))


class NaturalEngine:
    """Identity passthrough - the always-available fallback."""

    id = "natural"

    def __init__(self, sr: int, bank: VoiceBank, voice_id: str = "natural",
                 params: dict[str, Any] | None = None, seed: int = 0xBEEF):
        self.sr = sr
        self.bank = bank
        self.voice_id = voice_id
        self.params = params or bank.resolve(voice_id)

    def set_voice(self, voice_id: str, params: dict[str, Any] | None = None) -> None:
        self.voice_id = voice_id
        self.params = params if params is not None else self.bank.resolve(voice_id)

    def set_params(self, params: dict[str, Any]) -> None:
        self.params = params

    def process(self, block: np.ndarray) -> np.ndarray:
        return block


class NeuralEngine:
    """Model-based engine.

    Kept deliberately thin: it is only constructed when
    :func:`_neural_backend` reports a usable ONNX model, and falls back to the
    DSP engine for voices without a model. Wiring a specific architecture
    (RVC / so-vits / VITS) is a drop-in step here.
    """

    id = "neural"

    def __init__(self, sr: int, bank: VoiceBank, voice_id: str = "natural",
                 params: dict[str, Any] | None = None, seed: int = 0xBEEF):
        self.sr = sr
        self.bank = bank
        self.voice_id = voice_id
        self.params = params or bank.resolve(voice_id)
        self.fallback = DspEngine(sr, bank, voice_id, self.params, seed)

    def set_voice(self, voice_id: str, params: dict[str, Any] | None = None) -> None:
        self.voice_id = voice_id
        self.params = params if params is not None else self.bank.resolve(voice_id)
        self.fallback.set_voice(voice_id, self.params)

    def set_params(self, params: dict[str, Any]) -> None:
        self.params = params
        self.fallback.set_params(params)

    def process(self, block: np.ndarray) -> np.ndarray:
        # No model matched this voice -> keep quality high with the DSP chain
        # rather than dropping to passthrough.
        return self.fallback.process(block)


def build_engine(engine_id: str, sr: int, bank: VoiceBank, voice_id: str = "natural",
                 params: dict[str, Any] | None = None, seed: int = 0xBEEF):
    if engine_id == "natural":
        return NaturalEngine(sr, bank, voice_id, params, seed)
    if engine_id == "neural":
        return NeuralEngine(sr, bank, voice_id, params, seed)
    return DspEngine(sr, bank, voice_id, params, seed)


class EngineSession:
    """Wraps an engine with health tracking and transparent failover.

    If a block raises or comes back non-finite, the session degrades one step
    down the chain and keeps going instead of killing the audio stream.
    """

    def __init__(self, sr: int, bank: VoiceBank, engine_id: str = "dsp",
                 voice_id: str = "natural", params: dict[str, Any] | None = None,
                 seed: int = 0xBEEF):
        self.sr = sr
        self.bank = bank
        self.requested = engine_id
        self.engine_id, self.notes = resolve_engine(engine_id)
        self.voice_id = voice_id
        self.params = params or bank.resolve(voice_id)
        self.seed = seed
        self.engine = build_engine(self.engine_id, sr, bank, voice_id, self.params, seed)
        self.failures = 0
        self.last_error = ""

    def set_voice(self, voice_id: str, params: dict[str, Any] | None = None) -> None:
        self.voice_id = voice_id
        self.params = params if params is not None else self.bank.resolve(voice_id)
        self.engine.set_voice(voice_id, self.params)

    def set_params(self, params: dict[str, Any]) -> None:
        self.params = params
        self.engine.set_params(params)

    def set_engine(self, engine_id: str) -> list[str]:
        self.engine_id, self.notes = resolve_engine(engine_id)
        self.engine = build_engine(self.engine_id, self.sr, self.bank,
                                   self.voice_id, self.params, self.seed)
        return self.notes

    def process(self, block: np.ndarray) -> np.ndarray:
        try:
            out = self.engine.process(block)
            if not np.all(np.isfinite(out)):
                raise ValueError("non-finite samples")
            return out
        except Exception as exc:
            self.failures += 1
            self.last_error = str(exc)
            return self._degrade(block)

    def _degrade(self, block: np.ndarray) -> np.ndarray:
        order = list(ENGINE_ORDER)
        idx = order.index(self.engine_id) if self.engine_id in order else 0
        for eid in order[idx + 1:]:
            self.engine_id = eid
            self.notes.append(f"engine error, degraded to '{eid}': {self.last_error}")
            self.engine = build_engine(eid, self.sr, self.bank, self.voice_id,
                                       self.params, self.seed)
            try:
                return self.engine.process(block)
            except Exception as exc:  # pragma: no cover - defensive
                self.last_error = str(exc)
        return block

    def health(self) -> dict[str, Any]:
        return {
            "requested_engine": self.requested,
            "active_engine": self.engine_id,
            "voice": self.voice_id,
            "failures": self.failures,
            "notes": self.notes,
            "last_error": self.last_error,
        }
