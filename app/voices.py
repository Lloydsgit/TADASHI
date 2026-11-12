"""Voice bank: loading, validation and preset resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent / "data"
VOICES_FILE = DATA_DIR / "voices.json"

# Bumped whenever the /api/voices payload shape or voice set changes
# incompatibly, so a stale static build can be detected.
SCHEMA_VERSION = 1

NUMERIC_PARAMS = (
    "pitch_semitones", "pitch_jitter", "formant_shift", "body_freq", "body_gain_db",
    "presence_freq", "presence_gain_db", "presence_q", "brightness_freq",
    "brightness_gain_db", "vibrato_hz", "vibrato_depth_cents", "ring_mod_hz",
    "ring_mod_mix", "drive", "drive_mix", "echo_ms", "echo_feedback", "echo_mix",
    "noise_mix", "noise_tone_hz", "chorus_hz", "chorus_depth_ms", "chorus_mix",
    "stutter_hz", "output_gain_db", "dry_mix",
)

# Bounds keep a hand-edited or user-supplied preset from blowing up the DSP.
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "pitch_semitones": (-24.0, 24.0),
    "pitch_jitter": (0.0, 6.0),
    "formant_shift": (0.5, 2.0),
    "body_freq": (60.0, 600.0),
    "body_gain_db": (-24.0, 18.0),
    "presence_freq": (400.0, 8000.0),
    "presence_gain_db": (-18.0, 18.0),
    "presence_q": (0.2, 6.0),
    "brightness_freq": (1200.0, 12000.0),
    "brightness_gain_db": (-24.0, 18.0),
    "vibrato_hz": (0.0, 14.0),
    "vibrato_depth_cents": (0.0, 200.0),
    "ring_mod_hz": (0.0, 900.0),
    "ring_mod_mix": (0.0, 1.0),
    "drive": (0.0, 12.0),
    "drive_mix": (0.0, 1.0),
    "echo_ms": (0.0, 1000.0),
    "echo_feedback": (0.0, 0.92),
    "echo_mix": (0.0, 1.0),
    "noise_mix": (0.0, 0.4),
    "noise_tone_hz": (200.0, 12000.0),
    "chorus_hz": (0.0, 12.0),
    "chorus_depth_ms": (0.0, 20.0),
    "chorus_mix": (0.0, 1.0),
    "stutter_hz": (0.0, 60.0),
    "output_gain_db": (-24.0, 18.0),
    "dry_mix": (0.0, 1.0),
}


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    tagline: str
    category: str
    icon: str
    accent: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "tagline": self.tagline,
            "category": self.category,
            "icon": self.icon,
            "accent": self.accent,
            "params": self.params,
        }


def _sanitize(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in (params or {}).items():
        if key not in NUMERIC_PARAMS:
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            continue
        lo, hi = PARAM_BOUNDS.get(key, (-1e9, 1e9))
        out[key] = max(lo, min(hi, num))
    return out


class VoiceBank:
    def __init__(self, path: Path = VOICES_FILE):
        self.path = Path(path)
        self.sample_rate = 24000
        self.defaults: dict[str, Any] = {}
        self.voices: dict[str, Voice] = {}
        self.load()

    def load(self) -> None:
        raw = json.loads(self.path.read_text())
        self.sample_rate = int(raw.get("sample_rate", 24000))
        self.defaults = _sanitize(raw.get("defaults", {}))
        voices: dict[str, Voice] = {}
        for item in raw.get("voices", []):
            vid = str(item.get("id", "")).strip()
            if not vid:
                continue
            voices[vid] = Voice(
                id=vid,
                name=str(item.get("name", vid.upper())),
                tagline=str(item.get("tagline", "")),
                category=str(item.get("category", "custom")),
                icon=str(item.get("icon", "\u25cf")),
                accent=str(item.get("accent", "#8b95a5")),
                params=_sanitize(item.get("params", {})),
            )
        if "natural" not in voices:
            voices["natural"] = Voice(
                id="natural", name="NATURAL", tagline="Your true voice, untouched",
                category="fallback", icon="\u25cf", accent="#8b95a5", params={},
            )
        self.voices = voices

    def get(self, voice_id: str) -> Voice | None:
        return self.voices.get(voice_id)

    def require(self, voice_id: str) -> Voice:
        voice = self.voices.get(voice_id)
        if voice is None:
            raise KeyError(f"unknown voice: {voice_id}")
        return voice

    def resolve(self, voice_id: str) -> dict[str, Any]:
        """Full parameter set for a voice: defaults overlaid with the preset."""
        params = dict(self.defaults)
        voice = self.get(voice_id)
        if voice is not None:
            params.update(voice.params)
        return _sanitize(params)

    def list_dicts(self) -> list[dict[str, Any]]:
        order = {"fallback": 0, "male": 1, "female": 2, "robotic": 3, "creature": 4}
        return [v.to_dict() for v in sorted(
            self.voices.values(), key=lambda v: (order.get(v.category, 9), v.name))]

    def categories(self) -> list[str]:
        seen: list[str] = []
        for v in self.list_dicts():
            if v["category"] not in seen:
                seen.append(v["category"])
        return seen


_bank: VoiceBank | None = None


def bank() -> VoiceBank:
    global _bank
    if _bank is None:
        _bank = VoiceBank()
    return _bank
