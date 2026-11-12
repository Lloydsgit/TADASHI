"""FastAPI application: REST endpoints, WebSocket streaming and static UI."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__, audio, dsp, engines, synth, tts
from .voices import NUMERIC_PARAMS, PARAM_BOUNDS, SCHEMA_VERSION, bank

log = logging.getLogger("tadashi")

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_SR = 24000
MAX_SAMPLE_BYTES = 12 * 1024 * 1024  # 12 MB upload cap

app = FastAPI(
    title="TADASHI",
    version=__version__,
    description="Real-time AI speech agent with live voice morphing.",
)


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "sample_rate": DEFAULT_SR,
            "voices": len(bank().voices)}


@app.get("/api/voices")
def list_voices() -> dict[str, Any]:
    b = bank()
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_rate": b.sample_rate,
        "categories": b.categories(),
        "defaults": b.defaults,
        "bounds": {k: list(v) for k, v in PARAM_BOUNDS.items()},
        "voices": b.list_dicts(),
    }


@app.get("/api/voices/{voice_id}")
def get_voice(voice_id: str) -> dict[str, Any]:
    voice = bank().get(voice_id)
    if voice is None:
        raise HTTPException(status_code=404, detail=f"unknown voice '{voice_id}'")
    return {"voice": voice.to_dict(), "resolved_params": bank().resolve(voice_id)}


@app.get("/api/engines")
def list_engines() -> dict[str, Any]:
    active, notes = engines.resolve_engine("neural")
    return {"engines": [e.to_dict() for e in engines.available_engines()],
            "order": list(engines.ENGINE_ORDER),
            "best_available": active, "notes": notes}


@app.get("/api/params")
def param_schema() -> dict[str, Any]:
    return {"params": list(NUMERIC_PARAMS),
            "bounds": {k: list(v) for k, v in PARAM_BOUNDS.items()}}


# --------------------------------------------------------------------------- #
# offline conversion
# --------------------------------------------------------------------------- #
@app.post("/api/convert")
async def convert(
    file: UploadFile = File(...),
    voice: str = Query("natural"),
    engine: str = Query("dsp"),
) -> Response:
    """Upload audio, get it back transformed by the selected voice."""
    raw = await file.read()
    if len(raw) > MAX_SAMPLE_BYTES:
        raise HTTPException(status_code=413, detail="file too large (12 MB max)")
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        x, _ = audio.decode_audio(raw, DEFAULT_SR)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode audio: {exc}") from exc

    if x.size == 0:
        raise HTTPException(status_code=400, detail="no audio decoded")

    if bank().get(voice) is None:
        raise HTTPException(status_code=404, detail=f"unknown voice '{voice}'")

    session = engines.EngineSession(DEFAULT_SR, bank(), engine_id=engine, voice_id=voice)
    started = time.perf_counter()
    y = dsp.process_array(x, DEFAULT_SR, session.params)
    elapsed = time.perf_counter() - started

    wav = audio.encode_wav(y, DEFAULT_SR)
    headers = {
        "X-Engine": session.engine_id,
        "X-Voice": voice,
        "X-Process-Ms": f"{elapsed * 1000:.1f}",
        "X-Realtime-Factor": f"{(x.size / DEFAULT_SR) / max(elapsed, 1e-6):.2f}",
        "Access-Control-Expose-Headers": "X-Engine,X-Voice,X-Process-Ms,X-Realtime-Factor",
    }
    return Response(content=wav, media_type="audio/wav", headers=headers)


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> dict[str, Any]:
    """Measure a voice sample and propose a matching preset (voice mimicry)."""
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")
    if len(raw) > MAX_SAMPLE_BYTES:
        raise HTTPException(status_code=413, detail="file too large (12 MB max)")
    try:
        x, _ = audio.decode_audio(raw, DEFAULT_SR)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not decode audio: {exc}") from exc
    try:
        features = dsp.analyze_voice(x, DEFAULT_SR)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    preset = dsp.preset_from_analysis(features, DEFAULT_SR)
    return {"features": features, "preset": preset,
            "suggested_name": _name_from_features(features)}


def _name_from_features(features: dict[str, Any]) -> str:
    f0 = float((features.get("pitch") or {}).get("f0_median", 0.0) or 0.0)
    centroid = float((features.get("spectral") or {}).get("centroid", 1800.0) or 1800.0)
    if f0 <= 0:
        return "CAPTURED VOICE"
    if f0 < 110:
        base = "DEEP"
    elif f0 < 150:
        base = "LOW"
    elif f0 < 200:
        base = "MID"
    elif f0 < 260:
        base = "HIGH"
    else:
        base = "BRIGHT"
    tone = "SMOOTH" if centroid < 1500 else "CLEAR" if centroid < 2600 else "SHARP"
    return f"{base} {tone}"


@app.post("/api/preview")
def preview(
    voice: str = Query("natural"),
    engine: str = Query("dsp"),
    f0: float = Query(165.0, ge=60.0, le=400.0),
    phrase: str = Query("ah,eh,ee,oh,oo"),
) -> Response:
    """Audition a voice without a microphone: synthesise vowels through it."""
    if bank().get(voice) is None:
        raise HTTPException(status_code=404, detail=f"unknown voice '{voice}'")
    order = tuple(v.strip() for v in phrase.split(",") if v.strip() in synth.VOWELS)
    if not order:
        order = ("ah", "eh", "ee", "oh", "oo")

    session = engines.EngineSession(DEFAULT_SR, bank(), engine_id=engine, voice_id=voice)
    src = synth.synth_vowels(DEFAULT_SR, f0=f0, order=order)
    y = session.process(src)
    wav = audio.encode_wav(y, DEFAULT_SR)
    return Response(content=wav, media_type="audio/wav",
                    headers={"X-Engine": session.engine_id, "X-Voice": voice,
                             "Access-Control-Expose-Headers": "X-Engine,X-Voice"})


# --------------------------------------------------------------------------- #
# agent speech (text -> speech -> voice morph)
# --------------------------------------------------------------------------- #
@app.get("/api/tts/voices")
def tts_voices() -> dict[str, Any]:
    return {
        "active_tier": tts.active_tier(),
        "piper_available": tts.piper_available(),
        "espeak_available": tts.espeak_available() is not None,
        "voices": [{"id": v.id, "label": v.label, "available": v.available}
                   for v in tts.list_tts_voices()],
    }


@app.post("/api/tts")
def tts_speak(payload: dict[str, Any] = Body(...)) -> Response:
    """Speak text as the selected voice.

    Body: ``{"text": str, "voice": str, "engine": str, "base_voice": str|null}``
    Returns WAV of the agent's speech in the chosen voice.
    """
    text = str(payload.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > 600:
        raise HTTPException(status_code=413, detail="text too long (600 char max)")

    voice = str(payload.get("voice", "natural"))
    engine = str(payload.get("engine", "dsp"))
    base_voice = payload.get("base_voice")
    if bank().get(voice) is None:
        raise HTTPException(status_code=404, detail=f"unknown voice '{voice}'")

    session = engines.EngineSession(DEFAULT_SR, bank(), engine_id=engine, voice_id=voice)
    y, meta = tts.synthesize_and_morph(text, DEFAULT_SR, session,
                                       str(base_voice) if base_voice else None)
    if y.size == 0:
        raise HTTPException(status_code=500, detail="speech synthesis produced no audio")

    wav = audio.encode_wav(y, DEFAULT_SR)
    headers = {
        "X-Engine": str(meta.get("engine", session.engine_id)),
        "X-Voice": voice,
        "X-TTS-Tier": str(meta.get("tts_tier", "?")),
        "X-TTS-Note": str(meta.get("tts_voice", "")),
        "Access-Control-Expose-Headers": "X-Engine,X-Voice,X-TTS-Tier,X-TTS-Note",
    }
    return Response(content=wav, media_type="audio/wav", headers=headers)


# --------------------------------------------------------------------------- #
# real-time streaming
# --------------------------------------------------------------------------- #
@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket) -> None:
    """Bidirectional PCM streaming.

    Client sends 16-bit little-endian mono PCM frames; the server replies with
    transformed frames of the same length, tagged with a small JSON header.

    Protocol
    --------
    client -> server:
      * binary frame : raw PCM16 samples
      * text frame   : ``{"type": "config", ...}`` to change voice/engine/params
    server -> client:
      * text frame   : ``{"type": "ready"|"status"|"error", ...}``
      * binary frame : raw PCM16 (transformed), same sample count as the input
    """
    await websocket.accept()
    b = bank()
    session = engines.EngineSession(DEFAULT_SR, b, engine_id="dsp", voice_id="natural")
    processed = 0
    last_report = time.perf_counter()
    realtime_ratio = 0.0

    await websocket.send_text(json.dumps({
        "type": "ready",
        "sample_rate": DEFAULT_SR,
        "engine": session.engine_id,
        "voice": session.voice_id,
        "notes": session.notes,
        "version": __version__,
    }))

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if (payload := message.get("bytes")) is not None:
                x = audio.pcm16_to_float(payload)
                if x.size == 0:
                    continue
                started = time.perf_counter()
                y = session.process(x)
                elapsed = max(time.perf_counter() - started, 1e-6)
                audio_ms = x.size / DEFAULT_SR
                # exponential moving average of processing speed
                realtime_ratio = 0.85 * realtime_ratio + 0.15 * (audio_ms / elapsed)
                await websocket.send_bytes(audio.float_to_pcm16(y))
                processed += x.size

                now = time.perf_counter()
                if now - last_report > 0.5:
                    last_report = now
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "processed_samples": processed,
                        "realtime_factor": round(realtime_ratio, 2),
                        "active_engine": session.engine_id,
                        "latency_ms": round(elapsed * 1000, 2),
                    }))

            elif (text := message.get("text")) is not None:
                try:
                    msg = json.loads(text)
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps(
                        {"type": "error", "detail": "invalid JSON"}))
                    continue
                await _handle_ws_config(websocket, session, msg)

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("websocket error")
        try:
            await websocket.send_text(json.dumps({"type": "error", "detail": str(exc)}))
        except Exception:
            pass


async def _handle_ws_config(websocket: WebSocket, session: engines.EngineSession,
                            msg: dict[str, Any]) -> None:
    mtype = msg.get("type")
    if mtype == "config":
        if "engine" in msg:
            notes = session.set_engine(str(msg["engine"]))
            if notes:
                await websocket.send_text(json.dumps({"type": "notice", "notes": notes}))
        if "voice" in msg:
            vid = str(msg["voice"])
            if bank().get(vid) is None:
                await websocket.send_text(json.dumps(
                    {"type": "error", "detail": f"unknown voice '{vid}'"}))
            else:
                session.set_voice(vid)
        if "params" in msg and isinstance(msg["params"], dict):
            merged = dict(session.params)
            for k, v in msg["params"].items():
                if k in NUMERIC_PARAMS:
                    try:
                        lo, hi = PARAM_BOUNDS.get(k, (-1e9, 1e9))
                        merged[k] = max(lo, min(hi, float(v)))
                    except (TypeError, ValueError):
                        continue
            session.set_params(merged)
        await websocket.send_text(json.dumps({
            "type": "config_ack", "engine": session.engine_id, "voice": session.voice_id,
            "params": session.params, "notes": session.notes,
        }))
    elif mtype == "ping":
        await websocket.send_text(json.dumps({"type": "pong"}))
    else:
        await websocket.send_text(json.dumps(
            {"type": "error", "detail": f"unknown message type '{mtype}'"}))


# --------------------------------------------------------------------------- #
# static UI (mounted last so /api and /ws take precedence)
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
