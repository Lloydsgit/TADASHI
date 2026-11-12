import math

import numpy as np
from fastapi.testclient import TestClient

from app import audio
from app.engines import ENGINE_ORDER, available_engines, resolve_engine
from app.main import app
from app.voices import PARAM_BOUNDS, bank

SR = 24000
client = TestClient(app)


def sine(freq: float, seconds: float, sr: int = SR, amp: float = 0.4) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return amp * np.sin(2 * math.pi * freq * t)


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #
def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["voices"] >= 15


def test_voices_listing_has_categories_and_bounds():
    body = client.get("/api/voices").json()
    ids = {v["id"] for v in body["voices"]}
    assert "natural" in ids
    assert {"male", "female", "robotic", "creature"} <= set(body["categories"])
    assert body["sample_rate"] == SR
    assert "pitch_semitones" in body["bounds"]


def test_every_voice_has_a_resolvable_preset():
    for voice in bank().list_dicts():
        r = client.get(f"/api/voices/{voice['id']}")
        assert r.status_code == 200
        params = r.json()["resolved_params"]
        assert isinstance(params, dict) and params


def test_unknown_voice_is_404():
    assert client.get("/api/voices/does-not-exist").status_code == 404


def test_engines_endpoint_lists_the_chain():
    body = client.get("/api/engines").json()
    ids = [e["id"] for e in body["engines"]]
    assert ids == list(ENGINE_ORDER)
    assert body["best_available"] in ids
    assert any(e["available"] for e in body["engines"])


# --------------------------------------------------------------------------- #
# engine fallback
# --------------------------------------------------------------------------- #
def test_natural_engine_always_available():
    infos = {e.id: e for e in available_engines()}
    assert infos["natural"].available
    assert infos["dsp"].available


def test_unknown_engine_falls_back_with_a_note():
    engine_id, notes = resolve_engine("warp-drive")
    assert engine_id in ENGINE_ORDER
    assert notes


def test_neural_falls_back_to_dsp_when_unavailable():
    engine_id, notes = resolve_engine("neural")
    infos = {e.id: e for e in available_engines()}
    if not infos["neural"].available:
        assert engine_id == "dsp"
        assert notes


def test_session_degrades_instead_of_failing():
    from app.engines import EngineSession
    session = EngineSession(SR, bank(), engine_id="dsp", voice_id="titan")
    original = session.engine.process
    session.engine.process = lambda block: (_ for _ in ()).throw(RuntimeError("boom"))
    out = session.process(sine(200.0, 0.05))
    assert out.size > 0
    assert session.failures == 1
    assert session.engine_id == "natural"
    session.engine.process = original


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #
def _wav_upload(freq=200.0, seconds=0.5):
    return audio.encode_wav(sine(freq, seconds), SR)


def test_convert_returns_audio_and_headers():
    r = client.post("/api/convert?voice=titan&engine=dsp",
                    files={"file": ("in.wav", _wav_upload(), "audio/wav")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.headers["X-Voice"] == "titan"
    assert float(r.headers["X-Realtime-Factor"]) > 1.0


def test_convert_actually_changes_the_signal():
    src = _wav_upload(220.0, 0.6)
    out = client.post("/api/convert?voice=titan&engine=dsp",
                      files={"file": ("in.wav", src, "audio/wav")}).content
    y, _ = audio.decode_audio(out, SR)
    x, _ = audio.decode_audio(src, SR)
    assert y.size == x.size
    assert not np.allclose(x, y, atol=1e-3)


def test_convert_rejects_unknown_voice():
    r = client.post("/api/convert?voice=nope",
                    files={"file": ("in.wav", _wav_upload(), "audio/wav")})
    assert r.status_code == 404


def test_convert_rejects_garbage_upload():
    r = client.post("/api/convert?voice=natural",
                    files={"file": ("in.wav", b"not audio at all", "audio/wav")})
    assert r.status_code == 400


def test_convert_rejects_empty_upload():
    r = client.post("/api/convert?voice=natural",
                    files={"file": ("in.wav", b"", "audio/wav")})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def test_analyze_returns_features_and_preset():
    body = client.post("/api/analyze",
                       files={"file": ("in.wav", _wav_upload(150.0, 1.0), "audio/wav")}).json()
    assert abs(body["features"]["pitch"]["f0_median"] - 150.0) < 8.0
    assert "pitch_semitones" in body["preset"]
    assert body["suggested_name"]


def test_analyze_rejects_empty():
    assert client.post("/api/analyze", files={"file": ("in.wav", b"", "audio/wav")}).status_code == 400


# --------------------------------------------------------------------------- #
# preview / tts
# --------------------------------------------------------------------------- #
def test_preview_returns_playable_wav():
    r = client.post("/api/preview?voice=kaiju&engine=dsp")
    assert r.status_code == 200
    y, sr = audio.decode_audio(r.content, SR)
    assert sr == SR and y.size > SR // 2
    assert np.max(np.abs(y)) > 0.01


def test_preview_rejects_unknown_voice():
    assert client.post("/api/preview?voice=zzz").status_code == 404


def test_tts_voices_endpoint_reports_a_tier():
    body = client.get("/api/tts/voices").json()
    assert body["active_tier"] in {"piper", "espeak", "formant"}
    assert isinstance(body["voices"], list)


def test_tts_speaks_text_in_the_selected_voice():
    r = client.post("/api/tts", json={"text": "Hello there.", "voice": "luna", "engine": "dsp"})
    assert r.status_code == 200
    assert r.headers["X-Voice"] == "luna"
    assert r.headers["X-TTS-Tier"]
    y, _ = audio.decode_audio(r.content, SR)
    assert y.size > SR // 4
    assert np.max(np.abs(y)) > 0.02


def test_tts_requires_text():
    assert client.post("/api/tts", json={"voice": "luna"}).status_code == 400


def test_tts_rejects_unknown_voice():
    assert client.post("/api/tts", json={"text": "hi", "voice": "nope"}).status_code == 404


# --------------------------------------------------------------------------- #
# websocket streaming
# --------------------------------------------------------------------------- #
def test_ws_handshake_reports_engine_and_rate():
    with client.websocket_connect("/ws/stream") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["sample_rate"] == SR
        assert ready["engine"] in ENGINE_ORDER


def test_ws_streams_back_same_length_audio():
    frame = audio.float_to_pcm16(sine(200.0, 0.1))
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()  # ready
        ws.send_bytes(frame)
        back = ws.receive_bytes()
        assert len(back) == len(frame)
        y = audio.pcm16_to_float(back)
        assert np.max(np.abs(y)) > 0.01


def test_ws_config_changes_voice_and_params():
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        ws.send_json({"type": "config", "voice": "pixie", "engine": "dsp",
                      "params": {"pitch_semitones": 6.0}})
        ack = ws.receive_json()
        assert ack["type"] == "config_ack"
        assert ack["voice"] == "pixie"
        assert ack["params"]["pitch_semitones"] == 6.0


def test_ws_clamps_out_of_range_params():
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        ws.send_json({"type": "config", "params": {"pitch_semitones": 999.0}})
        ack = ws.receive_json()
        lo, hi = PARAM_BOUNDS["pitch_semitones"]
        assert lo <= ack["params"]["pitch_semitones"] <= hi


def test_ws_rejects_unknown_voice_but_stays_open():
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "config", "voice": "ghostbuster"})
        assert ws.receive_json()["type"] == "error"
        # the ack still reports the unchanged, valid state
        ack = ws.receive_json()
        assert ack["type"] == "config_ack"
        assert ack["voice"] == "natural"
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"


def test_ws_ping_pong():
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        ws.send_json({"type": "ping"})
        assert ws.receive_json()["type"] == "pong"


def test_ws_ignores_bad_json():
    with client.websocket_connect("/ws/stream") as ws:
        ws.receive_json()
        ws.send_text("{not json")
        assert ws.receive_json()["type"] == "error"


# --------------------------------------------------------------------------- #
# static UI
# --------------------------------------------------------------------------- #
def test_index_page_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "TADASHI" in r.text


def test_static_assets_are_served():
    for path in ("/static/styles.css", "/static/js/app.js", "/static/js/worklet.js"):
        assert client.get(path).status_code == 200, path
