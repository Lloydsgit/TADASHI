# TADASHI

A real-time AI speech agent that listens to your voice and changes it live.

Speak into your microphone and hear a different voice come out of the
speakers, in real time. Pick from 17 built-in voices, tune any of them with 21
parameters, or upload a recording of someone and have TADASHI build a matching
voice. The agent can also speak text in any of those voices.

Everything runs on your own machine. No cloud service, no API keys, no audio
leaving your computer.

---

## What it does

- **Real-time voice morphing.** Live microphone in, transformed voice out.
  Two engines: one runs in your browser's audio thread (lowest latency), one
  runs on the server over a WebSocket.
- **17 voices** across five families, plus a clean passthrough:
  - *male* — TITAN, OMEGA, RADIO, KAIJU
  - *female* — LUNA, ARIA, SABLE, PIXIE
  - *robotic* — ANDROID, CYBER, GLITCH, QUANTUM
  - *creature* — DEMON, GHOST, GIANT, GNOME
  - *fallback* — NATURAL (your own voice, bit-exact)
- **Voice mimicry.** Upload a recording; TADASHI measures its pitch and timbre
  and generates a matching preset you can apply live.
- **Agent speech.** Type text and the agent says it in the selected voice
  (neural TTS, then morphed through the voice engine).
- **Server conversion.** Upload a clip, download it re-voiced, offline and
  higher quality.
- **Live tuning.** 21 sliders — pitch, formant size, vibrato, ring modulation,
  drive, breath, chorus, echo, stutter, EQ, output gain — with live meters and
  a waveform scope.

## Quick start

```bash
pip install -r requirements.txt
./scripts/fetch_models.sh     # optional: neural TTS voices (~180 MB)
./run.sh                      # http://127.0.0.1:8000
```

Open the URL, click **Start listening**, allow microphone access, and pick a
voice. Headphones are strongly recommended — with speakers, your microphone
will pick up the transformed voice and you will get feedback.

Use `PORT=12000 ./run.sh` to serve on a different port.

## How it works

The container has no audio hardware, so the microphone and speakers have to
live in the browser. That single constraint shapes the whole design:

```
                    ┌─────────────────────── browser ───────────────────────┐
  microphone ──▶    │  getUserMedia                                         │
                    │      │                                                │
                    │      ├─ Local mode ──▶ AudioWorklet ──▶ speakers      │
                    │      │                 (voice DSP in the audio thread) │
                    │      │                                                │
                    │      └─ Server mode ─▶ WebSocket ─┐                   │
                    └───────────────────────────────────┼───────────────────┘
                                                        ▼
                                          ┌──────── Python server ────────┐
                                          │  /ws/stream  PCM frames       │
                                          │  NumPy DSP engine             │
                                          │  (pitch, formants, effects)   │
                                          └───────────────────────────────┘
```

### Two engines, same chain

The voice transformation is implemented twice, deliberately:

| | Local (AudioWorklet) | Server (WebSocket) |
|---|---|---|
| Where it runs | Browser audio thread | Python process |
| Added latency | ~22 ms (one pitch window) | ~1.5 ms round trip + frame time |
| Works offline | Yes | No |
| Best for | Everyday use, low latency | Weak devices, batch conversion |

Both implement the same signal chain, and both are covered by tests that assert
the same invariants (see *Tests* below).

### The signal chain

```
input ─▶ DC/rumble highpass
      ─▶ pitch shift (+ jitter, + vibrato)   ← two crossfaded delay taps
      ─▶ formant emphasis                     ← bank of peaking resonators
      ─▶ body EQ ─▶ presence EQ ─▶ brightness EQ
      ─▶ ring modulation                      ← robotic / metallic voices
      ─▶ drive (tanh saturation)
      ─▶ breath noise (envelope-gated)
      ─▶ chorus ─▶ echo ─▶ stutter
      ─▶ output gain ─▶ limiter ─▶ output
```

Stages whose amount is zero are skipped entirely, which is why the NATURAL
voice costs nothing.

### Pitch shifting

The pitch shifter is a classic two-tap granular shifter. Two read pointers
sweep a ring buffer at a rate set by the pitch ratio, offset by half a window
and crossfaded with a raised cosine, so the wrap point of one tap is always
hidden behind the peak of the other. A 44 ms window gives ~22 ms of latency.

### Fallback chain

Engines are tried in order of quality and degrade automatically:

1. **`neural`** — an ONNX voice-conversion model. Best quality, needs a GPU and
   model weights, so it is optional. Drop a `.onnx` file in `app/models/` to
   enable it.
2. **`dsp`** — the NumPy chain. Runs anywhere on CPU, works for every voice.
3. **`natural`** — pure passthrough. Always available, zero latency.

If the requested engine is unavailable, the next one down is used and the
reason is reported to the UI. If an engine raises mid-stream, the session
degrades a step rather than dropping your audio.

The same pattern applies to agent speech: `piper` (neural TTS) → `espeak-ng`
→ a built-in formant synthesiser, so the Speak button always produces sound.

## Voice mimicry

`POST /api/analyze` with an audio file returns measured voice character and a
matched preset:

- **Pitch** via autocorrelation, over voiced frames only. Reported as median
  plus a 10th–90th percentile range.
- **Timbre** via spectral centroid, rolloff, flatness and spectral tilt.

The preset maps those measurements onto DSP parameters: pitch difference from a
neutral 165 Hz reference becomes `pitch_semitones`, spectral centroid drives
`formant_shift`, spectral tilt drives `brightness_gain_db`, and so on.

This is a *character* match, not identity cloning. It captures pitch, size and
brightness. True speaker-identity cloning needs a trained voice-conversion model
— which is exactly what the optional `neural` engine slot is for.

## API

Interactive docs at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Status and version |
| `GET` | `/api/voices` | Voice bank, categories, parameter bounds |
| `GET` | `/api/voices/{id}` | One voice with resolved parameters |
| `GET` | `/api/engines` | Engine tiers and availability |
| `GET` | `/api/params` | Parameter names and bounds |
| `POST` | `/api/convert` | Upload audio, get it re-voiced (WAV) |
| `POST` | `/api/analyze` | Measure a voice sample, get a matched preset |
| `POST` | `/api/preview` | Audition a voice with synthesised vowels |
| `POST` | `/api/tts` | Speak text as a chosen voice (WAV) |
| `GET` | `/api/tts/voices` | Available TTS voices and active tier |
| `WS` | `/ws/stream` | Real-time PCM streaming |

### WebSocket protocol

Client sends 16-bit little-endian mono PCM frames; the server replies with
transformed frames of the same length.

```
client → server   binary : raw PCM16 samples
client → server   text   : {"type":"config","voice":"titan","engine":"dsp","params":{...}}
server → client   text   : {"type":"ready"|"status"|"config_ack"|"error"|"notice", ...}
server → client   binary : transformed PCM16, same sample count as the input
```

Server conversion returns headers describing the work done:
`X-Engine`, `X-Voice`, `X-Process-Ms`, `X-Realtime-Factor`.

## Configuration

- **Sample rate** — 24 kHz throughout (`DEFAULT_SR` in `app/main.py`, `SR` in
  `app/static/js/app.js`). Change both together.
- **Voices** — edit `app/data/voices.json`. Every parameter is optional and
  falls back to `defaults`. Out-of-range values are clamped on load, so a
  hand-edited preset cannot break the DSP.
- **Frame size** — 1024 samples (~43 ms) in `app/static/js/app.js`.
- **Upload limit** — 12 MB (`MAX_SAMPLE_BYTES`).

## Tests

```bash
python3 -m pytest          # 77 tests: DSP, API, WebSocket, static assets
node tests/worklet_test.js # 8 tests: the browser engine, under Node
```

The interesting tests are the invariant ones. Because this is a *streaming*
engine, output must not depend on how the audio happens to be split into
blocks:

- `test_output_is_independent_of_block_size` — runs every voice at block sizes
  128, 333, 1024 and 4096 and requires identical output.
- `test_modulation_sources_are_block_size_independent` — same for ring mod,
  vibrato, chorus and stutter.
- The Node suite asserts the same property for the AudioWorklet.

These caught six real bugs during development, including three separate
oscillators that double-counted their phase (making their frequency depend on
block size), a limiter that ramped up from silence instead of starting at unity
gain, output gain applied *after* the limiter so voices could clip, and a delay
line that aliased when the delay was shorter than the block.

## Layout

```
app/
  main.py            FastAPI app: REST + WebSocket + static hosting
  dsp.py             The voice engine (stateful, chunk-streamable)
  engines.py         Engine registry and the fallback chain
  voices.py          Voice bank loading, validation, preset resolution
  tts.py             Agent speech: piper → espeak → formant
  synth.py           Formant synthesiser (auditions + last-resort TTS)
  audio.py           PCM codecs, resampling, WAV I/O
  data/voices.json   The voice bank
  static/
    index.html       UI
    styles.css       Styling
    js/app.js        Controller: modes, UI, recording, streaming
    js/worklet.js    Real-time engine (runs in the browser audio thread)
tests/
  test_dsp.py        DSP correctness and invariants
  test_api.py        REST, WebSocket, static assets
  worklet_test.js    Browser engine, run under Node
  worklet_harness.js Node stub for the audio-thread globals
scripts/
  fetch_models.sh    Download Piper TTS models
```

## Keyboard shortcuts

- `Space` — start / stop listening
- `1`–`9` — select the first nine voices
- Double-click a slider — reset it to the preset value

## Troubleshooting

**No sound, or only my own voice.** Check the routing buttons. *Bypass* passes
your voice through untouched; *Local* and *Server* transform it.

**Feedback squeal.** Use headphones. With speakers the microphone hears the
transformed output.

**"Microphone blocked".** The page needs microphone permission, and browsers
only grant it on `localhost` or HTTPS. Use `http://127.0.0.1:8000`, not a LAN
IP over plain HTTP.

**The agent sounds robotic.** No TTS models are installed. Run
`./scripts/fetch_models.sh`.

**Crackling or dropouts in server mode.** The server is not keeping up.
Switch to Local mode, which runs on your machine's audio thread instead.
