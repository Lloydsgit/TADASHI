/* ==========================================================================
   TADASHI front-end controller
   - Local mode  : AudioWorklet runs the voice engine in the audio thread
   - Server mode : PCM frames stream to /ws/stream and come back transformed
   - Bypass      : raw passthrough
   ========================================================================== */

const SR = 24000;
const FRAME = 1024;          // samples per streaming frame (~43 ms)
const $ = (id) => document.getElementById(id);

const state = {
  voices: [],
  categories: [],
  bounds: {},
  currentVoice: 'natural',
  params: {},               // live parameter values
  presetParams: {},         // preset values for the selected voice (for reset)
  mode: 'local',            // local | server | natural
  engineTier: 'dsp',
  engines: [],
  running: false,
  monitor: true,
  recording: false,
  workletReady: false,
  ttsTier: null,
  serverOnline: false,
  metrics: { inDb: -120, outDb: -120, latency: 0, speed: 0, engine: '—' },
};

const audio = {
  ctx: null, stream: null, source: null, worklet: null,
  analyserIn: null, analyserOut: null, sink: null,
  recorder: null, chunks: [], inBuf: null, outBuf: null, sampleBuf: null,
};

let ws = null, wsReady = false, wsPending = [];

/* ------------------------------------------------------------------ toast */
let toastTimer = null;
function toast(msg, kind = '') {
  const el = $('toast');
  el.textContent = msg;
  el.className = `toast show ${kind}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.className = 'toast'; }, 3600);
}

/* --------------------------------------------------------------- fetch API */
async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* keep statusText */ }
    throw new Error(detail);
  }
  return res;
}

/* ============================== BOOTSTRAP ============================== */
async function boot() {
  try {
    const data = await (await api('/api/voices')).json();
    state.voices = data.voices;
    state.categories = data.categories;
    state.bounds = data.bounds;
    renderCategories();
    renderVoices();
    selectVoice('natural', { silent: true });
    setServer(true);
  } catch (err) {
    setServer(false);
    toast(`Backend unreachable: ${err.message}`, 'err');
  }

  try {
    const info = await (await api('/api/engines')).json();
    state.engines = info.engines;
    renderEngineSelect(info);
  } catch { /* engines are optional for local mode */ }

  try {
    const t = await (await api('/api/tts/voices')).json();
    state.ttsTier = t.active_tier;
  } catch { /* TTS optional */ }

  buildSliders();
  wireEvents();
  initCanvases();
  requestAnimationFrame(drawLoop);
}

function setServer(online) {
  state.serverOnline = online;
  $('serverDot').className = `dot ${online ? 'ok' : 'bad'}`;
  $('serverText').textContent = online ? 'server online' : 'server offline';
}

/* ============================== VOICE BANK ============================== */
function renderCategories() {
  const box = $('categoryChips');
  box.innerHTML = '';
  const all = document.createElement('button');
  all.className = 'chip active';
  all.textContent = 'All';
  all.onclick = () => { filterCategory(null, all); };
  box.appendChild(all);
  state.categories.forEach((cat) => {
    const b = document.createElement('button');
    b.className = 'chip';
    b.textContent = cat;
    b.onclick = () => filterCategory(cat, b);
    box.appendChild(b);
  });
}

let activeCategory = null;
function filterCategory(cat, btn) {
  activeCategory = cat;
  document.querySelectorAll('#categoryChips .chip').forEach((c) => c.classList.remove('active'));
  btn.classList.add('active');
  renderVoices();
}

function renderVoices() {
  const grid = $('voiceGrid');
  const query = ($('voiceSearch').value || '').trim().toLowerCase();
  grid.innerHTML = '';
  const list = state.voices.filter((v) => {
    if (activeCategory && v.category !== activeCategory) return false;
    if (!query) return true;
    return `${v.name} ${v.tagline} ${v.category}`.toLowerCase().includes(query);
  });

  if (!list.length) {
    grid.innerHTML = '<p class="hint">No voices match that filter.</p>';
    return;
  }

  for (const v of list) {
    const card = document.createElement('div');
    card.className = `voice-card${v.id === state.currentVoice ? ' active' : ''}`;
    card.dataset.id = v.id;
    card.setAttribute('role', 'option');
    card.style.setProperty('--accent', v.accent || '#00e5c0');
    card.innerHTML = `
      <button class="vc-try" title="Audition">&#9654;</button>
      <div class="vc-top"><span class="vc-icon" style="color:${v.accent}">${v.icon}</span>
        <span class="vc-name">${v.name}</span></div>
      <div class="vc-tag">${v.tagline}</div>
      <div class="vc-cat">${v.category}</div>`;
    card.onclick = (e) => {
      if (e.target.classList.contains('vc-try')) { selectVoice(v.id); audition(v.id); return; }
      selectVoice(v.id);
    };
    grid.appendChild(card);
  }
}

function selectVoice(id, opts = {}) {
  const voice = state.voices.find((v) => v.id === id);
  if (!voice) return;
  state.currentVoice = id;
  state.presetParams = { ...voice.params };
  state.params = { ...voice.params };

  document.querySelectorAll('.voice-card').forEach((c) => {
    c.classList.toggle('active', c.dataset.id === id);
  });
  $('cvIcon').textContent = voice.icon;
  $('cvIcon').style.color = voice.accent;
  $('cvName').textContent = voice.name;
  $('cvTagline').textContent = voice.tagline;
  $('statVoice').textContent = voice.name;

  syncSliders();
  pushParams();
  if (!opts.silent) toast(`Voice: ${voice.name}`, 'ok');
}

/* ============================== SLIDERS ============================== */
const SLIDER_DEFS = [
  ['pitch_semitones', 'Pitch', -24, 24, 0.5, ' st'],
  ['formant_shift', 'Formant size', 0.5, 2, 0.01, 'x'],
  ['pitch_jitter', 'Pitch jitter', 0, 6, 0.05, ''],
  ['vibrato_depth_cents', 'Vibrato depth', 0, 200, 1, ' c'],
  ['vibrato_hz', 'Vibrato rate', 0, 14, 0.1, ' Hz'],
  ['body_gain_db', 'Body / chest', -24, 18, 0.5, ' dB'],
  ['presence_gain_db', 'Presence', -18, 18, 0.5, ' dB'],
  ['presence_freq', 'Presence freq', 400, 8000, 50, ' Hz'],
  ['brightness_gain_db', 'Brightness', -24, 18, 0.5, ' dB'],
  ['ring_mod_hz', 'Ring mod rate', 0, 900, 1, ' Hz'],
  ['ring_mod_mix', 'Ring mod mix', 0, 1, 0.01, ''],
  ['drive', 'Drive', 0, 12, 0.1, ''],
  ['drive_mix', 'Drive mix', 0, 1, 0.01, ''],
  ['noise_mix', 'Breath / grit', 0, 0.4, 0.005, ''],
  ['chorus_mix', 'Chorus mix', 0, 1, 0.01, ''],
  ['chorus_hz', 'Chorus rate', 0, 12, 0.1, ' Hz'],
  ['echo_ms', 'Echo time', 0, 1000, 5, ' ms'],
  ['echo_feedback', 'Echo feedback', 0, 0.92, 0.01, ''],
  ['echo_mix', 'Echo mix', 0, 1, 0.01, ''],
  ['stutter_hz', 'Stutter', 0, 60, 1, ' Hz'],
  ['output_gain_db', 'Output gain', -24, 18, 0.5, ' dB'],
];

const DEFAULTS = {
  pitch_semitones: 0, formant_shift: 1, pitch_jitter: 0, vibrato_depth_cents: 0,
  vibrato_hz: 0, body_gain_db: 0, presence_gain_db: 0, presence_freq: 2800,
  brightness_gain_db: 0, ring_mod_hz: 0, ring_mod_mix: 0, drive: 0, drive_mix: 0,
  noise_mix: 0, chorus_mix: 0, chorus_hz: 0, echo_ms: 0, echo_feedback: 0,
  echo_mix: 0, stutter_hz: 0, output_gain_db: 0,
  presence_q: 0.9, body_freq: 180, brightness_freq: 5000, noise_tone_hz: 3200,
  chorus_depth_ms: 2.5, dry_mix: 1,
};

function buildSliders() {
  const box = $('sliders');
  box.innerHTML = '';
  for (const [key, label, min, max, step, unit] of SLIDER_DEFS) {
    const wrap = document.createElement('div');
    wrap.className = 'slider';
    wrap.innerHTML = `
      <div class="slider-head"><label for="sl_${key}">${label}</label>
        <output id="out_${key}">0</output></div>
      <input type="range" id="sl_${key}" min="${min}" max="${max}" step="${step}" />`;
    box.appendChild(wrap);
    const input = wrap.querySelector('input');
    input.addEventListener('input', () => {
      state.params[key] = parseFloat(input.value);
      updateSliderVisual(key);
      pushParams();
    });
    input.addEventListener('dblclick', () => {
      const base = key in state.presetParams ? state.presetParams[key] : DEFAULTS[key];
      state.params[key] = base;
      syncSliders();
      pushParams();
    });
  }
}

function sliderValue(key) {
  const v = key in state.params ? state.params[key] : DEFAULTS[key];
  return typeof v === 'number' ? v : DEFAULTS[key];
}

function updateSliderVisual(key) {
  const input = $(`sl_${key}`);
  const out = $(`out_${key}`);
  if (!input) return;
  const def = SLIDER_DEFS.find((d) => d[0] === key);
  const val = sliderValue(key);
  input.value = val;
  const pct = ((val - def[2]) / (def[3] - def[2])) * 100;
  input.style.setProperty('--fill', `${pct}%`);
  const unit = def[5] || '';
  out.textContent = (def[4] < 1 ? val.toFixed(2) : val.toFixed(1)) + unit;
}

function syncSliders() {
  for (const [key] of SLIDER_DEFS) updateSliderVisual(key);
}

/* ============================== AUDIO GRAPH ============================== */
async function start() {
  try {
    audio.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    });
  } catch (err) {
    toast(`Microphone blocked: ${err.message}`, 'err');
    return;
  }

  audio.ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SR });
  await audio.ctx.resume();
  const ctx = audio.ctx;

  audio.source = ctx.createMediaStreamSource(audio.stream);
  audio.sink = ctx.createGain();
  audio.sink.gain.value = state.monitor ? 1 : 0;

  if (state.mode === 'local') {
    await startLocalGraph(ctx);
  } else if (state.mode === 'server') {
    startServerGraph(ctx);
  } else {
    audio.source.connect(audio.sink);           // bypass
  }

  audio.analyserIn = ctx.createAnalyser();
  audio.analyserIn.fftSize = 2048;
  audio.source.connect(audio.analyserIn);

  audio.analyserOut = ctx.createAnalyser();
  audio.analyserOut.fftSize = 2048;
  audio.sink.connect(audio.analyserOut);
  audio.sink.connect(ctx.destination);

  state.running = true;
  $('micBtn').setAttribute('aria-pressed', 'true');
  $('micLabel').textContent = 'Stop listening';
  $('footNote').textContent = `Listening · ${state.mode} mode`;
  toast('Microphone live', 'ok');
}

async function startLocalGraph(ctx) {
  if (!state.workletReady) {
    await ctx.audioWorklet.addModule('/static/js/worklet.js');
    state.workletReady = true;
  }
  audio.worklet = new AudioWorkletNode(ctx, 'tadashi-voice', {
    numberOfInputs: 1, numberOfOutputs: 1,
    outputChannelCount: [1],
    channelCount: 1, channelCountMode: 'explicit',
  });
  audio.source.connect(audio.worklet);
  audio.worklet.connect(audio.sink);
  pushParams();
}

function stopLocalGraph() {
  if (audio.worklet) { try { audio.worklet.disconnect(); } catch { /* already gone */ } audio.worklet = null; }
}

function startServerGraph(ctx) {
  const url = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/stream`;
  ws = new WebSocket(url);
  ws.binaryType = 'arraybuffer';
  wsReady = false;

  ws.onopen = () => {
    sendConfig();
    // Pump microphone frames to the server. A ScriptProcessor only runs while
    // it is connected to the graph, so it is wired to a muted gain node: the
    // raw microphone must NOT reach the speakers (only the server's processed
    // audio does), otherwise the user hears their own unprocessed voice.
    const proc = ctx.createScriptProcessor(FRAME, 1, 1);
    const mute = ctx.createGain();
    mute.gain.value = 0;
    audio.pump = proc;
    audio.pumpMute = mute;
    proc.onaudioprocess = (e) => {
      if (!wsReady || ws.readyState !== WebSocket.OPEN) return;
      const inCh = e.inputBuffer.getChannelData(0);
      const pcm = new Int16Array(inCh.length);
      for (let i = 0; i < inCh.length; i++) {
        const s = Math.max(-1, Math.min(1, inCh[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      ws.send(pcm.buffer);
    };
    audio.source.connect(proc);
    proc.connect(mute);
    mute.connect(ctx.destination);
  };

  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') {
      handleWsMessage(JSON.parse(ev.data));
      return;
    }
    const pcm = new Int16Array(ev.data);
    const buf = audio.ctx.createBuffer(1, pcm.length, SR);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
    const node = audio.ctx.createBufferSource();
    node.buffer = buf;
    node.connect(audio.sink);
    node.start();
    if (state.recording) recordChunk(buf);
  };

  ws.onerror = () => toast('Stream socket error', 'err');
  ws.onclose = () => { wsReady = false; };
}

function handleWsMessage(msg) {
  if (msg.type === 'ready') {
    wsReady = true;
    state.metrics.engine = msg.engine;
    $('statEngine').textContent = msg.engine;
    $('engineText').textContent = msg.engine;
    if (msg.notes && msg.notes.length) toast(msg.notes.join(' · '), '');
  } else if (msg.type === 'config_ack') {
    state.metrics.engine = msg.engine;
    $('statEngine').textContent = msg.engine;
  } else if (msg.type === 'status') {
    state.metrics.speed = msg.realtime_factor;
    state.metrics.latency = msg.latency_ms;
  } else if (msg.type === 'notice') {
    toast(msg.notes.join(' · '), '');
  } else if (msg.type === 'error') {
    toast(msg.detail, 'err');
  }
}

function stopServerGraph() {
  if (audio.pump) { try { audio.pump.disconnect(); } catch { /* ignore */ } audio.pump = null; }
  if (audio.pumpMute) { try { audio.pumpMute.disconnect(); } catch { /* ignore */ } audio.pumpMute = null; }
  if (ws) { try { ws.close(); } catch { /* ignore */ } ws = null; }
  wsReady = false;
}

function teardown() {
  stopLocalGraph();
  stopServerGraph();
  if (audio.stream) audio.stream.getTracks().forEach((t) => t.stop());
  if (audio.ctx) audio.ctx.close();
  audio.ctx = null; audio.source = null; audio.sink = null;
  state.running = false;
  $('micBtn').setAttribute('aria-pressed', 'false');
  $('micLabel').textContent = 'Start listening';
  $('footNote').textContent = 'Stopped.';
}

/* ============================== PARAMS SYNC ============================== */
function fullParams() {
  return { ...DEFAULTS, ...state.params };
}

function pushParams() {
  const params = fullParams();
  if (state.mode === 'local' && audio.worklet) {
    audio.worklet.port.postMessage({ type: 'params', params });
  } else if (state.mode === 'server' && ws && ws.readyState === WebSocket.OPEN) {
    sendConfig();
  }
}

function sendConfig() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({
    type: 'config',
    engine: state.engineTier,
    voice: state.currentVoice,
    params: fullParams(),
  }));
}

/* ============================== MODES ============================== */
async function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll('#modeSwitch button').forEach((b) => {
    b.classList.toggle('active', b.dataset.mode === mode);
  });
  const hints = {
    local: 'Local runs the voice engine inside your browser audio thread — lowest latency.',
    server: 'Server streams your voice to the Python engine and back. Try this on a weak device.',
    natural: 'Bypass passes your voice through untouched, for comparison.',
  };
  $('modeHint').textContent = hints[mode];
  if (state.running) { teardown(); await start(); }
}

/* ============================== AUDITION / PREVIEW ============================== */
let auditionAudio = null;
async function audition(voiceId) {
  try {
    const res = await api(`/api/preview?voice=${encodeURIComponent(voiceId)}&engine=${state.engineTier}`);
    const blob = await res.blob();
    playBlob(blob);
  } catch (err) { toast(`Audition failed: ${err.message}`, 'err'); }
}

function playBlob(blob) {
  if (auditionAudio) { auditionAudio.pause(); }
  auditionAudio = new Audio(URL.createObjectURL(blob));
  auditionAudio.play().catch((e) => toast(`Playback blocked: ${e.message}`, 'err'));
  return auditionAudio;
}

/* ============================== RECORDING ============================== */
function recordChunk(buf) {
  audio.chunks.push(buf);
}

function toggleRecord() {
  state.recording = !state.recording;
  $('recordBtn').setAttribute('aria-pressed', String(state.recording));
  $('recordBtn').textContent = state.recording ? 'Stop rec' : 'Record';
  if (!state.recording) return;

  audio.chunks = [];
  if (state.mode !== 'server' && audio.ctx && audio.sink) {
    // capture the post-effect signal locally
    const dest = audio.ctx.createMediaStreamDestination();
    audio.sink.connect(dest);
    audio.recorder = new MediaRecorder(dest.stream);
    audio.recorder.ondataavailable = (e) => { if (e.data.size) audio.chunks.push(e.data); };
    audio.recorder.start();
    audio.localRecDest = dest;
  }
  toast('Recording…', 'ok');
}

function stopRecording() {
  if (audio.recorder && audio.recorder.state !== 'inactive') {
    audio.recorder.onstop = () => {
      const blob = new Blob(audio.chunks, { type: 'audio/webm' });
      const link = $('downloadLink');
      link.href = URL.createObjectURL(blob);
      link.classList.remove('hidden');
      toast('Take ready to download', 'ok');
    };
    audio.recorder.stop();
  } else if (audio.chunks.length) {
    const wav = encodeWav(audio.chunks);
    const link = $('downloadLink');
    link.href = URL.createObjectURL(wav);
    link.classList.remove('hidden');
    toast('Take ready to download', 'ok');
  }
  if (audio.localRecDest) { try { audio.localRecDest.disconnect(); } catch { /* ignore */ } }
}

function encodeWav(buffers) {
  const total = buffers.reduce((n, b) => n + b.length, 0);
  const data = new Float32Array(total);
  let o = 0;
  for (const b of buffers) { data.set(b.getChannelData(0), o); o += b.length; }
  const buf = new ArrayBuffer(44 + total * 2);
  const view = new DataView(buf);
  const str = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); view.setUint32(4, 36 + total * 2, true); str(8, 'WAVE');
  str(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, SR, true);
  view.setUint32(28, SR * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  str(36, 'data'); view.setUint32(40, total * 2, true);
  let p = 44;
  for (let i = 0; i < total; i++) {
    const s = Math.max(-1, Math.min(1, data[i]));
    view.setInt16(p, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    p += 2;
  }
  return new Blob([buf], { type: 'audio/wav' });
}

/* ============================== VOICE MIMICRY ============================== */
async function analyzeFile(file) {
  const zone = $('analyzeZone');
  zone.classList.add('busy');
  $('analyzeLabel').textContent = 'Analysing…';
  try {
    const fd = new FormData();
    fd.append('file', file);
    const data = await (await api('/api/analyze', { method: 'POST', body: fd })).json();
    showAnalysis(data, file.name);
  } catch (err) {
    toast(`Analysis failed: ${err.message}`, 'err');
  } finally {
    zone.classList.remove('busy');
    $('analyzeLabel').textContent = 'Drop an audio file or click to analyse';
  }
}

function showAnalysis(data, filename) {
  const f = data.features;
  const p = data.preset;
  const box = $('analyzeResult');
  box.classList.remove('hidden');
  box.innerHTML = `
    <strong style="letter-spacing:.06em">${data.suggested_name}</strong>
    <dl>
      <dt>Source</dt><dd>${filename}</dd>
      <dt>Median pitch</dt><dd>${f.pitch.f0_median.toFixed(1)} Hz</dd>
      <dt>Pitch range</dt><dd>${f.pitch.f0_p10.toFixed(0)}–${f.pitch.f0_p90.toFixed(0)} Hz</dd>
      <dt>Brightness</dt><dd>${f.spectral.centroid ? f.spectral.centroid.toFixed(0) : '—'} Hz</dd>
      <dt>Voiced</dt><dd>${(f.pitch.voiced_ratio * 100).toFixed(0)}%</dd>
      <dt>Match pitch</dt><dd>${p.pitch_semitones > 0 ? '+' : ''}${p.pitch_semitones} st</dd>
      <dt>Match formant</dt><dd>${p.formant_shift}x</dd>
    </dl>
    <button id="applyMatch" class="accent-btn">Apply matched voice to engine</button>`;

  $('applyMatch').onclick = () => {
    state.params = { ...state.params, ...p };
    state.presetParams = { ...p };
    syncSliders();
    pushParams();
    $('cvName').textContent = data.suggested_name;
    $('cvTagline').textContent = `matched from ${filename}`;
    $('cvIcon').textContent = '\u25c8';
    toast('Matched voice applied', 'ok');
  };
}

/* ============================== AGENT SPEECH ============================== */
async function speak() {
  const text = $('ttsText').value.trim();
  if (!text) { toast('Type something for the agent to say', ''); return; }
  const btn = $('speakBtn');
  btn.disabled = true;
  btn.textContent = 'Synthesising…';
  try {
    const res = await api('/api/tts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        voice: state.currentVoice,
        engine: state.engineTier === 'natural' ? 'dsp' : state.engineTier,
      }),
    });
    const tier = res.headers.get('X-TTS-Tier') || '?';
    const blob = await res.blob();
    const player = playBlob(blob);
    player.onended = () => { btn.disabled = false; btn.textContent = '🔊 Speak'; };
    toast(`Speaking as ${state.currentVoice} (TTS: ${tier})`, 'ok');
  } catch (err) {
    toast(`Speech failed: ${err.message}`, 'err');
    btn.disabled = false;
  } finally {
    if (btn.textContent === 'Synthesising…') btn.textContent = '🔊 Speak';
    setTimeout(() => { btn.disabled = false; }, 400);
  }
}

/* ============================== SERVER CONVERT ============================== */
async function convertFile(file) {
  const zone = $('convertZone');
  zone.classList.add('busy');
  $('convertLabel').textContent = 'Converting…';
  try {
    const fd = new FormData();
    fd.append('file', file);
    const url = `/api/convert?voice=${encodeURIComponent(state.currentVoice)}&engine=${state.engineTier}`;
    const res = await api(url, { method: 'POST', body: fd });
    const factor = res.headers.get('X-Realtime-Factor');
    const blob = await res.blob();
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `${file.name.replace(/\.[^.]+$/, '')}-${state.currentVoice}.wav`;
    link.click();
    toast(`Converted (${factor}x realtime) — check your downloads`, 'ok');
  } catch (err) {
    toast(`Conversion failed: ${err.message}`, 'err');
  } finally {
    zone.classList.remove('busy');
    $('convertLabel').textContent = 'Drop audio to convert on the server';
  }
}

/* ============================== METERS & SCOPE ============================== */
function initCanvases() {
  const scope = $('scope');
  const dpr = window.devicePixelRatio || 1;
  scope.width = scope.clientWidth * dpr;
  scope.height = 90 * dpr;
  audio.sampleBuf = new Float32Array(2048);
  window.addEventListener('resize', () => {
    scope.width = scope.clientWidth * dpr;
    scope.height = 90 * dpr;
  });
}

function drawLoop() {
  requestAnimationFrame(drawLoop);
  const canvas = $('scope');
  const ctx2d = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx2d.clearRect(0, 0, w, h);
  ctx2d.fillStyle = '#0a0c12';
  ctx2d.fillRect(0, 0, w, h);

  ctx2d.strokeStyle = 'rgba(255,255,255,.05)';
  ctx2d.lineWidth = 1;
  ctx2d.beginPath(); ctx2d.moveTo(0, h / 2); ctx2d.lineTo(w, h / 2); ctx2d.stroke();

  if (state.running && audio.analyserOut) {
    audio.analyserOut.getFloatTimeDomainData(audio.sampleBuf);
    ctx2d.strokeStyle = '#00e5c0';
    ctx2d.lineWidth = 1.6 * (window.devicePixelRatio || 1);
    ctx2d.beginPath();
    const n = audio.sampleBuf.length;
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * w;
      const y = h / 2 - audio.sampleBuf[i] * (h / 2) * 0.92;
      i ? ctx2d.lineTo(x, y) : ctx2d.moveTo(x, y);
    }
    ctx2d.stroke();

    updateMeters();
  }
}

let meterTick = 0;
function updateMeters() {
  if (++meterTick % 3) return;      // ~20 Hz is plenty for meters

  const inDb = rmsDb(audio.analyserIn);
  const outDb = rmsDb(audio.analyserOut);
  state.metrics.inDb = inDb;
  state.metrics.outDb = outDb;

  setMeter('inMeter', 'inDb', inDb);
  setMeter('outMeter', 'outDb', outDb);

  $('statLatency').textContent = state.mode === 'local'
    ? '~22 ms'
    : `${state.metrics.latency.toFixed(0)} ms`;
  $('statSpeed').textContent = state.mode === 'local'
    ? 'realtime'
    : `${state.metrics.speed.toFixed(1)}x`;
}

function rmsDb(analyser) {
  if (!analyser) return -120;
  const buf = new Float32Array(analyser.fftSize);
  analyser.getFloatTimeDomainData(buf);
  let sum = 0;
  for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
  const rms = Math.sqrt(sum / buf.length);
  return rms < 1e-6 ? -120 : 20 * Math.log10(rms);
}

function setMeter(barId, textId, db) {
  const pct = Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
  $(barId).style.width = `${pct}%`;
  $(textId).textContent = db <= -119 ? '−∞' : `${db.toFixed(0)} dB`;
}

/* ============================== EVENTS ============================== */
function wireEvents() {
  $('micBtn').onclick = async () => {
    if (state.running) teardown(); else await start();
  };

  $('monitorBtn').onclick = () => {
    state.monitor = !state.monitor;
    $('monitorBtn').setAttribute('aria-pressed', String(state.monitor));
    $('monitorBtn').textContent = state.monitor ? 'Monitor off' : 'Monitor on';
    if (audio.sink) audio.sink.gain.value = state.monitor ? 1 : 0;
  };

  $('recordBtn').onclick = () => {
    if (state.recording) {
      state.recording = false;
      $('recordBtn').setAttribute('aria-pressed', 'false');
      $('recordBtn').textContent = 'Record';
      stopRecording();
    } else {
      toggleRecord();
    }
  };

  $('previewBtn').onclick = () => audition(state.currentVoice);
  $('resetBtn').onclick = () => {
    state.params = { ...state.presetParams };
    syncSliders();
    pushParams();
    toast('Preset restored', 'ok');
  };

  document.querySelectorAll('#modeSwitch button').forEach((b) => {
    b.onclick = () => setMode(b.dataset.mode);
  });

  $('engineSelect').onchange = (e) => {
    state.engineTier = e.target.value;
    updateEngineHint();
    if (state.mode === 'server') sendConfig();
  };

  $('voiceSearch').oninput = renderVoices;
  $('speakBtn').onclick = speak;
  $('stopSpeakBtn').onclick = () => { if (auditionAudio) auditionAudio.pause(); };

  const analyzeInput = $('analyzeInput');
  analyzeInput.onchange = () => analyzeInput.files[0] && analyzeFile(analyzeInput.files[0]);
  setupDropZone($('analyzeZone'), analyzeFile);

  const convertInput = $('convertInput');
  convertInput.onchange = () => convertInput.files[0] && convertFile(convertInput.files[0]);
  setupDropZone($('convertZone'), convertFile);

  document.addEventListener('keydown', (e) => {
    if (e.target.matches('input, textarea, select')) return;
    if (e.code === 'Space') { e.preventDefault(); $('micBtn').click(); }
    if (e.key >= '1' && e.key <= '9') {
      const idx = parseInt(e.key, 10) - 1;
      if (state.voices[idx]) selectVoice(state.voices[idx].id);
    }
  });
}

function setupDropZone(zone, handler) {
  ['dragenter', 'dragover'].forEach((ev) => zone.addEventListener(ev, (e) => {
    e.preventDefault(); zone.classList.add('dragover');
  }));
  ['dragleave', 'drop'].forEach((ev) => zone.addEventListener(ev, (e) => {
    e.preventDefault(); zone.classList.remove('dragover');
  }));
  zone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file) handler(file);
  });
}

/* ============================== ENGINE SELECT ============================== */
function renderEngineSelect(info) {
  const sel = $('engineSelect');
  sel.innerHTML = '';
  for (const e of info.engines) {
    const opt = document.createElement('option');
    opt.value = e.id;
    opt.textContent = `${e.name}${e.available ? '' : ' (unavailable)'}`;
    opt.disabled = !e.available;
    sel.appendChild(opt);
  }
  const best = info.engines.find((e) => e.available && e.quality === Math.max(
    ...info.engines.filter((x) => x.available).map((x) => x.quality)));
  state.engineTier = best ? best.id : 'dsp';
  sel.value = state.engineTier;
  updateEngineHint();
}

function updateEngineHint() {
  const e = state.engines.find((x) => x.id === state.engineTier);
  if (!e) return;
  $('engineHint').textContent = e.available
    ? `${e.description} (~${e.latency_ms} ms)`
    : `${e.description} Unavailable: ${e.reason}. Falls back automatically.`;
}

/* ============================== GO ============================== */
boot();
