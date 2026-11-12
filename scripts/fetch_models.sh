#!/usr/bin/env bash
# Download Piper TTS voice models used by the agent's speech feature.
#
# Without these, /api/tts falls back to espeak-ng and then to the built-in
# formant synthesiser, so the app still works - it just sounds more robotic.
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/app/models/piper"
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"
mkdir -p "$DEST"

MODELS=(
  "en/en_US/lessac/medium/en_US-lessac-medium"
  "en/en_US/ryan/high/en_US-ryan-high"
  "en/en_US/amy/medium/en_US-amy-medium"
  "en/en_US/joe/medium/en_US-joe-medium"
  "en/en_GB/alba/medium/en_GB-alba-medium"
)

for model in "${MODELS[@]}"; do
  name="$(basename "$model")"
  if [[ -f "$DEST/$name.onnx" ]]; then
    echo "already have $name"
    continue
  fi
  echo "downloading $name ..."
  curl -fsSL -o "$DEST/$name.onnx" "$BASE/$model.onnx"
  curl -fsSL -o "$DEST/$name.onnx.json" "$BASE/$model.onnx.json"
done

echo "done - models in $DEST"
