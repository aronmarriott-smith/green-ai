#!/bin/bash
# Native deployment script — use this instead of Docker when you want hardware-
# specific acceleration (Apple Silicon Metal/MLX, or NVIDIA without Container Toolkit).
#
# On Apple Silicon (M1/M2/M3):
#   Ollama runs natively with Metal acceleration.
#   The MLX model variant is used automatically for best performance.
#
# On NVIDIA (without nvidia-container-toolkit):
#   Ollama runs natively with CUDA acceleration.
#   Uses the standard model. Install nvidia-container-toolkit to use Docker instead.
#
# Usage:
#   chmod +x run-native.sh
#   ./run-native.sh

set -e

ARCH=$(uname -m)
OS=$(uname -s)

# ── Detect hardware and set model variant ─────────────────────────────────────
if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
  PLATFORM="Apple Silicon"
  export CHAT_MODEL="${CHAT_MODEL:-gemma4:e2b-mlx}"
  export EMBED_MODEL="${EMBED_MODEL:-embeddinggemma:300m}"
else
  PLATFORM="$OS ($ARCH)"
  export CHAT_MODEL="${CHAT_MODEL:-gemma4:e2b}"
  export EMBED_MODEL="${EMBED_MODEL:-embeddinggemma:300m}"
fi

echo ""
echo "  Green AI — Native"
echo "  ================="
echo "  Platform:  $PLATFORM"
echo "  Chat model: $CHAT_MODEL"
echo "  Embed model: $EMBED_MODEL"
echo ""

# ── Check Ollama is installed ─────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
  echo "  Error: Ollama not found."
  echo "  Install it from https://ollama.com and re-run."
  exit 1
fi

# ── Start Ollama if not already running ───────────────────────────────────────
if ! curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
  echo "  Starting Ollama..."
  OLLAMA_KEEP_ALIVE=-1 ollama serve > /tmp/ollama.log 2>&1 &
  echo "  Waiting for Ollama to be ready..."
  until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do sleep 2; done
else
  echo "  Ollama already running"
fi

# ── Pull models if not already downloaded ─────────────────────────────────────
pull_if_missing() {
  local model=$1
  if curl -sf -X POST http://localhost:11434/api/show \
      -H "Content-Type: application/json" \
      -d "{\"name\":\"$model\"}" > /dev/null 2>&1; then
    echo "  [cached] $model"
  else
    echo "  Downloading $model — first run only, may take several minutes..."
    ollama pull "$model"
    echo "  [ready]  $model"
  fi
}

echo "  Checking models..."
pull_if_missing "$EMBED_MODEL"
pull_if_missing "$CHAT_MODEL"

# ── Set up Python environment ─────────────────────────────────────────────────
# Prefer Homebrew Python — the macOS system Python lacks SQLite extension support
# which is required by sqlite-vec.
PYTHON3="python3"
for candidate in /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3; do
  if [[ -x "$candidate" ]]; then
    PYTHON3="$candidate"
    break
  fi
done

# On macOS 26 beta the system libexpat is missing a symbol that all Homebrew Python
# versions require. Override it with Homebrew's own expat when available.
HOMEBREW_EXPAT="/opt/homebrew/opt/expat/lib"
if [[ -d "$HOMEBREW_EXPAT" ]]; then
  export DYLD_LIBRARY_PATH="${HOMEBREW_EXPAT}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

# Recreate venv if it was built with a Python that lacks SQLite extension support.
if [[ -d venv ]] && ! venv/bin/python3 -c "import sqlite3; sqlite3.connect(':memory:').enable_load_extension(True)" 2>/dev/null; then
  echo "  Recreating venv with a Python that supports SQLite extensions..."
  rm -rf venv
fi

if [[ ! -d venv ]]; then
  echo "  Creating Python virtual environment..."
  "$PYTHON3" -m venv venv
fi

if ! venv/bin/pip show fastapi &>/dev/null; then
  echo "  Installing Python dependencies..."
  venv/bin/pip install -q -r requirements.txt
fi

# ── Run ingestion if the knowledge base is empty ──────────────────────────────
echo ""
echo "  Checking knowledge base..."
if venv/bin/python3 src/check_db.py 2>/dev/null; then
  echo "  Knowledge base OK"
else
  echo "  Running ingestion — this may take a while on first run..."
  venv/bin/python3 -u src/ingest.py
fi

# ── Start the server ──────────────────────────────────────────────────────────
echo ""
echo "  Server ready → http://localhost:8000"
echo "  Press Ctrl+C to stop."
echo ""
exec venv/bin/uvicorn src.server:app --host 0.0.0.0 --port 8000
