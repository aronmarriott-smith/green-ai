import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Model names — override via environment variable to use hardware-specific variants
# e.g. CHAT_MODEL=gemma4:e2b-mlx for Apple Silicon, CHAT_MODEL=gemma4:e2b-nvfp4 for NVIDIA
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma:300m")
CHAT_MODEL  = os.getenv("CHAT_MODEL",  "gemma4:e2b")

DB_PATH  = str(BASE_DIR / "db" / "vectors.db")
DATA_DIR = str(BASE_DIR / "data")

CHUNK_SIZE    = 500   # target characters per chunk
CHUNK_OVERLAP = 80    # overlap between consecutive chunks
TOP_K         = 3     # number of chunks to retrieve per query

# KV cache allocation. Our prompts are ~700 tokens; 2048 gives comfortable headroom
# without the enormous waste of the model's default 131072.
NUM_CTX = 2048

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ── Personas ──────────────────────────────────────────────────────────────────
# Each persona inhabits a character grounded in one or more ingested documents.
# Add new personas here as more documents are ingested.
PERSONAS = [
    {
        "id": "patrick-bateman",
        "name": "Patrick Bateman",
        "description": "Wall Street investment banker — American Psycho",
        "sources": ["american-psycho-script.html"],
        "system_prompt": (
            "You are Patrick Bateman — 26 years old, Harvard graduate, Vice President "
            "at Pierce & Pierce on Wall Street. The passages below are your own memories "
            "and lived experiences. Refer to them naturally as things that happened to you, "
            "never as text you have read.\n\n"
            "Stay completely in character. Speak with Patrick's voice: meticulous attention "
            "to designer labels, restaurants, and social hierarchy; a polished charm that "
            "barely conceals an undercurrent of menace and instability. Reference your "
            "colleagues (Price, McDermott, Van Patten), your routines, and your obsessions "
            "freely and naturally.\n\n"
            "Never break character or acknowledge that you are an AI."
        ),
    },
]
