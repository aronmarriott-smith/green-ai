# Raspberry Pi 2B — Feasibility Research

Working notes on whether/how Green AI's tiers (0–3) can run on a Raspberry Pi 2B.
This is **research only** — no production deployment scripts yet.

## Hardware under test

| | |
|---|---|
| Model | Raspberry Pi 2 Model B Rev 1.1 |
| SoC | BCM2836 (4× Cortex-A7 @ ~900 MHz, ARMv7-A) |
| RAM | 1 GB (921 MiB usable) + 920 MiB swap |
| Storage | 32 GB microSD, 25 GB free |
| Network | 100 Mbps Ethernet, no Wi-Fi/Bluetooth |
| OS | Raspberry Pi OS (Raspbian) 13 "trixie", **32-bit (armv7l)** |
| Python | 3.13.5 |

**Important architectural constraint:** the BCM2836 (Cortex-A7) is a 32-bit-only
SoC. Unlike the Pi 2B rev 1.2 (BCM2837, Cortex-A53), this specific board
**cannot run a 64-bit OS at all** — armv7l/armhf is the ceiling regardless of
which distro is installed. Any solution that requires `aarch64`/`arm64`
binaries is out, permanently, for this board.

Raspbian ships with [piwheels.org](https://www.piwheels.org/) configured as an
extra pip index — this provides prebuilt armv7l wheels for many popular
packages that don't publish their own (e.g. `pydantic-core`, `lxml`), which
helps a lot here.

---

## Device hardening (applied 2026-06-13)

The device became unresponsive ("Host is down") after a heavy `pip install`
(`uvicorn[standard]`/`ollama`) likely OOM'd while compiling a dependency from
source (no armv7l wheel). Required a power cycle to recover. Before continuing,
applied the following mitigations:

1. **Hardware watchdog (auto-reboot on hang)** — `/etc/systemd/system.conf`:
   `RuntimeWatchdogSec=20s`, `RebootWatchdogSec=10min`. systemd pings
   `/dev/watchdog`; if the kernel/systemd hangs for 20s, the SoC force-resets.
   Applied live via `systemctl daemon-reexec` (no reboot needed).
2. **Extra overflow swap** — added a 1 GB `/swapfile` (disk-backed, priority
   10) below the existing zram swap (priority 100, ~920 MB compressed RAM).
   Total swap now ~1.9 GB. zram is used first (fast); the disk swapfile is a
   last-resort buffer to avoid an instant OOM kill during memory spikes (e.g.
   pip source builds). Persisted via `/etc/fstab`.
3. **Download integrity practice** — for any large file fetched onto the Pi
   (model weights, source tarballs), verify a checksum after download
   (`sha256sum` against a published digest) and prefer resumable transfers
   (`curl -fSL -C -` / `wget -c`) so an interrupted transfer is detected
   rather than silently leaving a truncated file.
4. **No-hang-service testing practice** — anything that runs as a foreground
   service (e.g. `uvicorn`) is started with `timeout <N>s` or as a background
   job with a captured PID that we explicitly `kill` afterwards, so a test
   session never leaves a stray process running or blocks the SSH session
   indefinitely. Heavy `pip install`s are run with conservative package sets
   (no unnecessary extras) and `free -h` is checked before/during installs.

---

## Tier 0 — Keyword search (BM25 only, no embeddings/LLM)

**Goal:** `rank-bm25` + FastAPI serving keyword search over ingested chunks,
no vector DB, no Ollama.

### Findings

- `numpy` (a transitive dep of `rank-bm25`) has **no PyPI wheel for
  armv7l + Python 3.13**, and building from source fails (missing
  `python3-dev`, and would be very slow on this CPU anyway).
  - **Fix:** `sudo apt-get install python3-numpy` (Debian ships a precompiled
    armhf build, 2.2.4) then create the venv with `--system-site-packages`.
- With that in place, `pip install rank-bm25 beautifulsoup4 fastapi` **succeeded
  cleanly** — piwheels provided prebuilt wheels for `pydantic-core` etc.
- `lxml` and `uvicorn[standard]` — install attempted, in progress / caused the
  Pi to become temporarily unresponsive (see below). Needs re-test, likely
  without the `[standard]` extras (uvloop/httptools/watchfiles are Rust/C
  extensions that may need source builds).

### Update — full Tier 0 dependency set installed

With the apt-installed `numpy` + `--system-site-packages` venv in place:

- `rank-bm25`, `beautifulsoup4`, `fastapi`, `lxml`, plain `uvicorn` (no
  `[standard]` extras), and the `ollama` Python client **all installed
  successfully** via prebuilt piwheels armv7l wheels. No source builds, no
  memory issues during install (stayed at ~120-150 MB used / 921 MB).
- `uvicorn[standard]` (uvloop/httptools/websockets/watchfiles) was the likely
  cause of the first crash — these have no armv7l wheels and would compile
  from source (watchfiles uses Rust/Cargo, which is very memory-hungry).
  **Avoid `[standard]` extras on this board.**

### Second crash — running a smoke-test script

After installing the full Tier 0 set, ran a tiny script that just *imports*
`rank_bm25`, `bs4`, and `fastapi` and does a trivial BM25 query. This made the
Pi unresponsive again (SSH timed out) even with `timeout 60` wrapping it and
the hardware watchdog enabled. Awaiting a power cycle to investigate via
`journalctl`/`dmesg` for OOM-killer or watchdog-reset evidence.

**Working hypothesis:** first-time import of `pydantic`/`pydantic-core` +
`numpy` + `fastapi` on a 4×900MHz Cortex-A7 involves a large amount of one-time
bytecode compilation/module initialisation, which combined with the binary
sizes of `pydantic_core` and `numpy` may be enough to cause severe swap
thrashing (and the watchdog's `RuntimeWatchdogSec=20s` may not fire if the
kernel itself — not just systemd — is the thing stalled on I/O).

### Update — second crash was likely a red herring

After the power cycle, `smoke.py` had **not survived on disk** (the heredoc
write was likely never `fsync`'d to the SD card before the hard power-loss —
confirms the OS really did go down hard, not just an SSH hiccup). Re-ran each
import individually with `timeout` safeguards and full `free -h` checks:

| Test | Time | Result |
|------|------|--------|
| `import numpy` | 1.6s | OK |
| `from fastapi import FastAPI; FastAPI()` | 5.3s | OK |
| `rank_bm25` end-to-end query | 1.2s | OK |
| Full smoke script (BM25 + BS4/lxml + FastAPI) | 6.0s | OK |
| **`uvicorn` server start + `curl /ping`** | ~8s to first response | **OK — 200 {"ok":true}** |

Memory stayed at **148 MB / 921 MB** with the server running. No further
crashes. **Conclusion: the second crash was very likely an unrelated transient
issue** (or a leftover effect of the *first* crash's filesystem state) —
individually, none of these imports or the running server are heavy enough to
threaten this device's memory budget.

### Verdict — Tier 0

**✅ Confirmed feasible.** A FastAPI + plain `uvicorn` app serving `rank-bm25`
keyword search over `beautifulsoup4`/`lxml`-parsed HTML runs comfortably
within ~150 MB RAM and starts in single-digit seconds. Required setup deltas
from the standard `requirements.txt`:
- `apt-get install python3-numpy` (no armv7l PyPI wheel for numpy), then
  create the venv with `--system-site-packages`
- Drop `uvicorn[standard]` → plain `uvicorn` (extras have no armv7l wheels and
  source-build with Rust/Cargo, which is what caused the *first* crash)
- Drop `sqlite-vec` and `ollama`-server-dependent code paths (see Tiers 1/2)

---

## Tier 1 — Semantic search (embeddings + hybrid search)

**Goal:** embed chunks, store/query vectors, combine with BM25.

### Findings

- **`sqlite-vec` has no PyPI wheel for this board** (`macosx`, `win_amd64`,
  `manylinux x86_64`/`aarch64` only — no armv7l, no sdist). **However, the
  underlying C extension compiles natively from source without issue** — see
  below. This unblocks vector storage entirely.
- Embedding model: `embeddinggemma:300m` is served via **Ollama**, which is
  not installable (see Tier 2). The current embedding pipeline is still
  blocked — but this is now an *embedding-generation* problem only, separate
  from vector storage.

### 🎉 sqlite-vec compiles natively on the Pi 2B

Following [the official compiling guide](https://alexgarcia.xyz/sqlite-vec/compiling.html):

```bash
git clone --depth 1 https://github.com/asg017/sqlite-vec
cd sqlite-vec
bash scripts/vendor.sh   # fetches the SQLite amalgamation (~11MB)
make loadable             # -> dist/vec0.so
```

Results on this Pi 2B:
- **Build time: ~49 seconds** (`make loadable`, single C file, gcc 14.2.0)
- **Peak RAM during build: ~170 MB** (well within budget)
- Output: `dist/vec0.so` — `ELF 32-bit LSB shared object, ARM, EABI5`, 212KB
- Only harmless `-Wmaybe-uninitialized` warnings, no errors

No cross-compilation needed — the Pi's own `gcc`/`make` (already present in
Raspberry Pi OS) handle it directly. The armv6-vs-armv7 cross-compile concerns
from the forum thread / `docker-arm-cross-toolchain` are **not needed** when
building natively on-device; the system compiler already targets the correct
ABI (`armhf`) for this OS.

### Functional + performance test

Loaded `dist/vec0` via Python's stdlib `sqlite3.Connection.load_extension()`
(no PyPI package needed — see integration note below):

```python
db.enable_load_extension(True)
db.load_extension('./dist/vec0')
db.enable_load_extension(False)
db.execute('select vec_version()')   # -> 'v0.1.10-alpha.4'
```

- Created a `vec0` virtual table with `float[768]` columns (matches the
  project's `EMBED_DIM = 768`).
- **Inserted 500 × 768-dim vectors in 0.79s**
- **KNN query (`k=3`) over those 500 rows: 18ms**

Both are completely negligible relative to this device's other constraints —
vector search is *not* the bottleneck on this board.

### Integration approach (no upstream package needed)

`src/rag.py`, `src/ingest.py`, `src/search.py`, and `src/benchmark.py` all do:
```python
import sqlite_vec
...
sqlite_vec.load(db)
```
The real `sqlite_vec` PyPI package is just a ~5-line shim:
```python
def loadable_path():
    return os.path.join(os.path.dirname(__file__), "vec0")
def load(db):
    db.load_extension(loadable_path())
```
For the Pi, we can drop in a **local module also named `sqlite_vec`** (placed
earlier on `sys.path`, e.g. alongside `src/`) containing the same `load()`
function, with our self-compiled `vec0.so` sitting next to it. **Zero changes
needed to `ingest.py`/`rag.py`/`search.py`/`benchmark.py`.**

### Remaining blocker: embedding generation

Vector storage is solved, but we still need a way to turn text into
768-dim (or whatever-dim) vectors without Ollama. Candidates to investigate
next:
- `fastembed` (ONNX Runtime) — check armv7l wheel availability for
  `onnxruntime`
- A small `sentence-transformers` model via PyTorch CPU — PyTorch's armv7l
  support is historically poor/dropped; needs checking
- Brute-force numpy cosine similarity isn't needed anymore (sqlite-vec works),
  but remains a fallback if a different storage format is preferred

### Verdict so far

**🟢 Vector storage: solved** (self-compiled `sqlite-vec`, drop-in shim, fast
enough). **🟡 Embedding generation: still open** — this is now the only
remaining piece for Tier 1, and it's an ML-runtime question, not a
SQLite/storage question.

---

## Tier 2 — Conversational AI (RAG Q&A via Ollama + gemma4:e2b)

### Findings

- Checked Ollama's latest GitHub release assets: only
  `ollama-linux-amd64*` and `ollama-linux-arm64*` (64-bit ARM) builds are
  published. **No 32-bit ARM (`armv7`/`armhf`) build exists.**
- Even if a community/self-built Ollama binary existed, `gemma4:e2b` is
  7.2 GB — far beyond this device's 1 GB RAM + 920 MB swap. Swap thrashing on
  a microSD card would also be extremely slow and wear the card.

### Verdict

**Hard blocker, on two independent grounds** (no Ollama binary for this arch,
and the model doesn't fit in RAM even if it ran). Not feasible on this board
under any configuration.

---

## Tier 3 — Multimodal (image search / vision)

Depends entirely on Tier 2 (Ollama + a vision-capable Gemma model). Since
Tier 2 is a hard blocker, **Tier 3 is also a hard blocker** — no further
investigation needed for this board.

---

## Summary

| Tier | Status | Notes |
|------|--------|-------|
| 0 — Keyword search | 🟢 Confirmed feasible | apt-installed numpy + `--system-site-packages` venv; plain `uvicorn` (no `[standard]`); FastAPI+BM25+lxml server runs at ~150MB RAM |
| 1 — Semantic search | 🟡 Partially solved | `sqlite-vec` self-compiles natively in ~49s, 18ms KNN queries at 500×768-dim — storage is no longer a blocker. **Embedding generation** (replacing Ollama) is the one remaining open question |
| 2 — Conversational AI | 🔴 Hard blocked | No Ollama armv7 build; `gemma4:e2b` (7.2 GB) doesn't fit in 1 GB RAM |
| 3 — Multimodal | 🔴 Hard blocked | Depends on Tier 2 |

**Headline conclusion:** the Pi 2B (32-bit, BCM2836) cannot run **Ollama**
under any configuration (no armv7 builds, and `gemma4:e2b` wouldn't fit in
1GB RAM anyway) — Tiers 2 and 3 are permanently out. However, **`sqlite-vec`'s
lack of an armv7l PyPI wheel turned out to be a non-issue**: the C extension
compiles natively on-device in under a minute with no special tooling. So
Tier 1 (semantic + hybrid search) is now plausible *if* a non-Ollama embedding
path can be found — that's the next investigation. Tier 0 (pure BM25 keyword
search) is fully confirmed working today.

---

## Open items / next steps

- [x] Confirm `lxml` and plain `uvicorn` install and run on the Pi
- [x] Smoke-test a minimal Tier 0 FastAPI app (BM25 + lxml parsing + uvicorn
      server) — works at ~150MB RAM, ~8s cold start
- [x] Resolve `sqlite-vec` for armv7l — **solved**: compiles natively in ~49s,
      drop-in `sqlite_vec`-shim integration needs zero changes to existing code
- [ ] Build out the real Tier 0 app: adapt `src/ingest.py` (drop embedding
      step) and `src/search.py` (BM25-only, drop `sqlite-vec`) for this board,
      and benchmark query latency on real ingested data
- [ ] Investigate `onnxruntime`/`fastembed` armv7l wheel availability — the
      remaining piece for a Tier 1 redesign (embedding generation without
      Ollama)
- [ ] If `onnxruntime` doesn't support armv7l either, investigate other
      lightweight CPU embedding options (e.g. smaller ONNX/ggml-based models,
      or generating embeddings off-device and only serving search on the Pi)
