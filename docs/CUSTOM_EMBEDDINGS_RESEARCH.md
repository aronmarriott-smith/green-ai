# Custom Embeddings & Inference Models — Research & Plan

**Status: research and planning only — no implementation in this doc.**

This is a tangential research track to [PI2B_FEASIBILITY.md](PI2B_FEASIBILITY.md).
That document identified an open problem for Tier 1 on the Raspberry Pi 2B:
`sqlite-vec` (vector storage) works great, but **embedding generation** has no
good answer because Ollama can't run on this board. The default plan there is
to look for an existing small model (`fastembed`/`onnxruntime`).

This document explores an alternative/complementary path: **build our own
embedding model from scratch**, trained off-device, small enough to run on the
Pi 2B via a tiny custom inference engine. It is written for a developer (or an
AI coding agent) picking this up later with no other context.

## Goals and non-goals

**Goals**
- A from-scratch embedding model, trained on the [Princeton WordNet](https://wordnet.princeton.edu/)
  database (a "Level 0" MVP), using **our own embedding space** — see
  [Part 2](#part-2--output-dimensionality) for the chosen dimensionality.
- An inference path that runs on the Pi 2B (1 GB RAM, armv7l, no GPU),
  following the precedent established in PI2B_FEASIBILITY.md that **native
  on-device C compilation works well** on this board (sqlite-vec compiled in
  ~49s with no issues).
- A lightweight roadmap for a future "Level 1" generative/inference model,
  in keeping with the project's tiered progressive-enhancement philosophy.

**Non-goals**
- **Not** attempting to match `embeddinggemma:300m`'s embedding space,
  dimensionality, or interface in any way. This was considered (an earlier
  draft of this doc targeted 768-dim "shape compatibility") and has been
  **explicitly dropped** — this model defines its own vector space and its
  own `EMBED_DIM`, independent of the main project's Gemma-based config. The
  Pi-tier deployment that uses this model creates its own `vec0` table at
  whatever dimension we choose.
- **No implementation work** — this doc is the research/plan; building it is
  a separate future task.
- Mojo, if used at all, is an experiment for training-time performance, not
  a requirement (see [Part 3](#part-3--training-infrastructure)).

### Scope: English only

Inherited from the training data — **Princeton WordNet is an
English-language resource**, so this model's vocabulary, tokeniser, and
lemmatiser are **English (plus standard symbols/punctuation)** only. This
isn't a deliberate design constraint so much as a direct consequence of
Level 0's data source; multilingual support is out of scope unless a future
level deliberately adds multilingual training data (e.g. multilingual
wordnets, Wiktionary — see [Part 6](#part-6--roadmap-beyond-level-0)). It
also means the open-source evaluation story (Part 5) targets MTEB's English
tasks, not MMTEB's multilingual expansion.

---

## Background: WordNet

[WordNet](https://wordnet.princeton.edu/) is a lexical database of English
maintained by Princeton University. The pieces that matter for this project:

- **Synsets** — a synset is a set of synonyms representing one sense of a
  concept, e.g. `{car, automobile, motorcar}`. WordNet 3.0 has roughly
  **117,000 synsets** covering **~155,000 unique words/phrases** (mostly
  single words, some multi-word phrases like `ice_cream`, joined with
  underscores).
- **Glosses** — each synset has a short human-readable definition and often
  example sentences, e.g. *"car: a motor vehicle with four wheels; usually
  propelled by an internal combustion engine"*. All glosses combined are
  roughly **2-3 million words** of text — small by NLP corpus standards.
- **Relations** — synsets are linked by typed, directed edges:
  - `hypernym` / `hyponym` — "is-a" hierarchy (car → vehicle)
  - `meronym` / `holonym` — "part-of" (wheel → car)
  - `antonym`, `similar_to`, `also_see`, `derivationally_related_form`, etc.
  - There are roughly **~15-20 distinct relation types** and on the order of
    **1 million typed edges (triples)** total across the database.
- **Access** — the easiest path is Python's `nltk.corpus.wordnet`
  (`nltk.download('wordnet')`), which gives programmatic access to synsets,
  lemmas, glosses, and all relations without needing to parse the raw DB
  files directly.
- **License** — WordNet 3.0 ships under a permissive BSD-style license
  (free for commercial and non-commercial use with attribution). A model
  trained purely on WordNet data carries no licensing encumbrance, which
  matters for the open-sourcing notes in [Part 5](#part-5--open-sourcing--evaluation-compatibility).

---

## Part 1 — Choosing a training approach

WordNet can be used to train embeddings in three broadly different ways.
Since this is new territory, here's each option explained from first
principles with pros and cons, followed by a recommendation.

### Option A — Graph-based ("knowledge graph embeddings")

**Idea:** WordNet's relations form a graph — synsets are nodes, relations are
typed edges. Train embeddings so that connected/related nodes end up close
together (or related by a consistent transformation) in vector space.

Two families of techniques apply:
- **node2vec / DeepWalk** — take many random walks across the graph, treat
  each walk as a "sentence" of node IDs, and run word2vec's skip-gram
  algorithm on those sentences. Simple, untyped (doesn't distinguish *why*
  two nodes are connected).
- **Knowledge-graph embedding models** (TransE, DistMult, ComplEx, RotatE) —
  designed for *typed* relations. TransE, for example, learns vectors such
  that `head_vector + relation_vector ≈ tail_vector` (e.g.
  `vec(car) + vec(hypernym) ≈ vec(vehicle)`). These explicitly model what
  *kind* of relationship connects two concepts, which is exactly the
  information WordNet's curators hand-annotated.

| | |
|---|---|
| ✅ Pros | Captures WordNet's best asset — the *hand-curated* semantic structure — for free. Bounded, manageable vocabulary (~120k synsets). Training is fast and well-understood (a graph this size trains in minutes-to-hours even on CPU, trivial on a GTX 1060). Typed models (TransE etc.) capture *relation semantics*, not just "these are similar". |
| ❌ Cons | Produces vectors for *concepts* (synsets), not arbitrary text. A real RAG chunk is full sentences with named entities, numbers, jargon, etc. that aren't WordNet concepts at all — need a strategy to go from "synset vectors" to "text chunk vector" (see below). Word-sense ambiguity: a word like "bank" belongs to multiple synsets. |

### Option B — Distributional, from gloss text only

**Idea:** Treat the ~150k gloss definitions as a small text corpus and train
classic **word2vec/GloVe**-style embeddings on it — the same kind of
algorithm used to train embeddings on Wikipedia, just on a tiny corpus.

| | |
|---|---|
| ✅ Pros | Familiar, simple, well-documented pipeline (e.g. `gensim.models.Word2Vec`). Produces vectors for *every* word that appears in a gloss, not just headwords — broader surface coverage than Option A. |
| ❌ Cons | 2-3 million words is **tiny** for word2vec — production word2vec/GloVe models train on billions of words. Quality would likely be noticeably weak, especially for less common words. Throws away WordNet's curated relational structure entirely — the part that makes WordNet *WordNet*. |

### Option C — Hybrid (graph + gloss text, "retrofitting")

**Idea:** Combine A and B. Train distributional embeddings from gloss text
(Option B), then adjust ("retrofit") them using the WordNet graph (Option A)
so that synonyms/hypernyms/etc. end up closer together — a documented
technique (Faruqui et al., 2015, *"Retrofitting Word Vectors to Semantic
Lexicons"*). Alternatively, train a single model with two loss terms (one
for gloss co-occurrence, one for graph relations) jointly.

| | |
|---|---|
| ✅ Pros | Best of both signals — relational precision from the graph plus broader lexical coverage from gloss text. Most "semantically grounded" of the three options. |
| ❌ Cons | Two training stages (or a more complex joint loss) — more moving parts, more to debug for a first attempt. Still bounded by the small gloss corpus for the distributional half. |

### Decision: Option A for Level 0

**Decided** — Level 0 uses **Option A**: a typed knowledge-graph embedding
(TransE or a similar PyKEEN model — see
[Part 3](#part-3--training-infrastructure)) trained directly on WordNet's
synset relation graph. Rationale:

1. WordNet's actual value-add over a generic dictionary is the curated,
   *typed* relational graph — Option A uses that directly.
2. It's the cheapest to train and validate (small graph, fast iteration).
3. Options B/C (gloss-text distributional, hybrid retrofitting) remain
   documented above for context and are revisited as **"Level 0.5"** once
   the Level 0 pipeline (training → export → Pi inference) is proven
   end-to-end — see [Part 6](#part-6--roadmap-beyond-level-0).

### From synset vectors to chunk vectors

Whichever option is chosen, there's a gap between "vectors for WordNet
synsets/words" and "a single 128-dim vector for an arbitrary RAG text chunk".
Proposed pipeline for Level 0:

1. **Train synset-level embeddings** via the chosen graph method (one vector
   per synset, e.g. ~117k vectors).
2. **Derive lemma-level embeddings** — for each word/phrase, average the
   embeddings of all synsets it belongs to (a word with one dominant sense
   gets a vector close to that sense; a genuinely ambiguous word gets a
   "blended" vector — a known simplification, revisited in Level 0.5+ via
   simple word-sense disambiguation).
3. **Tokenise and lemmatise the input chunk** — lowercase, strip punctuation,
   and run WordNet's morphological lemmatizer (`nltk`'s `WordNetLemmatizer`/
   `morphy`, e.g. "running" → "run") so surface forms map onto the lemma
   vocabulary. Also check for multi-word lemmas (`ice_cream`) before
   falling back to single tokens.
4. **Mean-pool** the lemma vectors present in the chunk into one vector.
   Optionally weight by inverse document frequency (the project already
   computes BM25 statistics — IDF weighting could reuse that).
5. **L2-normalise** the result to match the cosine-distance convention used
   by the existing `vec0` schema.

**Out-of-vocabulary (OOV) handling** — real documents contain proper nouns,
numbers, technical jargon, and words outside WordNet's ~150k vocabulary.

**Decision for Level 0:** add one extra row to the embedding table — a
dedicated `undefined` entry (initialised to, e.g., the corpus-mean vector,
or trained as its own entity — to be determined during training). Every OOV
token contributes this `undefined` vector to the mean-pool, exactly like any
other token. This keeps the lookup/pool/normalise logic in the inference
engine (Part 4) completely uniform — there's no special-case branch for "no
in-vocabulary words found"; every chunk pools at least one vector (its own
tokens, or `undefined` for any/all of them).
- **Future improvement (Level 0.5+):** add fastText-style character n-gram
  hashed embeddings as a fallback/blend for OOV words — more robust but
  adds training and inference complexity, not needed to validate the
  pipeline.

---

## Part 2 — Output dimensionality

Gemma compatibility has been dropped entirely — this model defines **its own
embedding space and its own dimensionality**, decided independently of the
main project's `EMBED_DIM = 768` (which stays tied to `embeddinggemma:300m`
for the Ollama/Gemma stack). The Pi-tier deployment that uses this custom
model creates its own `vec0` table at whatever dimension we choose here —
there's no existing schema to stay compatible with.

**Chosen dimension: 128.** Knowledge graph embeddings for a graph this size
(~117k entities, ~1M triples) are typically trained at 100-300 dimensions —
128 is a common, well-supported choice for PyKEEN models (TransE etc.) and
sits comfortably in that range. If evaluation later shows quality is
capacity-limited, bumping to 256 is a straightforward retrain — still just a
config change (`EMBED_DIM`), not a compatibility concern, since nothing
external depends on this model's dimensionality.

### Memory budget on the Pi

At 128 dimensions, the memory picture is comfortable even without the
mitigations that would have been required at 768:

| Format | Size of full table (~150k entries) |
|---|---|
| float32, 768-dim (original Gemma-shaped plan, since dropped) | ~460 MB |
| float32, 128-dim | ~77 MB |
| **int8, 128-dim** | **~19 MB** |

Even the float32/128-dim table (~77 MB) fits comfortably resident in RAM on
a 1 GB board alongside the rest of the stack — **int8 quantisation and
`mmap()` become optional optimisations rather than requirements** at this
size, though both remain cheap and worth doing anyway:

- **int8 quantisation** — store each dimension as a signed byte plus a
  scale factor; ~19 MB total. `sqlite-vec`'s `vec0` also supports `int8`
  columns directly.
- **Memory-mapped table** — even if not strictly necessary at ~19-77 MB,
  `mmap()`ing the flat binary file (word2vec `.bin`-style) is simple, free,
  and avoids a slow startup parse step.

---

## Part 3 — Training infrastructure

**Good news: Level 0 training is cheap.** A typed knowledge-graph embedding
over ~117k entities / ~1M triples is a small-data problem by modern ML
standards — **no GPU needed**.

- **PyKEEN** (`pip install pykeen`) is a mature Python library purpose-built
  for knowledge graph embedding models (TransE, DistMult, ComplEx, RotatE,
  etc.), built on PyTorch. A graph this size trains in **minutes on CPU**.
- **Recommended Level 0 training machine: the Intel MBP (i7, Iris 655)**
  from the project's hardware table — CPU-only is sufficient; no need to
  involve the GTX 1060 or a cloud instance at this stage.
- **gensim** (`Word2Vec`) is the standard tool if the node2vec or gloss-text
  routes (Options B/C) come into play at Level 0.5 — also CPU-friendly and
  fast at this scale.
- For **Level 0.5+** (joint graph+text training, small neural retrofitting
  models), reach for the **GTX 1060** (6 GB VRAM — more than sufficient for
  models of this size, low tens of millions of parameters at most) or a
  cloud GPU instance, if CPU iteration on the i7 becomes a bottleneck.

> **Mojo footnote:** [Mojo](https://www.modular.com/mojo) was considered as
> an optional training-performance experiment but isn't relevant for
> Level 0 — PyKEEN's CPU training loop is already fast enough at this scale,
> and Mojo's ecosystem (Linux/macOS/WSL only) is far less mature than
> PyTorch's for this kind of work. Worth revisiting only for Level 0.5+/
> Level 1 custom training loops, if ever. It's also irrelevant to the Pi
> inference side regardless — Mojo doesn't target 32-bit ARM, and Part 4 is
> plain C/C++.

---

## Part 4 — Pi 2B inference engine (custom C/C++)

Following the precedent from PI2B_FEASIBILITY.md — **native on-device
compilation works well on this board** (sqlite-vec: ~49s build, ~170 MB peak
RAM, no cross-compilation needed) — the plan is a small, purpose-built C
program/library, compiled directly on the Pi with the system `gcc`.

### Components

1. **Embedding table file** — flat binary file: a short header (vocab size,
   dimension, quantisation scale/zero-point) followed by one int8 record per
   vocabulary entry. Conceptually identical to the classic word2vec `.bin`
   format. Generated once, off-device, after training.
2. **Vocabulary index** — a hash map from lemma string → row index in the
   embedding table. For ~150k entries this is a small, simple structure
   (e.g. a sorted array + binary search, or a basic open-addressing hash
   table) — no external dependency needed.
3. **Tokeniser/lemmatiser** — a minimal C port of the lemmatisation rules
   needed to match WordNet's morphology (suffix-stripping rules, e.g.
   "-ing"/"-ed"/"-s"), plus a check for known multi-word lemmas. This is the
   one piece that benefits from being *generated* from the Python/NLTK
   training pipeline (e.g. export WordNet's exception lists and morphology
   rules as a small data table the C code reads), rather than reimplementing
   NLTK's logic by hand.
4. **Lookup + pool + normalise** — for each token: hash-lookup → int8 row →
   dequantise to float32 (using the header's scale) → accumulate into a
   running mean → after all tokens, L2-normalise the 128-dim result.

### Interface to Python

**Decision: shared library + `ctypes`**, mirroring the integration pattern
already validated for sqlite-vec on this Pi — compile to `libembed.so`, load
via `ctypes.CDLL` from `ingest.py`/`rag.py`/`search.py`, call a single
`embed_text(const char* text, float* out_vec)` function. Lowest overhead, no
subprocess spawning per chunk/query — and a small, clean C ABI is also the
easiest thing for **other people to bind from their own language** if the
model is open-sourced (Part 5): Python via `ctypes` today, equally callable
from Rust/Go/Node/etc. later.

A thin CLI wrapper (`embed-cli "some text"` → prints N floats) is still
worth building over the same library — useful for the tests below and for
manual debugging — but the library/`ctypes` interface is the primary one.

### Memory & latency budget

- Embedding table on disk (int8, ~150k × 128 bytes): **~19 MB**, small
  enough to be fully resident, though still `mmap()`'d for a fast startup
  (see Part 2).
- Vocabulary index: tens of MB at most for ~150k string keys.
- Per-query cost: tokenising a short chunk (tens of words) and looking up
  each in a hash map + mmap'd table is on the order of microseconds-to-low-
  milliseconds — for context, the existing sqlite-vec KNN benchmark on this
  Pi was **18ms for k=3 over 500 vectors**; embedding generation should not
  be the bottleneck.
- **NEON SIMD** — the Cortex-A7 supports NEON; the dequantise+mean-pool step
  could use NEON intrinsics for a modest speedup. Given the tiny per-chunk
  workload, this is a "nice to have" micro-optimisation, not required for an
  MVP — flag for later if profiling shows it matters.

### Testing the inference engine

These tests verify the **engine's mechanics** — tokenisation, lookup,
pooling, normalisation, the C ABI — not the trained model's *quality* (that
is a separate evaluation/QA process, e.g. held-out WordNet relations or
MTEB-style tasks — see Parts 5/6). Keeping these concerns separate means
engine tests can run against a tiny synthetic embedding table (a handful of
known words) and don't depend on the real ~150k-entry trained table.

- **C-level unit tests** for the core primitives: hash map lookup
  (hit/miss), int8 dequantisation against known scale values, mean-pooling
  over a known set of vectors, L2 normalisation, and the `undefined`-vector
  fallback for OOV tokens. A lightweight approach (a handful of
  `assert`-based test functions run from a `main()`, or a minimal framework
  like [Unity](https://github.com/ThrowTheSwitch/Unity)) is sufficient — no
  need for a heavy framework.
- **Python-level integration tests via `ctypes`**, following the existing
  project convention (`tests/`, pytest — see `requirements-dev.txt`): build
  a tiny fixture embedding table + vocab (5-10 words plus the `undefined`
  entry), load `libembed.so` via `ctypes`, and assert on `embed_text(...)`
  for known inputs:
  - known single-word input → expected vector (within floating-point
    tolerance)
  - multi-word chunk → correct mean-pooling
  - OOV word → `undefined` vector contribution
  - empty string / punctuation-only input → doesn't crash, returns the
    `undefined` vector
  - output vector has length 128 and is L2-normalised (norm ≈ 1.0)
- Both layers run on a normal dev machine (no Pi required), since the engine
  is plain portable C — the Pi is only needed to confirm it **compiles and
  runs there** (a smoke test, as already practiced for sqlite-vec), not for
  correctness testing.

---

## Part 5 — Open-sourcing & evaluation compatibility

The Pi's runtime (Part 4) is intentionally minimal and custom — that's
*good* for the Pi, but it means nobody else can easily load or evaluate the
model without our C code. The plan: **export the trained model into one or
more standard formats alongside the custom binary table**, so it's usable
and evaluable without any custom tooling.

- **gensim / word2vec format** — trivial export: dump the lemma→vector table
  as a standard word2vec `.txt`/`.bin` file. Anyone can then
  `KeyedVectors.load_word2vec_format(...)` and use it with any tool that
  speaks that (extremely common) format.
- **`sentence-transformers` / MTEB compatibility** — the
  [Massive Text Embedding Benchmark (MTEB)](https://github.com/embeddings-benchmark/mteb)
  is the standard way embedding models get evaluated and leaderboarded.
  MTEB loads unrecognised models via
  `SentenceTransformer(model_name)`. Since Level 0 is "embedding lookup +
  mean pooling" (no transformer layers), this maps cleanly onto a
  `sentence-transformers` model built from just an embedding/`Dense` layer
  and a `Pooling` module — `model.save_pretrained(...)` produces a directory
  loadable as `SentenceTransformer("our-model")`, which MTEB (and the public
  leaderboard) can then run directly.
- **GGUF / llama.cpp ecosystem** — for visibility in tools like Ollama/LM
  Studio (which our project already uses), a HF-format export can be
  converted with llama.cpp's `convert_hf_to_gguf.py` and served via
  `llama-embedding`. This is a "nice to have" for discoverability — **not**
  how the Pi itself runs the model (our custom C engine is far smaller and
  simpler than embedding a full ggml runtime on a 1 GB board), but it lets
  others run/evaluate the model with familiar tools.
- **Licensing** — a model trained solely on WordNet (permissive BSD-style
  license) carries no licensing restrictions inherited from training data.
  Pick a permissive license for the model weights/code (e.g. Apache 2.0,
  matching the rest of this project's generation model) when/if this is
  published.

---

## Part 6 — Roadmap beyond Level 0

Lighter detail — for future investigation once Level 0 is validated.

- **Level 0** (this doc's focus) — typed knowledge-graph embeddings
  (TransE/PyKEEN) over WordNet's synset graph, 128-dim, exported to a
  quantised table, served by a custom C lookup/pooling engine on the Pi.
- **Level 0.5** — add gloss-text signal (Option B/C from Part 1) via
  retrofitting or joint training, improving coverage of words/senses that
  the pure graph approach underrepresents. Also revisit OOV handling
  (fastText-style character n-gram fallback) and simple word-sense
  disambiguation (currently: lemma vectors are an average over all senses).
- **Level 1 — tiny generative/inference model (sketch only)** — a small
  (low single-digit millions of parameters) char- or subword-level
  RNN/transformer. Positioned as a *very* limited "fallback generation" for
  constrained hardware — nowhere near `gemma4:e2b` quality, but potentially
  enough for short extractive-style answers. Would reuse the same training
  infra (Part 3) and the same custom-C-engine philosophy (Part 4), likely
  with int4/int8 weight quantisation. Explicitly **not** investigated
  further here beyond the dataset/pre-tuning notes below — flagged as the
  next research topic once Level 0/0.5 embeddings are working end-to-end.

### Level 0.5 — additional datasets and pre-tuning notes

**Additional open datasets to consider** (all permissively licensed, English):
- **[Open English WordNet (OEWN)](https://github.com/globalwordnet/english-wordnet)**
  — an actively maintained, CC BY 4.0 successor/fork of Princeton WordNet
  with more contemporary vocabulary. Could supplement, or eventually
  replace, WordNet 3.0 as the base graph — worth evaluating as a drop-in
  upgrade to the Level 0 graph itself, not just a Level 0.5 addition.
- **[ConceptNet](https://conceptnet.io/)** — a common-sense knowledge graph
  (CC-licensed) with relation types that complement WordNet's (e.g.
  `UsedFor`, `CapableOf`, `AtLocation`). Could be combined with WordNet in a
  joint KG embedding for richer relational coverage — requires reconciling
  the two relation-type vocabularies (see pre-tuning notes below).
- **English Wiktionary** — definitions for far more contemporary, technical,
  and informal vocabulary than WordNet covers; CC BY-SA/GFDL. Good source of
  additional gloss-style text for the distributional half of Level 0.5.
- **Simple English Wikipedia** — a small, clean, CC BY-SA general-English
  corpus; useful if the gloss-text corpus (~2-3M words) proves too small for
  meaningful distributional signal.

**Pre-tuning considerations:**
- **Vocabulary alignment** — WordNet lemmas use underscores for multi-word
  expressions (`ice_cream`) while Wiktionary/Wikipedia text doesn't; need a
  consistent tokenisation/lemmatisation scheme shared across all data
  sources before joint training.
- **Retrofit-strength tuning** — the strength of the graph-based "pull"
  during retrofitting is a hyperparameter; too weak and Level 0.5 barely
  differs from Level 0.5's distributional component alone, too strong and
  it collapses back toward Level 0. Validate on a held-out
  similarity/relatedness task (e.g. a subset of WordNet relations held out
  from training).
- **Relation-schema reconciliation** — if combining WordNet + ConceptNet in
  one KG embedding model, define a unified relation-type set (a mapping or
  union of both schemas) before training; PyKEEN expects a single
  relation vocabulary.

### Level 1 — additional datasets and pre-tuning notes

**Additional open datasets to consider:**
- **[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)**
  (Eldan & Li, Microsoft Research) — a dataset *and methodology* specifically
  designed to train small (~1-10M parameter) language models that produce
  coherent text. This is the closest existing precedent to "Level 1" as
  sketched here and is the natural starting point for both data and
  training recipe.
- **Simple English Wikipedia** / **Project Gutenberg** (public-domain books)
  — general-English text if more volume or topic diversity is needed beyond
  TinyStories' children's-story domain.
- **The project's own `data/` corpus** — for a final domain-adaptation
  stage, since Level 1's actual job is answering questions about *this*
  project's ingested documents, not general text.

**Pre-tuning considerations:**
- **Tokeniser choice** — a small BPE vocabulary (e.g. 4-8k tokens) vs.
  character-level tokenisation is a real tradeoff for the Pi: smaller vocab
  → smaller embedding table → less RAM, but coarser character-level
  tokenisation may need a longer context window to express the same text.
- **Staged training** — (1) general pretraining on TinyStories-style data to
  learn basic English structure cheaply, (2) continued
  pretraining/fine-tuning on the project's own `data/` corpus for domain
  adaptation, and optionally (3) **distillation from `gemma3:270m`** (already
  planned in the README as a content-safety guard model) — running it over
  the project's domain corpus to generate higher-quality training targets
  for the tiny model, without needing large amounts of human-written domain
  text.
- **Quantisation** — plan for int4/int8 post-training quantisation (or
  quantisation-aware training if quality drops too much) to fit weights in
  the Pi's memory budget, consistent with the embedding table's approach in
  Part 2.
- **Evaluation** — generative quality is a different axis from embedding
  quality (Part 5/MTEB); a held-out perplexity metric plus qualitative spot
  checks on the project's own domain questions is the right starting point,
  not a full benchmark suite.

---

## Decisions made (this revision)

- [x] Training approach: **Option A** — typed knowledge-graph embedding
      (TransE/PyKEEN) over WordNet's synset relation graph
- [x] Output dimensionality: **128**, our own vector space — no Gemma
      shape/interface compatibility
- [x] Scope: **English only** (+ symbols/punctuation), inherited from
      WordNet
- [x] OOV strategy: dedicated `undefined` embedding-table entry, used
      uniformly for every OOV token
- [x] Level 0 training machine: **Intel MBP (i7, Iris 655)**, CPU-only —
      GTX 1060/cloud reserved for Level 0.5+
- [x] Pi inference interface: **shared library + `ctypes`** (with a thin
      CLI wrapper for testing/debugging)

## Open questions / next steps

- [ ] Decide synset-level graph construction details — which WordNet
      relation types to include as edges for the TransE/PyKEEN model
- [ ] Prototype training on the Intel i7 MBP (PyKEEN + WordNet via `nltk`) —
      validate training time and embedding quality at 128 dimensions
- [ ] Design the binary embedding-table format (int8 quantisation, header
      layout, the `undefined` entry, mmap-friendly record size)
- [ ] Export WordNet's lemmatisation rules into a small data table the C
      engine can consume (avoid reimplementing NLTK's morphology logic)
- [ ] Prototype the minimal C lookup/pool/normalise engine plus its test
      suite (C-level unit tests + `ctypes`/pytest integration tests); compile
      and smoke-test natively on the Pi 2B (following the sqlite-vec
      precedent)
- [ ] Prototype reference exports (word2vec format, `sentence-transformers`
      directory for MTEB) once Level 0 embeddings exist
- [ ] Revisit Part 6 (Level 0.5 / Level 1) once Level 0 is validated
      end-to-end — including evaluating Open English WordNet (OEWN) as a
      possible upgrade to the base graph
