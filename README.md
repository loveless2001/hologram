# Hologram Memory

Hologram is a memory sandbox for chunked document ingestion, glyph-routed retrieval, and optional gravity-field experiments.

The current design is centered on:
- chunking documents into deterministic trace units
- embedding those chunks with MiniLM or other encoders
- storing traces under glyphs
- routing queries through glyph-local shard indexes with `GlyphRouter`
- treating symbolic extraction and gravity-heavy behavior as optional layers, not the default ingestion path

## Current Design

### Default ingestion path

The main document path is:

`text -> chunk_text() -> optional normalization/coref -> batch embedding -> trace storage -> glyph shard routing`

Use:
- `Hologram.ingest_document(...)` for bulk document ingestion
- `Hologram.add_text(...)` for low-level single-trace ingestion

### Default retrieval path

The main retrieval path is:

`query -> infer glyph distribution -> transform with GlyphOperator -> search shard indexes -> merge results`

Use:
- `Hologram.search_routed(...)` for glyph-routed retrieval
- `POST /query/routed` for the REST equivalent

Global retrieval and probe/gravity retrieval still exist, but they are no longer the primary design story.

### Optional layers

These modules remain part of the repo, but they are optional or secondary to the routed retrieval path:
- gravity-field simulation in `hologram/gravity.py`
- symbolic extraction in `hologram/text_utils.py`
- GLiNER-backed symbolic enrichment when explicitly enabled
- code ingestion in `hologram/code_map/`
- MG scorer and cost engine diagnostics
- KG/drift experiments in `hologram/kg/` and `hologram/drift/`

## Core Modules

- `hologram/api.py` – main `Hologram` facade
- `hologram/chunking.py` – sentence chunking with deterministic `source_hash`
- `hologram/glyph_operator.py` – glyph-conditioned transform `T_g(z)`
- `hologram/glyph_router.py` – shard routing and cross-glyph merge
- `hologram/glyphs.py` – glyph registry and trace membership
- `hologram/store.py` – in-memory trace/glyph storage and FAISS-backed search
- `hologram/coref.py` – optional FastCoref integration
- `hologram/normalization.py` / `hologram/text_utils.py` – text cleanup and optional symbolic extraction
- `hologram/server.py` – FastAPI server

## Quick Start

```python
from hologram.api import Hologram

holo = Hologram.init(encoder_mode="minilm", use_gravity=True)
holo.project = "demo"
holo.glyphs.create("doc:guide", title="Guide")

chunks = holo.ingest_document(
    glyph_id="doc:guide",
    text="First sentence. Second sentence. Third sentence. Fourth sentence.",
    sentences_per_chunk=2,
    overlap=1,
)

results = holo.search_routed("What does the guide say?", top_k=3)
for trace, score in results:
    print(trace.trace_id, round(score, 4), trace.content)
```

### Server

```bash
python -m hologram.server
```

Current API endpoints include:
- `POST /ingest`
- `POST /ingest/document`
- `POST /query`
- `POST /query/routed`
- `POST /ingest/code`
- `POST /query/code`
- `POST /kg/build_batch`
- `POST /drift/compare`
- `POST /save/{project}`
- `POST /load/{project}`

## Benchmarks

Current benchmark entry points:
- `tests/benchmark-glyph-routing-vs-global.py` – synthetic routing control
- `tests/benchmark-glyph-routing-minilm-real-text.py` – real-text MiniLM routing benchmark
- `scripts/benchmark_beir.py` – external retrieval benchmark
- `scripts/benchmark_ragbench_e2e.py` – end-to-end QA benchmark
- `scripts/benchmark_timeqa_e2e.py` and `scripts/benchmark_timeqa_stream.py` – temporal retrieval/drift experiments

Read [docs/benchmarking_guide.md](/home/lenovo/projects/hologram/docs/benchmarking_guide.md) before interpreting latency numbers. Routed benchmarks have a meaningful cold-vs-warm distinction.

## Documentation

Current docs are indexed in [docs/README.md](/home/lenovo/projects/hologram/docs/README.md).

The main current-facing design doc is [docs/architecture_design.md](/home/lenovo/projects/hologram/docs/architecture_design.md). Older gravity-first and historical planning docs have been moved under `docs/legacy/`.
