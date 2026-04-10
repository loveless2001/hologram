# Hologram Memory

Hologram is a memory sandbox for chunked document ingestion, glyph-routed retrieval, and optional gravity-field experiments.

The current design is centered on:
- chunking documents into deterministic trace units
- embedding those chunks with MiniLM or other encoders
- storing traces under glyphs
- supporting two benchmark-backed fast retrieval paths:
  - same-space glyph routing with merge refinements
  - global PCA-64 compression
- treating symbolic extraction and gravity-heavy behavior as optional layers, not the default ingestion path

## Current Design

### Default ingestion path

The main document path is:

`text -> chunk_text() -> optional normalization/coref -> batch embedding -> trace storage -> glyph shard routing`

Use:
- `Hologram.ingest_document(...)` for bulk document ingestion
- `Hologram.add_text(...)` for low-level single-trace ingestion

### Default retrieval path

The main routed retrieval path is:

`query -> infer glyph distribution -> same-space shard search -> weighted/filter merge -> results`

Use:
- `Hologram.search_routed(...)` for glyph-routed retrieval
- `POST /query/routed` for the REST equivalent

The main compression-first fast path is:

`query -> global PCA-64 projection -> compressed global FAISS search`

Use:
- `Hologram.search_global_pca(...)` for benchmark-backed compressed global retrieval
- `Hologram.search_dynamic(...)` or `POST /query/dynamic` to allocate automatically between:
  - full global search for quality
  - global PCA for smaller corpora / balanced speed
  - refined routed search once scale and shard structure justify it

Current benchmark-backed retrieval defaults:
- same-space routing is the safe routed default; operator projection is now opt-in
- routed merge quality improves with:
  - `secondary_shard_weight`
  - `shard2_cutoff_rank`
- at the current LoTTE benchmark scale, `global PCA-64` is the strongest fast path overall

### Optional layers

These modules remain part of the repo, but they are optional or secondary to the current retrieval stack:
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
- `POST /query/adaptive`
- `POST /query/dynamic`
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
- `scripts/benchmark_lotte_multidomain.py` – mixed-domain LoTTE routing/compression benchmark
- `scripts/benchmark_beir.py` – external retrieval benchmark
- `scripts/benchmark_ragbench_e2e.py` – end-to-end QA benchmark
- `scripts/benchmark_timeqa_e2e.py` and `scripts/benchmark_timeqa_stream.py` – temporal retrieval/drift experiments

Read [docs/benchmarking_guide.md](/home/lenovo/projects/hologram/docs/benchmarking_guide.md) before interpreting latency numbers. Routed benchmarks have a meaningful cold-vs-warm distinction.

Current LoTTE snapshot:
- global 384D: `0.926` `NDCG@10`, `98 ms/query`
- global PCA-64: `0.907` `NDCG@10`, `17 ms/query`
- routed 384D with merge refinements: about `0.894` `NDCG@10`, `17-18 ms/query`

## Documentation

Current docs are indexed in [docs/README.md](/home/lenovo/projects/hologram/docs/README.md).

The main current-facing design doc is [docs/architecture_design.md](/home/lenovo/projects/hologram/docs/architecture_design.md). Older gravity-first and historical planning docs have been moved under `docs/legacy/`.
