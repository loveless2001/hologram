# Hologram Architecture

Last updated: 2026-04-10

## Overview

Hologram is currently organized around chunked trace ingestion and glyph-routed retrieval.

The default architecture is:

`document -> chunks -> embeddings -> glyph-local traces -> routed retrieval`

The project still contains gravity, symbolic extraction, and diagnostic layers, but those are now secondary to the routed retrieval path rather than the primary top-level story.

## Design Priorities

### 1. Chunk-first ingestion

Large documents should enter the system through `Hologram.ingest_document(...)`, not as one giant trace.

Current ingestion flow:
1. Split source text with `hologram/chunking.py`
2. Optionally normalize text and resolve coreference
3. Batch-embed chunk texts
4. Store chunk traces under a glyph
5. Invalidate and lazily rebuild glyph shards when needed

Important properties:
- chunk IDs are deterministic from `source_hash + chunk_index`
- re-ingesting the same document is content-idempotent
- changed document text produces a new `source_hash`, so replacement is not automatic

### 2. Glyph-local retrieval

The retrieval system is built around:
- `GlyphOperator` in `hologram/glyph_operator.py`
- `GlyphRouter` in `hologram/glyph_router.py`

Current routed retrieval flow:
1. Embed the query
2. Infer likely glyphs from glyph resonance
3. Transform the query with each selected glyph operator
4. Search that glyph's FAISS shard
5. Merge results across selected glyphs
6. Optionally fall back to global trace search

Current operator modes:
- identity / no projection
- random orthogonal projection
- shared discriminant basis learned from glyph centroids

The discriminant basis result is the first meaningful quality win in the current architecture. It should be described accurately as a shared discriminant projection combined with glyph shard routing, not as full per-glyph learned `R_g`.

### 3. Deterministic identifiers

Persistent auto-generated IDs now use stable digest-based helpers rather than Python `hash()`.

That applies to:
- auto-generated text/image/concept/system IDs
- auto-generated document glyph IDs
- chunk `source_hash`
- stable seeds used by glyph projection logic

### 4. Optional symbolic and physics layers

The repo still includes:
- gravity-field dynamics
- probe-based retrieval
- GLiNER-backed symbolic extraction
- code-map ingestion
- KG/drift experiments
- MG scorer and cost engine diagnostics

These are real modules, but they are not the default ingestion or retrieval story.

In particular:
- GLiNER is optional enrichment, not a required dependency of the main retrieval design
- gravity/probe retrieval remains experimental and parallel to routed retrieval
- code mapping is a specialized ingestion path, not the core document path

## Current Module Roles

### `hologram/api.py`

The top-level facade. It wires encoders, store, glyph registry, gravity field, and glyph router together.

Key current methods:
- `init(...)`
- `load(...)`
- `add_text(...)`
- `ingest_document(...)`
- `search_routed(...)`
- `retrieve(...)`

### `hologram/chunking.py`

Defines sentence chunking and chunk metadata:
- `chunk_index`
- `char_start`
- `char_end`
- `sentence_start`
- `sentence_end`
- `source_hash`

### `hologram/glyph_operator.py`

Defines the retrieval-time transform `T_g(z)` used by routed retrieval. The interface is intentionally doc-aligned even when using identity or simple projection modes.

### `hologram/glyph_router.py`

Owns shard inference, shard build/invalidation, and routed search behavior. It is the center of the current retrieval architecture.

### `hologram/store.py`

Stores traces and glyphs, and provides base vector search. Routed retrieval builds on top of this storage layer.

### `hologram/server.py`

Exposes the main REST surface, including:
- `/ingest`
- `/ingest/document`
- `/query`
- `/query/routed`
- `/ingest/code`
- `/query/code`

### Optional modules

- `hologram/coref.py` – neural coreference
- `hologram/text_utils.py` – extraction and normalization helpers
- `hologram/gravity.py` – gravity field and concept dynamics
- `hologram/code_map/` – source code ingestion and search
- `hologram/kg/` and `hologram/drift/` – batch graph and drift experiments
- `hologram/mg_scorer.py` and `hologram/cost_engine.py` – diagnostics

## Recommended Mental Model

Read the system in this order:

1. `api.py` for the top-level lifecycle
2. `chunking.py` for the default ingestion unit
3. `glyph_operator.py` and `glyph_router.py` for the main retrieval path
4. `store.py` for persistence/search substrate
5. optional modules only if you need those experiments

## Legacy Material

Older gravity-first framing, historical reports, and superseded planning notes were moved under `docs/legacy/`.

Those files may still be useful as research context, but they should not be treated as the current design description.
