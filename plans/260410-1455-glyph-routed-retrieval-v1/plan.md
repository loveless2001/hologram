# Glyph-Conditioned Spectral Memory — Implementation Plan

## Status: PHASES 1-4 COMPLETE
## Priority: HIGH
## North-Star Spec: "Glyph-Conditioned Spectral Memory"

## Summary
Redesign Hologram's retrieval architecture so glyphs act as **operators on retrieval geometry** — not labels. Queries route through glyph-conditioned subspaces; mass becomes context-dependent salience. Existing gravity/physics code becomes scaffolding, not the center.

## Target Architecture (Doc-Faithful)

The north-star from the spec doc:

> "Glyphs are not labels on memory; they are operators on mnemonic geometry."

Core formula: `T_g(z) = P_k R_g z` — each glyph g defines a low-rank retrieval operator via orthogonal rotation R_g and dimension projection P_k.

**Target components:**
- **GlyphOperator**: per-glyph transform (R_g rotation + P_k projection + optional D_g scaling / S_g phase mask)
- **GlyphShardIndex**: per-glyph FAISS index storing transformed vectors
- **GlyphRouter**: query → infer p(g|q) → transform → search shards → cross-glyph merge
- **EffectiveSalience**: replaces scalar mass with context-dependent scoring (frequency, centrality, stability, glyph_mass[g])
- **Soft overlap**: related glyphs share partial subspace dimensions; cross-glyph bridges prevent over-discretization

**Target retrieval flow:**
```
query → encode → infer glyph distribution p(g|q)
              → for top glyphs: transform query via T_g
              → search glyph shard indexes
              → score within glyph space: cos(q_g, m_g)
              → cross-glyph merge: Σ p(g|q) * s(m,q,g)
              → optional global residual fallback
              → results
```

## Transitional Implementation Path

| # | Phase | Status | File |
|---|-------|--------|------|
| 1 | Concept glyph_affinity + GlyphOperator interface | DONE | [phase-01](phase-01-concept-glyph-affinity.md) |
| 2 | GlyphRouter + GlyphShardIndex (identity transforms) | DONE | [phase-02](phase-02-glyph-router-module.md) |
| 3 | API integration + /query/routed endpoint | DONE | [phase-03](phase-03-api-integration.md) |
| 4 | Benchmark: global vs glyph-routed retrieval | DONE | [phase-04](phase-04-benchmark.md) |
| 5 | Subspace transforms: R_g + P_k (future) | TODO | — |
| 6 | Decouple mass → effective salience (future) | TODO | — |
| 7 | Geometry-driven physics ablation (future) | TODO | — |

Phases 1–4 = first provable slice. Phases 5–7 = planned in later iterations.

## Design Principle

Even in Phase 1 (identity transforms), interfaces are shaped around **glyph operators**, not "buckets of traces." This means Phase 2 swaps in R_g + P_k without rewriting the retrieval path.

```python
# Phase 1: identity transform (proves routing)
GlyphOperator.transform_query(vec) → vec  # no-op
GlyphOperator.transform_trace(vec) → vec  # no-op

# Phase 2+: real subspace transform
GlyphOperator.transform_query(vec) → P_k @ R_g @ vec
GlyphOperator.transform_trace(vec) → P_k @ R_g @ vec
```

## Key Dependencies
- `GlyphRegistry.resonance_score()` — reuse as p(g|q) proxy
- `GravityField.search()` — global fallback
- `MemoryStore` trace/glyph linking — already tracks trace→glyph membership

## Architecture Context
- Discussion: #hologram channel (2026-04-10)
- Participants: user, claude, codex
- Consensus: doc-faithful target architecture, incremental implementation, testable at each phase
