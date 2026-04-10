# Phase 4: Benchmark — Global vs Glyph-Routed Retrieval

## Status: TODO
## Priority: HIGH

## Overview
Falsification gate. Does glyph-routed retrieval (even with identity transforms) beat global retrieval on recall/precision? If yes, validates the routing thesis and justifies Phase 5+ (real subspace transforms). If no, investigate before proceeding.

## Related Code Files

### Create
- `tests/test_glyph_router.py` — unit tests for GlyphRouter + GlyphOperator
- `tests/benchmark_glyph_routing.py` — comparative benchmark

### Read
- `tests/benchmark.py` — existing benchmark patterns
- `scripts/seed_relativity.py` — KB seeding patterns

## Benchmark Protocol

### Setup
1. Seed KB with 3+ distinct domains (e.g., physics, biology, computing)
2. Assign each domain a dedicated glyph
3. Ingest 20-30 concepts per domain with traces attached to correct glyphs

### Test Queries
- **In-domain**: "time dilation near black holes" → should route to physics glyph
- **Cross-domain**: "how does DNA compute?" → should activate multiple glyphs
- **Ambiguous**: "field theory" → physics vs agriculture

### Control Condition
Benchmark must compare routed vs global with **same downstream behavior** — no probe reranking, no mass-weighted rerank on either path. Raw cosine retrieval only. This isolates the routing signal from post-processing effects.

### Metrics
1. **Recall@5**: fraction of relevant results in top-5 (routed vs global)
2. **Interference rate**: irrelevant-domain results in top-5
3. **Latency**: search time comparison
4. **Fallback rate**: how often global fallback needed

### Pass/Fail Criteria
- Routed recall@5 >= global recall@5 AND interference rate lower → **validated, proceed to Phase 5**
- Routed recall@5 < global → investigate glyph inference quality
- Interference rate unchanged → routing not adding value, reconsider

## Success Criteria
- [ ] Benchmark runs end-to-end
- [ ] Routed retrieval shows measurable interference reduction
- [ ] No recall regression vs global
- [ ] Results documented for Phase 5+ decision
