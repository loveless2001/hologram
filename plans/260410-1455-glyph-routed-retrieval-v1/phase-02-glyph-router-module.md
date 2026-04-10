# Phase 2: GlyphRouter + GlyphShardIndex

## Status: TODO
## Priority: HIGH

## Overview
New module `hologram/glyph_router.py` — centralized glyph-routed retrieval using GlyphOperator transforms. Single place for shard management, glyph inference, and score fusion. Uses operator interface so Phase 2+ R_g/P_k swap is seamless.

## Related Code Files

### Create
- `hologram/glyph_router.py` — GlyphRouter + GlyphShardIndex classes

### Read (dependencies)
- `hologram/glyph_operator.py` — GlyphOperator (Phase 1)
- `hologram/glyphs.py` — `GlyphRegistry.resonance_score()` for p(g|q)
- `hologram/gravity.py` — `GravityField.search()` for global fallback
- `hologram/store.py` — `MemoryStore` for trace access by glyph

## Architecture

```python
class GlyphShardIndex:
    """Per-glyph FAISS index storing operator-transformed trace vectors."""
    
    def __init__(self, glyph_id: str, operator: GlyphOperator):
        self.glyph_id = glyph_id
        self.operator = operator
        self.index: faiss.Index = None  # lazy-built
        self.trace_ids: List[str] = []
    
    def build(self, traces: List[Trace]) -> None:
        """Build FAISS index from traces, applying operator transform."""
        # vecs = [operator.transform_trace(t.vec) for t in traces]
        # self.index = faiss.IndexFlatIP(operator.output_dim)
        # self.index.add(np.stack(vecs))
    
    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Search this shard with an already-transformed query vector."""


class GlyphRouter:
    """Routes queries through glyph-conditioned subspaces."""
    
    def __init__(self, store, glyphs, gravity_field):
        self._store = store
        self._glyphs = glyphs
        self._gravity = gravity_field
        self._operators: Dict[str, GlyphOperator] = {}
        self._shards: Dict[str, GlyphShardIndex] = {}
        self._dirty: bool = True
    
    def infer_glyphs(self, query_vec, top_n=2, min_score=0.1) -> Dict[str, float]:
        """Predict glyph distribution p(g|q) using resonance_score."""
    
    def _ensure_shards(self) -> None:
        """Lazy-build all shard indexes if dirty."""
    
    def search_routed(self, query_vec, top_k=5, top_glyphs=2,
                      fallback_global=True, min_glyph_score=0.1) -> List[Tuple[str, float]]:
        """
        Main entry point — doc-faithful retrieval flow:
        1. infer p(g|q) via resonance_score
        2. for each top glyph: transform query via operator, search shard
        3. score fusion: glyph_weight * cos_sim
        4. deduplicate across shards (keep best score)
        5. global fallback if results < top_k
        """
    
    def invalidate(self) -> None:
        """Mark shards dirty (call after trace add/remove)."""
```

## Implementation Steps

1. Create `hologram/glyph_router.py` with GlyphShardIndex and GlyphRouter
2. `GlyphShardIndex.build()`: collect glyph traces, apply `operator.transform_trace()`, build IndexFlatIP
3. `GlyphShardIndex.search()`: search with pre-transformed query vector
4. `GlyphRouter.infer_glyphs()`: wrap `GlyphRegistry.resonance_score()`, return top-N above min_score
5. `GlyphRouter._ensure_shards()`: for each glyph, create GlyphOperator + GlyphShardIndex, build index
6. `GlyphRouter.search_routed()`:
   a. Ensure shards built
   b. Infer top glyphs + weights
   c. For each top glyph: `transformed_q = operator.transform_query(query_vec)`, search shard
   d. Weight results: `final_score = glyph_weight * shard_score`
   e. Deduplicate (keep best score per trace)
   f. If fallback_global and results < top_k: fill from `gravity_field.search()`
   g. Sort, return top_k
7. `invalidate()`: set `_dirty = True`; called from `GlyphRegistry.attach_trace()`

## Key Design Decisions
- GlyphShardIndex uses GlyphOperator.transform_* — Phase 1 is identity, Phase 2+ swaps in real transforms without touching router logic
- Score fusion is multiplicative (glyph_weight * cos_sim)
- Global fallback ensures no recall regression
- One GlyphOperator per glyph, created in _ensure_shards()
- Module target: under 150 lines

## Success Criteria
- [ ] GlyphRouter.search_routed() returns results
- [ ] Queries route to correct glyph shards based on resonance
- [ ] Global fallback fills gaps when no glyph matches
- [ ] Shard invalidation on trace add works
- [ ] All transforms go through GlyphOperator interface (future-proof)
- [ ] Module under 150 lines
