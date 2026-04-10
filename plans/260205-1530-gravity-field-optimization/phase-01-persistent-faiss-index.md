# Phase 1: Persistent FAISS Index for nearest_concepts()

## Context Links
- [Parent Plan](plan.md)
- [Brainstorm Report](../reports/brainstorm-260205-1530-gravity-field-optimization.md)
- Target: `hologram/gravity.py:1061-1099`

## Overview

| Field | Value |
|-------|-------|
| Priority | P1 - Critical path for probe performance |
| Status | pending |
| Effort | 1.5h |

Replace brute-force O(N) loop in `nearest_concepts()` with FAISS IndexFlatIP search + mass reranking.

## Key Insights

- Current implementation iterates all candidates (O(N)) per probe step
- Up to 8 probe steps × 10 neighbors = 80 concept lookups per query
- FAISS IndexFlatIP provides SIMD-optimized cosine search
- Mass reranking on top-k results is O(k) where k << N

## Requirements

### Functional
- [ ] Add persistent FAISS index fields to Gravity class
- [ ] Implement lazy `_rebuild_index()` method
- [ ] Modify `nearest_concepts()` to use FAISS + rerank
- [ ] Mark index dirty on mutations (add/fuse/mitosis)

### Non-Functional
- [ ] Thread-safe: rebuild under `_lock`, search lock-free
- [ ] Memory: O(N × D × 4 bytes) for IndexFlatIP
- [ ] No API changes to callers

## Architecture

```python
@dataclass
class Gravity:
    # Existing fields...

    # NEW: FAISS Index Cache
    _faiss_index: Optional[faiss.IndexFlatIP] = field(default=None, repr=False)
    _index_names: List[str] = field(default_factory=list, repr=False)
    _index_dirty: bool = field(default=True, repr=False)
```

### Index Rebuild Flow
```
add_concept() ──┐
fuse_concepts() ├──> _index_dirty = True
check_mitosis() ┘
                      │
                      ▼
nearest_concepts() ──> if dirty: _rebuild_index()
                      │
                      ▼
                 FAISS search (top_k * 3)
                      │
                      ▼
                 Mass rerank → return top_k
```

## Related Code Files

| File | Action | Lines |
|------|--------|-------|
| `hologram/gravity.py` | Modify | 242-264, 1061-1099 |

## Implementation Steps

### Step 1: Add Index Fields to Gravity Dataclass (line ~258)

```python
# After _pca_cache field
_faiss_index: Optional[faiss.IndexFlatIP] = field(default=None, repr=False)
_index_names: List[str] = field(default_factory=list, repr=False)
_index_dirty: bool = field(default=True, repr=False)
```

### Step 2: Implement _rebuild_index() Method

Add after `__post_init__`:

```python
def _rebuild_index(self):
    """Rebuild FAISS index from current TIER_DOMAIN concepts."""
    # Filter valid attractors
    candidates = [
        n for n in self.concepts.keys()
        if (self.concepts[n].tier == TIER_DOMAIN
            or self.concepts[n].tier == TIER_META
            or n.startswith("glyph:"))
        and self.concepts[n].canonical_id is None  # Skip aliases
    ]

    if not candidates:
        self._faiss_index = None
        self._index_names = []
        self._index_dirty = False
        return

    # Build index
    vecs = np.stack([self.concepts[n].vec for n in candidates]).astype('float32')
    self._faiss_index = faiss.IndexFlatIP(self.dim)
    self._faiss_index.add(vecs)
    self._index_names = candidates
    self._index_dirty = False
```

### Step 3: Modify nearest_concepts() (line 1061-1099)

Replace brute-force loop:

```python
def nearest_concepts(self, vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
    """Find nearest concepts using FAISS + mass-weighted reranking."""
    if not self.concepts:
        return []

    # Lazy rebuild if dirty
    with self._lock:
        if self._index_dirty or self._faiss_index is None:
            self._rebuild_index()

    if not self._index_names:
        return []

    # Normalize query vector
    vec = (vec / (np.linalg.norm(vec) + 1e-8)).reshape(1, -1).astype('float32')

    # FAISS search: fetch 3× to account for mass reranking
    fetch_k = min(top_k * 3, len(self._index_names))
    D, I = self._faiss_index.search(vec, fetch_k)

    # Rerank by mass-weighted similarity
    results = []
    for cos_sim, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(self._index_names):
            continue
        name = self._index_names[idx]
        if name not in self.concepts:
            continue
        mass = self.concepts[name].mass
        weighted_sim = cos_sim * np.log1p(mass)
        results.append((name, float(weighted_sim)))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
```

### Step 4: Mark Index Dirty on Mutations

Add `self._index_dirty = True` to:

1. `add_concept()` - after adding new concept
2. `fuse_concepts()` - after fusion
3. `check_mitosis()` - after split (after adding siblings)

## Todo List

- [ ] Add _faiss_index, _index_names, _index_dirty fields
- [ ] Implement _rebuild_index() method
- [ ] Rewrite nearest_concepts() with FAISS
- [ ] Add dirty flag to add_concept()
- [ ] Add dirty flag to fuse_concepts()
- [ ] Add dirty flag to check_mitosis()
- [ ] Run existing tests
- [ ] Add performance benchmark test

## Success Criteria

1. `nearest_concepts()` uses FAISS search
2. All existing tests pass
3. Index rebuilds only when dirty
4. Thread-safe operation maintained

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Index rebuild blocks queries | Medium | High | Rebuild under lock, queries wait briefly |
| Stale index returns wrong results | Low | Medium | Dirty flag on all mutations |
| Memory spike during rebuild | Low | Low | Old index GC'd after new one ready |

## Security Considerations

- No external input directly to FAISS
- Index contains only internal concept vectors
- No new attack surface

## Next Steps

After completion:
- Proceed to [Phase 2: Probe Neighbor Caching](phase-02-probe-neighbor-caching.md)
