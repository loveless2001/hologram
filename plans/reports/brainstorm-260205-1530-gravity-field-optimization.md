# Brainstorm: Gravity Field Mass & Propagation Optimization

**Date**: 2026-02-05
**Context**: Hologram gravity field uses scalar mass for vector propagation/clustering
**Question**: Is scalar mass sufficient? Can propagation be optimized?

---

## Problem Statement

Current gravity field implementation uses scalar mass (`float`) per concept. Mass influences:
- Fusion thresholds (black hole effect)
- Weighted similarity during probe navigation
- Mutual drift between concepts

**User priorities**:
1. Dynamic scale calibration (adapt to dataset size)
2. Performance optimization (especially propagation)
3. Fast probes (retrieval must be snappy)

---

## Current Implementation Analysis

### Scalar Mass Mechanics

| Operation | Formula | Location |
|-----------|---------|----------|
| Fusion threshold | `0.85 - log(mass) * 0.02` | `gravity.py:615` |
| Weighted similarity | `cos_sim * log(1 + mass)` | `gravity.py:1093` |
| Mutual drift step | `eta * sim` (mass-independent) | `gravity.py:348` |
| Glyph mass | `1.0 + 0.75 * log(1 + traces)` | `glyphs.py:59-62` |

### Identified Bottlenecks

1. **`nearest_concepts()`** (line 1061-1099)
   - Brute-force O(N) loop
   - No FAISS acceleration for mass-weighted search
   - Called per probe step (up to 8× per query)

2. **`_mutual_drift()`** (line 279-361)
   - Full matrix multiply `X @ v.T` on every add
   - Iterative vector updates can't vectorize

3. **`check_fusion_all()`** (line 533-638)
   - Rebuilds FAISS index every call
   - O(N²) worst case

---

## Evaluated Approaches

### Option A: Optimize Current Scalar (RECOMMENDED)

**Philosophy**: Scalar mass IS semantically sufficient. Optimize compute only.

#### A1. Persistent FAISS Index for `nearest_concepts()`

```python
class Gravity:
    _faiss_index: Optional[faiss.IndexFlatIP] = None
    _index_names: List[str] = []
    _index_dirty: bool = True

    def nearest_concepts(self, vec, top_k=10):
        if self._index_dirty:
            self._rebuild_index()

        # FAISS search (SIMD optimized)
        fetch_k = min(top_k * 3, len(self._index_names))
        D, I = self._faiss_index.search(vec.reshape(1,-1), fetch_k)

        # Rerank by mass in O(k)
        results = []
        for sim, idx in zip(D[0], I[0]):
            name = self._index_names[idx]
            weighted = sim * np.log1p(self.concepts[name].mass)
            results.append((name, weighted))

        return sorted(results, key=lambda x: -x[1])[:top_k]
```

**Gain**: 10-50× faster than Python loops

#### A2. Batch Mutual Drift

```python
_drift_buffer: List[str] = []
_drift_batch_size: int = 32

def _flush_drift_buffer(self):
    new_vecs = np.stack([self.concepts[n].vec for n in self._drift_buffer])
    X_existing = self.get_matrix()  # existing concepts
    sims = new_vecs @ X_existing.T  # single batch multiply
    # ... apply top-k drift per new concept
```

**Gain**: ~5-10× on large batches, amortizes matrix ops

#### A3. Probe Neighbor Caching

```python
def run_probe_optimized(self, vec, max_steps=8):
    # Pre-fetch neighbors ONCE at start
    initial_neighbors = self.nearest_concepts(vec, top_k=50)
    neighbor_cache = {n: self.concepts[n] for n, _ in initial_neighbors}

    # Steps use cache, no repeated FAISS calls
    for step in range(max_steps):
        # Use neighbor_cache for attraction computation
        ...
```

**Gain**: Eliminates repeated search per step

#### A4. Dynamic Quantization by Scale

```python
def calibrate_quantization(self) -> float:
    base = 0.05
    n = len(self.concepts)

    if n > 100_000: base += 0.03
    elif n > 10_000: base += 0.02
    elif n > 1_000: base += 0.01

    # ... hardware adjustments ...
    return max(0.001, min(base, 0.2))
```

**Effect**: More concepts → skip finer interactions

---

### Option B: Tensor Mass (EXPLORATORY)

**Philosophy**: Per-dimension mass captures directional semantic weight.

```python
@dataclass
class Concept:
    vec: np.ndarray       # [D] position
    mass_vec: np.ndarray  # [D] per-dimension mass

def weighted_sim_tensor(query, concept):
    cos_sim = np.dot(query, concept.vec)
    directional_mass = np.dot(np.abs(query), concept.mass_vec)
    return cos_sim * np.log1p(directional_mass)
```

**Use cases**:
- Multi-domain disambiguation ("Python" heavy in code dims, light in animal dims)
- Aspect-based retrieval (query direction weights relevant mass)

**Challenges**:
- D× memory overhead per concept
- Initialization and update logic unclear
- May be over-engineering

**Verdict**: Interesting but unproven. Defer until Option A profiled.

---

### Option C: Glyph-Mediated Gravity

**Philosophy**: Glyphs are primary attractors; concepts inherit their field.

```
GLYPH LAYER (sparse, ~100-1K)
    ↓ propagates field
CONCEPT LAYER (dense, N concepts)
    ↓
PROBE navigates glyph-mediated space
```

**Pros**: O(G) complexity where G << N
**Cons**: Major architectural change, glyph-centric redesign needed

---

### Option D: Cached Influence Fields

**Philosophy**: Pre-compute gravity zones, probes use cached fields.

```python
influence_cache: Dict[str, Set[str]]  # concept → influenced neighbors
field_vectors: np.ndarray             # [N, D] cached attraction

def update_field(self, changed: List[str]):
    # Only recompute local zones
```

**Pros**: O(1) probe lookup
**Cons**: Stale cache issues, memory overhead

---

## Recommendation

### Phase 1: Implement Option A (Low Risk, High Gain)

1. **A1**: Persistent FAISS index with lazy rebuild
2. **A3**: Probe neighbor caching (fast probes priority)
3. **A4**: Dynamic quantization calibration

**Expected outcome**: 10-50× probe speedup, graceful scale handling

### Phase 2: Profile & Evaluate

After A-series optimization:
- Profile `run_probe()` with real workloads
- Measure clustering quality metrics
- Identify remaining bottlenecks

### Phase 3: Consider Tensor Mass (If Needed)

Only if:
- Clustering quality degraded at scale
- Multi-domain disambiguation issues
- Profiling shows mass calculation as bottleneck

---

## Implementation Considerations

### Index Maintenance Strategy

| Event | Action |
|-------|--------|
| `add_concept()` | Set `_index_dirty = True` |
| `step_dynamics()` (fusion/mitosis) | Set `_index_dirty = True` |
| First `nearest_concepts()` after dirty | Rebuild index |
| Large-scale (>100K) | Use IVF index with incremental add |

### Memory Budget

| Optimization | Memory Cost |
|--------------|-------------|
| FAISS IndexFlatIP | O(N × D × 4 bytes) |
| Neighbor cache (per probe) | O(50 × D × 4 bytes) |
| Tensor mass (Option B) | +O(N × D × 4 bytes) |

### Thread Safety

Current `_lock` covers mutations. Index operations should:
- Rebuild under lock
- Search can be lock-free (FAISS is thread-safe for search)

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Index rebuild latency spike | Background rebuild thread, swap atomically |
| Cache staleness in probes | Short-lived cache per probe run |
| Memory pressure at scale | IVF index trades accuracy for memory |

---

## Success Criteria

1. **Probe latency**: <50ms for 100K concepts
2. **Scale handling**: Graceful degradation to 1M concepts
3. **Clustering quality**: No regression in fusion/mitosis behavior
4. **Code simplicity**: Minimal API changes

---

## Next Steps

1. Create implementation plan for Option A optimizations
2. Add profiling instrumentation to measure baseline
3. Implement A1 (FAISS index) and A3 (probe caching) first
4. Benchmark and iterate

---

## Unresolved Questions

1. Should probe cache be per-probe instance or shared warm cache?
2. Optimal `fetch_k` multiplier for mass reranking (3× proposed)?
3. IVF `nlist` parameter tuning for different scales?
4. Tensor mass initialization strategy if pursued?
