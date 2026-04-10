# Phase 2: Probe Neighbor Caching

## Context Links
- [Parent Plan](plan.md)
- [Phase 1: FAISS Index](phase-01-persistent-faiss-index.md) (dependency)
- [Brainstorm Report](../reports/brainstorm-260205-1530-gravity-field-optimization.md)
- Target: `hologram/gravity.py:1181-1208`

## Overview

| Field | Value |
|-------|-------|
| Priority | P1 - Fast probes is primary goal |
| Status | pending |
| Effort | 1h |
| Depends On | Phase 1 |

Pre-fetch neighbors once at probe start, cache vectors/masses, eliminate repeated `nearest_concepts()` calls during drift steps.

## Key Insights

- Current `run_probe()` calls `nearest_concepts()` per step (up to 8×)
- Even with FAISS, repeated searches add latency
- Probe drift is local - initial neighbors likely remain relevant
- Cache per-probe instance avoids staleness issues

## Requirements

### Functional
- [ ] Pre-fetch expanded neighbor set at probe start (top_k=50)
- [ ] Cache neighbor vectors and masses in local dict
- [ ] Modify `step_probe()` to accept optional cache
- [ ] Use cached data for attraction computation

### Non-Functional
- [ ] Per-probe cache, not shared (avoids staleness)
- [ ] Memory: O(50 × D × 4 bytes) per active probe
- [ ] Backward compatible: cache is optional

## Architecture

```
run_probe(vec)
    │
    ├──> nearest_concepts(vec, top_k=50)  # ONE search
    │
    ├──> neighbor_cache = {
    │        name: (vec, mass) for top-50 neighbors
    │    }
    │
    └──> for step in range(max_steps):
             step_probe_cached(probe, neighbor_cache)
                 │
                 └──> Use cache for similarity + attraction
                      (No nearest_concepts() call)
```

## Related Code Files

| File | Action | Lines |
|------|--------|-------|
| `hologram/gravity.py` | Modify | 1101-1208 |

## Implementation Steps

### Step 1: Add Cached Step Method

Add new method after `step_probe()`:

```python
def _step_probe_cached(
    self,
    probe: Probe,
    neighbor_cache: Dict[str, Tuple[np.ndarray, float]],
    min_sim: float = 0.2,
    alpha: float = 0.4,
    beta: float = 0.15,
) -> Probe:
    """
    Execute probe step using cached neighbor data.
    neighbor_cache: {name: (vec, mass)}
    """
    # Compute similarities from cache
    neighbors = []
    for name, (nvec, nmass) in neighbor_cache.items():
        cos_sim = float(np.dot(probe.vec, nvec))
        weighted_sim = cos_sim * np.log1p(nmass)
        neighbors.append((name, weighted_sim))

    neighbors.sort(key=lambda x: x[1], reverse=True)

    # Filter by minimum attraction
    used = [(name, sim) for name, sim in neighbors if sim >= min_sim]

    if not used:
        probe.history.append(ProbeStep(
            position=probe.vec.copy(),
            neighbors=neighbors[:10],
            chosen=[]
        ))
        return probe

    # Accumulate attraction force
    attraction = np.zeros_like(probe.vec, dtype=np.float32)
    total_weight = 0.0

    for name, weight in used[:10]:  # Limit to top-10 for force calc
        nvec, _ = neighbor_cache[name]
        w = max(weight, 0.0)
        total_weight += w
        attraction += w * (nvec - probe.vec)

    if total_weight > 0:
        attraction /= (total_weight + 1e-8)

    # Inertia/damping
    inertia = np.zeros_like(probe.vec)
    if probe.previous_vec is not None:
        inertia = probe.vec - probe.previous_vec

    # Update position
    new_pos = probe.vec + (alpha * attraction) - (beta * inertia)
    new_pos /= (np.linalg.norm(new_pos) + 1e-8)

    probe.previous_vec = probe.vec.copy()
    probe.vec = new_pos

    probe.history.append(ProbeStep(
        position=probe.vec.copy(),
        neighbors=neighbors[:10],
        chosen=[name for name, _ in used[:10]]
    ))

    return probe
```

### Step 2: Modify run_probe() to Use Cache

Update `run_probe()` (line 1181-1208):

```python
def run_probe(
    self,
    vec: np.ndarray,
    name: str = "probe",
    max_steps: int = 8,
    tol: float = 1e-3,
    top_k: int = 10,
    min_sim: float = 0.1,
    use_cache: bool = True,  # NEW parameter
) -> Probe:
    """Run probe simulation until convergence."""
    probe = Probe(
        name=name,
        vec=vec.astype("float32"),
        previous_vec=None
    )

    # Pre-fetch neighbors and build cache
    neighbor_cache = None
    if use_cache:
        initial_neighbors = self.nearest_concepts(vec, top_k=50)
        neighbor_cache = {}
        for n, _ in initial_neighbors:
            if n in self.concepts:
                c = self.concepts[n]
                neighbor_cache[n] = (c.vec.copy(), c.mass)

    for _ in range(max_steps):
        prev_vec = probe.vec.copy()

        if neighbor_cache:
            self._step_probe_cached(probe, neighbor_cache, min_sim=min_sim)
        else:
            self.step_probe(probe, top_k=top_k, min_sim=min_sim)

        # Check convergence
        delta = np.linalg.norm(probe.vec - prev_vec)
        if delta < tol:
            break

    return probe
```

### Step 3: Keep Original step_probe() Unchanged

The original `step_probe()` remains for backward compatibility and cases where caching is disabled.

## Todo List

- [ ] Implement `_step_probe_cached()` method
- [ ] Modify `run_probe()` with `use_cache` parameter
- [ ] Default `use_cache=True` for performance
- [ ] Run existing tests
- [ ] Add benchmark comparing cached vs uncached

## Success Criteria

1. `run_probe()` calls `nearest_concepts()` only once (with cache)
2. All existing tests pass
3. Probe convergence behavior unchanged
4. Measurable latency reduction

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Cache misses important neighbors | Low | Medium | Fetch 50 neighbors (5× default) |
| Probe drifts outside cached region | Low | Low | Local drift assumption holds |
| Memory per probe | Very Low | Low | 50 × 256 × 4 = 51KB per probe |

## Security Considerations

- Cache is local to probe instance
- No persistence, no external exposure
- No new attack surface

## Next Steps

After completion:
- Proceed to [Phase 3: Dynamic Quantization](phase-03-dynamic-quantization.md)
