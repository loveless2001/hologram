# Phase 3: Dynamic Quantization by Scale

## Context Links
- [Parent Plan](plan.md)
- [Phase 1: FAISS Index](phase-01-persistent-faiss-index.md)
- [Phase 2: Probe Caching](phase-02-probe-neighbor-caching.md)
- [Brainstorm Report](../reports/brainstorm-260205-1530-gravity-field-optimization.md)
- Target: `hologram/gravity.py:94-133`

## Overview

| Field | Value |
|-------|-------|
| Priority | P2 - Graceful scale handling |
| Status | pending |
| Effort | 30m |
| Depends On | None (independent) |

Extend `calibrate_quantization()` to factor in dataset size. More concepts → higher quantization level → skip finer interactions.

## Key Insights

- Current calibration only considers hardware (GPU, CPU, RAM)
- At scale (100K+), fine-grained drift becomes expensive
- Quantization level acts as "Planck constant" - minimum action threshold
- Higher threshold = fewer drift operations = faster ingestion

## Requirements

### Functional
- [ ] Add scale-based adjustment to `calibrate_quantization()`
- [ ] Thresholds: >100K +0.03, >10K +0.02, >1K +0.01
- [ ] Pass concept count as optional parameter
- [ ] Optional: Lazy recalibration on significant scale changes

### Non-Functional
- [ ] Backward compatible (works with no args)
- [ ] No performance regression for small datasets
- [ ] Preserves physics accuracy at small scale

## Architecture

```
calibrate_quantization(n_concepts=None)
    │
    ├──> base = 0.05 (conservative)
    │
    ├──> Hardware adjustments (existing)
    │    ├── GPU: -0.02
    │    ├── CPU ≥8: -0.01
    │    ├── CPU ≤2: +0.02
    │    ├── RAM ≥16GB: -0.01
    │    └── RAM ≤4GB: +0.02
    │
    ├──> NEW: Scale adjustments
    │    ├── n > 100K: +0.03
    │    ├── n > 10K: +0.02
    │    └── n > 1K: +0.01
    │
    └──> return clamp(base, 0.001, 0.2)
```

## Related Code Files

| File | Action | Lines |
|------|--------|-------|
| `hologram/gravity.py` | Modify | 94-133, 260-263 |

## Implementation Steps

### Step 1: Modify calibrate_quantization() (line 94-133)

```python
def calibrate_quantization(n_concepts: int = 0) -> float:
    """
    Calibrate quantization level based on hardware specs and dataset scale.
    Lower level = more propagation (better hardware, smaller dataset).
    Higher level = less propagation (weaker hardware, larger dataset).

    Args:
        n_concepts: Current number of concepts in field (0 = ignore scale)
    """
    # Base level (conservative)
    q_level = 0.05

    # 1. GPU Check (FAISS)
    try:
        num_gpus = faiss.get_num_gpus()
        if num_gpus > 0:
            q_level -= 0.02  # Significant boost for GPU
    except Exception:
        pass

    # 2. CPU Check
    try:
        cpu_count = os.cpu_count() or 1
        if cpu_count >= 8:
            q_level -= 0.01
        elif cpu_count <= 2:
            q_level += 0.02
    except Exception:
        pass

    # 3. RAM Check (if psutil available)
    if psutil:
        try:
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024 ** 3)
            if total_gb >= 16:
                q_level -= 0.01
            elif total_gb <= 4:
                q_level += 0.02
        except Exception:
            pass

    # 4. NEW: Scale-based adjustment
    if n_concepts > 100_000:
        q_level += 0.03
    elif n_concepts > 10_000:
        q_level += 0.02
    elif n_concepts > 1_000:
        q_level += 0.01

    return max(0.001, min(q_level, 0.2))  # Clamp between 0.001 and 0.2
```

### Step 2: Update __post_init__ to Use Scale

Modify `__post_init__` (line 260-263):

```python
def __post_init__(self):
    self._lock = threading.RLock()
    # Initialize FAISS index fields
    self._faiss_index = None
    self._index_names = []
    self._index_dirty = True
    # Calibrate with current scale
    if self.quantization_level is None:
        self.quantization_level = calibrate_quantization(len(self.concepts))
```

### Step 3: Optional - Recalibrate on Scale Milestones

Add method for lazy recalibration:

```python
def _maybe_recalibrate_quantization(self):
    """Recalibrate quantization if scale crossed a threshold."""
    n = len(self.concepts)
    thresholds = [1_000, 10_000, 100_000, 500_000, 1_000_000]

    # Check if we crossed a threshold since last calibration
    if not hasattr(self, '_last_calibration_scale'):
        self._last_calibration_scale = 0

    for t in thresholds:
        if self._last_calibration_scale < t <= n:
            self.quantization_level = calibrate_quantization(n)
            self._last_calibration_scale = n
            log_event("GRAVITY", f"Recalibrated quantization at {n} concepts",
                     {"new_level": f"{self.quantization_level:.4f}"})
            break
```

Call this in `add_concept()` after incrementing count.

## Todo List

- [ ] Add `n_concepts` parameter to `calibrate_quantization()`
- [ ] Add scale-based thresholds (+0.01/0.02/0.03)
- [ ] Update `__post_init__` to pass initial count
- [ ] (Optional) Add `_maybe_recalibrate_quantization()`
- [ ] (Optional) Call recalibrate in `add_concept()`
- [ ] Run existing tests
- [ ] Test with synthetic large dataset

## Success Criteria

1. Quantization level increases with dataset size
2. All existing tests pass
3. No regression for small datasets (<1K concepts)
4. Graceful behavior at 100K+ concepts

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Too aggressive quantization loses precision | Low | Medium | Conservative thresholds, floor at 0.001 |
| Recalibration overhead | Very Low | Low | Only at milestone crossings |
| Backward compatibility | Very Low | Low | Default n_concepts=0 ignores scale |

## Security Considerations

- No external input affects quantization
- Internal scaling only
- No new attack surface

## Next Steps

After all phases complete:
- Run full test suite
- Benchmark probe latency at 10K, 50K, 100K concepts
- Document performance improvements in `docs/performance_metrics.md`
