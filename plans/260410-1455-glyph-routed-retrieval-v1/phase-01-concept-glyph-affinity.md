# Phase 1: Concept glyph_affinity + GlyphOperator Interface

## Status: TODO
## Priority: HIGH

## Overview
Add `glyph_affinity` to Concept dataclass and define the GlyphOperator interface. The operator interface is designed doc-faithful from the start — Phase 1 implements identity transforms, Phase 2+ swaps in R_g + P_k without rewriting retrieval.

## Related Code Files

### Modify
- `hologram/gravity.py` — Concept dataclass (~line 192): add `glyph_affinity: Dict[str, float]`
- `hologram/gravity.py` — `Gravity.serialize()` (~line 1060): include glyph_affinity
- `hologram/gravity.py` — `Gravity.load_state()` (~line 1100): deserialize glyph_affinity
- `hologram/glyphs.py` — `GlyphRegistry.attach_trace()` (~line 34): update concept's glyph_affinity

### Create
- `hologram/glyph_operator.py` — GlyphOperator class (interface for glyph-conditioned transforms)

## GlyphOperator Interface

```python
class GlyphOperator:
    """
    Glyph-conditioned transform operator.
    
    Doc spec: T_g(z) = P_k R_g z
    Phase 1: identity (no rotation, no projection)
    Phase 2+: real orthogonal rotation + dimension projection
    """
    def __init__(self, glyph_id: str, dim: int):
        self.glyph_id = glyph_id
        self.dim = dim
        # Phase 1: identity — no R_g, no P_k
        # Phase 2+: self.R_g = random_orthogonal_matrix(dim)
        #           self.P_k = top_k_projection(k=16)
    
    def transform_query(self, vec: np.ndarray) -> np.ndarray:
        """Transform query vector into this glyph's subspace."""
        return vec  # Phase 1: identity
    
    def transform_trace(self, vec: np.ndarray) -> np.ndarray:
        """Transform trace vector for storage in this glyph's subspace."""
        return vec  # Phase 1: identity
    
    @property
    def output_dim(self) -> int:
        """Dimension of vectors after transform. Phase 1: same as input."""
        return self.dim
```

## Implementation Steps

1. Create `hologram/glyph_operator.py` with GlyphOperator class (identity transforms)
2. Add `glyph_affinity: Dict[str, float] = field(default_factory=dict)` to Concept dataclass in `gravity.py`
3. In `GlyphRegistry.attach_trace()`, after linking trace to glyph:
   - Look up the concept name in gravity field that corresponds to this trace's content
   - Compute affinity: normalized trace count per glyph (`affinity[g] = traces_in_g / total_traces`)
   - Update `concept.glyph_affinity`
4. Include `glyph_affinity` in `Gravity.serialize()` output dict
5. Load `glyph_affinity` in `Gravity.load_state()` from saved data

## Important Constraint
**Glyph membership is trace-level authority in Phase 1.** A trace belongs to a glyph because it was explicitly attached via `GlyphRegistry.attach_trace()`. Concept-level glyph affinity (`concept.glyph_affinity`) is computed as a derived aggregate from trace membership — it is informational, not authoritative. Concept-level routing (where a concept is auto-assigned to glyphs without explicit trace attachment) is deferred to later phases.

## Success Criteria
- [ ] GlyphOperator class exists with transform_query/transform_trace (identity)
- [ ] Concept.glyph_affinity populated as derived aggregate from trace-level membership
- [ ] Serialization round-trip preserves glyph_affinity
- [ ] No changes to physics behavior (mass, drift, decay unchanged)
- [ ] Interface shape supports future R_g + P_k swap without retrieval path rewrite
