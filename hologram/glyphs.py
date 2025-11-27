# hologram/glyphs.py
from typing import List, Tuple, Dict, Optional
import numpy as np
from numpy.linalg import norm
from .store import MemoryStore, Glyph, Trace


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Safe cosine similarity."""
    na = norm(a) + 1e-8
    nb = norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))


class GlyphRegistry:
    """
    High-level access layer between Glyphs and Traces.

    Responsibilities:
      - manage glyph–trace linking
      - handle recall and search
      - compute resonance scores (weighted by decay)
    """

    def __init__(self, store: MemoryStore):
        self.store = store
        self._cache: Dict[str, List[Trace]] = {}

    # --- CRUD ---
    def create(self, glyph_id: str, title: str, notes: str = ""):
        """Create a new glyph anchor."""
        self.store.upsert_glyph(Glyph(glyph_id=glyph_id, title=title, notes=notes))

    def attach_trace(self, glyph_id: str, trace: Trace):
        """Attach a new trace to an existing glyph."""
        self.store.add_trace(trace)
        self.store.link_trace(glyph_id, trace.trace_id)
        self._cache.pop(glyph_id, None)  # invalidate cache
        
        # --- Glyph Physics ---
        # 1. Recompute Centroid
        glyph = self.store.get_glyph(glyph_id)
        if not glyph:
            return

        vecs: List[np.ndarray] = []
        for tid in glyph.trace_ids:
            t = self.store.get_trace(tid)
            if t is not None and t.vec is not None:
                vecs.append(t.vec)

        if not vecs:
            return

        mat = np.stack(vecs, axis=0).astype("float32")
        centroid = mat.mean(axis=0)
        centroid /= (norm(centroid) + 1e-8)

        # 2. Compute Mass (Logarithmic growth)
        import math
        trace_count = len(vecs)
        mass = 1.0 + 0.75 * math.log1p(trace_count)

        # 3. Update GravityField
        # Access via store.sim (Gravity instance)
        gravity = getattr(self.store, "sim", None)
        if gravity is not None:
            name = f"glyph:{glyph_id}"
            # Use is_glyph=True to overwrite vector/mass
            gravity.add_concept(name, vec=centroid, mass=mass, is_glyph=True)

    def recall(self, glyph_id: str, top_k: int = 5) -> List[Trace]:
        """Recall traces for a given glyph."""
        g = self.store.get_glyph(glyph_id)
        if not g:
            return []
        traces = [self.store.get_trace(tid) for tid in g.trace_ids]
        traces = [t for t in traces if t is not None]
        self._cache[glyph_id] = traces
        return traces[:top_k]

    # --- Search ---
    def search_across(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[Trace, float]]:
        """Search globally across all traces."""
        hits = self.store.search_traces(query_vec, top_k=top_k)
        out = []
        for tid, score in hits:
            t = self.store.get_trace(tid)
            if t:
                out.append((t, score))
        return out

    # --- Resonance ---
    def resonance_score(
        self,
        query_vec: np.ndarray,
        decay_gamma: Optional[float] = None,
        normalize: bool = True,
    ) -> Dict[str, float]:
        """
        Compute resonance per glyph based on query vector similarity.

        - Takes mean vector of all traces for each glyph.
        - Applies decay weighting using Gravity counts.
        - Optionally normalizes scores to [0,1].
        """
        sims: Dict[str, float] = {}
        glyphs = self.store.get_all_glyphs()
        gamma = decay_gamma or getattr(self.store.sim, "gamma_decay", 0.98)

        for g in glyphs:
            traces = [self.store.get_trace(tid) for tid in g.trace_ids]
            traces = [t for t in traces if t and t.vec is not None]
            if not traces:
                continue
            avg_vec = np.mean([t.vec for t in traces], axis=0)
            sims[g.glyph_id] = cosine(query_vec, avg_vec)

        # Decay weighting based on activation count in Gravity sim
        if hasattr(self.store, "sim") and hasattr(self.store.sim, "concepts"):
            for gid, score in sims.items():
                concept = self.store.sim.concepts.get(gid)
                if concept:
                    sims[gid] = score * (gamma ** concept.count)

        # Optional normalization
        if normalize and sims:
            vals = np.array(list(sims.values()), dtype=np.float32)
            minv, maxv = vals.min(), vals.max()
            rng = (maxv - minv) + 1e-8
            for k, v in sims.items():
                sims[k] = float((v - minv) / rng)

        return sims

    # --- Drift / Decay control ---
    def decay(self, steps: int = 1):
        """Apply decay through MemoryStore (gravity field)."""
        self.store.step_decay(steps)

    # --- Introspection ---
    def summary(self) -> Dict[str, int]:
        """Return a quick overview of stored content."""
        return {
            "glyphs": len(self.store.glyphs),
            "traces": len(self.store.traces),
        }
