from typing import List, Tuple, Optional, Dict
import numpy as np
from .store import MemoryStore, Glyph, Trace
from numpy.linalg import norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Safe cosine similarity."""
    na = norm(a) + 1e-8
    nb = norm(b) + 1e-8
    return float(np.dot(a, b) / (na * nb))


class GlyphRegistry:
    """
    The registry maps glyphs to their traces (text, image, etc.)
    and allows recall and resonance-based search.
    """
    def __init__(self, store: MemoryStore):
        self.store = store
        # optional: cache glyph -> traces for quick lookup
        self._cache: Dict[str, List[Trace]] = {}

    # --- CRUD ---
    def create(self, glyph_id: str, title: str, notes: str = ""):
        self.store.upsert_glyph(Glyph(glyph_id=glyph_id, title=title, notes=notes))

    def attach_trace(self, glyph_id: str, trace: Trace):
        """Attach a trace to a glyph, link in memory store."""
        self.store.add_trace(trace)
        self.store.link_trace(glyph_id, trace.trace_id)
        self._cache.pop(glyph_id, None)  # invalidate cache

    def recall(self, glyph_id: str, top_k: int = 5) -> List[Trace]:
        """Recall traces for a given glyph."""
        g = self.store.get_glyph(glyph_id)
        if not g:
            return []
        traces = [self.store.get_trace(tid) for tid in g.trace_ids]
        traces = [t for t in traces if t is not None]
        self._cache[glyph_id] = traces
        return traces[:top_k]

    # --- Search / Resonance ---
    def search_across(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[Trace, float]]:
        """Search all traces globally by vector similarity."""
        hits = self.store.search_traces(query_vec, top_k=top_k)
        out = []
        for tid, score in hits:
            t = self.store.get_trace(tid)
            if t:
                out.append((t, score))
        return out

    def resonance_score(self, query_vec: np.ndarray, decay_gamma: float = 0.98) -> Dict[str, float]:
        """
        Compute per-glyph resonance relative to a query vector.
        We take mean of all trace vectors per glyph, then apply
        decay based on its age / activation count (if available).
        """
        sims = {}
        all_glyphs = self.store.get_all_glyphs()
        for g in all_glyphs:
            traces = [self.store.get_trace(tid) for tid in g.trace_ids]
            traces = [t for t in traces if t is not None and t.vec is not None]
            if not traces:
                continue
            avg_vec = np.mean([t.vec for t in traces], axis=0)
            sims[g.glyph_id] = cosine(query_vec, avg_vec)

        # optional decay weighting using GravitySim if available
        if hasattr(self.store, "sim") and hasattr(self.store.sim, "concepts"):
            for gid, score in sims.items():
                if gid in self.store.sim.concepts:
                    sims[gid] *= (decay_gamma ** self.store.sim.concepts[gid].count)

        return sims
