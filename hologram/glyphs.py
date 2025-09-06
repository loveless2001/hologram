
from typing import List, Tuple
from .store import MemoryStore, Glyph, Trace
import numpy as np

class GlyphRegistry:
    def __init__(self, store: MemoryStore):
        self.store = store

    def create(self, glyph_id: str, title: str, notes: str = ""):
        self.store.upsert_glyph(Glyph(glyph_id=glyph_id, title=title, notes=notes))

    def attach_trace(self, glyph_id: str, trace: Trace):
        # add trace + link
        self.store.add_trace(trace)
        self.store.link_trace(glyph_id, trace.trace_id)

    def recall(self, glyph_id: str, top_k: int = 5) -> List[Trace]:
        g = self.store.get_glyph(glyph_id)
        if not g:
            return []
        # Return traces in link order (could sort by recency/score later)
        return [self.store.get_trace(tid) for tid in g.trace_ids if self.store.get_trace(tid) is not None]

    def search_across(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[Trace, float]]:
        hits = self.store.search_traces(query_vec, top_k=top_k)
        out = []
        for tid, score in hits:
            t = self.store.get_trace(tid)
            if t:
                out.append((t, score))
        return out
