
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

# --- Data model ---
@dataclass
class Trace:
    trace_id: str
    kind: str            # 'text' | 'image' | 'token' | etc.
    content: str         # raw text, path, or small payload
    vec: np.ndarray      # vector representation
    meta: dict = field(default_factory=dict)

@dataclass
class Glyph:
    glyph_id: str        # e.g., '🝞' or 'anchor:memory-gravity'
    title: str
    notes: str = ""
    trace_ids: List[str] = field(default_factory=list)

class SimpleIndex:
    """A minimal cosine similarity index. Swap with FAISS later."""
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: Dict[str, np.ndarray] = {}

    def upsert(self, id: str, vec: np.ndarray):
        self.vectors[id] = vec.astype('float32')

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.vectors:
            return []
        # Stack all vectors
        ids = list(self.vectors.keys())
        mat = np.stack([self.vectors[i] for i in ids], axis=0)
        # cosine sim
        q = query / (np.linalg.norm(query) + 1e-8)
        matn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
        sims = (matn @ q).astype('float32')   # [N]
        top_idx = sims.argsort()[-top_k:][::-1]
        return [(ids[i], float(sims[i])) for i in top_idx]

class MemoryStore:
    def __init__(self, vec_dim: int):
        self.vec_dim = vec_dim
        self.traces: Dict[str, Trace] = {}
        self.glyphs: Dict[str, Glyph] = {}
        self.index = SimpleIndex(vec_dim)

    # --- Traces ---
    def add_trace(self, t: Trace):
        self.traces[t.trace_id] = t
        self.index.upsert(t.trace_id, t.vec)

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        return self.traces.get(trace_id)

    # --- Glyphs ---
    def upsert_glyph(self, g: Glyph):
        if g.glyph_id in self.glyphs:
            # merge traces if re‑upsert
            existing = self.glyphs[g.glyph_id]
            existing.title = g.title or existing.title
            existing.notes = g.notes or existing.notes
            # extend unique trace ids
            seen = set(existing.trace_ids)
            for tid in g.trace_ids:
                if tid not in seen:
                    existing.trace_ids.append(tid)
                    seen.add(tid)
        else:
            self.glyphs[g.glyph_id] = g

    def link_trace(self, glyph_id: str, trace_id: str):
        g = self.glyphs.setdefault(glyph_id, Glyph(glyph_id=glyph_id, title=glyph_id))
        if trace_id not in g.trace_ids:
            g.trace_ids.append(trace_id)

    def get_glyph(self, glyph_id: str) -> Optional[Glyph]:
        return self.glyphs.get(glyph_id)

    # --- Search ---
    def search_traces(self, query_vec, top_k=5):
        return self.index.search(query_vec, top_k=top_k)
