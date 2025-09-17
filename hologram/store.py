
import json
from dataclasses import dataclass, field
from pathlib import Path
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

    # --- Persistence ---
    def save(self, path: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vec_dim": self.vec_dim,
            "traces": [
                {
                    "trace_id": t.trace_id,
                    "kind": t.kind,
                    "content": t.content,
                    "vec": t.vec.tolist(),
                    "meta": t.meta,
                }
                for t in self.traces.values()
            ],
            "glyphs": [
                {
                    "glyph_id": g.glyph_id,
                    "title": g.title,
                    "notes": g.notes,
                    "trace_ids": g.trace_ids,
                }
                for g in self.glyphs.values()
            ],
        }
        path_obj.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "MemoryStore":
        path_obj = Path(path)
        data = json.loads(path_obj.read_text(encoding="utf-8"))

        vec_dim = int(data.get("vec_dim"))
        store = cls(vec_dim=vec_dim)

        for t_data in data.get("traces", []):
            vec = np.array(t_data["vec"], dtype=np.float32)
            trace = Trace(
                trace_id=t_data["trace_id"],
                kind=t_data["kind"],
                content=t_data["content"],
                vec=vec,
                meta=t_data.get("meta", {}),
            )
            store.add_trace(trace)

        for g_data in data.get("glyphs", []):
            glyph = Glyph(
                glyph_id=g_data["glyph_id"],
                title=g_data.get("title", g_data["glyph_id"]),
                notes=g_data.get("notes", ""),
                trace_ids=list(g_data.get("trace_ids", [])),
            )
            store.glyphs[glyph.glyph_id] = glyph

        return store
