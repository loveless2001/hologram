# hologram/store.py
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from .gravity import Gravity  # fixed import


# --- Data models ---
@dataclass
class Trace:
    trace_id: str
    kind: str              # 'text' | 'image' | 'token' | etc.
    content: str           # raw text, path, or payload
    vec: np.ndarray        # vector representation
    meta: dict = field(default_factory=dict)


@dataclass
class Glyph:
    glyph_id: str          # e.g. '🝞' or 'anchor:memory-gravity'
    title: str
    notes: str = ""
    trace_ids: List[str] = field(default_factory=list)


# --- Lightweight vector index ---
import faiss

class VectorIndex:
    def __init__(self, dim: int, use_gpu: bool = True):
        self.dim = dim
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0

        # CPU index first
        self.index = faiss.IndexFlatIP(dim)

        # Move to GPU if possible
        if self.use_gpu:
            print(f"[FAISS] Using GPU backend ({faiss.get_num_gpus()} GPUs available)")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        else:
            print("[FAISS] Using CPU backend")

        self.id_to_key = []

    def upsert(self, key: str, vec: np.ndarray):
        vec = vec.astype('float32').reshape(1, -1)
        self.index.add(vec)
        self.id_to_key.append(key)

    def search(self, query: np.ndarray, top_k: int = 5):
        if self.index.ntotal == 0:
            return []
        query = query.astype('float32').reshape(1, -1)
        D, I = self.index.search(query, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < len(self.id_to_key):
                results.append((self.id_to_key[idx], float(score)))
        return results

# --- Core store ---
class MemoryStore:
    """Persistent memory + glyph registry + gravity integration."""
    def __init__(self, vec_dim: int):
        self.vec_dim = vec_dim
        self.traces = {}
        self.glyphs = {}
        self.index = VectorIndex(vec_dim, use_gpu=True)
        self.sim = Gravity(dim=vec_dim)

    # --- Trace operations ---
    def add_trace(self, t: Trace):
        self.traces[t.trace_id] = t
        self.index.upsert(t.trace_id, t.vec)
        self.sim.add_concept(t.trace_id, vec=t.vec)

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        return self.traces.get(trace_id)

    # --- Glyph operations ---
    def upsert_glyph(self, g: Glyph):
        if g.glyph_id in self.glyphs:
            existing = self.glyphs[g.glyph_id]
            existing.title = g.title or existing.title
            existing.notes = g.notes or existing.notes
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

    def get_all_glyphs(self) -> List[Glyph]:
        """Expose all glyphs for resonance scoring."""
        return list(self.glyphs.values())

    # --- Search ---
    def search_traces(self, query_vec: np.ndarray, top_k: int = 5):
        return self.index.search(query_vec, top_k=top_k)

    # --- Gravity integration ---
    def step_decay(self, steps: int = 1):
        """Advance the gravity field’s decay (for lossy persistence)."""
        self.sim.step_decay(steps)

    # --- Persistence ---
    def save(self, path: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vec_dim": self.vec_dim,
            "gravity_state": self.sim.get_state(),  # Save gravity state
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
        if not path_obj.exists():
            raise FileNotFoundError(f"Store file not found: {path}")
        data = json.loads(path_obj.read_text(encoding="utf-8"))

        vec_dim = int(data.get("vec_dim"))
        store = cls(vec_dim=vec_dim)

        # Restore gravity state if available
        gravity_state = data.get("gravity_state")
        if gravity_state:
            store.sim.set_state(gravity_state)
            print(f"[MemoryStore] Restored gravity field state ({len(store.sim.concepts)} concepts)")

        for t_data in data.get("traces", []):
            vec = np.array(t_data["vec"], dtype=np.float32)
            trace = Trace(
                trace_id=t_data["trace_id"],
                kind=t_data["kind"],
                content=t_data["content"],
                vec=vec,
                meta=t_data.get("meta", {}),
            )
            # Add to store and index, but conditionally skip gravity update
            store.traces[trace.trace_id] = trace
            store.index.upsert(trace.trace_id, trace.vec)
            
            # Only add to sim if we didn't restore state (legacy replay mode)
            # OR if this specific trace isn't in the restored concepts (partial update?)
            # For simplicity: if gravity_state was loaded, we assume it covers the traces.
            # But to be safe against partial saves, we can check:
            if not gravity_state or trace.trace_id not in store.sim.concepts:
                store.sim.add_concept(trace.trace_id, vec=trace.vec)

        for g_data in data.get("glyphs", []):
            glyph = Glyph(
                glyph_id=g_data["glyph_id"],
                title=g_data.get("title", g_data["glyph_id"]),
                notes=g_data.get("notes", ""),
                trace_ids=list(g_data.get("trace_ids", [])),
            )
            store.glyphs[glyph.glyph_id] = glyph

        return store
