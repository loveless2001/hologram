# hologram/store.py
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

from .gravity import Gravity  # fixed import

# SqliteBackend import moved to bottom to avoid circular dependency



# --- Data models ---
@dataclass
class Trace:
    trace_id: str
    kind: str              # 'text' | 'image' | 'token' | etc.
    content: str           # raw text, path, or payload
    vec: np.ndarray        # vector representation
    meta: dict = field(default_factory=dict)
    resolved_text: Optional[str] = None  # Text with pronouns resolved
    coref_map: Optional[Dict[str, str]] = None  # Map of resolved pronouns
    span: Optional[Tuple[int, int]] = None      # Code span (start_line, end_line)
    source_file: Optional[str] = None           # Path to source file


@dataclass
class Glyph:
    glyph_id: str          # e.g. '🝞' or 'anchor:memory-gravity'
    title: str
    notes: str = ""
    trace_ids: List[str] = field(default_factory=list)
    code_meta: Optional[Dict] = None            # Metadata for code symbols


# --- Lightweight vector index ---
import faiss

class VectorIndex:
    def __init__(self, dim: int, use_gpu: bool = None):
        self.dim = dim
        self._lock = threading.RLock()  # Thread safety

        # Use config if not explicitly provided
        from .config import Config
        if use_gpu is None:
            use_gpu = Config.storage.USE_GPU

        # Try GPU first if requested
        self.use_gpu = False
        if use_gpu:
            try:
                # Check if GPU functions exist
                if hasattr(faiss, 'StandardGpuResources') and hasattr(faiss, 'index_cpu_to_gpu'):
                    # Create CPU index first
                    cpu_index = faiss.IndexFlatIP(dim)
                    
                    # Try to move to GPU
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                    self.use_gpu = True
                    print(f"[FAISS] Using GPU backend")
                else:
                    # GPU functions not available (CPU-only build)
                    self.index = faiss.IndexFlatIP(dim)
                    print("[FAISS] GPU functions not available, using CPU backend")
            except Exception as e:
                # GPU initialization failed (no GPU, CUDA issues, etc.)
                self.index = faiss.IndexFlatIP(dim)
                print(f"[FAISS] GPU initialization failed: {e}")
                print("[FAISS] Falling back to CPU backend")
        else:
            # User requested CPU
            self.index = faiss.IndexFlatIP(dim)
            print("[FAISS] Using CPU backend (user requested)")

        self.id_to_key = []

    def upsert(self, key: str, vec: np.ndarray):
        with self._lock:
            vec = vec.astype('float32')
            vec /= (np.linalg.norm(vec) + 1e-8)
            vec = vec.reshape(1, -1)
            self.index.add(vec)
            self.id_to_key.append(key)

    def search(self, query: np.ndarray, top_k: int = 5):
        with self._lock:
            if self.index.ntotal == 0:
                return []
            query = query.astype('float32')
            query /= (np.linalg.norm(query) + 1e-8)
            query = query.reshape(1, -1)
            D, I = self.index.search(query, top_k)
            results = []
            for score, idx in zip(D[0], I[0]):
                if idx < len(self.id_to_key):
                    results.append((self.id_to_key[idx], float(score)))
            return results

# --- Core store ---
class MemoryStore:
    """Persistent memory + glyph registry + gravity integration."""
    def __init__(self, vec_dim: int, backend: Optional[Any] = None):
        self.vec_dim = vec_dim
        self.traces = {}
        self.glyphs = {}
        self.index = VectorIndex(vec_dim, use_gpu=None)
        self.sim = Gravity(dim=vec_dim)
        self.backend = backend


    # --- Trace operations ---
    def add_trace(self, t: Trace):
        self.traces[t.trace_id] = t
        self.index.upsert(t.trace_id, t.vec)
        # Pass text content to gravity sim for negation detection
        text_for_negation = t.content if t.kind == "text" else None
        self.sim.add_concept(t.trace_id, text=text_for_negation, vec=t.vec)

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
        if path.endswith(".db") or path.endswith(".sqlite"):
            if SqliteBackend is None:
                raise RuntimeError("SqliteBackend not available.")
            backend = SqliteBackend(path)
            
            # Save Meta
            backend.save_meta("vec_dim", str(self.vec_dim))
            backend.save_meta("gravity_state", json.dumps(self.sim.get_state()))
            
            # Save Traces
            for t in self.traces.values():
                backend.save_trace(t)
                
            # Save Glyphs
            for g in self.glyphs.values():
                backend.save_glyph(g)
                
            backend.close()
            return

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
                    "resolved_text": t.resolved_text,
                    "coref_map": t.coref_map,
                    "resolved_text": t.resolved_text,
                    "coref_map": t.coref_map,
                    "span": t.span,
                    "source_file": t.source_file,
                }
                for t in self.traces.values()
            ],
            "glyphs": [
                {
                    "glyph_id": g.glyph_id,
                    "title": g.title,
                    "notes": g.notes,
                    "trace_ids": g.trace_ids,
                    "trace_ids": g.trace_ids,
                    "code_meta": g.code_meta,
                }
                for g in self.glyphs.values()
            ],
        }
        path_obj.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "MemoryStore":
        if path.endswith(".db") or path.endswith(".sqlite"):
            if SqliteBackend is None:
                raise RuntimeError("SqliteBackend not available.")
            backend = SqliteBackend(path)
            
            vec_dim_str = backend.load_meta("vec_dim")
            if not vec_dim_str:
                raise ValueError("Invalid SQLite store: missing vec_dim")
            vec_dim = int(vec_dim_str)
            
            store = cls(vec_dim=vec_dim, backend=backend)
            
            # Restore gravity
            gravity_state_json = backend.load_meta("gravity_state")
            if gravity_state_json:
                store.sim.set_state(json.loads(gravity_state_json))
                print(f"[MemoryStore] Restored gravity field state ({len(store.sim.concepts)} concepts)")
                
            # Restore traces
            traces = backend.load_traces()
            for t in traces:
                store.traces[t.trace_id] = t
                store.index.upsert(t.trace_id, t.vec)
                # Gravity sync (if not restored from state)
                if not gravity_state_json or t.trace_id not in store.sim.concepts:
                    store.sim.add_concept(t.trace_id, vec=t.vec)
                    
            # Restore glyphs
            glyphs = backend.load_glyphs()
            for g in glyphs:
                store.glyphs[g.glyph_id] = g
                
            backend.close()
            return store

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
                resolved_text=t_data.get("resolved_text"),
                coref_map=t_data.get("coref_map"),
                span=tuple(t_data["span"]) if t_data.get("span") else None,
                source_file=t_data.get("source_file"),
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
                code_meta=g_data.get("code_meta"),
            )
            store.glyphs[glyph.glyph_id] = glyph

        return store

# Lazy import to avoid circular dependency
try:
    from .storage.sqlite_store import SqliteBackend
except ImportError:
    SqliteBackend = None
