# hologram/glyph_router.py
"""
Glyph-routed retrieval — routes queries through glyph-conditioned subspaces.

Doc spec (north-star): query → infer p(g|q) → transform via T_g → search glyph shards
                       → cross-glyph merge → optional global fallback → results

Phase 1: identity transforms (GlyphOperator is no-op), proves routing thesis.
Phase 2+: real R_g + P_k transforms swap in without changing this module's logic.
"""
import numpy as np
import faiss
from typing import Dict, List, Tuple, Optional

from .glyph_operator import GlyphOperator
from .store import MemoryStore, Trace
from .glyphs import GlyphRegistry


class GlyphShardIndex:
    """Per-glyph FAISS index storing operator-transformed trace vectors."""

    def __init__(self, glyph_id: str, operator: GlyphOperator):
        self.glyph_id = glyph_id
        self.operator = operator
        self.index: Optional[faiss.Index] = None
        self.trace_ids: List[str] = []

    def build(self, traces: List[Trace]) -> None:
        """Build FAISS index from traces, applying operator transform."""
        vecs_with_ids = []
        for t in traces:
            if t is not None and t.vec is not None:
                transformed = self.operator.transform_trace(t.vec)
                vecs_with_ids.append((t.trace_id, transformed))

        if not vecs_with_ids:
            self.index = None
            self.trace_ids = []
            return

        self.trace_ids = [tid for tid, _ in vecs_with_ids]
        mat = np.stack([v for _, v in vecs_with_ids]).astype("float32")
        # Normalize for cosine similarity via inner product
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        mat /= norms

        self.index = faiss.IndexFlatIP(self.operator.output_dim)
        self.index.add(mat)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search this shard with an already-transformed query vector."""
        if self.index is None or len(self.trace_ids) == 0:
            return []

        q = query_vec.astype("float32").reshape(1, -1)
        q /= (np.linalg.norm(q) + 1e-8)

        k = min(top_k, len(self.trace_ids))
        D, I = self.index.search(q, k)

        results = []
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(self.trace_ids):
                results.append((self.trace_ids[idx], float(score)))
        return results


class GlyphRouter:
    """Routes queries through glyph-conditioned subspaces for targeted retrieval."""

    def __init__(self, store: MemoryStore, glyphs: GlyphRegistry,
                 gravity_field=None, use_projection: bool = True,
                 projection_k: int = None):
        self._store = store
        self._glyphs = glyphs
        self._gravity = gravity_field
        self._use_projection = use_projection
        self._projection_k = projection_k
        self._operators: Dict[str, GlyphOperator] = {}
        self._shards: Dict[str, GlyphShardIndex] = {}
        self._dirty: bool = True

    def invalidate(self) -> None:
        """Mark shards dirty — call after trace add/remove."""
        self._dirty = True

    def infer_glyphs(self, query_vec: np.ndarray, top_n: int = 2,
                     min_score: float = 0.0) -> Dict[str, float]:
        """Predict glyph distribution p(g|q) using raw cosine resonance.

        Always returns at least top-1 glyph (the routing question is
        "which shard is most likely" not "is any shard likely").
        Additional glyphs beyond top-1 are filtered by min_score.
        """
        scores = self._glyphs.resonance_score(query_vec, normalize=False)
        if not scores:
            return {}

        sorted_glyphs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Always include top-1 (best available shard)
        result = {sorted_glyphs[0][0]: sorted_glyphs[0][1]}
        # Additional glyphs filtered by min_score
        for g, s in sorted_glyphs[1:top_n]:
            if s >= min_score:
                result[g] = s
        return result

    def _ensure_shards(self) -> None:
        """Lazy-build all shard indexes if dirty."""
        if not self._dirty:
            return

        all_glyphs = self._store.get_all_glyphs()
        dim = getattr(self._store, "dim", 384)
        # Infer dim from first trace if available
        for g in all_glyphs:
            for tid in g.trace_ids:
                t = self._store.get_trace(tid)
                if t is not None and t.vec is not None:
                    dim = len(t.vec)
                    break
            if dim != 384:
                break

        self._operators.clear()
        self._shards.clear()

        for g in all_glyphs:
            # Create glyph-specific operator with rotation + projection
            op = GlyphOperator(g.glyph_id, dim, k=self._projection_k,
                               use_projection=self._use_projection)
            self._operators[g.glyph_id] = op

            # Collect traces for this glyph
            traces = [self._store.get_trace(tid) for tid in g.trace_ids]
            traces = [t for t in traces if t is not None]

            # Build shard index
            shard = GlyphShardIndex(g.glyph_id, op)
            shard.build(traces)
            self._shards[g.glyph_id] = shard

        self._dirty = False

    def search_routed(self, query_vec: np.ndarray, top_k: int = 5,
                      top_glyphs: int = 2, fallback_global: bool = True,
                      min_glyph_score: float = 0.0) -> List[Tuple[str, float]]:
        """
        Doc-faithful retrieval flow:
        1. Infer p(g|q) via resonance_score (always includes top-1)
        2. For each top glyph: transform query via operator, search shard
        3. Rank by raw within-shard cosine (glyph weight for routing only)
        4. Deduplicate across shards (keep best score per trace)
        5. Global fallback via store trace index if results < top_k
        """
        self._ensure_shards()

        # 1. Infer top glyphs
        glyph_weights = self.infer_glyphs(query_vec, top_n=top_glyphs,
                                          min_score=min_glyph_score)

        # 2-3. Search each glyph shard with transformed query
        # Glyph weight is used for ROUTING (which shards to search), not scoring.
        # Within-shard cosine similarity is the ranking signal.
        best_scores: Dict[str, float] = {}
        for glyph_id, glyph_weight in glyph_weights.items():
            shard = self._shards.get(glyph_id)
            op = self._operators.get(glyph_id)
            if shard is None or op is None:
                continue

            transformed_q = op.transform_query(query_vec)
            hits = shard.search(transformed_q, top_k=top_k)

            for trace_id, cos_sim in hits:
                # 4. Deduplicate: keep best score per trace (raw cosine, no weight)
                if trace_id not in best_scores or cos_sim > best_scores[trace_id]:
                    best_scores[trace_id] = cos_sim

        # Sort by score
        results = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)

        # 5. Global fallback via store's trace index (not gravity field)
        # Gravity field contains concept/glyph IDs, not just traces —
        # using store.search_traces() keeps return type consistent
        if fallback_global and len(results) < top_k:
            existing_ids = {tid for tid, _ in results}
            global_hits = self._store.search_traces(query_vec, top_k=top_k)
            for trace_id, score in global_hits:
                if trace_id not in existing_ids:
                    results.append((trace_id, score))
                    if len(results) >= top_k:
                        break

        return results[:top_k]
