# tests/test_glyph_router.py
"""Unit tests for GlyphOperator, GlyphShardIndex, and GlyphRouter."""
import numpy as np
import pytest
from hologram.glyph_operator import GlyphOperator
from hologram.glyph_router import GlyphRouter, GlyphShardIndex
from hologram.store import MemoryStore, Trace
from hologram.glyphs import GlyphRegistry


DIM = 32


def _make_biased_vec(dim, bias_start, bias_end, bias_strength=3.0):
    """Create a random vector biased toward specific dimensions."""
    vec = np.random.randn(dim).astype("float32")
    vec[bias_start:bias_end] += bias_strength
    return vec


@pytest.fixture
def multi_domain_setup():
    """Create a store with 2 glyphs and domain-separated traces."""
    store = MemoryStore(vec_dim=DIM)
    glyphs = GlyphRegistry(store)
    glyphs.create("physics", title="Physics")
    glyphs.create("biology", title="Biology")

    # Physics traces: biased toward dims 0-8
    for i in range(5):
        vec = _make_biased_vec(DIM, 0, 8)
        t = Trace(trace_id=f"phys_{i}", kind="text", content=f"physics_{i}", vec=vec)
        glyphs.attach_trace("physics", t)

    # Biology traces: biased toward dims 16-24
    for i in range(5):
        vec = _make_biased_vec(DIM, 16, 24)
        t = Trace(trace_id=f"bio_{i}", kind="text", content=f"biology_{i}", vec=vec)
        glyphs.attach_trace("biology", t)

    return store, glyphs


class TestGlyphOperator:
    def test_identity_transform_query(self):
        op = GlyphOperator("test", DIM)
        vec = np.random.randn(DIM).astype("float32")
        assert np.array_equal(op.transform_query(vec), vec)

    def test_identity_transform_trace(self):
        op = GlyphOperator("test", DIM)
        vec = np.random.randn(DIM).astype("float32")
        assert np.array_equal(op.transform_trace(vec), vec)

    def test_output_dim(self):
        op = GlyphOperator("test", 384)
        assert op.output_dim == 384


class TestGlyphShardIndex:
    def test_build_and_search(self):
        op = GlyphOperator("test", DIM)
        shard = GlyphShardIndex("test", op)

        traces = []
        for i in range(3):
            vec = np.random.randn(DIM).astype("float32")
            traces.append(Trace(trace_id=f"t_{i}", kind="text", content=f"c_{i}", vec=vec))

        shard.build(traces)
        assert shard.index is not None
        assert len(shard.trace_ids) == 3

        results = shard.search(traces[0].vec, top_k=2)
        assert len(results) > 0
        # First result should be the query trace itself (highest cosine)
        assert results[0][0] == "t_0"

    def test_empty_build(self):
        op = GlyphOperator("test", DIM)
        shard = GlyphShardIndex("test", op)
        shard.build([])
        assert shard.index is None
        assert shard.search(np.zeros(DIM), top_k=3) == []


class TestGlyphRouter:
    def test_infer_glyphs(self, multi_domain_setup):
        store, glyphs = multi_domain_setup
        router = GlyphRouter(store, glyphs)

        # Physics-biased query should rank physics glyph higher
        q = _make_biased_vec(DIM, 0, 8)
        inferred = router.infer_glyphs(q, top_n=2)
        assert len(inferred) > 0

    def test_search_routed_returns_results(self, multi_domain_setup):
        store, glyphs = multi_domain_setup
        router = GlyphRouter(store, glyphs)

        q = _make_biased_vec(DIM, 0, 8)
        results = router.search_routed(q, top_k=5)
        assert len(results) > 0

    def test_routing_prefers_correct_domain(self, multi_domain_setup):
        store, glyphs = multi_domain_setup
        router = GlyphRouter(store, glyphs)

        # Physics query
        q = _make_biased_vec(DIM, 0, 8, bias_strength=5.0)
        results = router.search_routed(q, top_k=3)
        phys_count = sum(1 for tid, _ in results if tid.startswith("phys_"))
        assert phys_count >= 2, f"Expected mostly physics, got {results}"

        # Biology query
        q = _make_biased_vec(DIM, 16, 24, bias_strength=5.0)
        results = router.search_routed(q, top_k=3)
        bio_count = sum(1 for tid, _ in results if tid.startswith("bio_"))
        assert bio_count >= 2, f"Expected mostly biology, got {results}"

    def test_invalidation(self, multi_domain_setup):
        store, glyphs = multi_domain_setup
        router = GlyphRouter(store, glyphs)

        # Force shard build
        router.search_routed(np.random.randn(DIM).astype("float32"), top_k=1)
        assert not router._dirty

        router.invalidate()
        assert router._dirty

    def test_single_glyph(self):
        """Single-glyph project should still route correctly."""
        store = MemoryStore(vec_dim=DIM)
        glyphs = GlyphRegistry(store)
        glyphs.create("only", title="Only")

        for i in range(3):
            vec = np.random.randn(DIM).astype("float32")
            t = Trace(trace_id=f"t_{i}", kind="text", content=f"c_{i}", vec=vec)
            glyphs.attach_trace("only", t)

        router = GlyphRouter(store, glyphs)
        results = router.search_routed(np.random.randn(DIM).astype("float32"), top_k=3)
        assert len(results) > 0

    def test_fallback_returns_trace_ids_only(self, multi_domain_setup):
        """Fallback should return trace IDs, not concept/glyph IDs."""
        store, glyphs = multi_domain_setup
        router = GlyphRouter(store, glyphs)

        # Force fallback by setting impossibly high min_glyph_score
        results = router.search_routed(
            np.random.randn(DIM).astype("float32"),
            top_k=5, min_glyph_score=99.0
        )
        for tid, _ in results:
            assert not tid.startswith("glyph:"), f"Got glyph ID in results: {tid}"

    def test_glyph_affinity_equal_weights(self):
        """glyph_affinity should have equal weights for multi-glyph concepts."""
        from hologram.gravity import Gravity

        store = MemoryStore(vec_dim=DIM)
        store.sim = Gravity(dim=DIM)
        glyphs = GlyphRegistry(store)
        glyphs.create("g1", title="G1")
        glyphs.create("g2", title="G2")

        vec = np.random.randn(DIM).astype("float32")
        tid = "shared_trace"

        # Add concept to gravity field first
        store.sim.add_concept(tid, vec=vec)

        # Attach same trace to two glyphs
        t = Trace(trace_id=tid, kind="text", content="shared", vec=vec)
        glyphs.attach_trace("g1", t)
        # Re-create trace for second glyph (same trace_id)
        t2 = Trace(trace_id=tid, kind="text", content="shared", vec=vec)
        glyphs.attach_trace("g2", t2)

        concept = store.sim.concepts[tid]
        assert len(concept.glyph_affinity) == 2
        assert abs(concept.glyph_affinity["g1"] - 0.5) < 1e-9
        assert abs(concept.glyph_affinity["g2"] - 0.5) < 1e-9
