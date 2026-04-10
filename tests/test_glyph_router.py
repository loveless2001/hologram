# tests/test_glyph_router.py
"""Unit tests for GlyphOperator, GlyphShardIndex, and GlyphRouter."""
import numpy as np
import pytest
from hologram.api import Hologram
from hologram.glyph_operator import GlyphOperator
from hologram.glyph_router import GlyphRouter, GlyphShardIndex, RoutingDecision
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
        op = GlyphOperator("test", DIM, use_projection=False)
        vec = np.random.randn(DIM).astype("float32")
        assert np.array_equal(op.transform_query(vec), vec)

    def test_identity_transform_trace(self):
        op = GlyphOperator("test", DIM, use_projection=False)
        vec = np.random.randn(DIM).astype("float32")
        assert np.array_equal(op.transform_trace(vec), vec)

    def test_identity_output_dim(self):
        op = GlyphOperator("test", 384, use_projection=False)
        assert op.output_dim == 384

    def test_projection_reduces_dim(self):
        op = GlyphOperator("test", DIM, k=8, use_projection=True)
        vec = np.random.randn(DIM).astype("float32")
        out = op.transform_query(vec)
        assert len(out) == 8
        assert op.output_dim == 8

    def test_different_glyphs_different_rotations(self):
        vec = np.random.randn(DIM).astype("float32")
        op_a = GlyphOperator("physics", DIM, k=8, use_projection=True)
        op_b = GlyphOperator("biology", DIM, k=8, use_projection=True)
        assert not np.allclose(op_a.transform_query(vec), op_b.transform_query(vec))

    def test_deterministic_rotation(self):
        vec = np.random.randn(DIM).astype("float32")
        op1 = GlyphOperator("test", DIM, k=8, use_projection=True)
        op2 = GlyphOperator("test", DIM, k=8, use_projection=True)
        assert np.allclose(op1.transform_query(vec), op2.transform_query(vec))


class TestGlyphShardIndex:
    def test_build_and_search(self):
        op = GlyphOperator("test", DIM, k=8, use_projection=True)
        shard = GlyphShardIndex("test", op)

        traces = []
        for i in range(3):
            vec = np.random.randn(DIM).astype("float32")
            traces.append(Trace(trace_id=f"t_{i}", kind="text", content=f"c_{i}", vec=vec))

        shard.build(traces)
        assert shard.index is not None
        assert len(shard.trace_ids) == 3

        # Search with transformed query (shard expects projected vectors)
        q_transformed = op.transform_query(traces[0].vec)
        results = shard.search(q_transformed, top_k=2)
        assert len(results) > 0
        # First result should be the query trace itself (highest cosine)
        assert results[0][0] == "t_0"

    def test_empty_build(self):
        op = GlyphOperator("test", DIM, use_projection=False)
        shard = GlyphShardIndex("test", op)
        shard.build([])
        assert shard.index is None
        assert shard.search(np.zeros(DIM), top_k=3) == []


class TestGlyphRouter:
    def test_hologram_init_defaults_to_same_space_router(self):
        holo = Hologram.init(
            encoder_mode="hash",
            use_clip=False,
            use_gravity=False,
            auto_ingest_system=False,
        )
        assert holo.router is not None
        assert holo.router._use_projection is False

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

    def test_secondary_shard_weight_downweights_non_top_shards(self):
        store = MemoryStore(vec_dim=DIM)
        glyphs = GlyphRegistry(store)
        glyphs.create("g1", title="G1")
        glyphs.create("g2", title="G2")

        q = _make_biased_vec(DIM, 0, 8, bias_strength=6.0)
        g1_vec = q.copy()
        g2_vec = q.copy()
        g2_vec[0] *= 0.8
        g2_vec /= (np.linalg.norm(g2_vec) + 1e-8)

        glyphs.attach_trace("g1", Trace(trace_id="g1_doc", kind="text", content="g1", vec=g1_vec))
        glyphs.attach_trace("g2", Trace(trace_id="g2_doc", kind="text", content="g2", vec=g2_vec))

        router = GlyphRouter(store, glyphs, use_projection=False)
        inferred = list(router.infer_glyphs(q, top_n=2).keys())
        assert inferred == ["g1", "g2"]

        unweighted = router.search_routed(q, top_k=2, top_glyphs=2, secondary_shard_weight=1.0)
        weighted = router.search_routed(q, top_k=2, top_glyphs=2, secondary_shard_weight=0.5)

        assert unweighted[0][0] == "g1_doc"
        assert weighted[0][0] == "g1_doc"
        assert weighted[1][1] < unweighted[1][1]

    def test_shard2_cutoff_filters_weak_secondary_hits(self):
        store = MemoryStore(vec_dim=DIM)
        glyphs = GlyphRegistry(store)
        glyphs.create("g1", title="G1")
        glyphs.create("g2", title="G2")

        q = _make_biased_vec(DIM, 0, 8, bias_strength=6.0)
        g1_a = q.copy()
        g1_b = q.copy()
        g1_b[0] *= 0.98
        g1_b /= (np.linalg.norm(g1_b) + 1e-8)
        g2 = q.copy()
        g2[0] *= 0.1
        g2[8:16] -= 3.0
        g2 /= (np.linalg.norm(g2) + 1e-8)

        glyphs.attach_trace("g1", Trace(trace_id="g1_a", kind="text", content="g1_a", vec=g1_a))
        glyphs.attach_trace("g1", Trace(trace_id="g1_b", kind="text", content="g1_b", vec=g1_b))
        glyphs.attach_trace("g2", Trace(trace_id="g2_a", kind="text", content="g2_a", vec=g2))

        router = GlyphRouter(store, glyphs, use_projection=False)
        unfiltered = router.search_routed(q, top_k=3, top_glyphs=2, fallback_global=False)
        filtered = router.search_routed(
            q,
            top_k=3,
            top_glyphs=2,
            fallback_global=False,
            shard2_cutoff_rank=2,
        )

        assert any(trace_id == "g2_a" for trace_id, _ in unfiltered)
        assert all(trace_id != "g2_a" for trace_id, _ in filtered)

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

    def test_decide_routing_rejects_sparse_doc_shards(self):
        """Doc-per-glyph layouts should stay global until shards are nontrivial."""
        store = MemoryStore(vec_dim=DIM)
        glyphs = GlyphRegistry(store)
        router = GlyphRouter(store, glyphs)

        for i in range(10):
            gid = f"doc_{i}"
            glyphs.create(gid, title=gid)
            vec = np.random.randn(DIM).astype("float32")
            t = Trace(trace_id=f"t_{i}", kind="text", content=f"c_{i}", vec=vec)
            glyphs.attach_trace(gid, t)

        decision = router.decide_routing()
        assert not decision.should_route
        assert decision.reason == "too_few_populated_glyphs"

    def test_search_adaptive_matches_global_when_sparse(self):
        store = MemoryStore(vec_dim=DIM)
        glyphs = GlyphRegistry(store)
        router = GlyphRouter(store, glyphs)

        target_vec = _make_biased_vec(DIM, 0, 8, bias_strength=6.0)
        for i in range(6):
            gid = f"doc_{i}"
            glyphs.create(gid, title=gid)
            vec = target_vec.copy() if i == 0 else np.random.randn(DIM).astype("float32")
            t = Trace(trace_id=f"t_{i}", kind="text", content=f"c_{i}", vec=vec)
            glyphs.attach_trace(gid, t)

        global_hits = store.search_traces(target_vec, top_k=3)
        adaptive_hits = router.search_adaptive(target_vec, top_k=3)
        assert adaptive_hits == global_hits

    def test_search_adaptive_routes_when_glyphs_are_populated(self, multi_domain_setup):
        store, glyphs = multi_domain_setup
        router = GlyphRouter(store, glyphs)

        decision = router.decide_routing()
        assert decision.should_route

        q = _make_biased_vec(DIM, 0, 8, bias_strength=5.0)
        results = router.search_adaptive(q, top_k=3)
        phys_count = sum(1 for tid, _ in results if tid.startswith("phys_"))
        assert phys_count >= 2, f"Expected adaptive routing to use shard path, got {results}"

    def test_search_dynamic_prefers_global_pca_for_small_corpus(self, monkeypatch):
        holo = Hologram.init(
            encoder_mode="hash",
            use_clip=False,
            use_gravity=False,
            auto_ingest_system=False,
        )
        holo.store.traces = {f"t{i}": None for i in range(10)}

        monkeypatch.setattr(
            holo,
            "search_global_pca",
            lambda query, top_k=5, pca_dim=64: [("pca", 1.0)],
        )

        strategy, results = holo.search_dynamic("test query", optimize_for="balanced")
        assert strategy == "global_pca"
        assert results == [("pca", 1.0)]

    def test_search_dynamic_prefers_routed_for_large_routable_corpus(self, monkeypatch):
        holo = Hologram.init(
            encoder_mode="hash",
            use_clip=False,
            use_gravity=False,
            auto_ingest_system=False,
        )
        holo.store.traces = {f"t{i}": None for i in range(6000)}
        monkeypatch.setattr(
            holo.router,
            "decide_routing",
            lambda **kwargs: RoutingDecision(
                should_route=True,
                reason="route",
                total_glyphs=10,
                routable_glyphs=5,
                total_traces=6000,
                min_routable_glyphs=2,
                min_traces_per_glyph=3,
                min_total_traces=8,
            ),
        )
        monkeypatch.setattr(
            holo,
            "search_routed",
            lambda query, top_k=5, top_glyphs=2, secondary_shard_weight=1.0, shard2_cutoff_rank=None: [("routed", 1.0)],
        )

        strategy, results = holo.search_dynamic("test query", optimize_for="speed")
        assert strategy == "routed"
        assert results == [("routed", 1.0)]

    def test_search_dynamic_quality_mode_uses_global(self, monkeypatch):
        holo = Hologram.init(
            encoder_mode="hash",
            use_clip=False,
            use_gravity=False,
            auto_ingest_system=False,
        )
        monkeypatch.setattr(holo, "search_text", lambda query, top_k=5: [("global", 1.0)])

        strategy, results = holo.search_dynamic("test query", optimize_for="quality")
        assert strategy == "global"
        assert results == [("global", 1.0)]
