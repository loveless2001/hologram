"""Tests for Hologram.search_scoped() — scoped retrieval helper.

Covers filter semantics (glyph_ids, doc_ids, trace_filter), mode dispatch,
oversampling behavior, and edge cases. Added for the holochat v1 integration
seam (plans/260420-0213-holochat-v1/phase-02).
"""
import os

os.environ["HOLOGRAM_QUIET"] = "1"

import pytest

from hologram.api import Hologram


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def holo_with_two_docs(isolated_hologram):
    """Seed a hologram with two distinct 'documents' under the same glyph.

    Each 'document' is a handful of text chunks carrying a distinct source_doc
    value in trace metadata — mirrors how ingest_parsed_document populates traces.
    """
    holo = isolated_hologram
    glyph = "aviation"
    holo.glyphs.create(glyph, title=glyph)

    doc_a_chunks = [
        "ETOPS requires twin-engine aircraft to operate safely away from diversion airports.",
        "Certification of ETOPS requires demonstration of engine reliability over extended periods.",
        "Crew training under ETOPS covers diversion procedures and fuel reserves.",
    ]
    doc_b_chunks = [
        "RVSM airspace requires vertical separation of 1000 feet between aircraft above FL290.",
        "RVSM approval depends on altimetry system performance and monitoring.",
        "Aircraft operating in RVSM airspace must maintain height-keeping performance standards.",
    ]

    for i, chunk in enumerate(doc_a_chunks):
        tid = holo.add_text(
            glyph, chunk, trace_id=f"doc_a:chunk:{i}", do_extract_concepts=False
        )
        trace = holo.store.get_trace(tid)
        trace.meta["source_doc"] = "doc_a.pdf"
        trace.meta["page_number"] = i + 1

    for i, chunk in enumerate(doc_b_chunks):
        tid = holo.add_text(
            glyph, chunk, trace_id=f"doc_b:chunk:{i}", do_extract_concepts=False
        )
        trace = holo.store.get_trace(tid)
        trace.meta["source_doc"] = "doc_b.pdf"
        trace.meta["page_number"] = i + 1

    return holo


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------
class TestModeDispatch:
    def test_global_mode_runs(self, holo_with_two_docs):
        results = holo_with_two_docs.search_scoped(
            "ETOPS certification", top_k=3, mode="global"
        )
        assert isinstance(results, list)
        assert len(results) <= 3
        assert all(len(r) == 2 for r in results)

    def test_global_pca_mode_runs(self, holo_with_two_docs):
        results = holo_with_two_docs.search_scoped(
            "ETOPS certification", top_k=3, mode="global_pca"
        )
        assert len(results) <= 3

    def test_dynamic_mode_runs(self, holo_with_two_docs):
        results = holo_with_two_docs.search_scoped(
            "ETOPS certification", top_k=3, mode="dynamic"
        )
        assert len(results) <= 3

    def test_invalid_mode_raises(self, holo_with_two_docs):
        with pytest.raises(ValueError, match="mode must be one of"):
            holo_with_two_docs.search_scoped("query", mode="nonsense")

    def test_no_filter_global_matches_search_text(self, holo_with_two_docs):
        """With no filters, global mode should return the same traces as search_text."""
        scoped = holo_with_two_docs.search_scoped(
            "ETOPS certification", top_k=3, mode="global"
        )
        baseline = holo_with_two_docs.search_text("ETOPS certification", top_k=3)
        scoped_ids = [t.trace_id for t, _ in scoped]
        baseline_ids = [t.trace_id for t, _ in baseline]
        assert scoped_ids == baseline_ids


# ---------------------------------------------------------------------------
# doc_ids filter — the chatbot's primary use case
# ---------------------------------------------------------------------------
class TestDocIdsFilter:
    def test_doc_ids_single_excludes_other_doc(self, holo_with_two_docs):
        results = holo_with_two_docs.search_scoped(
            "aviation regulation", top_k=5, doc_ids=["doc_a.pdf"], mode="global"
        )
        for trace, _ in results:
            assert trace.meta.get("source_doc") == "doc_a.pdf"

    def test_doc_ids_other_doc(self, holo_with_two_docs):
        results = holo_with_two_docs.search_scoped(
            "aviation regulation", top_k=5, doc_ids=["doc_b.pdf"], mode="global"
        )
        for trace, _ in results:
            assert trace.meta.get("source_doc") == "doc_b.pdf"

    def test_doc_ids_multi_includes_all_listed(self, holo_with_two_docs):
        results = holo_with_two_docs.search_scoped(
            "aviation regulation",
            top_k=6,
            doc_ids=["doc_a.pdf", "doc_b.pdf"],
            mode="global",
        )
        seen = {t.meta.get("source_doc") for t, _ in results}
        assert seen.issubset({"doc_a.pdf", "doc_b.pdf"})

    def test_doc_ids_missing_doc_returns_empty(self, holo_with_two_docs):
        results = holo_with_two_docs.search_scoped(
            "aviation regulation", top_k=5, doc_ids=["does_not_exist.pdf"], mode="global"
        )
        assert results == []


# ---------------------------------------------------------------------------
# glyph_ids filter
# ---------------------------------------------------------------------------
class TestGlyphIdsFilter:
    def test_glyph_ids_restricts_to_attached_traces(self, holo_with_two_docs):
        holo = holo_with_two_docs
        holo.glyphs.create("unrelated", title="unrelated")
        holo.add_text(
            "unrelated",
            "Completely unrelated content about gardening and tomatoes.",
            trace_id="unrelated:chunk:0",
            do_extract_concepts=False,
        )

        results = holo.search_scoped(
            "aviation regulation",
            top_k=5,
            glyph_ids=["aviation"],
            mode="global",
        )
        aviation_trace_ids = set(holo.store.get_glyph("aviation").trace_ids)
        for trace, _ in results:
            assert trace.trace_id in aviation_trace_ids

    def test_glyph_ids_unknown_glyph_returns_empty(self, holo_with_two_docs):
        results = holo_with_two_docs.search_scoped(
            "aviation", top_k=5, glyph_ids=["no_such_glyph"], mode="global"
        )
        assert results == []


# ---------------------------------------------------------------------------
# trace_filter escape hatch
# ---------------------------------------------------------------------------
class TestTraceFilter:
    def test_trace_filter_by_page_number(self, holo_with_two_docs):
        results = holo_with_two_docs.search_scoped(
            "ETOPS",
            top_k=5,
            trace_filter=lambda t: t.meta.get("page_number") == 1,
            mode="global",
        )
        for trace, _ in results:
            assert trace.meta.get("page_number") == 1

    def test_trace_filter_never_matches_returns_empty(self, holo_with_two_docs):
        results = holo_with_two_docs.search_scoped(
            "ETOPS",
            top_k=5,
            trace_filter=lambda t: False,
            mode="global",
        )
        assert results == []


# ---------------------------------------------------------------------------
# Combined filters
# ---------------------------------------------------------------------------
class TestCombinedFilters:
    def test_doc_ids_plus_trace_filter(self, holo_with_two_docs):
        results = holo_with_two_docs.search_scoped(
            "aviation",
            top_k=5,
            doc_ids=["doc_a.pdf"],
            trace_filter=lambda t: t.meta.get("page_number", 0) <= 2,
            mode="global",
        )
        for trace, _ in results:
            assert trace.meta.get("source_doc") == "doc_a.pdf"
            assert trace.meta.get("page_number", 0) <= 2

    def test_all_three_filters_combined(self, holo_with_two_docs):
        results = holo_with_two_docs.search_scoped(
            "aviation",
            top_k=5,
            glyph_ids=["aviation"],
            doc_ids=["doc_b.pdf"],
            trace_filter=lambda t: t.meta.get("page_number", 0) >= 2,
            mode="global",
        )
        for trace, _ in results:
            assert trace.meta.get("source_doc") == "doc_b.pdf"
            assert trace.meta.get("page_number", 0) >= 2


# ---------------------------------------------------------------------------
# top_k truncation + oversampling
# ---------------------------------------------------------------------------
class TestTopKTruncation:
    def test_returns_at_most_top_k(self, holo_with_two_docs):
        results = holo_with_two_docs.search_scoped(
            "aviation", top_k=2, mode="global"
        )
        assert len(results) <= 2

    def test_filter_preserves_recall_within_matching_traces(self, holo_with_two_docs):
        """Filters shouldn't starve top_k when enough matching traces exist.

        Codex's scoped helper performs exact cosine over the filtered candidate
        set, so top_k should always be saturated when the doc contains ≥top_k chunks.
        """
        results = holo_with_two_docs.search_scoped(
            "aviation regulation",
            top_k=3,
            doc_ids=["doc_a.pdf"],
            mode="global",
        )
        assert len(results) == 3
