import os

os.environ["HOLOGRAM_QUIET"] = "1"

from hologram.api import Hologram
from hologram.store import Trace


def _build_holo() -> Hologram:
    holo = Hologram.init(encoder_mode="hash", use_gravity=False, auto_ingest_system=False)
    holo.glyphs.create("docs", title="Docs")
    holo.glyphs.create("notes", title="Notes")
    return holo


def test_search_scoped_filters_to_requested_doc_ids():
    holo = _build_holo()
    holo.ingest_document(
        "docs",
        "Runway braking procedures for wet runway operations.",
        sentences_per_chunk=1,
        overlap=0,
        normalize=False,
        base_meta={"source_doc": "doc:runway", "section": "ops"},
    )
    holo.ingest_document(
        "docs",
        "Cabin service notes for beverage preparation.",
        sentences_per_chunk=1,
        overlap=0,
        normalize=False,
        base_meta={"source_doc": "doc:cabin", "section": "service"},
    )

    results = holo.search_scoped(
        "runway braking procedures",
        top_k=5,
        doc_ids=["doc:runway"],
    )

    assert results
    assert all(trace.meta.get("source_doc") == "doc:runway" for trace, _ in results)


def test_search_scoped_applies_trace_filter_after_scope():
    holo = _build_holo()
    holo.ingest_document(
        "docs",
        "Runway braking procedures for wet runway operations.",
        sentences_per_chunk=1,
        overlap=0,
        normalize=False,
        base_meta={"source_doc": "doc:runway", "section": "ops"},
    )
    holo.ingest_document(
        "docs",
        "Runway maintenance checklist for winter inspections.",
        sentences_per_chunk=1,
        overlap=0,
        normalize=False,
        base_meta={"source_doc": "doc:runway", "section": "maintenance"},
    )

    results = holo.search_scoped(
        "runway",
        top_k=5,
        doc_ids=["doc:runway"],
        trace_filter=lambda trace: trace.meta.get("section") == "maintenance",
    )

    assert len(results) == 1
    assert results[0][0].meta["section"] == "maintenance"


def test_search_scoped_narrows_to_specific_glyph_ids():
    holo = _build_holo()
    holo.add_text("docs", "Flight dispatch fuel planning", trace_id="docs:1", skip_nlp=True)
    holo.add_text("notes", "Personal grocery list", trace_id="notes:1", skip_nlp=True)

    results = holo.search_scoped(
        "grocery list",
        top_k=5,
        glyph_ids=["notes"],
    )

    assert results
    assert all(isinstance(trace, Trace) for trace, _ in results)
    assert all(trace.trace_id.startswith("notes:") for trace, _ in results)
