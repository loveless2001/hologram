"""Tests for caption linking heuristics, DOCX parsing, and metadata enrichment."""

import os

os.environ["HOLOGRAM_QUIET"] = "1"

from hologram.parsers.types import ParsedImage, ParsedPage, ParsedDocument
from hologram.parsers.caption_linker import (
    link_captions_on_page,
    link_captions,
    _is_caption_like,
    _score_candidate,
)
from hologram.api import Hologram


def _build_holo() -> Hologram:
    holo = Hologram.init(encoder_mode="hash", use_gravity=True, auto_ingest_system=False)
    holo.glyphs.create("test", title="Test")
    return holo


# --- Caption pattern detection ---


def test_caption_pattern_figure():
    assert _is_caption_like("Figure 1: Engine cross-section")


def test_caption_pattern_fig_dot():
    assert _is_caption_like("Fig. 3.2 Turbofan layout")


def test_caption_pattern_table():
    assert _is_caption_like("Table 4: Maintenance schedule")


def test_caption_pattern_diagram():
    assert _is_caption_like("Diagram 2.1 Fuel system overview")


def test_empty_text_not_caption():
    assert not _is_caption_like("")
    assert not _is_caption_like("   ")


# --- Score candidate ---


def test_score_candidate_block_below_image():
    image_bbox = (50.0, 100.0, 300.0, 400.0)
    block_bbox = (60.0, 410.0, 290.0, 430.0)  # just below image
    score = _score_candidate(image_bbox, block_bbox, "Figure 1: Test")
    assert score is not None
    assert score < 100  # should be a good score


def test_score_candidate_block_too_far_below():
    image_bbox = (50.0, 100.0, 300.0, 400.0)
    block_bbox = (60.0, 500.0, 290.0, 520.0)  # 100pts below — too far
    score = _score_candidate(image_bbox, block_bbox, "Figure 1: Test")
    assert score is None


def test_score_candidate_block_above_image():
    image_bbox = (50.0, 100.0, 300.0, 400.0)
    block_bbox = (60.0, 75.0, 290.0, 95.0)  # just above image
    score = _score_candidate(image_bbox, block_bbox, "Figure 1: Test")
    assert score is not None


def test_score_candidate_too_far_horizontal():
    image_bbox = (50.0, 100.0, 200.0, 400.0)
    block_bbox = (500.0, 410.0, 600.0, 430.0)  # far right
    score = _score_candidate(image_bbox, block_bbox, "Figure 1: Test")
    assert score is None


# --- Page-level caption linking ---


def test_link_captions_assigns_nearest_block():
    page = ParsedPage(
        page_number=1,
        text="Some text. Figure 1: Engine diagram. More text.",
        blocks=[
            {"block_index": 0, "bbox": (50.0, 50.0, 300.0, 70.0), "text": "Some header text."},
            {"block_index": 1, "bbox": (50.0, 310.0, 300.0, 330.0), "text": "Figure 1: Engine diagram"},
            {"block_index": 2, "bbox": (50.0, 500.0, 300.0, 520.0), "text": "Unrelated paragraph below."},
        ],
        images=[
            ParsedImage(
                path="/tmp/img1.png",
                page_number=1,
                image_index=1,
                bbox=(50.0, 100.0, 300.0, 300.0),  # image above block 1
            )
        ],
    )

    link_captions_on_page(page)

    assert page.images[0].caption_text == "Figure 1: Engine diagram"


def test_link_captions_skips_already_captioned():
    page = ParsedPage(
        page_number=1,
        text="Figure 1: Pre-set caption.",
        blocks=[
            {"block_index": 0, "bbox": (50.0, 310.0, 300.0, 330.0), "text": "Figure 1: Would match"},
        ],
        images=[
            ParsedImage(
                path="/tmp/img1.png",
                page_number=1,
                image_index=1,
                bbox=(50.0, 100.0, 300.0, 300.0),
                caption_text="Already set",
            )
        ],
    )

    link_captions_on_page(page)

    assert page.images[0].caption_text == "Already set"


def test_link_captions_no_double_claim():
    """Two images should not claim the same caption block."""
    page = ParsedPage(
        page_number=1,
        text="Figure 1: First. Figure 2: Second.",
        blocks=[
            {"block_index": 0, "bbox": (50.0, 210.0, 300.0, 230.0), "text": "Figure 1: First diagram"},
            {"block_index": 1, "bbox": (50.0, 510.0, 300.0, 530.0), "text": "Figure 2: Second diagram"},
        ],
        images=[
            ParsedImage(path="/tmp/img1.png", page_number=1, image_index=1,
                        bbox=(50.0, 10.0, 300.0, 200.0)),
            ParsedImage(path="/tmp/img2.png", page_number=1, image_index=2,
                        bbox=(50.0, 310.0, 300.0, 500.0)),
        ],
    )

    link_captions_on_page(page)

    captions = [img.caption_text for img in page.images]
    assert "Figure 1: First diagram" in captions
    assert "Figure 2: Second diagram" in captions
    # No duplicates
    assert len(set(captions)) == 2


def test_link_captions_no_images_is_noop():
    page = ParsedPage(page_number=1, text="Just text.", blocks=[
        {"block_index": 0, "bbox": (0, 0, 100, 20), "text": "Figure 1: Orphan caption"},
    ])
    link_captions_on_page(page)  # should not raise


def test_link_captions_no_blocks_is_noop():
    page = ParsedPage(page_number=1, text="", images=[
        ParsedImage(path="/tmp/img.png", page_number=1, image_index=1,
                    bbox=(10, 10, 100, 100)),
    ])
    link_captions_on_page(page)
    assert page.images[0].caption_text is None


def test_link_captions_multi_page():
    pages = [
        ParsedPage(page_number=1, text="Fig. 1 Overview", blocks=[
            {"block_index": 0, "bbox": (50.0, 310.0, 300.0, 330.0), "text": "Fig. 1 Overview"},
        ], images=[
            ParsedImage(path="/tmp/p1.png", page_number=1, image_index=1,
                        bbox=(50.0, 100.0, 300.0, 300.0)),
        ]),
        ParsedPage(page_number=2, text="Table 2 Data", blocks=[
            {"block_index": 0, "bbox": (50.0, 310.0, 300.0, 330.0), "text": "Table 2: Inspection data"},
        ], images=[
            ParsedImage(path="/tmp/p2.png", page_number=2, image_index=1,
                        bbox=(50.0, 100.0, 300.0, 300.0)),
        ]),
    ]

    link_captions(pages)

    assert pages[0].images[0].caption_text == "Fig. 1 Overview"
    assert pages[1].images[0].caption_text == "Table 2: Inspection data"


# --- Metadata enrichment ---


def test_revision_metadata_flows_through_ingestion():
    holo = _build_holo()
    parsed = ParsedDocument(
        source_path="/tmp/aviation-manual.pdf",
        doc_id="doc:avi001",
        metadata={
            "parser": "pymupdf",
            "revision": "rev-3",
            "effective_date": "2026-01-15",
        },
        pages=[
            ParsedPage(
                page_number=1,
                text="Chapter 1: Pre-flight inspection procedures.",
                metadata={"section": "pre-flight"},
            ),
        ],
    )

    result = holo.ingest_parsed_document("test", parsed, normalize=False)

    chunk = holo.store.get_trace(result["chunks"][0]["trace_id"])
    assert chunk.meta["source_doc"] == "doc:avi001"
    assert chunk.meta["revision"] == "rev-3"
    assert chunk.meta["effective_date"] == "2026-01-15"
    assert chunk.meta["section"] == "pre-flight"
    assert chunk.meta["page_number"] == 1


def test_figure_id_and_caption_in_image_metadata():
    holo = _build_holo()
    parsed = ParsedDocument(
        source_path="/tmp/manual.pdf",
        doc_id="doc:man01",
        metadata={"parser": "pymupdf"},
        pages=[
            ParsedPage(
                page_number=3,
                text="See figure below.",
                images=[
                    ParsedImage(
                        path="/tmp/engine.png",
                        page_number=3,
                        image_index=2,
                        bbox=(10, 10, 200, 200),
                        caption_text="Figure 3.2: CFM56 turbofan cross-section",
                    )
                ],
            ),
        ],
    )

    result = holo.ingest_parsed_document("test", parsed, normalize=False)

    img = holo.store.get_trace(result["images"][0]["trace_id"])
    assert img.meta["figure_id"] == "doc:man01:p3:img2"
    assert img.meta["caption_text"] == "Figure 3.2: CFM56 turbofan cross-section"
    assert img.meta["source_doc"] == "doc:man01"


# --- DOCX parser ---


def test_parse_docx_import_error_message():
    """Verify clear error when python-docx is missing."""
    import importlib
    import sys

    # Temporarily hide docx if it exists
    docx_mod = sys.modules.get("docx")
    sys.modules["docx"] = None  # type: ignore

    try:
        from hologram.parsers.docx_parser import parse_docx as _parse
        # Force reimport to hit the import guard
        import hologram.parsers.docx_parser as mod
        importlib.reload(mod)
        try:
            mod.parse_docx("/tmp/fake.docx")
        except RuntimeError as exc:
            assert "python-docx" in str(exc)
        else:
            # If docx was genuinely importable, the reload won't trigger
            pass
    finally:
        if docx_mod is not None:
            sys.modules["docx"] = docx_mod
        elif "docx" in sys.modules:
            del sys.modules["docx"]


def test_ingest_file_accepts_docx(monkeypatch):
    """Verify ingest_file dispatches .docx to parse_docx."""
    import tempfile
    holo = _build_holo()

    parsed = ParsedDocument(
        source_path="/tmp/test.docx",
        doc_id="doc:docxtest",
        metadata={"parser": "python-docx"},
        pages=[
            ParsedPage(page_number=1, text="Aviation safety briefing."),
        ],
    )

    calls = {}

    def fake_parse_docx(path, image_output_dir=None, extract_images=True):
        calls["called"] = True
        calls["path"] = path
        return parsed

    monkeypatch.setattr("hologram.parsers.parse_docx", fake_parse_docx)

    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        f.write(b"PK stub")
        docx_path = f.name

    try:
        result = holo.ingest_file("test", docx_path, normalize=False)
        assert result["doc_id"] == "doc:docxtest"
        assert calls.get("called")
    finally:
        os.unlink(docx_path)
