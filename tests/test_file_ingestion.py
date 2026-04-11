import os

os.environ["HOLOGRAM_QUIET"] = "1"

from hologram.api import Hologram
from hologram.parsers import ParsedDocument, ParsedImage, ParsedPage


def _build_holo() -> Hologram:
    holo = Hologram.init(encoder_mode="hash", use_gravity=True, auto_ingest_system=False)
    holo.glyphs.create("test", title="Test")
    return holo


def _sample_parsed_document() -> ParsedDocument:
    return ParsedDocument(
        source_path="/tmp/sample.pdf",
        doc_id="doc:test",
        metadata={"parser": "stub", "revision": "rev-1"},
        pages=[
            ParsedPage(
                page_number=1,
                text="Alpha section. Beta details.",
                metadata={"section": "intro"},
                images=[
                    ParsedImage(
                        path="/tmp/figure-1.png",
                        page_number=1,
                        image_index=1,
                        bbox=(1.0, 2.0, 3.0, 4.0),
                        caption_text="Figure 1",
                        metadata={"image_role": "figure"},
                    )
                ],
            ),
            ParsedPage(
                page_number=2,
                text="Gamma findings.",
                metadata={"section": "results"},
            ),
        ],
    )


def test_ingest_document_preserves_base_metadata():
    holo = _build_holo()

    results = holo.ingest_document(
        "test",
        "Alpha sentence. Beta sentence.",
        sentences_per_chunk=1,
        overlap=0,
        normalize=False,
        base_meta={"source_doc": "doc:test", "page_number": 3, "section": "intro"},
    )

    assert results
    for item in results:
        trace = holo.store.get_trace(item["trace_id"])
        assert trace is not None
        assert trace.meta["source_doc"] == "doc:test"
        assert trace.meta["page_number"] == 3
        assert trace.meta["section"] == "intro"


def test_ingest_parsed_document_ingests_text_and_images():
    holo = _build_holo()
    parsed = _sample_parsed_document()

    result = holo.ingest_parsed_document("test", parsed, normalize=False)

    assert result["doc_id"] == "doc:test"
    assert result["pages_ingested"] == 2
    assert len(result["chunks"]) >= 2
    assert len(result["images"]) == 1

    first_chunk = holo.store.get_trace(result["chunks"][0]["trace_id"])
    assert first_chunk is not None
    assert first_chunk.meta["source_doc"] == "doc:test"
    assert first_chunk.meta["source_path"] == "/tmp/sample.pdf"
    assert first_chunk.meta["page_number"] == 1
    assert first_chunk.meta["parser"] == "stub"
    assert first_chunk.meta["revision"] == "rev-1"
    assert first_chunk.meta["section"] == "intro"

    image_trace = holo.store.get_trace(result["images"][0]["trace_id"])
    assert image_trace is not None
    assert image_trace.kind == "image"
    assert image_trace.meta["source_doc"] == "doc:test"
    assert image_trace.meta["figure_id"] == "doc:test:p1:img1"
    assert image_trace.meta["caption_text"] == "Figure 1"
    assert image_trace.meta["image_role"] == "figure"


def test_ingest_file_dispatches_pdf_parser(monkeypatch, tmp_path):
    holo = _build_holo()
    parsed = _sample_parsed_document()
    pdf_path = tmp_path / "manual.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 stub")
    calls = {}

    def fake_parse_pdf(path: str, image_output_dir=None, extract_images=True):
        calls["path"] = path
        calls["image_output_dir"] = image_output_dir
        calls["extract_images"] = extract_images
        return parsed

    monkeypatch.setattr("hologram.api.parse_pdf", fake_parse_pdf)

    result = holo.ingest_file(
        glyph_id="test",
        path=str(pdf_path),
        normalize=False,
        ingest_images=False,
        image_output_dir=str(tmp_path / "images"),
    )

    assert result["doc_id"] == "doc:test"
    assert calls["path"] == str(pdf_path.resolve())
    assert calls["image_output_dir"] == str((tmp_path / "images").resolve())
    assert calls["extract_images"] is False


def test_ingest_file_rejects_unsupported_suffix(tmp_path):
    holo = _build_holo()
    pptx_path = tmp_path / "slides.pptx"
    pptx_path.write_text("stub", encoding="utf-8")

    try:
        holo.ingest_file(glyph_id="test", path=str(pptx_path))
    except ValueError as exc:
        assert "Unsupported file type" in str(exc)
    else:
        raise AssertionError("Expected ingest_file to reject unsupported suffix")


def test_ingest_file_endpoint_wires_into_hologram(monkeypatch, test_client):
    holo = _build_holo()
    parsed = _sample_parsed_document()

    monkeypatch.setattr("hologram.server.get_or_create_hologram", lambda project: holo)
    monkeypatch.setattr("hologram.api.parse_pdf", lambda *args, **kwargs: parsed)

    response = test_client.post(
        "/ingest/file",
        json={
            "project": "parser-test",
            "path": "/tmp/sample.pdf",
            "glyph_id": "test",
            "normalize": False,
            "ingest_images": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["glyph_id"] == "test"
    assert data["doc_id"] == "doc:test"
    assert data["pages_ingested"] == 2
