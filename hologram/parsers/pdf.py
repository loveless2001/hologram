from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional

from .types import ParsedDocument, ParsedImage, ParsedPage


def _doc_id_for_path(path: Path) -> str:
    digest = hashlib.blake2b(str(path.resolve()).encode("utf-8"), digest_size=8).hexdigest()
    return f"doc:{digest}"


def parse_pdf(
    path: str,
    image_output_dir: Optional[str] = None,
    extract_images: bool = True,
) -> ParsedDocument:
    """Parse a PDF into page text plus extracted images.

    Requires `PyMuPDF` (`fitz`) at runtime. The dependency is optional so the
    wider project can still run without PDF ingestion support installed.
    """
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised only when dependency missing
        raise RuntimeError(
            "PDF ingestion requires PyMuPDF. Install it with `pip install pymupdf`."
        ) from exc

    pdf_path = Path(path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    doc_id = _doc_id_for_path(pdf_path)
    output_dir = Path(image_output_dir).expanduser().resolve() if image_output_dir else (
        pdf_path.parent / ".hologram_extracted_images" / doc_id
    )
    if extract_images:
        output_dir.mkdir(parents=True, exist_ok=True)

    pages: List[ParsedPage] = []
    for page_idx, page in enumerate(doc, start=1):
        blocks = []
        raw_blocks = page.get_text("blocks")
        raw_blocks = sorted(raw_blocks, key=lambda block: (block[1], block[0]))
        text_parts: List[str] = []
        for block_idx, block in enumerate(raw_blocks):
            x0, y0, x1, y1, text = block[:5]
            block_text = str(text).strip()
            if not block_text:
                continue
            text_parts.append(block_text)
            blocks.append(
                {
                    "block_index": block_idx,
                    "bbox": (float(x0), float(y0), float(x1), float(y1)),
                    "text": block_text,
                }
            )

        images: List[ParsedImage] = []
        if extract_images:
            for image_idx, image in enumerate(page.get_images(full=True), start=1):
                xref = image[0]
                extracted = doc.extract_image(xref)
                if not extracted:
                    continue
                ext = extracted.get("ext", "png")
                image_name = f"page-{page_idx:04d}-image-{image_idx:03d}.{ext}"
                image_path = output_dir / image_name
                image_path.write_bytes(extracted["image"])
                rects = page.get_image_rects(xref)
                bbox = None
                if rects:
                    rect = rects[0]
                    bbox = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
                images.append(
                    ParsedImage(
                        path=str(image_path),
                        page_number=page_idx,
                        image_index=image_idx,
                        bbox=bbox,
                    )
                )

        pages.append(
            ParsedPage(
                page_number=page_idx,
                text="\n\n".join(text_parts).strip(),
                blocks=blocks,
                images=images,
                metadata={"source_path": str(pdf_path)},
            )
        )

    # Run heuristic caption linking across all pages
    from .caption_linker import link_captions
    link_captions(pages)

    return ParsedDocument(
        source_path=str(pdf_path),
        doc_id=doc_id,
        pages=pages,
        metadata={"parser": "pymupdf", "page_count": len(pages)},
    )
