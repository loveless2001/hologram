"""Heuristic caption-to-image linking based on page layout proximity.

Scans text blocks near each image bounding box and assigns the closest
caption-like block as the image's caption_text. Works with the block/image
data produced by the PDF parser.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .types import BBox, ParsedImage, ParsedPage

# Patterns that signal a block is a caption (case-insensitive)
_CAPTION_PATTERNS = re.compile(
    r"^(fig(ure|\.)?|image|diagram|table|chart|exhibit|plate|photo)\s*[\d.:]+",
    re.IGNORECASE,
)

# Maximum vertical distance (points) between image bottom edge and candidate
# caption block top edge for a "below" match
_MAX_VERTICAL_GAP_BELOW = 40.0

# Maximum vertical distance (points) between candidate caption block bottom
# edge and image top edge for an "above" match
_MAX_VERTICAL_GAP_ABOVE = 30.0

# Maximum horizontal drift allowed between image center and caption center
_MAX_HORIZONTAL_DRIFT = 150.0


def _bbox_center_x(bbox: BBox) -> float:
    return (bbox[0] + bbox[2]) / 2.0


def _vertical_distance_below(image_bbox: BBox, block_bbox: BBox) -> float:
    """Distance from image bottom to block top (positive = block is below)."""
    return block_bbox[1] - image_bbox[3]


def _vertical_distance_above(image_bbox: BBox, block_bbox: BBox) -> float:
    """Distance from block bottom to image top (positive = block is above)."""
    return image_bbox[1] - block_bbox[3]


def _horizontal_drift(image_bbox: BBox, block_bbox: BBox) -> float:
    return abs(_bbox_center_x(image_bbox) - _bbox_center_x(block_bbox))


def _is_caption_like(text: str) -> bool:
    """Check if text looks like a figure/table caption."""
    stripped = text.strip()
    if not stripped:
        return False
    # Explicit caption pattern match
    if _CAPTION_PATTERNS.match(stripped):
        return True
    # Short descriptive text near an image is also a candidate
    if len(stripped) < 200 and stripped[0].isupper():
        return True
    return False


def _score_candidate(
    image_bbox: BBox,
    block_bbox: BBox,
    block_text: str,
) -> Optional[float]:
    """Score a text block as a caption candidate for an image.

    Lower score is better. Returns None if the block is not a viable candidate.
    """
    h_drift = _horizontal_drift(image_bbox, block_bbox)
    if h_drift > _MAX_HORIZONTAL_DRIFT:
        return None

    # Prefer blocks below the image (most common caption placement)
    v_below = _vertical_distance_below(image_bbox, block_bbox)
    if 0 <= v_below <= _MAX_VERTICAL_GAP_BELOW:
        # Bonus for explicit caption pattern
        pattern_bonus = 0.0 if _CAPTION_PATTERNS.match(block_text.strip()) else 50.0
        return v_below + h_drift * 0.5 + pattern_bonus

    # Also accept blocks above the image (less common but valid)
    v_above = _vertical_distance_above(image_bbox, block_bbox)
    if 0 <= v_above <= _MAX_VERTICAL_GAP_ABOVE:
        pattern_bonus = 0.0 if _CAPTION_PATTERNS.match(block_text.strip()) else 50.0
        # Slight penalty for above-placement (less common)
        return v_above + h_drift * 0.5 + pattern_bonus + 10.0

    return None


def link_captions_on_page(page: ParsedPage) -> None:
    """Assign caption_text to images on a page using layout proximity.

    Mutates each ParsedImage in-place, setting caption_text if a suitable
    nearby text block is found. Blocks already claimed by another image on
    the same page are not reused.
    """
    if not page.images or not page.blocks:
        return

    claimed_block_indices: set = set()

    for image in page.images:
        if image.bbox is None:
            continue
        # Already has a caption (e.g. set by caller) — skip
        if image.caption_text:
            continue

        best_score: Optional[float] = None
        best_block_idx: Optional[int] = None
        best_text: Optional[str] = None

        for block in page.blocks:
            block_idx = block.get("block_index")
            if block_idx in claimed_block_indices:
                continue
            block_bbox = block.get("bbox")
            block_text = block.get("text", "")
            if not block_bbox or not block_text:
                continue
            if not _is_caption_like(block_text):
                continue

            score = _score_candidate(image.bbox, block_bbox, block_text)
            if score is not None and (best_score is None or score < best_score):
                best_score = score
                best_block_idx = block_idx
                best_text = block_text

        if best_text is not None and best_block_idx is not None:
            image.caption_text = best_text.strip()
            claimed_block_indices.add(best_block_idx)


def link_captions(pages: List[ParsedPage]) -> None:
    """Run caption linking across all pages of a parsed document."""
    for page in pages:
        link_captions_on_page(page)
