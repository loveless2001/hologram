from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


BBox = Tuple[float, float, float, float]


@dataclass
class ParsedImage:
    path: str
    page_number: int
    image_index: int
    bbox: Optional[BBox] = None
    caption_text: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ParsedPage:
    page_number: int
    text: str
    blocks: List[Dict[str, object]] = field(default_factory=list)
    images: List[ParsedImage] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    source_path: str
    doc_id: str
    pages: List[ParsedPage]
    metadata: Dict[str, object] = field(default_factory=dict)
