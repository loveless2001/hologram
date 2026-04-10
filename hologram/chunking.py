# hologram/chunking.py
"""
Text chunking for document ingestion.

Splits text into sentence-based chunks with configurable overlap
for context continuity. Chunks include metadata for reconstruction.
"""
import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Chunk:
    """A text chunk with positional metadata."""
    text: str
    index: int              # chunk position in document (0-based)
    char_start: int         # start offset in source text
    char_end: int           # end offset in source text
    sentence_start: int     # first sentence index in this chunk
    sentence_end: int       # last sentence index (exclusive)
    source_hash: str        # hash of full source text for provenance


# Sentence boundary pattern: split on .!? followed by whitespace or end
_SENTENCE_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z"\'])|(?<=[.!?])$', re.MULTILINE
)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation boundaries.

    Handles common abbreviations conservatively — splits on .!?
    followed by whitespace + uppercase letter or end of string.
    """
    text = text.strip()
    if not text:
        return []

    # Split on sentence boundaries
    parts = _SENTENCE_RE.split(text)
    # Filter empty strings and strip whitespace
    sentences = [s.strip() for s in parts if s.strip()]
    return sentences


def chunk_text(text: str, sentences_per_chunk: int = 3,
               overlap: int = 1) -> List[Chunk]:
    """Split text into overlapping chunks of grouped sentences.

    Args:
        text: Source text to chunk
        sentences_per_chunk: Number of sentences per chunk
        overlap: Number of sentences shared between adjacent chunks

    Returns:
        List of Chunk objects with text and metadata
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    source_hash = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    step = max(1, sentences_per_chunk - overlap)
    chunks = []

    # Pre-compute sentence char offsets in source text
    offsets = []
    search_start = 0
    for sent in sentences:
        idx = text.find(sent, search_start)
        if idx == -1:
            idx = search_start  # fallback
        offsets.append((idx, idx + len(sent)))
        search_start = idx + len(sent)

    chunk_idx = 0
    for start in range(0, len(sentences), step):
        end = min(start + sentences_per_chunk, len(sentences))
        chunk_sentences = sentences[start:end]
        chunk_text_str = " ".join(chunk_sentences)

        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]

        chunks.append(Chunk(
            text=chunk_text_str,
            index=chunk_idx,
            char_start=char_start,
            char_end=char_end,
            sentence_start=start,
            sentence_end=end,
            source_hash=source_hash,
        ))
        chunk_idx += 1

        if end >= len(sentences):
            break

    return chunks
