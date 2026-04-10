# tests/test_chunking_and_ingestion.py
"""Tests for text chunking and document ingestion pipeline."""
import os
os.environ["HOLOGRAM_QUIET"] = "1"

import numpy as np
import pytest
from hologram.chunking import split_sentences, chunk_text, Chunk


class TestSplitSentences:
    def test_basic_split(self):
        text = "Hello world. This is a test. Final sentence."
        sents = split_sentences(text)
        assert len(sents) == 3

    def test_empty_text(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []

    def test_single_sentence(self):
        sents = split_sentences("Just one sentence.")
        assert len(sents) == 1

    def test_preserves_content(self):
        text = "First sentence. Second sentence."
        sents = split_sentences(text)
        assert "First sentence." in sents[0]
        assert "Second sentence." in sents[1]


class TestChunkText:
    def test_basic_chunking(self):
        text = "A. B. C. D. E. F."
        chunks = chunk_text(text, sentences_per_chunk=2, overlap=0)
        assert len(chunks) == 3
        assert chunks[0].index == 0
        assert chunks[1].index == 1

    def test_overlap(self):
        text = "A. B. C. D. E. F."
        chunks = chunk_text(text, sentences_per_chunk=3, overlap=1)
        # With overlap=1 and step=2, sentences overlap
        assert len(chunks) >= 2
        # Second chunk should share a sentence with first
        assert chunks[1].sentence_start < chunks[0].sentence_end

    def test_metadata_offsets(self):
        text = "First. Second. Third."
        chunks = chunk_text(text, sentences_per_chunk=1, overlap=0)
        for c in chunks:
            # char offsets should point into source text
            assert c.char_start >= 0
            assert c.char_end <= len(text)
            assert c.char_start < c.char_end

    def test_source_hash_consistent(self):
        text = "Hello. World."
        chunks = chunk_text(text, sentences_per_chunk=1, overlap=0)
        hashes = {c.source_hash for c in chunks}
        assert len(hashes) == 1  # all chunks share same source hash

    def test_empty_text(self):
        assert chunk_text("") == []

    def test_single_sentence_chunk(self):
        chunks = chunk_text("Only one.", sentences_per_chunk=3, overlap=1)
        assert len(chunks) == 1
        assert chunks[0].text == "Only one."


class TestIngestDocument:
    @pytest.fixture
    def holo(self):
        from hologram.api import Hologram
        h = Hologram.init(encoder_mode="hash", use_gravity=True,
                          auto_ingest_system=False)
        h.glyphs.create("test", title="Test")
        return h

    def test_basic_ingest(self, holo):
        text = "Alpha sentence. Beta sentence. Gamma sentence."
        results = holo.ingest_document("test", text, sentences_per_chunk=2,
                                       overlap=0, normalize=False)
        assert len(results) > 0
        assert all("trace_id" in r for r in results)
        assert all("chunk_index" in r for r in results)

    def test_chunks_stored_as_traces(self, holo):
        text = "First part. Second part. Third part."
        results = holo.ingest_document("test", text, sentences_per_chunk=1,
                                       overlap=0, normalize=False)
        for r in results:
            trace = holo.store.get_trace(r["trace_id"])
            assert trace is not None
            assert trace.kind == "chunk"

    def test_idempotent_reingest(self, holo):
        text = "Repeat me. Again and again."
        r1 = holo.ingest_document("test", text, normalize=False)
        n_traces_after_first = len(holo.store.traces)
        n_faiss_after_first = holo.store.index.index.ntotal

        r2 = holo.ingest_document("test", text, normalize=False)
        n_traces_after_second = len(holo.store.traces)
        n_faiss_after_second = holo.store.index.index.ntotal

        # Same trace IDs (deterministic from source hash + chunk index)
        assert [x["trace_id"] for x in r1] == [x["trace_id"] for x in r2]
        # FAISS should NOT have duplicate entries
        assert n_faiss_after_second == n_faiss_after_first

    def test_router_invalidated(self, holo):
        if holo.router is None:
            pytest.skip("No router")
        # Force shard build
        holo.router._ensure_shards()
        assert not holo.router._dirty

        holo.ingest_document("test", "New content here.", normalize=False)
        assert holo.router._dirty

    def test_search_after_ingest(self, holo):
        text = "Quantum mechanics describes atomic behavior. Electrons orbit the nucleus."
        holo.ingest_document("test", text, sentences_per_chunk=1,
                             overlap=0, normalize=False)
        results = holo.search_routed("quantum atomic", top_k=2)
        assert len(results) > 0
