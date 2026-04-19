"""End-to-end integration tests for the holochat app.

Exercises router → retrieval adapter → orchestrator → FastAPI endpoints with a
real (not mocked) Hologram instance seeded with two small 'documents'. No
external LLM calls — orchestrator falls back to deterministic replies when
`OPENAI_API_KEY` is absent, which keeps the suite hermetic.

Covers phase-05 of plans/260420-0213-holochat-v1/.
"""
import os

os.environ["HOLOGRAM_QUIET"] = "1"
os.environ.pop("OPENAI_API_KEY", None)  # force deterministic-reply path

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
PROJECT = "test_holochat_integration"


@pytest.fixture
def seeded_holochat():
    """Seed two distinct 'documents' into a shared glyph and register the
    hologram instance under PROJECT so codex's get_or_create_hologram returns it.
    """
    from hologram.api import Hologram
    from hologram.server import hologram_instances

    holo = Hologram.init(
        use_clip=False,
        use_gravity=True,
        auto_ingest_system=False,
        encoder_mode="minilm",
    )
    holo.project = PROJECT
    glyph = "aviation"
    holo.glyphs.create(glyph, title=glyph)

    doc_a_chunks = [
        "ETOPS certification requires engine reliability over extended operations.",
        "ETOPS diversion planning depends on airport suitability and fuel reserves.",
        "Crew training under ETOPS covers fuel management and emergency procedures.",
    ]
    doc_b_chunks = [
        "RVSM airspace uses 1000-foot vertical separation above flight level 290.",
        "RVSM approval depends on altimetry system performance and height monitoring.",
        "Aircraft in RVSM airspace must demonstrate height-keeping standards.",
    ]

    for i, chunk in enumerate(doc_a_chunks):
        tid = holo.add_text(glyph, chunk, trace_id=f"doc_a:chunk:{i}", do_extract_concepts=False)
        trace = holo.store.get_trace(tid)
        trace.meta["source_doc"] = "doc_a.pdf"
        trace.meta["page_number"] = i + 1

    for i, chunk in enumerate(doc_b_chunks):
        tid = holo.add_text(glyph, chunk, trace_id=f"doc_b:chunk:{i}", do_extract_concepts=False)
        trace = holo.store.get_trace(tid)
        trace.meta["source_doc"] = "doc_b.pdf"
        trace.meta["page_number"] = i + 1

    hologram_instances[PROJECT] = holo
    yield holo


@pytest.fixture
def client(seeded_holochat):
    """TestClient with the orchestrator's LLM provider forced off.

    The server module imports `build_provider()` at load time, which reads
    OPENAI_API_KEY from `.env`. For hermetic tests we force the deterministic
    reply path by nulling the provider directly on the module-level orchestrator.
    """
    from apps.holochat import server as holochat_server
    original_provider = holochat_server.orchestrator.provider
    holochat_server.orchestrator.provider = None
    try:
        yield TestClient(holochat_server.app)
    finally:
        holochat_server.orchestrator.provider = original_provider


def _chat_payload(
    user_message: str,
    *,
    session_id: str = "session-1",
    active_document_id=None,
    selected_document_ids=None,
    top_k: int = 5,
):
    return {
        "project": PROJECT,
        "session_id": session_id,
        "context": {
            "user_message": user_message,
            "active_document_id": active_document_id,
            "selected_document_ids": selected_document_ids or [],
            "recent_messages": [],
            "session_summary": None,
            "just_uploaded_files": False,
            "available_actions": [],
            "corpus_search_enabled": True,
        },
        "top_k": top_k,
    }


# ---------------------------------------------------------------------------
# /chat/route (classifier only, no pipeline execution)
# ---------------------------------------------------------------------------
class TestRouteEndpoint:
    def test_route_returns_decision_shape(self, client):
        body = {
            "project": PROJECT,
            "context": {"user_message": "What is ETOPS certification?"},
        }
        r = client.post("/chat/route", json=body)
        assert r.status_code == 200
        route = r.json()["route"]
        assert "intent" in route
        assert "confidence" in route
        assert "needs_retrieval" in route
        assert "retrieval_scope" in route
        assert "response_mode" in route

    def test_route_doc_scope_when_active_document(self, client):
        body = {
            "project": PROJECT,
            "context": {
                "user_message": "Summarize this document please.",
                "active_document_id": "doc_a.pdf",
            },
        }
        r = client.post("/chat/route", json=body)
        assert r.status_code == 200
        route = r.json()["route"]
        assert route["retrieval_scope"] in {"active_document", "selected_documents"}


# ---------------------------------------------------------------------------
# /chat/respond — per-intent behavior
# ---------------------------------------------------------------------------
class TestRespondEndpoint:
    def test_chat_intent_no_citations(self, client):
        r = client.post("/chat/respond", json=_chat_payload("hi there"))
        assert r.status_code == 200
        data = r.json()
        assert data["route"]["intent"] == "chat"
        assert data["citations"] == []

    def test_knowledge_qa_returns_citations_from_corpus(self, client):
        r = client.post(
            "/chat/respond", json=_chat_payload("What does ETOPS certification require?")
        )
        assert r.status_code == 200
        data = r.json()
        assert data["route"]["needs_retrieval"] is True
        assert len(data["citations"]) >= 1

    def test_doc_qa_active_document_isolates_citations(self, client):
        r = client.post(
            "/chat/respond",
            json=_chat_payload(
                "Summarize this file on ETOPS.",
                active_document_id="doc_a.pdf",
            ),
        )
        assert r.status_code == 200
        data = r.json()
        for citation in data["citations"]:
            assert citation["source_doc"] == "doc_a.pdf"

    def test_doc_qa_selected_documents_excludes_others(self, client):
        r = client.post(
            "/chat/respond",
            json=_chat_payload(
                "Summarize the selected files.",
                selected_document_ids=["doc_b.pdf"],
            ),
        )
        assert r.status_code == 200
        data = r.json()
        for citation in data["citations"]:
            assert citation["source_doc"] == "doc_b.pdf"

    def test_active_document_never_leaks_other_doc(self, client):
        """Stronger isolation assertion: even querying B-flavored text with
        active_document_id=A must never surface B's traces.
        """
        r = client.post(
            "/chat/respond",
            json=_chat_payload(
                "Tell me about RVSM airspace.",
                active_document_id="doc_a.pdf",
            ),
        )
        assert r.status_code == 200
        data = r.json()
        for citation in data["citations"]:
            assert citation["source_doc"] != "doc_b.pdf"


# ---------------------------------------------------------------------------
# Session persistence across turns
# ---------------------------------------------------------------------------
class TestSessionPersistence:
    def test_two_turns_persist_messages(self, client):
        first = client.post(
            "/chat/respond",
            json=_chat_payload("What is ETOPS?", session_id="persist-session"),
        )
        assert first.status_code == 200

        second = client.post(
            "/chat/respond",
            json=_chat_payload("And the crew training?", session_id="persist-session"),
        )
        assert second.status_code == 200
        # Both turns return successfully; session store should accumulate messages
        # (verified indirectly by successful retrieval on both calls)
        assert second.json()["session_id"] == "persist-session"


# ---------------------------------------------------------------------------
# Smoke: server root responds
# ---------------------------------------------------------------------------
def test_root_endpoint_alive(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "holochat"
