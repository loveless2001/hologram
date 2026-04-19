import os

os.environ["HOLOGRAM_QUIET"] = "1"

import numpy as np
from fastapi.testclient import TestClient

from apps.holochat.orchestrator import ChatOrchestrator
from apps.holochat.router.classifier import PrototypeIntentClassifier
from apps.holochat.router.decision import decide_route
from apps.holochat.schemas import RouterContext
from apps.holochat.server import app
from hologram.api import Hologram


def _dummy_embed(text: str) -> np.ndarray:
    vec = np.zeros(8, dtype="float32")
    lowered = text.lower()
    if "summar" in lowered:
        vec[0] += 1.0
    if "compare" in lowered or "changed" in lowered:
        vec[1] += 1.0
    if "what did we" in lowered or "plan" in lowered:
        vec[2] += 1.0
    if "index" in lowered or "delete" in lowered:
        vec[3] += 1.0
    if "document" in lowered or "section" in lowered:
        vec[4] += 1.0
    if "what" in lowered or "explain" in lowered:
        vec[5] += 1.0
    if not vec.any():
        vec[6] = 1.0
    return vec


def _build_holo() -> Hologram:
    holo = Hologram.init(encoder_mode="hash", use_gravity=False, auto_ingest_system=False)
    holo.glyphs.create("docs", title="Docs")
    return holo


def test_route_prefers_summarization_for_active_document():
    ctx = RouterContext(
        user_message="Summarize this",
        active_document_id="doc:manual",
    )
    classifier = PrototypeIntentClassifier(_dummy_embed)

    decision = decide_route(ctx, classifier)

    assert decision.intent == "summarization"
    assert decision.retrieval_scope == "active_document"


def test_route_falls_back_for_unresolved_reference_without_doc_context():
    ctx = RouterContext(user_message="Summarize this")
    classifier = PrototypeIntentClassifier(_dummy_embed)

    decision = decide_route(ctx, classifier)

    assert decision.intent == "fallback"


def test_orchestrator_uses_active_document_scope():
    holo = _build_holo()
    holo.ingest_document(
        "docs",
        "Runway braking procedures are required before dispatch.",
        sentences_per_chunk=1,
        overlap=0,
        normalize=False,
        base_meta={"source_doc": "doc:runway"},
    )
    holo.ingest_document(
        "docs",
        "Cabin beverage service begins after climb.",
        sentences_per_chunk=1,
        overlap=0,
        normalize=False,
        base_meta={"source_doc": "doc:cabin"},
    )
    orchestrator = ChatOrchestrator()

    response = orchestrator.respond(
        holo,
        "session-1",
        RouterContext(
            user_message="What does this document say about braking?",
            active_document_id="doc:runway",
        ),
        top_k=3,
    )

    assert response.route.intent == "doc_qa"
    assert response.citations
    assert all(citation.source_doc == "doc:runway" for citation in response.citations)
    assert "Grounded answer:" in response.reply


def test_holochat_route_endpoint(monkeypatch):
    holo = _build_holo()
    monkeypatch.setattr("apps.holochat.server.get_or_create_hologram", lambda project: holo)
    client = TestClient(app)

    response = client.post(
        "/chat/route",
        json={
            "project": "demo",
            "context": {
                "user_message": "Summarize this",
                "active_document_id": "doc:manual",
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["route"]["intent"] == "summarization"
